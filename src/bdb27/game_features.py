from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .data import SCHEMA, regular_plays
from .combine_features import cuts, aggregate_cuts

def game_features(root, chunksize=50000):
    plays=regular_plays(root)
    eligible=plays.loc[plays.route_ran.notna() | plays.lined_up_position.isin(['CB','DB','SS','FS'])]
    wanted=set(map(tuple,eligible[['game_id','play_id','nfl_id']].to_numpy()))
    parts=[]
    # Hash partition by player keeps entire plays together even for interleaved or unsorted CSVs.
    # Only one partition is loaded at a time; temporary filtered shards are removed on exit.
    with tempfile.TemporaryDirectory(prefix='bdb27-shards-',dir=Path.cwd()) as temp:
        for path in sorted(Path(root).glob('game_tracking_*.csv')):
            for chunk in pd.read_csv(path,usecols=SCHEMA['game_tracking'].split(),chunksize=chunksize):
                mask=[key in wanted for key in chunk[['game_id','play_id','nfl_id']].itertuples(index=False,name=None)]
                chunk=chunk.loc[mask]
                for bucket,g in chunk.groupby(chunk.nfl_id%32):
                    out=Path(temp)/f'{int(bucket)}.csv'
                    g.to_csv(out,mode='a',header=not out.exists(),index=False)
        for path in Path(temp).glob('*.csv'):
            shard=pd.read_csv(path)
            for (gid,play,pid),g in shard.groupby(['game_id','play_id','nfl_id']):
                g=g.sort_values('time').reset_index(drop=True)
                snap=g.index[g.event.eq('ball_snap')]; throw=g.index[g.event.eq('pass_forward')]
                if not len(snap) or not len(throw) or throw[-1]<=snap[0]: continue
                c=cuts(g.iloc[snap[0]:throw[-1]+1])
                if len(c): parts.append(c.assign(nfl_id=pid,game_id=gid,play_id=play))
    table=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    return aggregate_cuts(table,'game_'),plays,table

def exposure_outcomes(plays,career):
    rows=[]
    for pid,g in plays.groupby('nfl_id'):
        routes=g.route_ran.notna(); targets=g.target.eq(True) & routes
        seasons=g.season.nunique()
        active=career.loc[career.nfl_id.eq(pid)]
        if active.empty: continue
        c=active.iloc[0]; ratio=c.career_games_started/max(1,c.career_games_active)
        rows.append(dict(nfl_id=pid,route_count=int(routes.sum()),target_count=int(targets.sum()),
            separation=g.loc[targets,'separation_at_pass_forward'].mean(),
            yards_per_route=g.loc[routes,'rec_yards'].fillna(0).sum()/max(1,routes.sum()),
            target_epa=g.loc[targets,'expected_points_added'].mean(),tackles_per_snap=g.tackle.sum()/len(g),
            reg_snaps_per_season=len(g)/seasons,
            # 17 games x 65 team snaps is explicitly an opportunity proxy, not true team snap share.
            snap_share_proxy=(len(g)/seasons)/(17*65),start_ratio=ratio,starter=int(ratio>=.5)))
    return pd.DataFrame(rows)
