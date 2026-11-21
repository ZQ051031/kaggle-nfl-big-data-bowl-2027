from pathlib import Path
import numpy as np
import pandas as pd
from .data import SCHEMA

def trajectory(retention, speed=7., cut=True, noise=0.):
    t = np.arange(51 if cut else int(np.ceil((40/speed+1.1)*10))+1) / 10
    if cut:
        s = speed * (1 - (1-retention)*np.maximum(0, 1-abs(t-2)/.8))
        direction = 90 * np.clip((t-1.7)/.6, 0, 1)
    else:
        s = speed * (1-np.exp(-t))
        direction = np.zeros(len(t))
        finish=np.flatnonzero(np.cumsum(s*.1)>=40)
        if len(finish):
            end=finish[0]+1
            t=t[:end]; s=s[:end]; direction=direction[:end]
    rad = np.deg2rad(direction)
    return pd.DataFrame(dict(time=pd.date_range('2025-03-01', periods=len(t), freq='100ms').astype(str),
        x=30+np.cumsum(s*np.sin(rad)*.1), y=5+np.cumsum(s*np.cos(rad)*.1),
        s=s, a=np.gradient(s,.1), dis=s*.1, dir=direction))

def generate(root, seed=2027, n=150):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    players=[]; results=[]; career=[]; combine=[]; plays=[]; games=[]
    tracking={y:[] for y in [2023,2024,2025]}
    for i in range(n):
        pid=55000+i; year=2023+i%3; pos=['WR','TE','RB','CB','SS'][i%5]
        agility=rng.normal(); fast=rng.normal(); talent=.6*agility+.5*fast+rng.normal(0,.9)
        pick=int(np.clip(140-30*talent+rng.normal(0,55),1,259))
        players.append(dict(nfl_id=pid, display_name=f'Synthetic Prospect {i:03}', draft_year=year,
            nfl_position=pos,birth_date='2001-02-01',college_name='Synthetic University',college_conference='Synthetic',
            draft_round=min(7,1+pick//37) if i%11 else np.nan,draft_pick_within_round=1+pick%32 if i%11 else np.nan,draft_overall_pick=pick if i%11 else np.nan))
        results.append(dict(nfl_id=pid,draft_year=year,combine_position=pos,combine_height=rng.normal(74,2),combine_weight=int(rng.normal(235 if pos=='TE' else 205,12)),hand_size=9.5,arm_length=32.,wing_span=77.,
            ten_yd_split=1.6-.04*fast+rng.normal(0,.04),forty=4.6-.1*fast+rng.normal(0,.08),vertical=34+2*fast+rng.normal(),broad_jump=120+4*fast+rng.normal(0,3),three_cone=7-.05*agility+rng.normal(0,.14) if i%9 else np.nan,short_shuttle=4.3-.03*agility+rng.normal(0,.1),bench_reps=18.,ngs_athleticism_score=60,ngs_college_production_score=65.,ngs_final_score=65.))
        active=(2026-year)*15; ratio=1/(1+np.exp(-(.6*talent+rng.normal(0,.6))))
        career.append(dict(nfl_id=pid,career_offensive_snaps=int(active*35*ratio) if pos in ['WR','TE','RB'] else 0,career_defensive_snaps=int(active*35*ratio) if pos in ['CB','SS'] else 0,career_special_teams_snaps=80,career_games_active=active,career_games_started=int(active*ratio),ap_all_pro_1st_team=0,ap_all_pro_2nd_team=0,pro_bowl_original_ballot=0))
        for j,drill in enumerate(['THREE_CONE_DRILL','SHORT_SHUTTLE','SKILL_DRILLS_'+('DB' if pos in ['CB','SS'] else pos),'FORTY_YARD_DASH']):
            z=trajectory(np.clip(.56+.065*agility+rng.normal(0,.025),.3,.8),speed=(10.7 if j==3 else 7.5)+.3*fast,cut=j!=3)
            z.time=z.time.str.replace('2025-',f'{year}-',regex=False)
            z=z.assign(nfl_id=pid,draft_year=year,event_id=f'{pid}_{j}',entity_type='PLAYER',drill_type=drill,drill_name='FORTY_YARD_DASH' if j==3 else 'SLOT_STRIKE_ROUTE_LEFT',attempt=1)
            combine.append(z)
        for j in range(18):
            season=year+j%(2026-year); gid=season*1000000+91000+i+(30000 if j==17 else 0); playid=j+1
            games.append(dict(game_id=gid,game_key=gid,season=season,season_type='POST' if j==17 else 'REG',week=1+j%18,home_team_abbr='KC',visitor_team_abbr='BUF'))
            target=bool(rng.random()<.65); route=rng.choice(['OUT','SLANT','POST']); cov=rng.choice(['MAN_COVERAGE','ZONE_COVERAGE']); cushion=rng.uniform(3,9)
            sep=max(.1,2.3+.28*agility+.12*fast+.10*cushion+.25*(cov=='ZONE_COVERAGE')+.15*(route=='POST')+rng.normal(0,.85))
            row=dict.fromkeys(SCHEMA['player_play'].split(),0.)
            row.update(game_id=gid,play_id=playid,nfl_id=pid,team_abbr='KC',lined_up_position=pos,play_description='Synthetic pass',quarter=1,down=1+j%3,yards_to_go=10,possession_team='KC',yardline_side='KC',yardline_number=25,game_clock='12:00',pre_snap_home_score=0,pre_snap_visitor_score=0,play_direction='right',pre_snap_home_team_win_probability=.5,pre_snap_visitor_team_win_probability=.5,offense_formation='SHOTGUN',receiver_alignment='2x2',pass_result='C',pass_length=8.,play_nullified_by_penalty='N',penalty_yards=np.nan,pre_penalty_yards_gained=8,yards_gained=8,team_coverage_man_zone=cov,team_coverage_type='COVER_1' if cov=='MAN_COVERAGE' else 'COVER_3',in_motion_at_ball_snap=False,target=target,rec_yards=max(0,rng.normal(8+talent,5)) if target else 0.,rush_yards=np.nan,yards_after_catch=3.,route_ran=route if pos in ['WR','TE','RB'] else np.nan,separation_at_pass_forward=sep if target and pos in ['WR','TE','RB'] else np.nan,cushion=cushion,expected_yards_after_catch=3.,expected_points_added=rng.normal(.15*talent,.7),expected_points=1.,dropback_duration=5.,pressure_allowed=False,blitzing=False,coverage_assignment='MAN',coverage_assignment_at_snap='MAN',tackle=float(rng.random()<.15))
            plays.append(row)
            z=trajectory(np.clip(.55+.055*agility+rng.normal(0,.06),.25,.85),speed=7+.3*fast)
            z.time=z.time.str.replace('2025-03-01',f'{season}-09-10',regex=False)
            z=z.assign(game_id=gid,play_id=playid,nfl_id=pid,o=z.dir,event='')
            z.loc[0,'event']='ball_snap'; z.loc[50,'event']='pass_forward'
            tracking[season].append(z)
    tables=dict(players=pd.DataFrame(players),combine_results=pd.DataFrame(results),player_career_successes=pd.DataFrame(career),combine_tracking=pd.concat(combine),player_play=pd.DataFrame(plays),games=pd.DataFrame(games).drop_duplicates('game_id'))
    for name,df in tables.items(): df[SCHEMA[name].split()].to_csv(root/f'{name}.csv',index=False)
    for year,parts in tracking.items(): pd.concat(parts)[SCHEMA['game_tracking'].split()].to_csv(root/f'game_tracking_{year}.csv',index=False)
    (root/'SYNTHETIC.txt').write_text(f'Artificial validation fixture; seed={seed}; not NFL observations.\n')
    return root
