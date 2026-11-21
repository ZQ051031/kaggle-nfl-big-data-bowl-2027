import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .data import load_table

CUT_METRICS=['entry_speed','min_speed','exit_speed','peak_deceleration','recovery_time','csr']

def cuts(frames):
    f=frames.sort_values('time').reset_index(drop=True)
    if len(f)<21: return pd.DataFrame(columns=CUT_METRICS)
    s=f['s'].to_numpy(); direction=f['dir'].to_numpy()
    times=(pd.to_datetime(f.time)-pd.to_datetime(f.time.iloc[0])).dt.total_seconds().to_numpy()
    candidates,_=find_peaks(-s,distance=10,prominence=.25)
    rows=[]
    for k in candidates:
        if k<5 or k+10>=len(f): continue
        delta=abs((direction[k+3]-direction[k-3]+180)%360-180)
        entry=float(np.median(s[k-5:k-2]))
        if delta<45 or entry<2: continue
        recovery=np.flatnonzero(s[k+1:]>=.8*entry)
        rows.append(dict(entry_speed=entry,min_speed=s[k],exit_speed=s[k+10],
            peak_deceleration=max(0,-np.min(np.gradient(s[k-5:k+1],times[k-5:k+1]))),
            recovery_time=times[k+1+recovery[0]]-times[k] if len(recovery) else np.nan,csr=s[k]/entry,
            cut_frame=k,turn_degrees=delta))
    return pd.DataFrame(rows)

def aggregate_cuts(table, prefix):
    if table.empty: return pd.DataFrame(columns=['nfl_id'])
    grouped=table.groupby('nfl_id')[CUT_METRICS]
    med=grouped.median().add_prefix(prefix)
    iqr=(grouped.quantile(.75)-grouped.quantile(.25)).add_prefix(prefix).add_suffix('_iqr')
    return med.join(iqr).join(table.groupby('nfl_id').size().rename(prefix+'cut_count')).reset_index()

def combine_features(root):
    f=load_table(root,'combine_tracking'); f=f.loc[f.entity_type.eq('PLAYER') & f.nfl_id.notna()]
    parts=[]; sprints=[]
    allowed=['THREE_CONE_DRILL','SHORT_SHUTTLE','SKILL_DRILLS_WR','SKILL_DRILLS_TE','SKILL_DRILLS_DB']
    for (pid,event),g in f.groupby(['nfl_id','event_id']):
        drill=g.drill_type.iloc[0]
        if drill in allowed:
            # Position skill drills must describe a route or transition, not catching-only drills.
            if drill.startswith('SKILL') and not g.drill_name.str.contains('ROUTE|TRANSITION|PEDAL',case=False,regex=True).any(): continue
            c=cuts(g)
            if len(c): parts.append(c.assign(nfl_id=pid,event_id=event))
        if drill=='FORTY_YARD_DASH':
            g=g.sort_values('time'); speed=g.s.to_numpy(); t=pd.to_datetime(g.time)
            sprints.append(dict(nfl_id=pid,sprint_max_speed=speed.max(),sprint_time90=(t.iloc[np.flatnonzero(speed>=.9*speed.max())[0]]-t.iloc[0]).total_seconds(),sprint_peak_accel=np.max(np.gradient(speed,.1))))
    cut_table=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    agg=aggregate_cuts(cut_table,'combine_')
    if sprints: agg=agg.merge(pd.DataFrame(sprints).groupby('nfl_id').median().reset_index(),on='nfl_id',how='outer')
    return agg,cut_table
