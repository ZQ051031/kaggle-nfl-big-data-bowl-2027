from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .combine_features import cuts

BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'

def figures(root,frame,trace,importance,summary):
    out=Path(root); out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,
        'axes.prop_cycle':plt.cycler(color=[BLUE,ORANGE,GREEN,'#CC79A7','#E69F00'])})
    def save(name):
        plt.tight_layout(); plt.savefig(out/name,bbox_inches='tight'); plt.close()
    c=cuts(trace).iloc[0]; k=int(c.cut_frame); t=np.arange(len(trace))*.1
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(t,trace.s,color=BLUE,lw=2)
    ax.scatter([t[k],t[k+10]],[c.min_speed,c.exit_speed],color=[ORANGE,GREEN],zorder=4)
    ax.axhline(c.entry_speed,ls='--',color='gray',label=f'Entry {c.entry_speed:.2f} yd/s')
    ax.annotate(f'Minimum {c.min_speed:.2f} yd/s\nCSR {c.csr:.2f}',(t[k],c.min_speed),xytext=(.2,.5),textcoords='axes fraction',arrowprops={'arrowstyle':'->'})
    ax.annotate(f'Exit +1 s: {c.exit_speed:.2f}\nRecovery: {c.recovery_time:.1f} s\nPeak braking: {c.peak_deceleration:.1f} yd/s²',(t[k+10],c.exit_speed),xytext=(.6,.15),textcoords='axes fraction',arrowprops={'arrowstyle':'->'})
    ax.set(xlabel='Seconds from attempt start',ylabel='Speed (yards/second)',title='Brake & Burst | one detected cut'); ax.legend(loc='upper left'); save('cut_trace.png')
    fig,ax=plt.subplots(figsize=(7,5))
    for pos,g in frame.groupby('nfl_position'):
        ax.scatter(g.combine_csr,g.game_csr,label=pos,alpha=.7,s=24)
    ax.set(xlabel='Combine cut speed retention (median)',ylabel='In-game cut speed retention (median)',title='Does the movement trait transfer?'); ax.legend(); save('translation.png')
    fig,ax=plt.subplots(figsize=(8,5)); imp=importance.groupby('feature').importance.mean().sort_values().tail(10)
    ax.barh(imp.index,imp,color=BLUE); ax.axvline(0,color='black',lw=.8)
    ax.set(xlabel='Held-out MAE increase after permutation (yards)',title='Which inputs add predictive information?'); save('importance.png')
    fig,ax=plt.subplots(figsize=(7,4))
    vals=[summary[x]['r2'] for x in ['A','B']]
    ax.bar(['A: stopwatch + size','B: A + tracking'],vals,color=[BLUE,ORANGE],width=.55)
    lo,hi=summary['r2_improvement_ci']; ax.set(xlabel='Leave-one-draft-class-out Ridge',ylabel='Out-of-class R²',title=f'Incremental R²: {vals[1]-vals[0]:+.3f} | 95% CI [{lo:+.3f}, {hi:+.3f}]')
    ax.axhline(0,color='gray',lw=.8); save('validation.png')
    fig,ax=plt.subplots(figsize=(7,4))
    groups=list(frame.groupby('nfl_position'))
    ax.boxplot([g.combine_csr.dropna() for _,g in groups],tick_labels=[p for p,_ in groups],showfliers=False)
    ax.set(xlabel='Roster position',ylabel='Combine CSR',title='Compare movement within position, not across roles'); save('positions.png')
