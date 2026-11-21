import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .data import load_table
from .synthetic import generate
from .combine_features import combine_features
from .game_features import game_features, exposure_outcomes
from .modelling import evaluate
from .plotting import figures

def markdown_table(df):
    values=df.copy()
    for c in values.select_dtypes('number'): values[c]=values[c].map(lambda x:f'{x:.3f}' if isinstance(x,float) else str(x))
    return '| '+' | '.join(values.columns)+' |\n| '+' | '.join(['---']*len(values.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in values.itertuples(index=False,name=None))

def run(data_dir=None,synthetic=False,output_dir='.',report_template=None):
    dest=Path(output_dir); dest.mkdir(parents=True,exist_ok=True)
    root=Path(data_dir) if data_dir else Path('/kaggle/input/nfl-big-data-bowl-2027')
    if synthetic or (data_dir is None and not root.exists()): root=generate(dest/'data/synthetic')
    is_synthetic=(root/'SYNTHETIC.txt').exists()
    out=dest/'outputs'; out.mkdir(exist_ok=True)
    combine,combine_cuts=combine_features(root); game,plays,game_cuts=game_features(root)
    players=load_table(root,'players'); career=load_table(root,'player_career_successes')
    frame=players.merge(load_table(root,'combine_results').drop(columns='draft_year'),on='nfl_id',how='left').merge(combine,on='nfl_id',how='left').merge(game,on='nfl_id',how='left').merge(exposure_outcomes(plays,career),on='nfl_id',how='inner')
    metrics,pred,importance,talent,summary,audits=evaluate(frame,plays)
    valid=frame[['combine_csr','game_csr']].dropna()
    summary.update(synthetic=is_synthetic,translation_spearman=float(spearmanr(valid.combine_csr,valid.game_csr).statistic),translation_n=len(valid),regular_player_plays=len(plays),combine_cuts=len(combine_cuts),game_cuts=len(game_cuts))
    for name,table in dict(player_features=frame,combine_cuts=combine_cuts,game_cuts=game_cuts,cv_metrics=metrics,predictions=pred,permutation_importance=importance,hidden_talent=talent.sort_values('surplus',ascending=False).head(8),bust_risk=talent.sort_values('surplus').head(8)).items():
        table.to_csv(out/f'{name}.csv',index=False)
        if name in ['hidden_talent','bust_risk']: (out/f'{name}.md').write_text(markdown_table(table))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)); (out/'cv_audit.json').write_text(json.dumps(audits,indent=2))
    trace=load_table(root,'combine_tracking'); trace=trace.loc[trace.event_id.eq(combine_cuts.event_id.iloc[0]) & trace.entity_type.eq('PLAYER')]
    figures(dest/'figures',frame,trace,importance,summary)
    if report_template is None: report_template=Path('REPORT_TEMPLATE.md').read_text()
    values={**{k:str(v) for k,v in summary.items()},'status':'SYNTHETIC DATA ONLY — not NFL evidence' if is_synthetic else 'REAL DATA RUN — review diagnostics and all claims before submission',
        'a_r2':f"{summary['A']['r2']:.3f}",'b_r2':f"{summary['B']['r2']:.3f}",'a_mae':f"{summary['A']['mae']:.3f}",'b_mae':f"{summary['B']['mae']:.3f}",'a_rho':f"{summary['A']['spearman']:.3f}",'b_rho':f"{summary['B']['spearman']:.3f}",
        'rho':f"{summary['translation_spearman']:.3f}",'ci':f"[{summary['r2_improvement_ci'][0]:.3f}, {summary['r2_improvement_ci'][1]:.3f}]",'talent_table':markdown_table(talent.sort_values('surplus',ascending=False).head(4)[['display_name','nfl_position','draft_overall_pick','surplus']])}
    for key,value in values.items(): report_template=report_template.replace('{{'+key+'}}',value)
    if re.search(r'\{\{.*?\}\}',report_template): raise ValueError('Unfilled report placeholders')
    (dest/'REPORT.md').write_text(report_template)
    print('Data mode:', 'SYNTHETIC' if is_synthetic else 'REAL')
    print('Leave-one-draft-class-out Ridge:')
    for name in ['A','B']: print(name,json.dumps(summary[name],sort_keys=True))
    print('B−A R² 95% bootstrap CI:',summary['r2_improvement_ci'])
    print('Combine ↔ game CSR Spearman:',round(summary['translation_spearman'],3),'n =',len(valid))
    return summary
