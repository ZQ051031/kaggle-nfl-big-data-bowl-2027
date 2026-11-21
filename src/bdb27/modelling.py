import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance

BASE=['forty','ten_yd_split','three_cone','short_shuttle','vertical','broad_jump','combine_height','combine_weight','nfl_position']
CONTEXT=['route_ran','team_coverage_man_zone','cushion','down','yards_to_go']

def estimator(columns, family='ridge', classification=False):
    cat=[x for x in columns if x in ['nfl_position','route_ran','team_coverage_man_zone']]
    num=[x for x in columns if x not in cat]
    transform=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    model=(LogisticRegression(C=.1,max_iter=1000) if classification else Ridge(alpha=20)) if family=='ridge' else (GradientBoostingClassifier(n_estimators=45,max_depth=2,min_samples_leaf=8,random_state=2027) if classification else GradientBoostingRegressor(n_estimators=45,max_depth=2,min_samples_leaf=8,random_state=2027))
    return Pipeline([('preprocess',transform),('model',model)])

def fold_outcome(players,plays,train_ids):
    p=plays.loc[plays.target.eq(True) & plays.nfl_id.isin(players.nfl_id) & plays.separation_at_pass_forward.notna()].copy()
    train=p.nfl_id.isin(train_ids)
    if train.sum()<5: raise ValueError('Too few training targets for context adjustment')
    context=estimator(CONTEXT)
    context.fit(p.loc[train,CONTEXT],p.loc[train,'separation_at_pass_forward'])
    p['residual']=p.separation_at_pass_forward-context.predict(p[CONTEXT])
    p=p.merge(players[['nfl_id','nfl_position']],on='nfl_id')
    train=p.nfl_id.isin(train_ids)
    prior=p.loc[train].groupby('nfl_position').residual.mean()
    stats=p.groupby('nfl_id').residual.agg(['sum','count'])
    joined=players[['nfl_id','nfl_position']].merge(stats,on='nfl_id',how='left').fillna({'sum':0,'count':0})
    joined['outcome']=(joined['sum']+10*joined.nfl_position.map(prior).fillna(0))/(joined['count']+10)
    return joined.set_index('nfl_id').outcome.reindex(players.nfl_id).to_numpy()

def scores(y,p,classification=False):
    if classification:
        return dict(auc=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,brier=float(brier_score_loss(y,p)))
    return dict(r2=float(r2_score(y,p)),spearman=float(spearmanr(y,p).statistic),mae=float(mean_absolute_error(y,p)))

def evaluate(frame,plays):
    df=frame.loc[frame.nfl_position.isin(['WR','TE']) & frame.target_count.ge(3)].reset_index(drop=True)
    if len(df)<20: raise ValueError('Need at least 20 WR/TE players with >=3 targets')
    tracking=[c for c in df if c.startswith(('combine_','sprint_')) and c!='combine_position' and c not in BASE]
    # No NFL outcomes or draft class are permitted predictors.
    variants={'A':BASE,'B':BASE+tracking,'A_draft':BASE+['draft_overall_pick'],'B_draft':BASE+['draft_overall_pick']+tracking}
    df['draft_overall_pick']=df.draft_overall_pick.fillna(260)
    splits={'draft_class':list(LeaveOneGroupOut().split(df,groups=df.draft_year)),
        'repeated_stratified':list(RepeatedStratifiedKFold(n_splits=3,n_repeats=2,random_state=2027).split(df,df.starter))}
    rows=[]; predictions=[]; importances=[]; audits=[]
    for validation,folds in splits.items():
        for fold,(tr,te) in enumerate(folds):
            y=fold_outcome(df,plays,set(df.iloc[tr].nfl_id))
            audits.append(dict(validation=validation,fold=fold,train_ids=df.iloc[tr].nfl_id.tolist(),test_ids=df.iloc[te].nfl_id.tolist()))
            for family in ['ridge','gradient_boosting']:
                for name,cols in variants.items():
                    for classification in [False,True]:
                        target=df.starter.to_numpy() if classification else y
                        model=estimator(cols,family,classification)
                        model.fit(df.iloc[tr][cols],target[tr])
                        pred=model.predict_proba(df.iloc[te][cols])[:,1] if classification else model.predict(df.iloc[te][cols])
                        rows.append(dict(validation=validation,fold=fold,family=family,variant=name,task='starter' if classification else 'separation',**scores(target[te],pred,classification)))
                        for j,idx in enumerate(te): predictions.append(dict(validation=validation,fold=fold,family=family,variant=name,task='starter' if classification else 'separation',nfl_id=int(df.nfl_id.iloc[idx]),truth=float(target[idx]),prediction=float(pred[j])))
                        if validation=='draft_class' and name=='B' and family=='ridge' and not classification:
                            imp=permutation_importance(model,df.iloc[te][cols],target[te],n_repeats=5,random_state=2027,scoring='neg_mean_absolute_error')
                            importances.extend(dict(feature=c,importance=float(v),fold=fold) for c,v in zip(cols,imp.importances_mean))
    metrics=pd.DataFrame(rows); pred=pd.DataFrame(predictions)
    paired=pred.query("validation == 'draft_class' and family == 'ridge' and task == 'separation' and variant in ['A','B']").pivot(index='nfl_id',columns='variant',values='prediction')
    truth=pred.query("validation == 'draft_class' and family == 'ridge' and task == 'separation' and variant == 'A'").set_index('nfl_id').truth.reindex(paired.index).to_numpy()
    rng=np.random.default_rng(2027); deltas=[]
    for _ in range(500):
        idx=rng.integers(0,len(truth),len(truth)); yy=truth[idx]
        deltas.append([r2_score(yy,paired.B.to_numpy()[idx])-r2_score(yy,paired.A.to_numpy()[idx]),mean_absolute_error(yy,paired.B.to_numpy()[idx])-mean_absolute_error(yy,paired.A.to_numpy()[idx])])
    ci=np.quantile(deltas,[.025,.975],axis=0)
    summary={name:scores(truth,paired[name]) for name in ['A','B']}
    summary.update(r2_improvement_ci=ci[:,0].tolist(),mae_difference_ci=ci[:,1].tolist(),n_players=len(df))
    # Draft expectation is also evaluated out of class, never fit on these players' outcomes.
    expectation=np.zeros(len(df))
    for tr,te in splits['draft_class']:
        y=fold_outcome(df,plays,set(df.iloc[tr].nfl_id))
        model=estimator(['draft_overall_pick','nfl_position']); model.fit(df.iloc[tr][['draft_overall_pick','nfl_position']],y[tr])
        expectation[te]=model.predict(df.iloc[te][['draft_overall_pick','nfl_position']])
    talent=df[['nfl_id','display_name','nfl_position','draft_year','draft_overall_pick','target_count']].copy()
    talent['tracking_prediction']=paired.B.reindex(df.nfl_id).to_numpy(); talent['draft_expectation']=expectation
    talent['surplus']=talent.tracking_prediction-talent.draft_expectation
    return metrics,pred,pd.DataFrame(importances),talent,summary,audits
