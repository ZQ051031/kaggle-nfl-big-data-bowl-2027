import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from bdb27.synthetic import generate, trajectory
from bdb27.combine_features import cuts, combine_features
from bdb27.game_features import game_features
from bdb27.modelling import estimator, fold_outcome
from bdb27.data import regular_plays
from sklearn.model_selection import LeaveOneGroupOut

@pytest.fixture(scope='module')
def fixture_data(tmp_path_factory):
    return generate(tmp_path_factory.mktemp('synthetic'),n=30)

def test_handmade_cut():
    t=np.arange(51)/10
    frame=pd.DataFrame({'time':pd.date_range('2025-01-01',periods=51,freq='100ms'),
        's':np.where(abs(t-2)<.8,7*(1-.5*(1-abs(t-2)/.8)),7),
        'dir':90*np.clip((t-1.7)/.6,0,1)})
    found=cuts(frame)
    assert len(found)==1
    assert found.iloc[0].min_speed==pytest.approx(3.5)
    assert found.iloc[0].exit_speed==pytest.approx(7)
    assert 0<found.iloc[0].csr<1
    assert found.iloc[0].recovery_time==pytest.approx(.2)
    assert cuts(frame.assign(dir=0)).empty
    assert cuts(frame.iloc[:5]).empty
    with pytest.raises(KeyError): cuts(frame.drop(columns='s'))

def test_documented_schema_and_physics(fixture_data):
    reference=json.loads((Path(__file__).parent/'documented_schema.json').read_text())
    players=pd.read_csv(fixture_data/'players.csv')
    assert players.loc[players.draft_overall_pick.isna(),['draft_round','draft_pick_within_round']].isna().all().all()
    for path in fixture_data.glob('*.csv'):
        key='game_tracking_2023' if path.stem.startswith('game_tracking') else path.stem
        df=pd.read_csv(path)
        assert list(df)==list(reference[key]),path.name
        for col,dtype in reference[key].items():
            if dtype in ['int64','float64']: assert pd.api.types.is_numeric_dtype(df[col]),col
            if dtype=='bool': assert pd.api.types.is_bool_dtype(df[col]),col
        if 'dir' in df:
            assert df.x.between(0,120).all() and df.y.between(0,53.33).all()
            assert df.s.between(0,13).all() and df.dir.between(0,360).all()
            group=['event_id'] if key=='combine_tracking' else ['game_id','play_id','nfl_id']
            for _,g in df.groupby(group):
                assert np.allclose(pd.to_datetime(g.time).diff().dt.total_seconds().dropna(),.1)
                assert np.max(np.abs(g.a))<12
                assert np.allclose(np.hypot(g.x.diff().iloc[1:],g.y.diff().iloc[1:]),g.dis.iloc[1:])

def test_chunk_boundaries_and_regular_season(fixture_data):
    # A deliberately awkward chunk length bisects every play and attempt.
    features,plays,table=game_features(fixture_data,chunksize=307)
    assert len(table)==30*17
    assert len(features)==30
    assert not plays.play_id.eq(18).any()
    features,table=combine_features(fixture_data)
    assert features.combine_csr.notna().all()

def test_no_leakage_cv_smoke(fixture_data):
    players=pd.read_csv(fixture_data/'players.csv')
    players=players[players.nfl_position.isin(['WR','TE'])].reset_index(drop=True)
    plays=regular_plays(fixture_data)
    for tr,te in LeaveOneGroupOut().split(players,groups=players.draft_year):
        assert set(players.iloc[tr].draft_year).isdisjoint(players.iloc[te].draft_year)
        train_ids=set(players.iloc[tr].nfl_id)
        original=fold_outcome(players,plays,train_ids)
        perturbed=plays.copy(); perturbed.loc[~perturbed.nfl_id.isin(train_ids),'separation_at_pass_forward']=100
        changed=fold_outcome(players,perturbed,train_ids)
        assert np.allclose(original[tr],changed[tr])
        assert not np.allclose(original[te],changed[te])
        x=pd.DataFrame({'forty':np.arange(len(players),dtype=float)})
        x.loc[te,'forty']=1e6
        model=estimator(['forty']).fit(x.iloc[tr],original[tr])
        mean=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].mean_[0]
        assert mean==pytest.approx(x.iloc[tr].forty.mean())
        assert len(model.predict(x.iloc[te]))==len(te)
