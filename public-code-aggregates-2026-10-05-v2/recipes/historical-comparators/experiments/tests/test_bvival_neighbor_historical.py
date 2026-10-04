import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bvival_neighbor_available import AvailableNeighborGain
from bvival_neighbor_loss import CachedNeighborLoss
from run_bvival_neighbor_historical import (
    allocate, context_table, development_gains, freeze_allocations, summarize, validate_config,
)
from generate_bvival_prediction_pairs import stable_fold


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/bvival_neighbor_historical_v1_2026-10-01.json').read_text())


def contexts(n):
    return [{'cat':'same','num':str(i)} for i in range(n)]


def test_complete_cache_equivalent_to_before_after_loss():
    before=np.array([10.,20.,30.]); after=np.array([[15,5,10],[40,30,20],[0,0,0]])
    keys=['c','a','b']; c=contexts(3)
    original=CachedNeighborLoss(['cat'],['num']).fit(keys,c,before,after)
    new=AvailableNeighborGain(['cat'],['num']).fit_gains(keys,c,before[:,None]-after)
    for k, expected in original.predict(c,[1,2,3]).items():
        np.testing.assert_allclose(new.predict(c,[1,2,3])[k],expected)


def test_unavailable_action_omitted_not_zero_and_empty_uses_global():
    m=AvailableNeighborGain(['cat'],['num']).fit_gains(['a','b','c'],contexts(3),
        [[-5,10,np.nan,np.nan],[5,np.nan,7,np.nan],[15,30,9,8]])
    scored=m.predict([{'cat':'same','num':'0'}],[1,2,3])
    np.testing.assert_allclose(scored[1], [[-5,10,8,8]])
    np.testing.assert_allclose(scored[2], [[0,10,7,8]])
    np.testing.assert_allclose(scored[3], [[5,20,8,8]])
    assert m.last_fallback_counts_[1].tolist()==[0,0,1,1]
    assert m.last_fallback_counts_[2].tolist()==[0,0,0,1]


@pytest.mark.parametrize('gain', [[[1,np.inf]], [[1,np.nan]], [[np.nan,np.nan]], [1,2]])
def test_invalid_cache_rejected(gain):
    with pytest.raises(ValueError):
        AvailableNeighborGain(['cat'],['num']).fit_gains(['a'],contexts(1),gain)


def test_query_outcomes_rejected_and_train_only_transform_unchanged():
    m=AvailableNeighborGain(['cat'],['num']).fit_gains(['a','b'],contexts(2),[[1,2],[3,4]])
    before=m.train_.copy()
    with pytest.raises(ValueError,match='allowlist'):
        m.predict([{'cat':'same','num':'0','target_log':10}],[1])
    m.predict([{'cat':'new','num':'1000'}],[1])
    np.testing.assert_array_equal(before,m.train_)


def test_exact_negative_capacity_lexical_ties_and_single_action():
    frame=pd.DataFrame({'listing_id':['b','a','a','b'],'action_id':['z','z','a','a'],'score':[-1]*4})
    assert allocate(frame,'score',2,.1)==[('a','a')]
    assert allocate(frame,'score',2,1.)==[('a','a'),('b','a')]


@pytest.mark.parametrize('mutation', ['hidden','inconsistent','duplicate','outcome_field'])
def test_context_rejections(mutation):
    frame=pd.DataFrame({'listing_id':['a','a'],'action_id':['f','g'],'f':[np.nan]*2,'g':[np.nan]*2,'cat':['x']*2,'num':[1]*2})
    fields=['cat','num']
    if mutation=='hidden': frame.loc[0,'f']=4
    if mutation=='inconsistent': frame.loc[1,'num']=2
    if mutation=='duplicate': frame.loc[1,'action_id']='f'
    if mutation=='outcome_field': fields+=['target_log']; frame['target_log']=4
    with pytest.raises(ValueError):
        context_table(frame,actions=['f','g'],fields=fields,categorical=['cat'])


def test_development_oof_and_signed_loss_identity():
    pairs=pd.DataFrame({'listing_id':['a','a','b'],'action_id':['f','g','f'],
        'fold_id':[stable_fold(k,3) for k in ['a','a','b']], 'prediction_is_oof':[True]*3,
        'before_prediction_log':np.log1p([10,10,10]),'after_prediction_log':np.log1p([18,8,15])})
    labels=pd.DataFrame({'listing_id':['a','b'],'target_log':np.log1p([20,20])})
    gains=development_gains(pairs,labels,['a','b'],['f','g'],3)
    np.testing.assert_allclose(gains,[[8,-2],[5,np.nan]],equal_nan=True)
    pairs.loc[0,'prediction_is_oof']=False
    with pytest.raises(ValueError,match='OOF'): development_gains(pairs,labels,['a','b'],['f','g'],3)


def test_numeric_string_coercion_matches_original_value_head():
    frame=pd.DataFrame({'listing_id':['a'],'action_id':['f'],'f':[np.nan],'g':[np.nan],
                        'cat':['x'],'num':['More than 9']})
    keys,rows=context_table(frame,actions=['f','g'],fields=['cat','num'],categorical=['cat'])
    assert keys==['a'] and rows==[{'cat':'x','num':''}]


def test_original_numeric_boolean_predictor_becomes_zero_one():
    frame=pd.DataFrame({'listing_id':['a','b'],'action_id':['f','f'],'f':[np.nan]*2,'g':[np.nan]*2,
                        'cat':['x']*2,'num':[True,False]})
    keys,rows=context_table(frame,actions=['f','g'],fields=['cat','num'],categorical=['cat'])
    assert keys==['a','b'] and [r['num'] for r in rows]==['1.0','0.0']


def test_outcome_intervention_changes_only_metrics_not_choices():
    cfg=config(); keys=['a','b','c']; actions=['f','g','h','i']
    frame=pd.DataFrame([(k,a) for k in keys for a in actions],columns=['listing_id','action_id'])
    frame['score_mean_value']=np.arange(12)+1
    frame['score_uncertainty_only']=[10 if a=='f' else -np.inf for k,a in zip(frame.listing_id,frame.action_id)]
    neighbor={k:np.arange(12).reshape(3,4)+1 for k in cfg['neighbor_counts']}
    first=freeze_allocations(frame,neighbor,keys,actions,cfg)
    outcomes=frame[['listing_id','action_id']].copy()
    outcomes['target_log']=np.log1p(20); outcomes['before_prediction_log']=np.log1p(10); outcomes['after_prediction_log']=np.log1p(18)
    r1=summarize(outcomes,first,keys,actions)
    outcomes['target_log']=np.log1p(100); outcomes['after_prediction_log']=np.log1p(200)
    second=freeze_allocations(frame,neighbor,keys,actions,cfg)
    r2=summarize(outcomes,second,keys,actions)
    assert first==second and r1!=r2
    for name,allocation in first.items(): assert len(allocation)==(0 if name=='no_acquisition' else 1)


@pytest.mark.parametrize('key,value',[('select_neighbor_count',True),('compute_confirmatory_inference',True),
    ('budget',.2),('neighbor_counts',[64]),('historically_unopened_test',True)])
def test_frozen_contract_rejects_selection_or_confirmation(key,value):
    cfg=config(); validate_config(cfg); cfg[key]=value
    with pytest.raises(ValueError): validate_config(cfg)


def test_actual_output_manifest_and_historical_replay():
    from generate_bvival_prediction_pairs import file_sha256
    root=Path(__file__).resolve().parents[2]
    directory=root/'experiments/outputs/bvival_neighbor_historical_2026-10-01_r3'
    if not directory.exists(): pytest.skip('Private post-test replay is not distributed')
    manifest=json.loads((directory/'artifact_hashes.json').read_text())
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    assert actual==set(manifest)|{'artifact_hashes.json'}
    for name,expected in manifest.items(): assert file_sha256(directory/name)==expected
    summary=json.loads((directory/'summary.json').read_text())
    assert len(summary['sources'])==3 and summary['pooled_mae_or_significance_computed'] is False
    for source in summary['sources']:
        assert source['historical_reference_replay_verified'] is True
        assert source['selected_neighbor_count'] is None
        assert len(source['rows'])==9
        for name,row in source['rows'].items():
            assert row['actions']==(0 if name=='no_acquisition' else int(np.ceil(source['evaluation_listings']*.1)))
