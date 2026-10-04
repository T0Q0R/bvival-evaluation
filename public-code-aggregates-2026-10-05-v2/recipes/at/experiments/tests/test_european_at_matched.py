"""Synthetic matched-task tests, not evidence of empirical performance."""
import json
from pathlib import Path

import numpy as np
import pytest

import european_at_matched as m
import run_european_at_matched as runner
from build_european_at_cohort import ACTIONS
from test_european_at_audit import fixture, cell


def records(n=20):
    return [{"record_key": f"K{i:03}", "context": {"brand": "ford", "family": "transit", "year": float(2000+i)},
             "actions": {"mileage_km": 1000.*i, "power_kw": 80.+i, "fuel": "diesel", "transmission": "manual"}}
            for i in range(n)]


def test_only_revealed_field_enters_after_state():
    states = m.states(records(1)).to_dict('records')
    assert len(states)==5
    for i,s in enumerate(states):
        assert sum(1-s['hidden_'+a] for a in ACTIONS)==int(i>0)
        for a in ACTIONS:
            if s['hidden_'+a]:
                assert s[a]=='__HIDDEN__' if a in {'fuel','transmission'} else np.isnan(s[a])


def test_state_refuses_extra_title_or_price_context():
    r=records(1)
    r[0]['context']['raw_title']='DSG'
    with pytest.raises(ValueError,match='allowlists'): m.states(r)


def test_policy_context_refuses_revealed_value_and_labels():
    ctx=m.contexts(records(1),np.ones((1,5)))
    assert set(ctx[0])==set(m.CONTEXT_FIELDS)
    for extra in ('price','mileage_km','gain','after_prediction_log'):
        with pytest.raises(ValueError,match='pre-action'):m.context_frame([{**ctx[0],extra:1}])


def test_exact_capacity_negative_scores_no_abstention_and_canonical_ties():
    keys=['K3','K1','K2']
    a=m.allocation(keys,[-1,-1,-1],[0,1,2],.1)
    assert a.tolist()==[-1,1,-1]


def test_same_predictor_same_fixed_field_p1_p2_and_all_policies_budget():
    keys=[f'K{i:02}' for i in range(20)]
    gain=np.tile([1,4,2,3],(20,1));risk=np.arange(20);near=np.tile([8,1,2,3],(20,1))
    p=m.policies(keys,gain,risk,near,'fuel',.1)
    assert all((a>=0).sum()==2 for a in p.values())
    assert set(p['P1'][p['P1']>=0])==set(p['P2'][p['P2']>=0])=={2}
    assert set(p['P3'][p['P3']>=0])=={1}
    assert set(p['P4'][p['P4']>=0])=={0}


def test_unselected_errors_remain_before_not_best_observed_action():
    e=np.array([[10,1,2,3,4],[20,9,8,7,6]])
    assert m.selected_errors(e,[-1,2]).tolist()==[10,7]


def test_invalid_label_does_not_reallocate_budget_or_change_actions():
    logs=np.log1p(np.array([[100,90,90,90,90],[200,210,210,210,210],[300,310,310,310,310]]))
    acts={p:[0,-1,-1] for p in ('P1','P2','P3','P4')}
    cfg={'budget_fraction':.1,'bootstrap':{'seed':2026,'replicates':100}}
    before=json.dumps(acts)
    result=m.summarize(logs,[None,220,320],acts,cfg)
    assert result['nonprice_candidates_N']==3 and result['valid_observed_target_n']==2
    assert all(x['actions_all_nonprice_candidates']==1 and x['actions_on_valid_target_records']==0 for x in result['policies'].values())
    assert all(x['mae_EUR_valid_target_cohort']==pytest.approx(20) for x in result['policies'].values())
    assert json.dumps(acts)==before


def test_bootstrap_pair_is_paired_and_seed_reproducible():
    a=m.bootstrap_pair([9,18,27],[10,20,30],replicates=100)
    b=m.bootstrap_pair([9,18,27],[10,20,30],replicates=100)
    assert a==b and a['relative_mae_improvement_percent']==pytest.approx(10)
    assert a['nominal_97_5_percent_interval']==pytest.approx([10,10])


@pytest.mark.parametrize('value,expected',[('100',100.),(' 12.5 ',12.5),('0',None),('-1',None),('NaN',None),('Infinity',None),('12,000',None),('EUR 100',None),('sNaN',None)])
def test_target_validity_does_not_guess_or_impute(value,expected):
    assert runner.positive_price(value)==expected


def test_development_price_access_never_decodes_eval_or_reserved_cells(monkeypatch,tmp_path):
    rowdev='<row r="2">'+cell('C',2,'100')+'</row>'
    poison='<row r="2"><c r="B2"><v>UNAUTHORIZED_TARGET &invalid;</v></c></row>'
    with fixture(month_rows={'02':[rowdev],'09':[poison],'11':[poison]}) as z:
        path=tmp_path/'fixture.xlsx';path.write_bytes(z.fp.getvalue())
    monkeypatch.setattr(runner,'probe',lambda p:None)
    original=runner.decode_approved_cell
    def guard(block,col,approved):
        assert b'UNAUTHORIZED_TARGET' not in block
        return original(block,col,approved)
    monkeypatch.setattr(runner,'decode_approved_cell',guard)
    data=[{'record_key':'train1','month':'02','source_row':2,'role':'development'}]
    prices,access=runner.load_prices(path,data,{'development','validation'})
    assert prices=={'train1':100.}
    assert access['entire_unrequested_months_skipped']==11


def test_label_loader_refuses_out_of_phase_role_before_source_io(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'probe',lambda p:pytest.fail('Source should not open'))
    with pytest.raises(ValueError,match='not authorized'):
        runner.load_prices(tmp_path/'missing',[{'role':'evaluation'}],{'development','validation'})


def test_scores_has_no_target_loader_call():
    import ast,inspect
    tree=ast.parse(inspect.getsource(runner.scores))
    assert not any(isinstance(n,ast.Name) and n.id in {'load_prices','positive_price','source'} for n in ast.walk(tree))


def test_tiny_actual_catboost_fit_smoke_on_synthetic_targets_only():
    cfg={'model_seeds':[2026], 'base_predictor':{'iterations':5,'depth':2,'learning_rate':.05,'l2_leaf_reg':10,
         'thread_count':1,'loss_function':'MAE'},'gain_head':{'iterations':5,'depth':2,'learning_rate':.05,
         'l2_leaf_reg':10,'thread_count':1,'loss_function':'RMSE'},'risk_head':{'iterations':5,'depth':2,
         'learning_rate':.05,'l2_leaf_reg':10,'thread_count':1,'loss_function':'RMSE'}}
    r=records(20);prices=np.arange(20)*1000+5000
    models=m.fit_base(r,prices,cfg)
    logs=m.predict_base(models,r);ctx=m.contexts(r,logs)
    g,rr,gains,scale=m.fit_heads(ctx,m.errors(prices,logs),cfg)
    scores,risk=m.predict_heads(g,rr,ctx)
    assert logs.shape==(20,5) and scores.shape==(20,4) and risk.shape==(20,)
    assert np.isfinite(scores).all() and scale>0
