"""Replay existing three-source tests, with new scores saved before label parsing.

All tests historically opened; this is never confirmatory. No valuation refit,
raw source download, calibration fit, neighbor selection or pooled inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import sklearn

from bvival_neighbor_available import AvailableNeighborGain
from generate_bvival_prediction_pairs import file_sha256, stable_fold
from bvi_val import select_unit_cost_actions


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def validate_config(cfg):
    expected = {
        'stage': 'POST_TEST_EXPLORATORY_ESTIMATOR_COMPARISON',
        'identity': 'action_eligible_singleton_cached_neighbor_gain_not_full_TAFA_ACO',
        'neighbor_counts': [16, 64, 128], 'budget': .1, 'max_categories': 64,
        'working_memory_mib': 64,
        'neighbor_missing_action_rule': 'omit_unavailable_then_development_global_mean_if_zero',
        'capacity_rule': 'exact_ceil_listing_count_for_all_active_policies_keep_negative_scores',
        'reference_policy': 'score_uncertainty_only', 'select_neighbor_count': False,
        'compute_confirmatory_inference': False, 'evaluation_outcomes_in_scores': False,
        'historically_unopened_test': False,
    }
    if any(cfg.get(k) != v for k,v in expected.items()):
        raise ValueError('Changed historical diagnostic contract')
    if [s['name'] for s in cfg['sources']] != ['MUCars', 'JUCars', 'AutoScout24']:
        raise ValueError('Retain every source in declared order')


def checked_file(path, expected, hashes):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f'Cached input differs from historical audit: {path.name}')
    hashes[str(path)] = actual
    return path


def read_csv(path):
    return pd.read_csv(path, dtype={'listing_id': str, 'action_id': str})


def context_table(frame, *, actions, fields, categorical):
    if frame.duplicated(['listing_id','action_id']).any():
        raise ValueError('Duplicate pre-action pairs')
    if not set(frame.action_id).issubset(actions) or not frame[actions].isna().all().all():
        raise ValueError('Hidden fields must remain unavailable in every context')
    if not set(fields).issubset(frame.columns) or set(fields) & (set(actions)|{'action_id','listing_id','target_log','after_prediction_log'}):
        raise ValueError('Strict original observable input allowlist required')
    normalized = frame[['listing_id', *fields]].copy()
    for field in fields:
        if field in categorical:
            normalized[field] = normalized[field].fillna('__MISSING__').astype(str)
        else:
            # Match the original fitted value head's numeric coercion exactly.
            numbers = pd.to_numeric(normalized[field], errors='coerce')
            normalized[field] = numbers.map(lambda x: '' if pd.isna(x) else str(float(x)))
    unique = normalized.drop_duplicates()
    if unique.listing_id.duplicated().any():
        raise ValueError('Action copies disagree about pre-action listing context')
    unique = unique.sort_values('listing_id').reset_index(drop=True)
    return unique.listing_id.tolist(), unique[fields].to_dict('records')


def development_gains(pairs, labels, keys, actions, folds):
    if pairs.duplicated(['listing_id','action_id']).any() or labels.listing_id.duplicated().any():
        raise ValueError('Unique development pairs and labels required')
    if (not pairs.prediction_is_oof.astype(str).str.lower().isin(['true','1']).all()
            or not (pairs.fold_id == pairs.listing_id.map(lambda k: stable_fold(k, folds))).all()):
        raise ValueError('Historical train/validation OOF folds required')
    if set(pairs.listing_id) != set(keys) or not set(pairs.action_id).issubset(actions):
        raise ValueError('Development key/action universe mismatch')
    joined = pairs.merge(labels[['listing_id','target_log']], on='listing_id', how='left', validate='many_to_one')
    values = joined[['target_log','before_prediction_log','after_prediction_log']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Finite aligned development values required')
    prices = np.maximum(np.expm1(values), 0)
    gain = np.abs(prices[:,0]-prices[:,1])-np.abs(prices[:,0]-prices[:,2])
    if not np.isfinite(gain).all():
        raise ValueError('Nonfinite development gain')
    joined['gain'] = gain
    return joined.pivot(index='listing_id', columns='action_id', values='gain').reindex(index=keys, columns=actions).to_numpy(float)


def allocate(scores, column, n, budget):
    """Exact common capacity, lexical ties, no outcome or positivity gate."""
    work = scores[['listing_id','action_id',column]].copy()
    if work.duplicated(['listing_id','action_id']).any() or work.listing_id.nunique() != n:
        raise ValueError('Unique complete eligible action universe required')
    values = pd.to_numeric(work[column], errors='raise').to_numpy(float)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('Invalid routing scores')
    work = work[np.isfinite(values)]
    best = work.sort_values(['listing_id',column,'action_id'], ascending=[True,False,True], kind='mergesort').drop_duplicates('listing_id')
    capacity = int(np.ceil(n*budget))
    if len(best) < capacity:
        raise ValueError('Insufficient eligible listings for original capacity')
    selected = best.sort_values([column,'listing_id','action_id'], ascending=[False,True,True], kind='mergesort').head(capacity)
    return list(zip(selected.listing_id, selected.action_id))


def freeze_allocations(scores, neighbor, keys, actions, cfg):
    if scores.duplicated(['listing_id','action_id']).any() or set(scores.listing_id) != set(keys):
        raise ValueError('Frozen score universe mismatch')
    allocations = {'no_acquisition': []}
    for name, column in [('frozen_benefit','score_mean_value'),('frozen_risk','score_uncertainty_only')]:
        allocations[name] = allocate(scores, column, len(keys), cfg['budget'])
        finite = scores[np.isfinite(scores[column])][['listing_id','action_id',column]].rename(columns={column:'lower_value'})
        original = select_unit_cost_actions(finite, budget_fraction=cfg['budget'])
        if set(allocations[name]) != set(zip(original.listing_id,original.action_id)):
            raise ValueError('Exact-capacity diagnostic differs from original baseline allocation; stop')
    risk_keys = {key for key,_ in allocations['frozen_risk']}
    for k in cfg['neighbor_counts']:
        matrix = neighbor[k]
        mapping = {(key,action):float(matrix[i,j]) for i,key in enumerate(keys) for j,action in enumerate(actions)}
        new = scores[['listing_id','action_id']].copy()
        new['neighbor'] = [mapping[(key,action)] for key,action in zip(new.listing_id,new.action_id)]
        allocations[f'neighbor_joint_k{k}'] = allocate(new,'neighbor',len(keys),cfg['budget'])
        same = new[new.listing_id.isin(risk_keys)]
        allocations[f'risk_neighbor_field_k{k}'] = allocate(same,'neighbor',len(risk_keys),1.)
    return allocations


def summarize(outcomes, allocations, keys, actions):
    if outcomes.duplicated(['listing_id','action_id']).any() or set(outcomes.listing_id) != set(keys):
        raise ValueError('Unique matching evaluation outcome keys required')
    values = outcomes[['target_log','before_prediction_log','after_prediction_log']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Finite evaluation outcomes required')
    prices = np.maximum(np.expm1(values),0)
    work = outcomes[['listing_id','action_id']].copy()
    work['before'] = np.abs(prices[:,0]-prices[:,1]); work['after'] = np.abs(prices[:,0]-prices[:,2])
    if work.groupby('listing_id').before.nunique().gt(1).any():
        raise ValueError('Before loss inconsistent across action copies')
    before = work.drop_duplicates('listing_id').set_index('listing_id').before.reindex(keys).to_numpy()
    lookup = {(r.listing_id,r.action_id): (r.before,r.after) for r in work.itertuples()}
    index = {k:i for i,k in enumerate(keys)}
    rows = {}
    for name, selected in allocations.items():
        post = before.copy(); deltas=[]
        for key,action in selected:
            b,a = lookup[(key,action)]; post[index[key]]=a; deltas.append(b-a)
        delta = np.asarray(deltas)
        rows[name] = {'post_mae':float(post.mean()), 'actions':len(selected),
            'harmful_action_rate':float(np.mean(delta<0)) if len(delta) else None,
            'gross_harm':float(-delta[delta<0].sum()),
            'action_mix':{a:Counter(a2 for _,a2 in selected)[a] for a in actions}}
    for row in rows.values():
        row['mae_reduction_vs_frozen_benefit_percent'] = 100*(rows['frozen_benefit']['post_mae']-row['post_mae'])/rows['frozen_benefit']['post_mae']
        row['mae_reduction_vs_frozen_risk_percent'] = 100*(rows['frozen_risk']['post_mae']-row['post_mae'])/rows['frozen_risk']['post_mae']
    return rows


def run_source(root, spec, cfg, output):
    output.mkdir(mode=0o700)
    started=time.perf_counter(); base=root/'experiments/outputs'; hashes={}
    pair_dir=base/spec['pairs']; value_dir=base/spec['value']; score_dir=base/spec['scores']; outcome_dir=base/spec['outcomes']
    audits = {name:json.loads(path.read_text()) for name,path in {
        'pairs':pair_dir/'prediction_pair_generation_audit.json', 'value':value_dir/'value_policy_fit_audit.json',
        'scores':score_dir/'policy_score_assembly_audit_absolute_price_error.json',
        'outcomes':outcome_dir/'evaluation_outcome_join_audit.json',
        'historical':base/spec['historical_evaluation']/'bvival_evaluation_audit.json'}.items()}
    for folder,name in [(pair_dir,'prediction_pair_generation_audit.json'),(value_dir,'value_policy_fit_audit.json'),
            (score_dir,'policy_score_assembly_audit_absolute_price_error.json'),(outcome_dir,'evaluation_outcome_join_audit.json'),
            (base/spec['historical_evaluation'],'bvival_evaluation_audit.json')]:
        path=folder/name; hashes[str(path)]=file_sha256(path)
    action_dir=base/spec['development_actions']
    action_audit_file=action_dir/'bvival_action_dataset_audit.json'
    action_audit=json.loads(action_audit_file.read_text()); hashes[str(action_audit_file)]=file_sha256(action_audit_file)
    action_file=checked_file(action_dir/'bvival_action_dataset.csv',audits['value']['inputs']['train_actions_sha256'],hashes)
    if action_audit['output_sha256'] != file_sha256(action_file) or not action_audit['all_predictions_oof']:
        raise ValueError('Original value-head training target provenance mismatch')
    for kind,suffix in [('pre_action_features','features'),('prediction_pairs','pairs'),('labels','labels')]:
        if action_audit['inputs'][kind]['sha256'] != audits['pairs']['outputs']['development'][suffix]:
            raise ValueError('Value head and neighbor do not share original OOF training inputs')
    if audits['value']['inputs']['evaluation_features_sha256'] != audits['pairs']['outputs']['evaluation']['features']:
        raise ValueError('Original value head evaluation contexts differ')
    for partition in ['development','evaluation']:
        for kind,suffix in [('features','pre_action_features'),('pairs','prediction_pairs')]:
            checked_file(pair_dir/f'{partition}_{suffix}.csv',audits['pairs']['outputs'][partition][kind],hashes)
    label_file=checked_file(pair_dir/'development_labels.csv',audits['pairs']['outputs']['development']['labels'],hashes)
    score_file=checked_file(score_dir/'frozen_policy_scores_absolute_price_error.csv',audits['historical']['frozen_policy_scores_sha256'],hashes)
    outcome_file=checked_file(outcome_dir/'evaluation_outcomes.csv',audits['outcomes']['output_sha256'],hashes)
    if not audits['outcomes']['labels_opened'] or audits['historical']['evaluation_outcomes_sha256'] != file_sha256(outcome_file):
        raise ValueError('Must use exactly historically opened evaluation outcomes')
    actions=audits['pairs']['action_fields']
    fields=[f for f in audits['value']['feature_columns'] if f not in ['action_id',*actions]]
    categorical=[f for f in audits['value']['categorical_columns'] if f in fields]
    numeric=[f for f in fields if f not in categorical]
    train_pre=read_csv(pair_dir/'development_pre_action_features.csv'); query_pre=read_csv(pair_dir/'evaluation_pre_action_features.csv')
    train_keys,contexts=context_table(train_pre,actions=actions,fields=fields,categorical=categorical)
    keys,query=context_table(query_pre,actions=actions,fields=fields,categorical=categorical)
    if set(train_keys)&set(keys):
        raise ValueError('Development/evaluation IDs overlap')
    train_pairs=read_csv(pair_dir/'development_prediction_pairs.csv')
    gains=development_gains(train_pairs,read_csv(label_file),train_keys,actions,audits['pairs']['folds'])
    original_train=read_csv(action_file)
    original_gain=original_train.pivot(index='listing_id',columns='action_id',values='value_absolute_price_error').reindex(index=train_keys,columns=actions).to_numpy(float)
    if not np.allclose(gains,original_gain,rtol=1e-12,atol=1e-8,equal_nan=True):
        raise ValueError('Recomputed neighbor targets differ from original benefit-head targets')
    model=AvailableNeighborGain(categorical,numeric,max_categories=cfg['max_categories'],working_memory=cfg['working_memory_mib'])
    fit_start=time.perf_counter(); model.fit_gains(train_keys,contexts,gains); fit_seconds=time.perf_counter()-fit_start
    score_start=time.perf_counter(); neighbor=model.predict(query,cfg['neighbor_counts']); score_seconds=time.perf_counter()-score_start
    scores=read_csv(score_file)
    if set(zip(scores.listing_id,scores.action_id)) != set(zip(query_pre.listing_id,query_pre.action_id)):
        raise ValueError('Score and pre-action candidate universes differ')
    allocations=freeze_allocations(scores,neighbor,keys,actions,cfg)
    additional=scores[['listing_id','action_id']].copy(); indices={key:i for i,key in enumerate(keys)}
    for k in cfg['neighbor_counts']:
        additional[f'neighbor_k{k}']=[neighbor[k][indices[key],actions.index(action)] for key,action in zip(additional.listing_id,additional.action_id)]
    additional.to_csv(output/'scores.csv',index=False)
    write_json(output/'allocations.json',allocations)
    frozen_scores_hash=file_sha256(output/'scores.csv'); frozen_actions_hash=file_sha256(output/'allocations.json')
    # Only now parse existing opened evaluation labels / after predictions.
    outcomes=read_csv(outcome_file)
    if set(zip(outcomes.listing_id,outcomes.action_id)) != set(zip(scores.listing_id,scores.action_id)):
        raise ValueError('Evaluation action universe differs')
    rows=summarize(outcomes,allocations,keys,actions)
    historical=base/spec['historical_evaluation']/'bvival_error_budget_curve.csv'
    checked_file(historical,audits['historical']['outputs'][historical.name],hashes)
    old=read_csv(historical)
    for name,policy in [('no_acquisition','score_no_acquisition'),('frozen_benefit','score_mean_value'),('frozen_risk','score_uncertainty_only')]:
        match=old[(old.policy==policy)&np.isclose(old.budget,cfg['budget'])]
        if len(match)!=1 or not np.isclose(rows[name]['post_mae'],float(match.iloc[0].post_mae_price),rtol=1e-12,atol=1e-8):
            raise ValueError('Historical MAE replay differs; stop')
    if file_sha256(output/'scores.csv')!=frozen_scores_hash or file_sha256(output/'allocations.json')!=frozen_actions_hash:
        raise ValueError('Evaluation mutated frozen choices')
    result={'source':spec['name'],'stage':cfg['stage'],'development_listings':len(train_keys),'evaluation_listings':len(keys),
        'development_partition':'historical_train_plus_validation_OOF_not_train_only',
        'action_fields':actions,'categorical_fields':categorical,'numeric_fields':numeric,
        'development_eligible_counts':dict(zip(actions,(~np.isnan(gains)).sum(axis=0).tolist())),
        'fallback_counts_all_query_action_scores':{str(k):dict(zip(actions,model.last_fallback_counts_[k].tolist())) for k in cfg['neighbor_counts']},
        'rows':rows,'preprocessed_dimensions':int(model.train_.shape[1]),
        'timing_seconds':{'representation_and_cache_fit':fit_seconds,'retrieval_all_three_k':score_seconds,'total':time.perf_counter()-started},
        'selected_neighbor_count':None,'valuation_models_refitted':False,'full_published_method_reproduction':False,
        'all_scores_and_actions_saved_before_evaluation_label_parsing':True,'historical_reference_replay_verified':True,
        'p_values_or_intervals_computed':False,'new_confirmatory_validation':False}
    write_json(output/'results.json',result); write_json(output/'input_hashes.json',hashes)
    write_json(output/'artifact_hashes.json',{p.name:file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({'source':spec['name'],'development_listings':len(train_keys),'evaluation_listings':len(keys),
                     'rows':{n:{'mae':r['post_mae'],'vs_benefit_percent':r['mae_reduction_vs_frozen_benefit_percent'],'actions':r['actions']} for n,r in rows.items()}}),flush=True)
    return result


def run(root, config, output):
    cfg=json.loads(config.read_text()); validate_config(cfg)
    if output.exists():
        raise ValueError('New output directory required; never overwrite evidence')
    output.mkdir(parents=True,mode=0o700); write_json(output/'config_snapshot.json',cfg)
    for name in ['run_bvival_neighbor_historical.py','bvival_neighbor_available.py','bvival_neighbor_loss.py','tests/test_bvival_neighbor_historical.py']:
        path=root/'experiments'/name; shutil.copyfile(path,output/(path.name+'.snapshot'))
    results=[run_source(root,s,cfg,output/s['name']) for s in cfg['sources']]
    write_json(output/'summary.json',{'stage':cfg['stage'],'sources':results,'pooled_mae_or_significance_computed':False,
        'config_sha256':file_sha256(config),'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}})
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(Path(__file__).resolve().parents[1],args.config.resolve(),args.output.resolve())
