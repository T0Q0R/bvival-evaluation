"""Separated freeze/development/score/evaluate phases for the AT comparison.

No target values printed. The scores phase has no source-workbook argument.
Evaluation joins labels ONLY to hashed, already selected actions; no training,
selection, reallocation or target-dependent capacity change is allowed there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import joblib
import numpy as np

from audit_european_at_schema import AT_SHA256, probe, recognized_header, workbook_sheets
from audit_turkey_feature_only import CELL, fragments, cell_reference, row_number, decode_approved_cell, resolve_shared_strings
from build_european_at_cohort import ACTIONS, ROLE_MONTHS
import european_at_matched as common

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["audit_european_at_schema.py", "audit_european_at_features.py", "build_european_at_cohort.py",
           "european_at_matched.py", "run_european_at_matched.py", "audit_turkey_feature_only.py",
           "archive_turkey_candidate.py", "bvival_neighbor_available.py", "bvival_neighbor_loss.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    path.chmod(0o600)


def fresh(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError("New empty output directory required")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def config_and_cohort(config_path, cohort_path):
    cfg, records = json.loads(config_path.read_text()), json.loads(cohort_path.read_text())
    if sha(cohort_path) != cfg['cohort_sha256'] or cfg['source_sha256'] != AT_SHA256 or cfg['actions'] != list(ACTIONS):
        raise ValueError("Fixed source/action/cohort hash mismatch")
    if cfg['role_months'] != {k: list(v) for k, v in ROLE_MONTHS.items()}:
        raise ValueError("Roles differ from price-free construction")
    if Counter(r['role'] for r in records) != Counter(cfg['expected_nonprice_counts']):
        raise ValueError("Cohort counts differ from price-free receipt")
    if len(records) != len({r['record_key'] for r in records}) or len(records) != len({r['group_hash'] for r in records}):
        raise ValueError("Each key and title-year proxy must be unique across roles")
    for r in records:
        if r['month'] not in cfg['role_months'][r['role']] or set(r['context']) != set(cfg['initial_context']) or set(r['actions']) != set(ACTIONS):
            raise ValueError("Cohort role/features disagree with protocol")
        if r['role'] == 'development':
            expected = int(hashlib.sha256((cfg['oof_namespace'] + r['group_hash']).encode()).hexdigest()[:16], 16) % 5
            if r['oof_fold'] != expected:
                raise ValueError("OOF fold differs from frozen hash")
    return cfg, records


def freeze(config, cohort, output):
    cfg, records = config_and_cohort(config, cohort)
    fresh(output)
    (output / 'config.json').write_bytes(config.read_bytes())
    (output / 'protocol.md').write_bytes((ROOT/'notes/design/bvival-european-at-matched-protocol-2026-10-04.md').read_bytes())
    (output / 'code').mkdir(mode=0o700)
    hashes = {}
    for name in SOURCES:
        src = ROOT/'experiments'/name
        (output/'code'/name).write_bytes(src.read_bytes())
        hashes[name] = sha(src)
    import catboost, sklearn, pandas
    import sys
    write(output/'freeze_receipt.json', {
        'stage': 'FIXED_TASK_AND_CODE_BEFORE_ANY_SOURCE_TARGET_DECODE', 'time_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': sha(output/'config.json'), 'protocol_sha256': sha(output/'protocol.md'),
        'cohort_sha256': sha(cohort), 'source_sha256': cfg['source_sha256'], 'source_hashes': hashes,
        'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pandas.__version__,
                     'sklearn': sklearn.__version__, 'catboost': catboost.__version__},
        'any_source_target_values_decoded_before_freeze': False,
        'external_preregistration_or_independent_label_custody': False,
        'formal_human_author_approval_claimed': False,
        'reserved_months_prices_will_not_be_read': True,
    })
    print(json.dumps({'stage': 'TASK_CODE_FROZEN_BEFORE_LABELS', 'freeze_receipt_sha256': sha(output/'freeze_receipt.json')}))


def bound(freeze_dir, cohort):
    receipt = json.loads((freeze_dir/'freeze_receipt.json').read_text())
    if sha(freeze_dir/'config.json') != receipt['config_sha256'] or sha(freeze_dir/'protocol.md') != receipt['protocol_sha256']:
        raise ValueError("Protocol/config changed after freeze")
    for name, expected in receipt['source_hashes'].items():
        if sha(ROOT/'experiments'/name) != expected or sha(freeze_dir/'code'/name) != expected:
            raise ValueError("Bound source code changed after freeze")
    return config_and_cohort(freeze_dir/'config.json', cohort)


def positive_price(value):
    # Excel numeric payloads use dot decimals. No guessed separators/currency.
    try:
        n = Decimal(value.strip())
        f = float(n)
    except (InvalidOperation, ValueError, OverflowError):
        return None
    return f if n.is_finite() and np.isfinite(f) and f > 0 else None


def load_prices(source, records, allowed_roles):
    if not records or any(r['role'] not in allowed_roles for r in records):
        raise ValueError("Requested record roles not authorized for this label phase")
    probe(source)  # Exact hash + header-only read before target selection.
    wanted = {(r['month'], r['source_row']): r['record_key'] for r in records}
    if len(wanted) != len(records):
        raise ValueError("Duplicate requested source coordinates")
    raw, needed, counts = {}, set(), Counter()
    with zipfile.ZipFile(source) as z:
        for month, member in workbook_sheets(z):
            if not any(m == month for m, _ in wanted):
                counts['entire_unrequested_months_skipped'] += 1
                continue
            header = recognized_header(z, member, month)
            cols = [c for c, label in header.items() if label == 'price']
            if len(cols) != 1:
                raise ValueError("One exact price header required")
            price_col = cols[0]
            with z.open(member) as stream:
                for block in fragments(stream, b'row'):
                    number = row_number(block)
                    if (month, number) not in wanted:
                        # Crucially: do not call the cell payload decoder here.
                        counts['unrequested_rows_skipped_before_cell_decode'] += 1
                        continue
                    key = wanted[(month, number)]
                    if key in raw:
                        raise ValueError("Duplicate target record")
                    cells = [c for c in CELL.findall(block) if cell_reference(c) == (price_col, number)]
                    if len(cells) > 1:
                        raise ValueError("Duplicate price cell")
                    if not cells:
                        raw[key] = ('text', '')
                    else:
                        raw[key] = decode_approved_cell(cells[0], price_col, {price_col})
                        counts['requested_price_cells_decoded'] += 1
                        if raw[key][0] == 'shared':
                            needed.add(raw[key][1])
        strings = resolve_shared_strings(z, needed)
    if set(raw) != {r['record_key'] for r in records}:
        raise ValueError("Requested target universe incomplete")
    prices = {k: positive_price(strings[v] if kind == 'shared' else v) for k, (kind, v) in raw.items()}
    counts['invalid_requested_targets'] = sum(v is None for v in prices.values())
    return prices, dict(counts)


def artifact_hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def development(source, freeze_dir, cohort, output):
    cfg, records = bound(freeze_dir, cohort)
    permitted = [r for r in records if r['role'] in {'development','validation'}]
    fresh(output)
    prices, access = load_prices(source, permitted, {'development','validation'})
    train = [r for r in permitted if r['role']=='development' and prices[r['record_key']] is not None]
    val = [r for r in permitted if r['role']=='validation']
    if not train or not val:
        raise ValueError("Nonempty training and validation required")
    oof, fit_audit, models_dir = np.empty((len(train), 5)), [], output/'models'
    models_dir.mkdir(mode=0o700)
    for fold in range(5):
        ix = [i for i,r in enumerate(train) if r['oof_fold']==fold]
        fitting = [r for r in train if r['oof_fold']!=fold]
        if not ix or not fitting:
            raise ValueError("Every planned fold must be nonempty")
        start=time.perf_counter()
        print(f'OOF {fold+1}/5: shared five-state predictor, {len(fitting)} fitting proxy groups', flush=True)
        models = common.fit_base(fitting, [prices[r['record_key']] for r in fitting], cfg)
        oof[ix] = common.predict_base(models, [train[i] for i in ix])
        fit_audit.append({'stage':f'oof_{fold}', 'fit_groups':len(fitting), 'heldout_groups':len(ix),
                          'fit_heldout_overlap':0, 'seconds':time.perf_counter()-start})
    print('Final development-only shared predictor; no validation refit', flush=True)
    models=common.fit_base(train,[prices[r['record_key']] for r in train],cfg)
    joblib.dump(models, models_dir/'base.joblib')
    val_logs=common.predict_base(models,val)
    train_ctx=common.contexts(train,oof)
    loss=common.errors([prices[r['record_key']] for r in train],oof)
    gain_models,risk_models,gains,scale=common.fit_heads(train_ctx,loss,cfg)
    joblib.dump({'gain':gain_models,'risk':risk_models,'scale':scale}, models_dir/'heads.joblib')
    near=common.fit_neighbor([r['record_key'] for r in train],train_ctx,gains,cfg)
    joblib.dump(near,models_dir/'neighbor.joblib')
    val_ctx=common.contexts(val,val_logs)
    gv,rv=common.predict_heads(gain_models,risk_models,val_ctx)
    nvalues=near.predict(val_ctx,cfg['neighbor']['k_candidates'])
    keys=[r['record_key'] for r in val]; vp=[prices[k] for k in keys]
    fields={}
    for i,a in enumerate(ACTIONS):
        acts=common.allocation(keys,gv[:,i],np.full(len(keys),i),cfg['budget_fraction'])
        fields[a]=common.observed_mae(val_logs,vp,acts)
    fixed=min(ACTIONS,key=lambda a:(fields[a],ACTIONS.index(a)))
    ks={}
    for k,ng in nvalues.items():
        choices=common.policies(keys,gv,rv,ng,fixed,cfg['budget_fraction'])
        ks[k]=common.observed_mae(val_logs,vp,choices['P4'])
    k=min(ks,key=lambda k:(ks[k],k))
    write(output/'selected_parameters.json',{'fixed_field':fixed,'neighbor_k':k,
            'validation_fixed_field_P2_mae_EUR':fields,'validation_P4_mae_EUR_by_k':ks,
            'validation_candidates':len(val),'validation_valid_targets':sum(v is not None for v in vp),
            'common_oof_currency_scale':scale})
    write(output/'development_receipt.json',{'stage':'DEVELOPMENT_AND_SELECTION_COMPLETE_TEST_LABELS_UNREAD',
            'freeze_receipt_sha256':sha(freeze_dir/'freeze_receipt.json'), 'cohort_sha256':sha(cohort),
            'permitted_label_roles':['development','validation'], 'label_access_counts':access,
            'development_valid_targets':len(train), 'base_fit_count':18,'gain_head_fit_count':3,'risk_head_fit_count':3,
            'fit_audit':fit_audit, 'reserved_or_evaluation_price_cells_decoded':0,
            'artifact_hashes_before_receipt':artifact_hashes(output)})
    print(json.dumps({'stage':'DEVELOPMENT_COMPLETE_NO_TEST_LABELS_READ','fixed_field':fixed,'neighbor_k':k}),flush=True)


def verify_development(directory, freeze_dir):
    receipt=json.loads((directory/'development_receipt.json').read_text())
    if receipt['freeze_receipt_sha256']!=sha(freeze_dir/'freeze_receipt.json'):
        raise ValueError('Development bound to different protocol')
    for name,h in receipt['artifact_hashes_before_receipt'].items():
        if sha(directory/name)!=h:
            raise ValueError('Development model/selection changed')


def scores(freeze_dir,cohort,development_dir,output):
    cfg, records=bound(freeze_dir,cohort)
    verify_development(development_dir,freeze_dir)
    fresh(output)
    evaluation=[r for r in records if r['role']=='evaluation']
    # This phase has no source path and no target-loading function call.
    logs=common.predict_base(joblib.load(development_dir/'models/base.joblib'),evaluation)
    heads=joblib.load(development_dir/'models/heads.joblib')
    ctx=common.contexts(evaluation,logs)
    gains,risk=common.predict_heads(heads['gain'],heads['risk'],ctx)
    selected=json.loads((development_dir/'selected_parameters.json').read_text())
    neighbor=joblib.load(development_dir/'models/neighbor.joblib').predict(ctx,[selected['neighbor_k']])[selected['neighbor_k']]
    keys=[r['record_key'] for r in evaluation]
    choices=common.policies(keys,gains,risk,neighbor,selected['fixed_field'],cfg['budget_fraction'])
    write(output/'frozen_predictions_scores_and_actions.json',{'record_keys':keys,'five_state_logs':logs.tolist(),
            'gain_head_scores':gains.tolist(),'risk_scores':risk.tolist(),'neighbor_scores':neighbor.tolist(),
            'exact_actions':{p:a.tolist() for p,a in choices.items()},'selected_parameters':selected})
    write(output/'score_freeze_receipt.json',{'stage':'ALL_EVALUATION_PREDICTIONS_SCORES_ACTIONS_FROZEN_BEFORE_TARGETS',
            'time_utc':datetime.now(timezone.utc).isoformat(),'freeze_receipt_sha256':sha(freeze_dir/'freeze_receipt.json'),
            'development_receipt_sha256':sha(development_dir/'development_receipt.json'),
            'predictions_scores_actions_sha256':sha(output/'frozen_predictions_scores_and_actions.json'),
            'evaluation_candidates':len(keys),'actions_each_policy':{p:int((a>=0).sum()) for p,a in choices.items()},
            'source_workbook_opened_here':False,'evaluation_targets_read':False,
            'independent_label_custody_claimed':False})
    print(json.dumps({'stage':'SCORES_AND_ACTIONS_FROZEN_NO_TARGET_ACCESS','receipt_sha256':sha(output/'score_freeze_receipt.json')}))


def evaluate(source,freeze_dir,cohort,score_dir,output):
    cfg,records=bound(freeze_dir,cohort)
    fr=json.loads((score_dir/'score_freeze_receipt.json').read_text())
    predictions_path=score_dir/'frozen_predictions_scores_and_actions.json'
    if fr['freeze_receipt_sha256']!=sha(freeze_dir/'freeze_receipt.json') or fr['predictions_scores_actions_sha256']!=sha(predictions_path):
        raise ValueError('Frozen score artifact changed or belongs to different plan')
    frozen=json.loads(predictions_path.read_text())
    evaluation=[r for r in records if r['role']=='evaluation']
    if frozen['record_keys']!=[r['record_key'] for r in evaluation]:
        raise ValueError('Frozen evaluation key order differs')
    fresh(output)
    marker=freeze_dir/'evaluation_label_access_started.json'
    # Durable first-access marker written before the target-decoder entry point.
    write(marker,{'stage':'EVALUATION_LABEL_ACCESS_STARTED_NOT_RESULT','time_utc':datetime.now(timezone.utc).isoformat(),
                  'score_receipt_sha256':sha(score_dir/'score_freeze_receipt.json'),'output':str(output)})
    prices,access=load_prices(source,evaluation,{'evaluation'})
    result=common.summarize(frozen['five_state_logs'],[prices[k] for k in frozen['record_keys']],frozen['exact_actions'],cfg)
    result.update(stage='FIXED_NEW_TIME_COHORT_MATCHED_COMPARISON_COMPLETE',price_unit='EUR',
                  selected_parameters=frozen['selected_parameters'],label_access_counts=access,
                  plan_freeze_receipt_sha256=sha(freeze_dir/'freeze_receipt.json'),
                  score_freeze_receipt_sha256=sha(score_dir/'score_freeze_receipt.json'),
                  predictions_actions_unchanged_after_label_join=True,
                  source_rows_or_prices_publicly_exported=False, independent_label_custody_or_preregistration_claimed=False)
    write(output/'aggregate_results.json',result)
    print(json.dumps({'stage':result['stage'],'valid_observed_target_n':result['valid_observed_target_n'],
                      'primary_comparisons':result['primary_comparisons']}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['freeze','development','scores','evaluate'],required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--freeze-dir',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--development-dir',type=Path)
    p.add_argument('--score-dir',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    required={'freeze':['config'],'development':['source','freeze_dir'],'scores':['freeze_dir','development_dir'],
              'evaluate':['source','freeze_dir','score_dir']}[args.stage]
    if any(getattr(args,name) is None for name in required):
        p.error('Missing required stage input')
    if args.stage=='scores' and args.source is not None:
        p.error('Scores phase must not receive source workbook')
    if args.stage=='freeze': freeze(args.config,args.cohort,args.output_dir)
    elif args.stage=='development': development(args.source,args.freeze_dir,args.cohort,args.output_dir)
    elif args.stage=='scores': scores(args.freeze_dir,args.cohort,args.development_dir,args.output_dir)
    else: evaluate(args.source,args.freeze_dir,args.cohort,args.score_dir,args.output_dir)


if __name__=='__main__':
    main()
