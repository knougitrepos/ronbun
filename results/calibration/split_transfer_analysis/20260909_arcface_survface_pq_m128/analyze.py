"""Read-only source audit; writes only new diagnostic outputs beside this script.

This is an analysis companion, not a change to the research pipeline.
Run from C:/ronbun: py -3.11 results/calibration/split_transfer_analysis/
20260909_arcface_survface_pq_m128/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from research.calibration.conditional import (  # noqa: E402
    apply_threshold_model, assign_quality_groups,
    deterministic_calibration_partition, fit_conditional_threshold,
    fit_global_threshold,
)
from research.evaluation.metrics import wilson_score_interval  # noqa: E402
from research.experiments.fiqa_threshold_calibration import (  # noqa: E402
    _validate_comparison_contract, join_fiqa_score_artifacts,
    load_condition_score_artifact,
)
from research.fiqa import load_fiqa_score_artifact  # noqa: E402
from research.runtime.hashing import sha256_file  # noqa: E402

# Fixed before inspecting multi-seed outcomes. No best-seed selection.
SEEDS = (*range(18), 42, 8972)
TARGETS = (.01, .05, .10)
OUTER_HOLDOUT_SEED = 314159
OUTER_HOLDOUT_FRACTION = .20
SAFETY_FRACTION = .30
RUN_ID = '20260902-R001-61915edf'
BASE = ROOT / 'results/calibration'
CONDITION = BASE / 'condition_scores' / RUN_ID / 'pq_512_m128_b8__pq_adc_exhaustive/genuine-score-topk-v2'
FIQA_DIRS = {
    'S': BASE / 'fiqa_scores/survface/cr-fiqa-s-b9f457a6f00e0363a0cf',
    'L': BASE / 'fiqa_scores/survface/cr-fiqa-l-5fca24736e4f8df5fbfc',
}
REFERENCE = BASE / 'fiqa_priority_diagnostics' / RUN_ID / 'fiqa-priority-f78e2baa54436e2f09c95f15'
RUN = ROOT / 'runs/survface_20260902/20260902-R001-61915edf_step4_survface_arcface-7972a704552df378345f'
OUTPUT = Path(__file__).parent / 'audit-v1'


def frozen_groups(frame, model):
    if model.quality_column is None:
        return np.full(len(frame), 'all')
    return assign_quality_groups(frame[model.quality_column],
                                 cutpoints=model.quality_cutpoints,
                                 labels=model.group_labels)


def evaluate(frame, model, *, genuine=False):
    groups = frozen_groups(frame, model)
    tau = pd.Series(groups).map(model.thresholds).to_numpy(float)
    nm = ~frame.is_mated.to_numpy(bool)
    accept = frame.score.to_numpy(float) >= tau
    count, total = int(accept[nm].sum()), int(nm.sum())
    low, high = wilson_score_interval(count, total)
    out = dict(non_mated_count=total, false_accept_count=count,
               fpir=count/total, wilson95_low=low, wilson95_high=high,
               target_met=count/total <= model.target_fpir)
    if genuine:
        success = ((~nm) & frame.top_k_correct.to_numpy(bool)
                   & (frame.true_identity_score.to_numpy(float) >= tau))
        out.update(tpir20_count=int(success.sum()), mated_count=int((~nm).sum()),
                   tpir20=float(success.sum()/(~nm).sum()))
    return out


def group_rates(frame, model, split):
    frame = frame.loc[~frame.is_mated]
    groups = frozen_groups(frame, model)
    rows = []
    for label in model.group_labels:
        scores = frame.loc[groups == label, 'score'].to_numpy(float)
        assert len(scores)
        rows.append(dict(split=split, group=label, count=len(scores),
                         fraction=len(scores)/len(frame),
                         false_accept_count=int((scores >= model.thresholds[label]).sum()),
                         fpir=float(np.mean(scores >= model.thresholds[label])),
                         threshold=model.thresholds[label]))
    return rows


def main():
    started = time.perf_counter()
    if OUTPUT.exists():
        raise FileExistsError(f'Analysis outputs are immutable: {OUTPUT}')
    condition = load_condition_score_artifact(CONDITION)
    assert condition.condition_uid == 'compressed-scores-2548cd9ab45de52c3b614a9a'
    source_files = [CONDITION / 'manifest.json']
    source_files += [CONDITION / name for name in condition.manifest['files']]
    reference_manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    for name, digest in reference_manifest['files'].items():
        assert sha256_file(REFERENCE / name) == digest
        source_files.append(REFERENCE / name)
    source_files.append(REFERENCE / 'manifest.json')
    for path, field in ((RUN / 'run_manifest.json', 'source_run_manifest_sha256'),
                        (RUN / 'artifacts/step2_workflow/freeze_manifest.json', 'source_freeze_manifest_sha256')):
        assert sha256_file(path) == condition.manifest[field]
        source_files.append(path)
    core = Path(condition.manifest['upgrade_provenance']['test_genuine_scores_source'])
    assert sha256_file(core) == condition.manifest['persisted_test_core_sha256']
    source_files.append(core)
    joined = {}
    for variant, path in FIQA_DIRS.items():
        artifact = load_fiqa_score_artifact(path)
        assert artifact.manifest['variant'] == variant
        cal, test = join_fiqa_score_artifacts(condition, artifact)
        _validate_comparison_contract(cal, test, condition_manifest=condition.manifest,
                                      fiqa_manifest=artifact.manifest)
        joined[variant] = (cal, test)
        source_files.append(path / 'manifest.json')
        source_files += [path / name for name in artifact.manifest['files']]
    for column in ['sample_id', 'identity_id', 'score', 'is_mated']:
        for split in (0, 1):
            assert joined['S'][split][column].equals(joined['L'][split][column])
    source_files += [ROOT / p for p in ('research/calibration/conditional.py',
        'research/calibration/rejection.py', 'research/experiments/fiqa_threshold_calibration.py',
        'research/protocols/open_set.py')]
    source_files.append(Path(__file__))
    original_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    ref = pd.read_csv(REFERENCE / 'method_summary.csv')
    summaries, groups, models_saved, partitions, gaps = [], [], [], [], []
    print('Verified sources; seeds locked:', SEEDS, flush=True)
    cal = joined['S'][0]
    outer = deterministic_calibration_partition(cal, safety_fraction=OUTER_HOLDOUT_FRACTION,
                                                seed=OUTER_HOLDOUT_SEED, partition_column='identity_id')
    outer_mask = outer.eq('safety').to_numpy()
    assert not set(cal.loc[outer_mask, 'identity_id']) & set(cal.loc[~outer_mask, 'identity_id'])
    inventory = []
    for name, frame in [('calibration', cal), ('test', joined['S'][1]),
                        ('calibration_holdout', cal.loc[outer_mask])]:
        for mated in (False, True):
            subset = frame.loc[frame.is_mated.eq(mated)]
            inventory.append(dict(population=name, is_mated=mated, rows=len(subset),
                                  identity_key_count=subset.identity_id.nunique(),
                                  identity_semantics='synthetic_unknown' if name=='test' and not mated else 'labelled'))
    slim = ['sample_id', 'identity_id', 'is_mated', 'score', 'fiqa_score']
    for regime in ['full_calibration', 'independent_calibration_holdout']:
        train = {v: frames[0].loc[~outer_mask if regime.endswith('holdout') else np.ones(len(cal), bool), slim].reset_index(drop=True)
                 for v, frames in joined.items()}
        test_frames = {v: frames[1] for v, frames in joined.items()}
        holdout = {v: frames[0].loc[outer_mask].reset_index(drop=True) for v, frames in joined.items()}
        for seed in SEEDS:
            partition = deterministic_calibration_partition(train['S'], safety_fraction=SAFETY_FRACTION,
                                                            seed=seed, partition_column='identity_id')
            for label in ('fit', 'safety'):
                f = train['S'].loc[partition.eq(label)]
                partitions.append(dict(regime=regime, seed=seed, split=label, rows=len(f),
                                       non_mated_count=int((~f.is_mated).sum()),
                                       non_mated_identities=f.loc[~f.is_mated, 'identity_id'].nunique()))
            for target in TARGETS:
                common = dict(target_fpir=target, partition_seed=seed, partition_column='identity_id',
                              score_space=condition.manifest['score_space'])
                fitted = {'global_empirical': fit_global_threshold(train['S'], safety_fraction=0, **common),
                          'global_safe': fit_global_threshold(train['S'], safety_fraction=SAFETY_FRACTION, **common)}
                for v in ('S', 'L'):
                    fitted['fiqa_'+v.lower()] = fit_conditional_threshold(
                        train[v], safety_fraction=SAFETY_FRACTION, bin_count=2,
                        shrinkage_strength=200, minimum_group_non_mated=100, **common)
                for method, model in fitted.items():
                    variant = 'L' if method == 'fiqa_l' else 'S'
                    key = dict(regime=regime, seed=seed, target_fpir=target, method=method)
                    result = evaluate(test_frames[variant], model, genuine=True)
                    summaries.append({**key, 'split': 'test', **result})
                    models_saved.append({**key, 'model': model.as_dict()})
                    if regime=='full_calibration' and seed==8972:
                        r = ref.loc[ref.method.eq(method) & np.isclose(ref.target_fpir, target)].iloc[0]
                        assert result['fpir'] == r.realized_fpir or abs(result['fpir']-r.realized_fpir) < 1e-14
                        assert abs(result['tpir20']-r.tpir_at_rank_k) < 1e-14
                        canonical = apply_threshold_model(test_frames[variant], model).summary
                        assert result['false_accept_count'] == canonical['false_accept_count']
                        assert abs(result['tpir20']-canonical['tpir_at_rank_k']) < 1e-14
                    if regime.endswith('holdout'):
                        summaries.append({**key, 'split': 'calibration_holdout', **evaluate(holdout[variant], model)})
                    transfer = []
                    for label in ('fit', 'safety'):
                        transfer += group_rates(train[variant].loc[partition.eq(label)], model, label)
                    transfer += group_rates(test_frames[variant], model, 'test')
                    if regime.endswith('holdout'):
                        transfer += group_rates(holdout[variant], model, 'calibration_holdout')
                    groups.extend([{**key, **r} for r in transfer])
                    transfer = pd.DataFrame(transfer)
                    reference_split = 'calibration_holdout' if regime.endswith('holdout') else 'safety'
                    left = transfer.loc[transfer.split.eq(reference_split)].set_index('group')
                    right = transfer.loc[transfer.split.eq('test')].set_index('group')
                    mixture = float(((right.fraction-left.fraction)*left.fpir).sum())
                    within = float((right.fraction*(right.fpir-left.fpir)).sum())
                    gap = float((right.fraction*right.fpir).sum()-(left.fraction*left.fpir).sum())
                    assert abs(gap-mixture-within) < 1e-12
                    gaps.append({**key, 'reference_split': reference_split,
                                 'test_minus_reference': gap, 'mixture_component': mixture,
                                 'within_group_component': within})
            print(f'{regime} seed={seed} done; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    summary = pd.DataFrame(summaries)
    ranges = summary.groupby(['regime', 'method', 'target_fpir', 'split'], sort=False).agg(
        seeds=('seed', 'count'), target_met_seeds=('target_met', 'sum'),
        fpir_min=('fpir', 'min'), fpir_median=('fpir', 'median'), fpir_max=('fpir', 'max'),
        fpir_sd=('fpir', 'std'), min_query_wilson95_low=('wilson95_low', 'min'),
        tpir20_min=('tpir20', 'min'), tpir20_median=('tpir20', 'median'),
        tpir20_max=('tpir20', 'max')).reset_index()
    for path, digest in original_hashes.items():
        assert sha256_file(ROOT / path) == digest, f'Source changed during analysis: {path}'
    OUTPUT.mkdir()
    tables = {'seed_metrics': summary, 'seed_ranges': ranges,
              'group_transfer': pd.DataFrame(groups), 'gap_decomposition': pd.DataFrame(gaps),
              'partition_counts': pd.DataFrame(partitions), 'population_inventory': pd.DataFrame(inventory)}
    for name, table in tables.items():
        table.to_csv(OUTPUT / f'{name}.csv', index=False)
    (OUTPUT / 'models.json').write_text(json.dumps(models_saved, indent=2, allow_nan=False), encoding='utf-8')
    manifest = dict(status='completed', kind='exploratory_split_transfer_diagnostic',
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    condition_uid=condition.condition_uid, seeds=list(SEEDS), targets=list(TARGETS),
                    outer_holdout_seed=OUTER_HOLDOUT_SEED, outer_holdout_fraction=OUTER_HOLDOUT_FRACTION,
                    safety_fraction=SAFETY_FRACTION, calibration_gallery_seed=condition.manifest['calibration_seed'],
                    metric_contract=condition.manifest['metric_contract'], source_sha256=original_hashes,
                    wall_seconds=time.perf_counter()-started,
                    checks=['source hashes unchanged', 'S/L aligned', 'calibration/test disjoint',
                            'holdout/fit identity disjoint', 'seed 8972 matches stored and canonical results',
                            'mixture plus within equals total gap'],
                    limitations=['Same underlying calibration cohort, fixed gallery and codec',
                                 'Repeated-seed range is descriptive, NOT 95% CI or independent experiments',
                                 'Query-level Wilson ignores unlabelled non-mated identity correlation',
                                 'Outer holdout was not used for threshold fitting but shared frozen codec/gallery',
                                 'No best-seed selection and no threshold fit using test'])
    manifest['files'] = {p.name: sha256_file(p) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(ranges.to_string(index=False), flush=True)
    print('SAVED', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
