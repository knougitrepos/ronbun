"""Secondary check prompted by the low FPIR of the first outer holdout.

All listed outer/inner seeds are reported; none is selected for deployment.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import (BASE, CONDITION, FIQA_DIRS, ROOT, evaluate,
                     deterministic_calibration_partition, fit_global_threshold,
                     fit_conditional_threshold, load_condition_score_artifact,
                     load_fiqa_score_artifact, join_fiqa_score_artifacts,
                     _validate_comparison_contract, sha256_file)

OUTER_SEEDS = (0, 1, 2, 3, 314159)
INNER_SEEDS = (42, 8972)
TARGETS = (.01, .05)
output = Path(__file__).parent / 'holdout-sensitivity-v1'
if output.exists():
    raise FileExistsError(output)
condition = load_condition_score_artifact(CONDITION)
joined = {}
for variant, path in FIQA_DIRS.items():
    artifact = load_fiqa_score_artifact(path)
    cal, test = join_fiqa_score_artifacts(condition, artifact)
    _validate_comparison_contract(cal, test, condition_manifest=condition.manifest,
                                  fiqa_manifest=artifact.manifest)
    joined[variant] = (cal, test)
rows, concentration = [], []
cal = joined['S'][0]
for target in TARGETS:
    model = fit_global_threshold(cal, target_fpir=target, safety_fraction=0,
                                 score_space=condition.manifest['score_space'])
    nm = cal.loc[~cal.is_mated].copy()
    nm['fa'] = nm.score.ge(model.global_final_threshold)
    counts = nm.groupby('identity_id').fa.sum().sort_values(ascending=False)
    concentration.append({'target_fpir': target, 'identities': len(counts),
                          'non_mated_rows': len(nm), 'false_accept_count': int(counts.sum()),
                          'identities_with_false_accept': int(counts.gt(0).sum()),
                          'top1_share': float(counts.head(1).sum()/counts.sum()),
                          'top5_share': float(counts.head(5).sum()/counts.sum()),
                          'top10_share': float(counts.head(10).sum()/counts.sum()),
                          'top10_probe_fraction': float(nm.identity_id.isin(counts.head(10).index).mean())})
for outer_seed in OUTER_SEEDS:
    mask = deterministic_calibration_partition(cal, safety_fraction=.2, seed=outer_seed,
                                                partition_column='identity_id').eq('safety').to_numpy()
    for inner_seed in INNER_SEEDS:
        for target in TARGETS:
            common = dict(target_fpir=target, partition_seed=inner_seed,
                          partition_column='identity_id', safety_fraction=.3,
                          score_space=condition.manifest['score_space'])
            slim = ['sample_id','identity_id','is_mated','score','fiqa_score']
            fitted = {'global_safe': fit_global_threshold(cal.loc[~mask, slim], **common)}
            for variant in ('S','L'):
                fitted['fiqa_'+variant.lower()] = fit_conditional_threshold(
                    joined[variant][0].loc[~mask, slim], bin_count=2,
                    shrinkage_strength=200, minimum_group_non_mated=100, **common)
            for method, model in fitted.items():
                v = 'L' if method=='fiqa_l' else 'S'
                h = evaluate(joined[v][0].loc[mask], model)
                t = evaluate(joined[v][1], model, genuine=True)
                rows.append(dict(outer_seed=outer_seed, inner_seed=inner_seed,
                                 target_fpir=target, method=method,
                                 holdout_fpir=h['fpir'], test_fpir=t['fpir'],
                                 test_minus_holdout=t['fpir']-h['fpir'],
                                 holdout_target_met=h['target_met'], test_target_met=t['target_met'],
                                 holdout_false_accept_count=h['false_accept_count'],
                                 holdout_non_mated_count=h['non_mated_count'],
                                 test_false_accept_count=t['false_accept_count'],
                                 test_non_mated_count=t['non_mated_count']))
    print('outer seed', outer_seed, 'done', flush=True)
frame = pd.DataFrame(rows)
output.mkdir()
frame.to_csv(output/'metrics.csv', index=False)
pd.DataFrame(concentration).to_csv(output/'calibration_tail_concentration.csv', index=False)
manifest = dict(status='completed', outer_seeds=OUTER_SEEDS, inner_seeds=INNER_SEEDS,
                target_fpirs=TARGETS, condition_uid=condition.condition_uid,
                exploratory_followup=True, script_sha256=sha256_file(Path(__file__)),
                source_analysis_manifest_sha256=sha256_file(Path(__file__).parent/'audit-v1/manifest.json'),
                files={p.name: sha256_file(p) for p in output.iterdir()})
(output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(frame.groupby(['method','target_fpir']).agg(
    count=('inner_seed','count'),holdout_min=('holdout_fpir','min'),holdout_max=('holdout_fpir','max'),
    test_min=('test_fpir','min'),test_max=('test_fpir','max'),
    gap_min=('test_minus_holdout','min'),gap_max=('test_minus_holdout','max'),
    test_met=('test_target_met','sum')).to_string())
print(pd.DataFrame(concentration).to_string(index=False))
