"""Verify the uncompressed control using saved thresholds and raw test scores."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.runtime.hashing import sha256_file  # noqa: E402

base = ROOT / 'results/paper/survface/20260902-R001-61915edf/search_space_v6_query_gallery_conditions/pq'
family = json.loads((base / 'family_manifest.json').read_text())
diag_path = base / 'origin_calibration_diagnostics.json'
assert sha256_file(diag_path) == family['outputs'][diag_path.name]['sha256']
diagnostics = json.loads(diag_path.read_text())
condition_path = ROOT / 'results/calibration/condition_scores/20260902-R001-61915edf/pq_512_m128_b8__pq_adc_exhaustive/genuine-score-topk-v2/manifest.json'
condition = json.loads(condition_path.read_text())
core_path = Path(condition['upgrade_provenance']['test_genuine_scores_source'])
assert sha256_file(core_path) == condition['persisted_test_core_sha256']
core = pd.read_parquet(core_path, columns=['is_mated', 'origin_top1_score'])
scores = core.loc[~core.is_mated, 'origin_top1_score'].to_numpy(float)
rows = []
for target in (.01, .05, .10):
    d = diagnostics['diagnostics_by_target'][str(target)]
    c, t = d['splits']['calibration'], d['splits']['test']
    tau = d['origin_decision_threshold']
    count = int(np.sum(scores >= tau))
    assert count == t['origin_false_accept_count']
    assert len(scores) == t['non_mated_count']
    rows.append(dict(target_fpir=target, threshold=tau,
                     calibration_fpir=c['origin_fpir'], test_fpir=count/len(scores),
                     test_false_accept_count=count, test_non_mated_count=len(scores),
                     calibration_templates=c['template_count'], test_templates=t['template_count'],
                     calibration_gallery_images=c['source_image_count'], test_gallery_images=t['source_image_count']))
result = {'rows': rows, 'test_counts_recomputed': True,
          'calibration_threshold_recomputed_this_turn': False,
          'threshold_source': 'verified saved calibration diagnostic',
          'score_space': 'origin cosine, independently calibrated; not reused for ADC',
          'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p)
                            for p in (diag_path, base/'family_manifest.json', core_path, condition_path, Path(__file__))}}
output = Path(__file__).parent / 'origin_control.json'
with output.open('x', encoding='utf-8') as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
