"""Real producer/export/consumer integration using synthetic weights, not a benchmark."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("kind", ["MobileFaceNet", "Resnet"])
@pytest.mark.parametrize("family", ["arcface", "cosface", "magface", "adaface"])
def test_actual_facex_producer_and_ronbun_consumer(kind, family, tmp_path, monkeypatch):
    root = Path("F:/FaceX-Zoo")
    if not (root / "fine-tune/modules/model_builder.py").exists():
        pytest.skip("FaceX producer checkout unavailable")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "fine-tune"))
    import torch
    from modules.model_builder import build_model
    from modules.checkpoint_manager import CheckpointManager, compute_config_hash, extract_model_config
    from research.embeddings.facex_zoo import model_spec_from_facex_export
    from research.embeddings.registry import create_pytorch_adapter_from_spec
    from research.runtime.hashing import sha256_file
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb = dict(type=kind, feat_dim=1024, dropout_rate=.6, out_h=7, out_w=7)
    if kind == "Resnet":
        bb.update(num_layers=50, mode="ir")
    head = dict(type=family, num_class=4)
    model = build_model(bb, head).to(device)
    # Bounded forward/backward validates both backbone and actual head wiring.
    model.train()
    logits, auxiliary = model(torch.randn(2, 3, 112, 112, device=device), torch.tensor([0, 1], device=device))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device=device))
    if auxiliary is not None:
        loss = loss + auxiliary.mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model.eval()
    cfg = extract_model_config(model)
    cfg.update(dataset=dict(name="scface"), preprocessing=dict(input_size=[112, 112], pixel_mean=127.5, pixel_std=128.0, color="BGR"))
    metadata = dict(completed=True, is_smoke=False, feat_dim=1024, run_id="synthetic_format_fixture",
                    head=family, dataset="scface", epochs=1, parent_hash="b" * 64,
                    source_sha256="a" * 64, protocol_record=dict(fixture=True),
                    source_config=cfg, source_config_hash=compute_config_hash(cfg),
                    preprocessing={**cfg["preprocessing"], "color": "BGR"})
    path = CheckpointManager(str(tmp_path)).export_backbone_only(model, "fixture.pt", metadata, target_feat_dim=1024)
    path = Path(path)
    before = sha256_file(path)
    spec = model_spec_from_facex_export(path)
    adapter = create_pytorch_adapter_from_spec(spec, device=device)
    images = np.random.default_rng(10).integers(0, 256, (2, 112, 112, 3), dtype=np.uint8)
    result = adapter.embed(images)
    with torch.inference_mode():
        native = model.backbone(adapter.preprocess(images)).detach().cpu().numpy()
    assert result.raw_embedding.shape == (2, 1024)
    np.testing.assert_allclose(result.raw_embedding, native, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(result.raw_norm, np.linalg.norm(native, axis=1), atol=1e-5, rtol=1e-5)
    assert sha256_file(path) == before
    del model, adapter, logits, loss, native, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if family == "arcface":
        # The producer is used only to make the fixture above. A fresh ronbun
        # consumer must work with the exported files and no FaceX source access.
        code = r'''
import json, sys
from pathlib import Path
def audit(event, args):
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes)):
        value = str(args[0]).replace('\\', '/').lower()
        if value.startswith('f:/facex-zoo/'):
            raise PermissionError('FaceX source access forbidden in standalone consumer')
sys.addaudithook(audit)
assert not any('facex-zoo' in str(p).lower() for p in sys.path)
import numpy as np
import torch
torch.set_num_threads(2)
from research.embeddings.facex_zoo import model_spec_from_facex_export
from research.embeddings.registry import create_pytorch_adapter_from_spec
spec = model_spec_from_facex_export(Path(sys.argv[1]))
adapter = create_pytorch_adapter_from_spec(spec, device='cpu')
result = adapter.embed(np.full((1, 112, 112, 3), 100, dtype=np.uint8))
assert result.normalized_embedding.shape == (1, 1024)
assert np.isfinite(result.normalized_embedding).all()
assert not any(n == 'modules' or n == 'backbone' for n in sys.modules)
print(json.dumps({'dimension':1024, 'facex_source_access':False}))
'''
        project = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(path)],
            cwd=project, env={**os.environ, "PYTHONPATH": str(project)}, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout.splitlines()[-1])["facex_source_access"] is False
