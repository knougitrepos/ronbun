"""Seed migration must not alter historical notebook outputs or collapse panels."""
import ast
import json
from pathlib import Path

from research.evaluation.metrics import PAIRED_BOOTSTRAP_RANDOM_SEED
from research.experiments.fiqa_split_stability import DEFAULT_PARTITION_SEEDS

ROOT = Path(__file__).resolve().parents[2]


def test_new_seed_and_distinct_partition_panel():
    assert PAIRED_BOOTSTRAP_RANDOM_SEED == 8972
    assert len(DEFAULT_PARTITION_SEEDS) == len(set(DEFAULT_PARTITION_SEEDS)) == 20
    assert 8972 in DEFAULT_PARTITION_SEEDS
    assert 42 not in DEFAULT_PARTITION_SEEDS


def test_no_legacy_seed_defaults_in_python_or_notebook_code():
    sources = [(str(p), p.read_text(encoding="utf-8"))
               for folder in ("research", "scripts") for p in (ROOT / folder).rglob("*.py")]
    for p in (ROOT / "notebooks").rglob("*.ipynb"):
        for i, cell in enumerate(json.loads(p.read_text(encoding="utf-8"))["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if not any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                    sources.append((f"{p}:{i}", source))
    for name, source in sources:
        tree = ast.parse(source, filename=name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args.posonlyargs + node.args.args
                defaults = list(zip(args[-len(node.args.defaults):], node.args.defaults))
                defaults += list(zip(node.args.kwonlyargs, node.args.kw_defaults))
                for arg, value in defaults:
                    if "seed" in arg.arg.lower() or arg.arg == "random_state":
                        assert not (isinstance(value, ast.Constant) and value.value == 42), name
