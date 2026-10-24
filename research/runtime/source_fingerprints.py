"""Fingerprint local Python dependencies without importing their runtimes.

Static imports (including imports inside functions) are followed transitively.
Callers must also name factories loaded dynamically from configuration. Package
initializers and this resolver are included; third-party packages are not source
files in this contract and their runtime versions belong in run manifests.
"""
import ast
from functools import lru_cache
from importlib.util import resolve_name
from pathlib import Path

from research.runtime.hashing import sha256_file


@lru_cache(maxsize=512)
def _imports(source, package):
    """Cache parsing by exact bytes, never by modification time or module name."""
    required, candidates = [], []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            required.extend(alias.name for alias in node.names if alias.name.startswith("research."))
        elif isinstance(node, ast.ImportFrom):
            name = (resolve_name("." * node.level + (node.module or ""), package)
                    if node.level else node.module or "")
            if name == "research" or name.startswith("research."):
                required.append(name)
                candidates.extend(f"{name}.{alias.name}" for alias in node.names)
    return tuple(required), tuple(candidates)


def source_fingerprints(*modules, project_root=None):
    root = Path(project_root or Path(__file__).resolve().parents[2])
    pending = [*modules, "research.runtime.source_fingerprints"]
    visited, result = set(), {}

    def path_for(module):
        path = root.joinpath(*module.split("."))
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py")
        if (path / "__init__.py").is_file():
            return path / "__init__.py"
        return None

    while pending:
        module = pending.pop()
        if module in visited or not (module == "research" or module.startswith("research.")):
            continue
        visited.add(module)
        path = path_for(module)
        if path is None:
            raise FileNotFoundError(f"Local source dependency is missing: {module}")
        source = path.read_bytes()
        result[path.relative_to(root).as_posix()] = sha256_file(path)
        parts = module.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts))
                       if path_for(".".join(parts[:i])))
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        required, candidates = _imports(source, package)
        pending.extend(required)
        pending.extend(name for name in candidates if path_for(name))
    return dict(sorted(result.items()))
