import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_IMPORT_ROOTS = {
    "aiohttp",
    "httpx",
    "requests",
    "subprocess",
}

FORBIDDEN_IMPORTS = {
    "urllib.request",
}

FORBIDDEN_CALLS = {
    "os.popen",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.system",
    "urllib.request.urlretrieve",
    "urllib.request.urlopen",
}


def dataset_script_paths() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "datasets").glob("*/*.py")
        if path.name != "__init__.py"
    )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def test_dataset_loaders_use_download_manager_for_downloads():
    violations = []

    for path in dataset_script_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", maxsplit=1)[0]
                    if root in FORBIDDEN_IMPORT_ROOTS or alias.name in FORBIDDEN_IMPORTS:
                        violations.append((path, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", maxsplit=1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS or module in FORBIDDEN_IMPORTS:
                    violations.append((path, node.lineno, f"from {module} import ..."))
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                if call_name in FORBIDDEN_CALLS:
                    violations.append((path, node.lineno, f"{call_name}(...)"))

    assert not violations, (
        "Dataset loaders must use datasets.DownloadManager/dl_manager for network "
        "downloads and extraction, not custom HTTP clients or shell commands:\n"
        + "\n".join(
            f"{path.relative_to(ROOT)}:{line}: {usage}"
            for path, line, usage in violations
        )
    )
