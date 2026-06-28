import ast
from pathlib import Path


def test_push_to_hub_calls_define_shard_policy():
    repo_root = Path(__file__).resolve().parents[1]
    missing_policy = []

    for path in sorted((repo_root / "datasets").glob("*/tests/*_test.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "push_to_hub":
                continue

            keyword_names = {keyword.arg for keyword in node.keywords}
            if {"max_shard_size", "num_shards"} & keyword_names:
                continue

            missing_policy.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert missing_policy == []
