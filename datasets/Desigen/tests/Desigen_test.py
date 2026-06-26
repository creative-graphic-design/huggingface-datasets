import os

import datasets as ds
import pytest


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def dataset_name() -> str:
    return "Desigen"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


def test_load_dataset_builder(dataset_path: str):
    builder = ds.load_dataset_builder(path=dataset_path, trust_remote_code=True)
    assert {
        "image",
        "prompt",
        "region",
        "description",
        "elements",
        "size",
    } <= set(builder.info.features)


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    expected_num_train: int = 36322,
    expected_num_validation: int = 999,
):
    dataset = ds.load_dataset(path=dataset_path, trust_remote_code=True)
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_validation

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(repo_id=repo_id, private=True)
