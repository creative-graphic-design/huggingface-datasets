import os

import pytest
from huggingface_hub import snapshot_download

import datasets as ds


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def original_dataset_name() -> str:
    return "lrzjason/ObjectRemovalAlpha"


@pytest.fixture
def dataset_name() -> str:
    return "ObjectRemovalAlpha"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    expected_num_train: int = 20,  # = 60 (original) / 3 (G, F, M)
    trust_remote_code: bool = True,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        trust_remote_code=trust_remote_code,
    )
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train

    dataset.push_to_hub(repo_id=repo_id, private=True)
