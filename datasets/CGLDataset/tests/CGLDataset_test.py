import os

import datasets as ds
import pytest


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "CGLDataset"


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str) -> str:
    return f"{org_name}/CGL-Dataset"


def test_load_dataset_builder(dataset_path: str):
    builder = ds.load_dataset_builder(path=dataset_path, trust_remote_code=True)
    assert {"image", "annotations"} <= set(builder.info.features)


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
def test_load_dataset(
    dataset_path: str,
    expected_num_train: int = 54546,
    expected_num_valid: int = 6002,
    expected_num_test: int = 1000,
):
    dataset = ds.load_dataset(path=dataset_path, trust_remote_code=True)
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["test"].num_rows == expected_num_test


def test_push_to_hub(
    repo_id: str,
    dataset_path: str,
):
    if not os.environ.get("HF_WRITE_TESTS"):
        pytest.skip("Set HF_WRITE_TESTS=1 to push to Hugging Face Hub.")

    dataset = ds.load_dataset(
        path=dataset_path,
        rename_category_names=True,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)

    dataset.push_to_hub(repo_id=repo_id, private=True)
