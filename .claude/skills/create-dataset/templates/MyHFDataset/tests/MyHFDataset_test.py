import os

import pytest
from huggingface_hub import HfApi

import datasets as ds


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "your-org"


@pytest.fixture
def dataset_name() -> str:
    return "MyHFDataset"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


# FIRST: This test creates the repository on HF Hub
@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
def test_load_dataset(dataset_path: str, repo_id: str):
    dataset = ds.load_dataset(
        path=dataset_path,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)

    # Uncomment to push to hub (creates the repository):
    # dataset.push_to_hub(repo_id=repo_id, private=True)


# SECOND: This test uploads README to the repository created above
def test_push_readme_to_hub(
    hf_api: HfApi,
    repo_id: str,
    script_dir: str,
):
    readme_path = os.path.join(script_dir, "README.md")

    hf_api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
