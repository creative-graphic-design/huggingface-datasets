import os

import pytest
from huggingface_hub import HfApi

import datasets as ds


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def dataset_name() -> str:
    return "BannerRequest400"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


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


@pytest.mark.parametrize(
    argnames=("request_type", "expected_num_train"),
    argvalues=(
        ("abstract_400", 400),
        ("concrete_5k", 100),
    ),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    request_type: str,
    expected_num_train: int,
    trust_remote_code: bool = True,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=request_type,
        trust_remote_code=trust_remote_code,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train

    # Verify key fields exist
    if request_type == "abstract_400":
        assert "banner_request" in dataset["train"].features
        assert "logo_png" in dataset["train"].features
        assert "logo_svg" in dataset["train"].features
        assert "id" in dataset["train"].features
    elif request_type == "concrete_5k":
        assert "advertising_variations" in dataset["train"].features
        assert "logo_description" in dataset["train"].features
        assert "advertiser" in dataset["train"].features
        assert "logo_name" in dataset["train"].features

    # Optionally push to hub (commented by default)
    # dataset.push_to_hub(
    #     repo_id=repo_id,
    #     config_name=request_type,
    #     private=True,
    # )
