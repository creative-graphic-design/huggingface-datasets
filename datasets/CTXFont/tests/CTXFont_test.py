import os

import pytest
from huggingface_hub import HfApi

import datasets as ds


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "CTXFont"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


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
    assert "train" in dataset
    assert "test" in dataset
    assert dataset["train"].num_rows == 4268
    assert dataset["test"].num_rows == 625

    # Check features
    expected_features = [
        "design_name",
        "design_url",
        "awwward_url",
        "design_tags",
        "text_content",
        "html_tags",
        "font_face",
        "font_size",
        "font_color_r",
        "font_color_g",
        "font_color_b",
        "font_color_a",
        "font_face_embedding",
        "center_x",
        "center_y",
        "width",
        "height",
    ]
    for feature in expected_features:
        assert feature in dataset["train"].features

    # dataset.push_to_hub(repo_id=repo_id, private=True)
