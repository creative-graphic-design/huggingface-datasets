import os

import pytest
from huggingface_hub import HfApi

import datasets as ds

_HUB_NUM_SHARDS = {"test": 120}


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "AesEvalBench"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


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
    expected_num_test: int = 1198,
    trust_remote_code: bool = True,
):
    load_kwargs = {
        "path": dataset_path,
        "trust_remote_code": trust_remote_code,
    }
    if local_archive := os.environ.get("AESEVAL_BENCH_ARCHIVE"):
        load_kwargs["data_dir"] = local_archive

    dataset = ds.load_dataset(**load_kwargs)
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["test"].num_rows == expected_num_test

    features = dataset["test"].features
    assert "preview" in features
    assert "preview_highlight" in features
    assert "element_images" in features
    assert "task_labels" in features
    assert "gt_annotations" in features

    sample = dataset["test"][0]
    assert sample["sample_name"].endswith("-perturbs_new")
    assert len(sample["task_labels"]) == 12
    assert {"dimension", "task", "key", "has_issue"} <= set(sample["task_labels"][0])
    assert len(sample["elements"]) >= 1
    assert len(sample["element_images"]) >= 1
    assert sample["preview"] is not None

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(
            repo_id=repo_id,
            private=True,
            num_shards=_HUB_NUM_SHARDS,
        )


def test_push_readme_to_hub(
    hf_api: HfApi,
    repo_id: str,
    script_dir: str,
):
    if not os.environ.get("HF_WRITE_TESTS"):
        pytest.skip("Set HF_WRITE_TESTS=1 to upload files to Hugging Face Hub.")

    readme_path = os.path.join(script_dir, "README.md")

    hf_api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
