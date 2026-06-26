import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import HfApi

import datasets as ds

_HUB_MAX_SHARD_SIZE = "50MB"
_SOURCE_EXPECTED_NUM_TEST = 215
_SOURCE_EXPECTED_ASPECT_RATIO_COUNTS = {
    "wide": 173,
    "square": 36,
    "tall": 6,
}


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "DEsignBenchPrompts"


@pytest.fixture
def hub_dataset_name() -> str:
    return "DEsignBench-Prompts"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str, hub_dataset_name: str) -> str:
    return f"{org_name}/{hub_dataset_name}"


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


@pytest.fixture
def dataset_module(dataset_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("DEsignBenchPrompts", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["DEsignBenchPrompts"] = module
    spec.loader.exec_module(module)
    return module


def test_iter_prompt_rows_reads_tsv_with_pandas(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    tsv_path = tmp_path / "DesignBench_Prompts.tsv"
    tsv_path.write_text(
        "\t".join(
            [
                "demo_imgname",
                "userinput",
                "expandprompt",
                "aspectratio",
                "",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "text_0_0",
                'a graffiti art of the text "free the pink" on a wall',
                "Photo of a smooth stone wall.",
                "wide",
                "",
            ]
        )
        + "\n"
        + "\t".join(["text_1_0", "missing expanded prompt", "", "square", ""])
        + "\n",
        encoding="utf-8",
    )

    examples = list(dataset_module._iter_prompt_rows(str(tsv_path)))

    assert len(examples) == 2
    key, example = examples[0]
    assert key == 0
    assert example == {
        "id": "text_0_0",
        "demo_imgname": "text_0_0",
        "userinput": 'a graffiti art of the text "free the pink" on a wall',
        "expandprompt": "Photo of a smooth stone wall.",
        "aspectratio": "wide",
    }
    assert examples[1][1]["expandprompt"] == ""
    assert "" not in examples[0][1]


def test_source_reported_counts_are_consistent():
    assert sum(_SOURCE_EXPECTED_ASPECT_RATIO_COUNTS.values()) == _SOURCE_EXPECTED_NUM_TEST


def test_load_dataset(dataset_path: str, repo_id: str):
    dataset = ds.load_dataset(path=dataset_path, trust_remote_code=True)
    assert isinstance(dataset, ds.DatasetDict)
    assert list(dataset) == ["test"]
    assert dataset["test"].num_rows == _SOURCE_EXPECTED_NUM_TEST

    sample = dataset["test"][0]
    assert sample["id"]
    assert sample["demo_imgname"] == sample["id"]
    assert sample["userinput"]
    assert sample["expandprompt"]
    assert sample["aspectratio"] in _SOURCE_EXPECTED_ASPECT_RATIO_COUNTS

    aspect_ratio_counts = {
        aspect_ratio: dataset["test"]
        .filter(lambda row, value=aspect_ratio: row["aspectratio"] == value)
        .num_rows
        for aspect_ratio in _SOURCE_EXPECTED_ASPECT_RATIO_COUNTS
    }
    assert aspect_ratio_counts == _SOURCE_EXPECTED_ASPECT_RATIO_COUNTS

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(
            repo_id=repo_id,
            max_shard_size=_HUB_MAX_SHARD_SIZE,
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
