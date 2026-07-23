import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import HfApi

import datasets as ds

_HUB_MAX_SHARD_SIZE = "50MB"
_SOURCE_EXPECTED_IMAGE_ADS = 64832
_SOURCE_EXPECTED_VIDEO_ADS = 3477
_INSPECTED_EXPECTED_ANNOTATED_IMAGE_ROWS = 64454
_INSPECTED_EXPECTED_VIDEO_ROWS = 3477
_PITT_IMAGE_URL_PREFIX = "https://people.cs.pitt.edu/~mzhang/image_ads/"


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "PittImageVideoAdsDataset"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


@pytest.fixture
def dataset_module(dataset_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "PittImageVideoAdsDataset",
        dataset_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["PittImageVideoAdsDataset"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    argnames=("config_name", "expected_num_train"),
    argvalues=(
        ("image_ads", _INSPECTED_EXPECTED_ANNOTATED_IMAGE_ROWS),
        ("video_ads", _INSPECTED_EXPECTED_VIDEO_ROWS),
    ),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    config_name: str,
    expected_num_train: int,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=config_name,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert list(dataset) == ["train"]
    assert dataset["train"].num_rows == expected_num_train

    if config_name == "image_ads":
        sample = dataset["train"][0]
        assert sample["source_path"]
        assert sample["image_url"].startswith(
            "https://people.cs.pitt.edu/~mzhang/image_ads/"
        )
        assert sample["image"].size == (400, 566)
        assert "image" in dataset["train"].features
        assert "topics" in dataset["train"].features
        assert "symbols" in dataset["train"].features
    else:
        sample = dataset["train"][0]
        assert sample["youtube_id"]
        assert sample["youtube_url"] == (
            f"https://www.youtube.com/watch?v={sample['youtube_id']}"
        )
        assert "raw_qa_action" in dataset["train"].features
        assert "clean_effective" in dataset["train"].features

    if os.environ.get("HF_WRITE_TESTS"):
        push_kwargs = {
            "repo_id": repo_id,
            "config_name": config_name,
            "max_shard_size": _HUB_MAX_SHARD_SIZE,
        }

        dataset.push_to_hub(**push_kwargs)


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


def test_source_reported_counts_are_consistent():
    assert _SOURCE_EXPECTED_IMAGE_ADS > _INSPECTED_EXPECTED_ANNOTATED_IMAGE_ROWS
    assert _SOURCE_EXPECTED_VIDEO_ADS == _INSPECTED_EXPECTED_VIDEO_ROWS


def test_image_url_uses_original_pitt_source(dataset_module: ModuleType):
    assert dataset_module._image_url("10/170489.png") == (
        "https://people.cs.pitt.edu/~mzhang/image_ads/10/170489.png"
    )


def test_load_video_ids_strips_single_quotes(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    video_ids_path = tmp_path / "final_video_id_list.csv"
    video_ids_path.write_text("'KONL05sae4E'\n\"VZ9lXgcYL50\"\n", encoding="utf-8")

    assert dataset_module._load_video_ids(video_ids_path) == [
        "KONL05sae4E",
        "VZ9lXgcYL50",
    ]


def test_iter_image_examples_normalizes_missing_annotations(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    paths = {}
    payloads = {
        "qa_action": {"10/170489.png": ["Buy it."]},
        "qa_reason": {},
        "qa_combined_action_reason": {},
        "sentiments": {"10/170489.png": [["14"], ["12"]]},
        "slogans": {},
        "strategies": {},
        "symbols": {"10/170489.png": [[58.0, 14.0, 430.0, 466.0, "electronics"]]},
        "topics": {"10/170489.png": ["2", "3"]},
    }
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)

    image_roots = {"10": str(tmp_path)}
    image_path = tmp_path / "10" / "170489.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"placeholder")

    examples = list(dataset_module._iter_image_examples(paths, image_roots))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "image-00000"
    assert example["id"] == "image-00000"
    assert example["source_path"] == "10/170489.png"
    assert example["image"].endswith("/10/170489.png")
    assert example["image_url"].endswith("/image_ads/10/170489.png")
    assert example["qa_action"] == ["Buy it."]
    assert example["qa_reason"] == []
    assert example["sentiments"] == [["14"], ["12"]]
    assert example["symbols"][0] == {
        "x1": 58.0,
        "y1": 14.0,
        "x2": 430.0,
        "y2": 466.0,
        "label": "electronics",
    }


def test_iter_video_examples_builds_urls_and_defaults_missing_clean_language(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    paths = {}
    payloads = {
        "raw_qa_action": {"KONL05sae4E": ["Buy this brand."]},
        "raw_qa_reason": {"KONL05sae4E": ["Because it is useful."]},
        "raw_topics": {"KONL05sae4E": ["media", "media"]},
        "raw_sentiments": {"KONL05sae4E": ["active"]},
        "raw_funny": {"KONL05sae4E": [0, 1]},
        "raw_exciting": {"KONL05sae4E": [1, 1]},
        "raw_language": {"KONL05sae4E": ["1"]},
        "clean_topic": {"KONL05sae4E": 27},
        "clean_sentiment": {"KONL05sae4E": 12},
        "clean_funny": {"KONL05sae4E": 0.0},
        "clean_exciting": {"KONL05sae4E": 1.0},
        "clean_language": {},
        "clean_effective": {"KONL05sae4E": 1.0},
    }
    video_ids_path = tmp_path / "final_video_id_list.csv"
    video_ids_path.write_text("'KONL05sae4E'\n", encoding="utf-8")
    paths["video_ids"] = str(video_ids_path)

    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)

    examples = list(dataset_module._iter_video_examples(paths))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "video-00000"
    assert example["youtube_id"] == "KONL05sae4E"
    assert example["youtube_url"] == "https://www.youtube.com/watch?v=KONL05sae4E"
    assert example["raw_funny"] == [0.0, 1.0]
    assert example["clean_topic"] == 27
    assert example["clean_language"] == ""
