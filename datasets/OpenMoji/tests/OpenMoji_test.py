import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import HfApi

import datasets as ds

_EXPECTED_ROWS = 4_495


@pytest.fixture
def script_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset_path(script_dir: Path) -> Path:
    return script_dir / "OpenMoji.py"


@pytest.fixture
def dataset_module(dataset_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("OpenMoji", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


def test_normalize_metadata_record_maps_empty_scalars_to_none(
    dataset_module: ModuleType,
):
    record = {
        "hexcode": "1F600",
        "emoji": "",
        "group": "",
        "subgroups": "",
        "annotation": "",
        "tags": "",
        "openmoji_tags": "",
        "openmoji_author": "",
        "openmoji_date": "",
        "skintone": "",
        "skintone_combination": "",
        "skintone_base_emoji": "",
        "skintone_base_hexcode": "",
        "unicode": "",
        "order": "",
    }

    normalized = dataset_module.normalize_metadata_record(record)
    scalar_values = [
        value
        for key, value in normalized.items()
        if key not in {"hexcode", "tags", "openmoji_tags"}
    ]

    assert normalized["hexcode"] == "1F600"
    assert normalized["tags"] == []
    assert normalized["openmoji_tags"] == []
    assert all(value is None for value in scalar_values)


def test_normalize_metadata_record_splits_tags_and_normalizes_mixed_types(
    dataset_module: ModuleType,
):
    record = {
        "hexcode": "1F44B-1F3FB",
        "emoji": "👋🏻",
        "group": "people-body",
        "subgroups": "hand-fingers-open",
        "annotation": "waving hand: light skin tone",
        "tags": " wave, , waving hand , hello ",
        "openmoji_tags": "hand, greeting",
        "openmoji_author": "OpenMoji",
        "openmoji_date": "2024-01-01",
        "skintone": 1,
        "skintone_combination": "1F3FB",
        "skintone_base_emoji": "👋",
        "skintone_base_hexcode": 128075,
        "unicode": 0.6,
        "order": "7",
    }

    normalized = dataset_module.normalize_metadata_record(record)

    assert normalized["tags"] == ["wave", "waving hand", "hello"]
    assert normalized["openmoji_tags"] == ["hand", "greeting"]
    assert normalized["skintone"] == "1"
    assert normalized["skintone_base_hexcode"] == "128075"
    assert normalized["unicode"] == "0.6"
    assert normalized["order"] == 7


@pytest.mark.skipif(
    os.environ.get("OPENMOJI_RUN_DOWNLOAD_TESTS") != "1",
    reason="Set OPENMOJI_RUN_DOWNLOAD_TESTS=1 to download and load the full release.",
)
def test_load_dataset(dataset_path: Path, dataset_module: ModuleType):
    dataset = ds.load_dataset(path=str(dataset_path), trust_remote_code=True)

    assert isinstance(dataset, ds.DatasetDict)
    assert set(dataset) == {"train"}
    train = dataset["train"]
    assert train.num_rows == _EXPECTED_ROWS, (
        f"Expected {_EXPECTED_ROWS:,} rows for OpenMoji "
        f"{dataset_module._OPENMOJI_VERSION}, found {train.num_rows:,}."
    )
    assert len(set(train["hexcode"])) == _EXPECTED_ROWS
    for row in train:
        assert isinstance(row["svg_color"], str) and row["svg_color"]
        assert isinstance(row["svg_black"], str) and row["svg_black"]
        assert row["image_color"] is not None
        assert row["image_black"] is not None

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(
            repo_id="creative-graphic-design/OpenMoji",
            max_shard_size="50MB",
        )


@pytest.mark.skipif(
    not os.environ.get("HF_WRITE_TESTS")
    or os.environ.get("OPENMOJI_RUN_DOWNLOAD_TESTS") != "1",
    reason="Set HF_WRITE_TESTS=1 and OPENMOJI_RUN_DOWNLOAD_TESTS=1 to publish the card.",
)
def test_push_readme_to_hub(
    hf_api: HfApi,
    script_dir: Path,
):
    hf_api.upload_file(
        path_or_fileobj=script_dir / "README.md",
        path_in_repo="README.md",
        repo_id="creative-graphic-design/OpenMoji",
        repo_type="dataset",
    )
