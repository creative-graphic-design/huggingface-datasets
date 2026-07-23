import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import HfApi
from PIL import Image

import datasets as ds

_HUB_MAX_SHARD_SIZE = "50MB"
_SOURCE_EXPECTED_NUM_TEST_BY_CONFIG = {
    "alignment": 200,
    "composition_understanding": 117,
    "empty_space": 167,
    "font_attributes": 1813,
    "font_effect": 450,
    "font_effect_2": 125,
    "font_matching": 400,
    "font_size_ocr": 1400,
    "hard_ocr": 400,
    "intention_understanding": 202,
    "layout_comparison": 256,
    "layout_generation": 145,
    "logo_ocr": 600,
    "overall_rating": 219,
    "poster_ocr": 205,
    "rotation": 205,
    "simple_ocr": 400,
    "style_understanding": 256,
    "text_localization": 205,
    "gen_composition": 117,
    "gen_dense": 114,
    "gen_font": 135,
    "gen_intention": 200,
    "gen_style": 256,
}
_SOURCE_EXPECTED_NUM_UNDERSTANDING = 7765
_SOURCE_EXPECTED_NUM_GENERATION = 822
_SOURCE_EXPECTED_TOTAL_ROWS = 8587


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "PosterIQ"


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
    spec = importlib.util.spec_from_file_location("PosterIQ", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["PosterIQ"] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(path)


def test_source_reported_counts_are_consistent(dataset_module: ModuleType):
    assert len(dataset_module.PosterIQ.BUILDER_CONFIGS) == 24
    assert sum(_SOURCE_EXPECTED_NUM_TEST_BY_CONFIG.values()) == (
        _SOURCE_EXPECTED_TOTAL_ROWS
    )

    understanding_count = sum(
        task.count for task in dataset_module._TASKS.values() if task.has_image
    )
    generation_count = sum(
        task.count for task in dataset_module._TASKS.values() if not task.has_image
    )

    assert understanding_count == _SOURCE_EXPECTED_NUM_UNDERSTANDING
    assert generation_count == _SOURCE_EXPECTED_NUM_GENERATION
    assert understanding_count + generation_count == _SOURCE_EXPECTED_TOTAL_ROWS
    assert {
        config.name.value: dataset_module._TASKS[config.name].count
        for config in dataset_module.PosterIQ.BUILDER_CONFIGS
    } == _SOURCE_EXPECTED_NUM_TEST_BY_CONFIG


def test_normalize_path_converts_backslashes(dataset_module: ModuleType):
    assert dataset_module._normalize_path(r"dense\1000.jpg") == "dense/1000.jpg"


def test_find_data_root_finds_nested_directory(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    root = tmp_path / "archive" / "nested" / "data"
    root.mkdir(parents=True)

    assert dataset_module._find_data_root(tmp_path) == root


def test_find_data_root_raises_for_missing_directory(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    with pytest.raises(FileNotFoundError):
        dataset_module._find_data_root(tmp_path)


def test_iter_examples_resolves_images_and_preserves_metadata(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    data_root = tmp_path / "data"
    image_path = data_root / "alignment" / "000_30_center_.png"
    _write_image(image_path)

    rows = [
        {
            "task": "alignment",
            "subtask": "",
            "name": "000_30_center_.png",
            "path": "alignment/000_30_center_.png",
            "prompt": "Choose the text alignment.",
            "gt": ["center-aligned"],
            "alignment": ["center-aligned"],
        }
    ]
    task = dataset_module._TASKS[dataset_module.PosterIQType.alignment]

    examples = list(
        dataset_module._iter_examples(
            rows,
            dataset_module.PosterIQType.alignment,
            task,
            data_root,
        )
    )

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "alignment-00000"
    assert example["id"] == "alignment-00000"
    assert example["task"] == "alignment"
    assert example["subtask"] == ""
    assert example["name"] == "000_30_center_.png"
    assert example["path"] == "alignment/000_30_center_.png"
    assert example["prompt"] == "Choose the text alignment."
    assert json.loads(example["gt_json"]) == ["center-aligned"]
    assert json.loads(example["metadata_json"]) == {"alignment": ["center-aligned"]}
    assert Path(example["image"]).is_file()
    assert Path(example["image_path"]).is_file()


def test_iter_examples_resolves_original_image(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    data_root = tmp_path / "data"
    image_path = data_root / "poster_ocr_1024" / "sample.png"
    original_image_path = data_root / "poster_ocr" / "sample.png"
    _write_image(image_path)
    _write_image(original_image_path)

    rows = [
        {
            "task": "poster ocr",
            "subtask": "",
            "name": "sample.png",
            "path": "poster_ocr_1024/sample.png",
            "path_original": "poster_ocr/sample.png",
            "prompt": "Extract text.",
            "size": [2700, 3450],
            "texts": ["REVOLUTION"],
            "text_bbox": [[432, 729, 1842, 1026]],
            "bbox_areas": [418770],
        }
    ]
    task = dataset_module._TASKS[dataset_module.PosterIQType.poster_ocr]

    examples = list(
        dataset_module._iter_examples(
            rows,
            dataset_module.PosterIQType.poster_ocr,
            task,
            data_root,
        )
    )

    _, example = examples[0]
    assert "gt_json" not in example
    assert Path(example["image"]).is_file()
    assert Path(example["original_image"]).is_file()
    assert Path(example["original_image_path"]).is_file()
    assert json.loads(example["metadata_json"]) == {
        "size": [2700, 3450],
        "texts": ["REVOLUTION"],
        "text_bbox": [[432, 729, 1842, 1026]],
        "bbox_areas": [418770],
    }


def test_iter_examples_keeps_generation_configs_image_free(dataset_module: ModuleType):
    rows = [
        {
            "task": "poster dense",
            "subtask": "",
            "name": "1000.jpg",
            "path": r"dense\1000.jpg",
            "theme": "NBA",
            "elements": [["Kevin Durant", "Stephen Curry"]],
            "prompt": "Generate a poster.",
            "gt": ["Aspect ratio 2:3"],
        }
    ]
    task = dataset_module._TASKS[dataset_module.PosterIQType.gen_dense]

    examples = list(
        dataset_module._iter_examples(
            rows,
            dataset_module.PosterIQType.gen_dense,
            task,
        )
    )

    _, example = examples[0]
    assert example["path"] == "dense/1000.jpg"
    assert "image" not in example
    assert json.loads(example["gt_json"]) == ["Aspect ratio 2:3"]
    assert json.loads(example["metadata_json"]) == {
        "theme": "NBA",
        "elements": [["Kevin Durant", "Stephen Curry"]],
    }


def test_iter_examples_requires_source_fields(dataset_module: ModuleType):
    task = dataset_module._TASKS[dataset_module.PosterIQType.alignment]

    with pytest.raises(KeyError):
        list(
            dataset_module._iter_examples(
                [
                    {
                        "task": "alignment",
                        "subtask": "",
                        "name": "000_30_center_.png",
                        "path": "alignment/000_30_center_.png",
                        "gt": ["center-aligned"],
                    }
                ],
                dataset_module.PosterIQType.alignment,
                task,
                "/tmp/data",
            )
        )


@pytest.mark.skipif(
    condition=os.environ.get("POSTER_IQ_RUN_DOWNLOAD_TESTS") != "1",
    reason="Set POSTER_IQ_RUN_DOWNLOAD_TESTS=1 to download and load the full dataset.",
)
@pytest.mark.parametrize(
    argnames=("config_name", "expected_num_test"),
    argvalues=tuple(_SOURCE_EXPECTED_NUM_TEST_BY_CONFIG.items()),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    config_name: str,
    expected_num_test: int,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=config_name,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert list(dataset) == ["test"]
    assert dataset["test"].num_rows == expected_num_test

    sample = dataset["test"][0]
    assert sample["prompt"]
    if config_name.startswith("gen_"):
        assert "image" not in dataset["test"].features
    else:
        assert sample["image"] is not None

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
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
