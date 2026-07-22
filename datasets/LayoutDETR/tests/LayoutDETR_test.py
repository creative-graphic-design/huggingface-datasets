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

_EXPECTED_TOTAL_SAMPLES = 7672
_EXPECTED_NUM_TRAIN = int(_EXPECTED_TOTAL_SAMPLES * 0.90)
_EXPECTED_NUM_VALIDATION = _EXPECTED_TOTAL_SAMPLES - _EXPECTED_NUM_TRAIN


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "LayoutDETR"


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
    spec = importlib.util.spec_from_file_location("LayoutDETR", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["LayoutDETR"] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_annotation(path: Path, text: str = "Shop now") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotation = [
        {
            "xyxy_word_fit": [10, 5, 50, 25],
            "str": text,
            "label": "button",
        }
    ]
    path.write_text(json.dumps(annotation), encoding="utf-8")


@pytest.fixture
def tiny_layoutdetr_root(tmp_path: Path) -> Path:
    root = tmp_path / "ads_banner_dataset"
    gt_dir = root / "png_json_gt"
    background_1x_dir = root / "1x_inpainted_background_png"
    background_3x_dir = root / "3x_inpainted_background_png"

    for index in range(10):
        stem = f"sample_{index:02d}"
        _write_image(gt_dir / f"{stem}.png")
        _write_annotation(gt_dir / f"{stem}.json", text=f"Copy {index}")
        if index != 8:
            _write_image(background_1x_dir / f"{stem}_inpainted.png")
        if index != 9:
            _write_image(background_3x_dir / f"{stem}_inpainted.png")

    return root


def test_find_dataset_root_finds_nested_png_json_gt(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    root = tmp_path / "archive" / "nested" / "ads_banner_dataset"
    (root / "png_json_gt").mkdir(parents=True)

    assert dataset_module._find_dataset_root(tmp_path) == root
    assert dataset_module._find_dataset_root(root / "png_json_gt") == root


def test_find_dataset_root_raises_for_missing_directory(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    with pytest.raises(FileNotFoundError):
        dataset_module._find_dataset_root(tmp_path)


def test_xyxy_to_normalized_cxcywh(dataset_module: ModuleType):
    assert dataset_module._xyxy_to_cxcywh_normalized(
        [10, 5, 50, 25],
        width=100,
        height=50,
    ) == pytest.approx([0.3, 0.3, 0.4, 0.4])


def test_split_json_paths_uses_upstream_9_to_1_rule(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    paths = [tmp_path / f"{index:02d}.json" for index in range(10)]

    assert dataset_module._split_json_paths(paths, "train") == paths[:9]
    assert dataset_module._split_json_paths(paths, "validation") == paths[9:]


def test_iter_examples_reads_raw_fixture_and_optional_backgrounds(
    tiny_layoutdetr_root: Path,
    dataset_module: ModuleType,
):
    train_examples = list(dataset_module._iter_examples(tiny_layoutdetr_root, "train"))
    validation_examples = list(
        dataset_module._iter_examples(tiny_layoutdetr_root, "validation")
    )

    assert len(train_examples) == 9
    assert len(validation_examples) == 1

    key, example = train_examples[0]
    assert key == "sample_00"
    assert example["id"] == "sample_00"
    assert Path(example["image_path"]).is_file()
    assert Path(example["background_1x_path"]).is_file()
    assert Path(example["background_3x_path"]).is_file()
    assert example["width"] == 100
    assert example["height"] == 50
    assert example["num_elements"] == 1
    assert example["elements"][0] == {
        "text": "Copy 0",
        "label": "button",
        "bbox_xyxy": [10.0, 5.0, 50.0, 25.0],
        "bbox_cxcywh_normalized": pytest.approx([0.3, 0.3, 0.4, 0.4]),
    }
    assert json.loads(example["raw_annotation"])[0]["str"] == "Copy 0"

    _, validation_example = validation_examples[0]
    assert validation_example["id"] == "sample_09"
    assert Path(validation_example["background_1x_path"]).is_file()
    assert validation_example["background_3x"] is None
    assert validation_example["background_3x_path"] == ""


def test_load_dataset_with_tiny_data_dir(
    dataset_path: str,
    tiny_layoutdetr_root: Path,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        data_dir=str(tiny_layoutdetr_root),
        trust_remote_code=True,
    )

    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == 9
    assert dataset["validation"].num_rows == 1
    sample = dataset["train"][0]
    assert sample["image"] is not None
    assert sample["background_1x"] is not None
    assert sample["background_3x"] is not None
    assert sample["elements"][0]["label"] == "button"

    validation_sample = dataset["validation"][0]
    assert validation_sample["id"] == "sample_09"
    assert validation_sample["background_3x"] is None
    assert validation_sample["background_3x_path"] == ""


def test_invalid_or_oversized_element_counts_are_filtered(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    root = tmp_path / "ads_banner_dataset"
    gt_dir = root / "png_json_gt"
    _write_image(gt_dir / "sample.png")
    annotation = [
        {"xyxy_word_fit": [10, 5, 50, 25], "str": "", "label": "button"},
        {"xyxy_word_fit": [10, 5, 50, 25], "str": "Unknown", "label": "unknown"},
    ]
    (gt_dir / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    assert list(dataset_module._iter_examples(root, "validation")) == []


def test_expected_full_split_counts_are_consistent():
    assert _EXPECTED_NUM_TRAIN == 6904
    assert _EXPECTED_NUM_VALIDATION == 768
    assert _EXPECTED_NUM_TRAIN + _EXPECTED_NUM_VALIDATION == _EXPECTED_TOTAL_SAMPLES


@pytest.mark.skipif(
    condition=os.environ.get("LAYOUT_DETR_RUN_DOWNLOAD_TESTS") != "1",
    reason=(
        "Set LAYOUT_DETR_RUN_DOWNLOAD_TESTS=1 to download and load the full "
        "14.7GB LayoutDETR ad banner dataset."
    ),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
):
    load_kwargs = {
        "path": dataset_path,
        "trust_remote_code": True,
    }
    if local_archive_or_dir := os.environ.get("LAYOUT_DETR_DATA_DIR"):
        load_kwargs["data_dir"] = local_archive_or_dir

    dataset = ds.load_dataset(**load_kwargs)
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == _EXPECTED_NUM_TRAIN
    assert dataset["validation"].num_rows == _EXPECTED_NUM_VALIDATION

    sample = dataset["train"][0]
    assert sample["image"] is not None
    assert 1 <= sample["num_elements"] <= 9
    assert sample["elements"][0]["bbox_xyxy"]
    assert sample["elements"][0]["bbox_cxcywh_normalized"]
    assert sample["raw_annotation"]

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(repo_id=repo_id)


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
