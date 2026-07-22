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
_QB_POSTER_EXPECTED_NUM_ROWS = {
    "train": 4675,
    "validation": 513,
}
_QB_POSTER_EXPECTED_TOTAL_ROWS = 5188
_USER_CONSTRAINED_EXPECTED_NUM_ROWS = {
    "train": 54546 + 9973,
    "validation": 6002,
}
_USER_CONSTRAINED_EXPECTED_TOTAL_ROWS = 70521


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "PosterLLaVA"


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
    spec = importlib.util.spec_from_file_location("PosterLLaVA", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["PosterLLaVA"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_qb_poster_data(tmp_path: Path) -> Path:
    raw = tmp_path / "data" / "qbposter" / "raw"
    for dirname in ("original_poster", "inpainted_1d5x", "inpainted_1x"):
        (raw / dirname).mkdir(parents=True)

    for dirname, image_name in (
        ("original_poster", "poster_train.png"),
        ("inpainted_1d5x", "poster_train.png"),
        ("original_poster", "poster_val.png"),
        ("inpainted_1x", "poster_val.png"),
    ):
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(
            raw / dirname / image_name
        )

    annotations = {
        "poster_train": {
            "split": "train",
            "width": 800,
            "height": 600,
            "boxes": [
                {
                    "xc": 400,
                    "yc": 300,
                    "width": 201,
                    "height": 101,
                    "label": "text",
                }
            ],
        },
        "poster_val": {
            "split": "val",
            "width": 1000,
            "height": 500,
            "boxes": [
                {
                    "xc": 250,
                    "yc": 250,
                    "width": 100,
                    "height": 200,
                    "label": "logo",
                }
            ],
        },
    }
    (raw / "annotations.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def tiny_user_constrained_data(tmp_path: Path) -> Path:
    root = tmp_path / "ucposter"
    root.mkdir()
    payloads = {
        "cgl_train.json": {
            "cgl_item": {
                "user constraints": [
                    "text_0 needs to be centered horizontally in the background image."
                ]
            }
        },
        "cgl_val.json": {
            "cgl_val_item": {
                "user constraints": [
                    "logo_0 needs to be placed at the top of the background image.",
                    "text_0 needs to be placed above text_1.",
                ]
            }
        },
        "posterlayout_train.json": {
            "0": {
                "user constraints": [
                    "All elements should be aligned to the left of the background image."
                ]
            }
        },
    }
    for filename, payload in payloads.items():
        (root / filename).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_find_qb_poster_root_finds_nested_raw_directory(
    dataset_module: ModuleType,
    tiny_qb_poster_data: Path,
):
    expected = tiny_qb_poster_data / "data" / "qbposter" / "raw"
    assert dataset_module._find_qb_poster_root(tiny_qb_poster_data) == expected


def test_iter_qb_poster_examples_normalizes_boxes_and_conversations(
    dataset_module: ModuleType,
    tiny_qb_poster_data: Path,
):
    root = dataset_module._find_qb_poster_root(tiny_qb_poster_data)
    examples = list(dataset_module._iter_qb_poster_examples(root, "train"))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "poster_train"
    assert example["id"] == "poster_train"
    assert example["image_path"].endswith("inpainted_1d5x/poster_train.png")
    assert example["elements"][0]["left"] == 300
    assert example["elements"][0]["top"] == 250
    assert example["elements"][0]["right"] == 500
    assert example["elements"][0]["bottom"] == 350
    assert example["elements"][0]["box"] == [0.375, 0.4167, 0.625, 0.5833]
    assert "place 1 foreground elements" in example["prompt"]
    assert example["conversations"][0]["from"] == "human"
    assert example["conversations"][1]["from"] == "gpt"
    conversation_prefix = "Sure! Here is the design results: "
    answer = example["conversations"][1]["value"].removeprefix(conversation_prefix)
    assert json.loads(answer)[0]["box"] == [0.375, 0.4167, 0.625, 0.5833]


def test_iter_user_constrained_examples_preserves_source_and_constraints(
    dataset_module: ModuleType,
    tiny_user_constrained_data: Path,
):
    root = dataset_module._find_user_constrained_root(tiny_user_constrained_data)
    train_examples = list(dataset_module._iter_user_constrained_examples(root, "train"))
    validation_examples = list(
        dataset_module._iter_user_constrained_examples(root, "validation")
    )

    assert [key for key, _ in train_examples] == ["cgl-cgl_item", "posterlayout-0"]
    assert validation_examples[0][0] == "cgl-cgl_val_item"
    assert validation_examples[0][1]["source_dataset"] == "cgl"
    assert validation_examples[0][1]["num_constraints"] == 2
    assert validation_examples[0][1]["user_constraints"][0].startswith("logo_0")


def test_load_dataset_with_tiny_data_dir(
    dataset_path: str,
    tiny_qb_poster_data: Path,
    tiny_user_constrained_data: Path,
):
    qb_dataset = ds.load_dataset(
        path=dataset_path,
        name="qb_poster",
        data_dir=str(tiny_qb_poster_data),
        trust_remote_code=True,
    )
    assert isinstance(qb_dataset, ds.DatasetDict)
    assert list(qb_dataset) == ["train", "validation"]
    assert qb_dataset["train"].num_rows == 1
    assert qb_dataset["validation"].num_rows == 1
    assert qb_dataset["train"][0]["image"] is not None

    uc_dataset = ds.load_dataset(
        path=dataset_path,
        name="user_constrained",
        data_dir=str(tiny_user_constrained_data),
        trust_remote_code=True,
    )
    assert isinstance(uc_dataset, ds.DatasetDict)
    assert uc_dataset["train"].num_rows == 2
    assert uc_dataset["validation"].num_rows == 1
    assert uc_dataset["validation"][0]["source_id"] == "cgl_val_item"


def test_source_reported_counts_are_consistent():
    assert sum(_QB_POSTER_EXPECTED_NUM_ROWS.values()) == (
        _QB_POSTER_EXPECTED_TOTAL_ROWS
    )
    assert sum(_USER_CONSTRAINED_EXPECTED_NUM_ROWS.values()) == (
        _USER_CONSTRAINED_EXPECTED_TOTAL_ROWS
    )


def test_loader_uses_python_310_compatible_enum(dataset_path: str):
    loader_text = Path(dataset_path).read_text(encoding="utf-8")
    assert "StrEnum" not in loader_text


@pytest.mark.skipif(
    condition=os.environ.get("POSTER_LLAVA_RUN_DOWNLOAD_TESTS") != "1",
    reason=(
        "Set POSTER_LLAVA_RUN_DOWNLOAD_TESTS=1 to download and load the full "
        "PosterLLaVA dataset releases."
    ),
)
@pytest.mark.parametrize("config_name", ("qb_poster", "user_constrained"))
def test_load_dataset(dataset_path: str, repo_id: str, config_name: str):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=config_name,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert list(dataset) == ["train", "validation"]

    if config_name == "qb_poster":
        for split_name, expected_num_rows in _QB_POSTER_EXPECTED_NUM_ROWS.items():
            assert dataset[split_name].num_rows == expected_num_rows
        sample = dataset["train"][0]
        assert sample["image"] is not None
        assert sample["elements"]
        assert sample["conversations"][0]["from"] == "human"
    else:
        for split_name, expected_num_rows in _USER_CONSTRAINED_EXPECTED_NUM_ROWS.items():
            assert dataset[split_name].num_rows == expected_num_rows
        sample = dataset["train"][0]
        assert sample["source_dataset"] in {"cgl", "posterlayout"}
        assert sample["user_constraints"]

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
