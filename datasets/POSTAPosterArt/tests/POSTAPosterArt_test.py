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

# Keep Hub Parquet row groups comfortably below the Dataset Viewer scan limit
# (300MB). The default push_to_hub shard size can create single row groups above
# that limit for image-heavy examples, which makes the Hub preview fail.
_HUB_DESIGN_MAX_SHARD_SIZE = "100MB"

# POSTA text has heavy image triplets. Even with a small max_shard_size, the
# writer still emitted 100-row Parquet row groups just over the viewer limit.
# Use num_shards for this config so each shard stays below that 100-row boundary.
_HUB_TEXT_NUM_SHARDS = {"train": 40}


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "POSTAPosterArt"


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
    spec = importlib.util.spec_from_file_location("POSTAPosterArt", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["POSTAPosterArt"] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(path)


def test_iter_text_examples_groups_complete_records(
    tmp_path: Path, dataset_module: ModuleType
):
    root = tmp_path / "PosterArt-Text"
    _write_image(root / "sample.jpg")
    _write_image(root / "sample_mask.png")
    _write_image(root / "sample_mask_img_single.png")
    (root / "sample.caption").write_text("baroque font made of coral", encoding="utf-8")
    (root / ".DS_Store").write_text("", encoding="utf-8")
    macosx_dir = tmp_path / "__MACOSX" / "PosterArt-Text"
    macosx_dir.mkdir(parents=True)
    (macosx_dir / "._sample.jpg").write_text("", encoding="utf-8")

    examples = list(dataset_module._iter_text_examples(tmp_path))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "sample"
    assert example["id"] == "sample"
    assert example["caption"] == "baroque font made of coral"
    assert example["image"].endswith("sample.jpg")
    assert example["mask"].endswith("sample_mask.png")
    assert example["mask_img_single"].endswith("sample_mask_img_single.png")


def test_iter_text_examples_skips_incomplete_records(
    tmp_path: Path, dataset_module: ModuleType
):
    root = tmp_path / "PosterArt-Text"
    _write_image(root / "sample.jpg")
    (root / "sample.caption").write_text("missing masks", encoding="utf-8")

    assert list(dataset_module._iter_text_examples(tmp_path)) == []


def test_flatten_text_layers_recurses_nested_groups(dataset_module: ModuleType):
    layers = [
        {
            "name": "group",
            "children": [
                {
                    "name": "Title",
                    "path": "group/Title",
                    "visible": True,
                    "kind": "type",
                    "opacity": 255,
                    "blend_mode": "BlendMode.NORMAL",
                    "position": {
                        "left": 1,
                        "top": 2,
                        "right": 11,
                        "bottom": 22,
                        "width": 10,
                        "height": 20,
                        "center_x": 6.0,
                        "center_y": 12.0,
                    },
                    "text_content": "POSTA",
                    "font_info": {
                        "font_name": "PlayfairDisplay",
                        "font_size": 45.5,
                        "color_values": [1.0, 1.0, 1.0, 1.0],
                        "alignment": "center",
                        "rotation": 12.0,
                    },
                }
            ],
        }
    ]

    flattened = dataset_module._flatten_text_layers(layers)

    assert len(flattened) == 1
    assert flattened[0]["path"] == "group/Title"
    assert flattened[0]["text_content"] == "POSTA"
    assert flattened[0]["font_name"] == "PlayfairDisplay"
    assert flattened[0]["left"] == 1
    assert flattened[0]["color_values"] == [1.0, 1.0, 1.0, 1.0]


def test_iter_design_examples_matches_assets_by_stem(
    tmp_path: Path, dataset_module: ModuleType
):
    root = tmp_path / "Part1"
    _write_image(root / "background" / "0002.jpg")
    _write_image(root / "JPG" / "0002.jpg")
    (root / "PSD").mkdir(parents=True)
    (root / "PSD" / "0002.psd").write_bytes(b"psd")
    annotation = {
        "filename": "0002.psd",
        "width": 2048,
        "height": 3072,
        "layers": [
            {
                "name": "Title",
                "path": "Title",
                "kind": "type",
                "text_content": "POSTA",
            }
        ],
    }
    (root / "json").mkdir(parents=True)
    (root / "json" / "0002.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )

    examples = list(dataset_module._iter_design_examples(tmp_path))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "0002"
    assert example["id"] == "Part1/0002"
    assert example["background_image"].endswith("background/0002.jpg")
    assert example["poster_image"].endswith("JPG/0002.jpg")
    assert example["psd_filename"] == "0002.psd"
    assert json.loads(example["annotation"]) == annotation
    assert example["text_layers"][0]["text_content"] == "POSTA"


@pytest.mark.skipif(
    condition=os.environ.get("POSTA_POSTER_ART_RUN_DOWNLOAD_TESTS") != "1",
    reason="Set POSTA_POSTER_ART_RUN_DOWNLOAD_TESTS=1 to download and load the full dataset.",
)
def test_load_text_dataset(dataset_path: str):
    dataset = ds.load_dataset(
        path=dataset_path,
        name="text",
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == 2218

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(
            repo_id="creative-graphic-design/POSTAPosterArt",
            config_name="text",
            num_shards=_HUB_TEXT_NUM_SHARDS,
        )


@pytest.mark.skipif(
    condition=os.environ.get("POSTA_POSTER_ART_RUN_DOWNLOAD_TESTS") != "1",
    reason="Set POSTA_POSTER_ART_RUN_DOWNLOAD_TESTS=1 to download and load the full dataset.",
)
def test_load_design_dataset(dataset_path: str):
    dataset = ds.load_dataset(
        path=dataset_path,
        name="design",
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == 353

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(
            repo_id="creative-graphic-design/POSTAPosterArt",
            config_name="design",
            max_shard_size=_HUB_DESIGN_MAX_SHARD_SIZE,
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
