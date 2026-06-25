import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import HfApi
from PIL import Image

import datasets as ds


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "GenPoster100K"


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
    spec = importlib.util.spec_from_file_location("GenPoster100K", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["GenPoster100K"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder(dataset_module: ModuleType):
    return dataset_module.GenPoster100K()


def test_builder_info(builder):
    info = builder.info
    assert info.features is not None

    expected_root_features = {
        "id",
        "background_image",
        "background_image_relpath",
        "layers",
        "regions",
        "psd_path",
    }
    assert expected_root_features.issubset(info.features.keys())
    assert "merged_image" not in info.features

    layer_feature = info.features["layers"].feature
    assert layer_feature["label"].__class__.__name__ == "Value"
    assert "layer_image" in layer_feature
    assert "layer_image_relpath" in layer_feature
    assert "fill_color" in layer_feature
    assert "bbox" in layer_feature


def test_normalize_relative_image_path(builder):
    signed_url = (
        "https://example.com/big_poster/poster_metadata/sample_bg.png"
        "?Expires=123&Signature=abc"
    )

    assert (
        builder._normalize_relative_image_path(signed_url)
        == "big_poster/poster_metadata/sample_bg.png"
    )
    assert (
        builder._normalize_relative_image_path("/poster_metadata/sample.png")
        == "big_poster/poster_metadata/sample.png"
    )


def test_build_image_index_filters_image_files(builder, tmp_path: Path):
    image_path = (
        tmp_path
        / "big_poster"
        / "poster_metadata"
        / "poster_metadata_split"
        / "part_0"
        / "sample_bg.png"
    )
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (255, 255, 255)).save(image_path)
    (image_path.parent / "notes.txt").write_text("not an image", encoding="utf-8")

    image_index, basename_index = builder._build_image_index([tmp_path.as_posix()])

    assert (
        image_index[
            "big_poster/poster_metadata/poster_metadata_split/part_0/sample_bg.png"
        ]
        == image_path.as_posix()
    )
    assert basename_index["sample_bg.png"] == image_path.as_posix()
    assert "notes.txt" not in basename_index


@pytest.mark.skipif(
    condition=os.environ.get("RUN_HEAVY_DATASET_TESTS") != "1",
    reason=(
        "Set RUN_HEAVY_DATASET_TESTS=1 to run full GenPoster100K smoke test. "
        "This test downloads/extracts a very large dataset."
    ),
)
def test_load_dataset(dataset_path: str, repo_id: str):
    dataset = ds.load_dataset(path=dataset_path, trust_remote_code=True)
    assert isinstance(dataset, ds.DatasetDict)
    assert "train" in dataset
    assert dataset["train"].num_rows > 0

    sample = dataset["train"][0]
    assert sample["background_image"] is not None
    assert sample["background_image_relpath"]
    assert len(sample["regions"]) >= 0
    assert len(sample["layers"]) > 0
    assert isinstance(sample["layers"][0]["layer_name"], str)
    assert isinstance(sample["layers"][0]["layer_image_relpath"], str)

    if os.environ.get("RUN_HEAVY_DATASET_TESTS_PUSH"):
        dataset.push_to_hub(
            repo_id=repo_id,
            private=True,
            max_shard_size="50MB",
        )


def test_push_readme_to_hub(
    hf_api: HfApi,
    repo_id: str,
    script_dir: str,
):
    if not os.environ.get("RUN_HEAVY_DATASET_TESTS_PUSH"):
        pytest.skip("Set RUN_HEAVY_DATASET_TESTS_PUSH=1 to enable Hub push.")

    readme_path = os.path.join(script_dir, "README.md")

    hf_api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
