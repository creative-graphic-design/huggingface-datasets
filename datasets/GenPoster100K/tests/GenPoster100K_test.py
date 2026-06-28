import importlib.util
import math
import os
import pickle
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import HfApi, HfFileSystem
from PIL import Image
import pyarrow.parquet as pq

import datasets as ds

# Keep Hub Parquet row groups below the Dataset Viewer scan limit (300MB).
# GenPoster100K has image-heavy examples because each row embeds background,
# merged, and layer images. 1024 shards still emitted ~100-row Parquet files
# that could exceed the viewer scan limit, so use ~25 rows per shard instead.
_HUB_NUM_SHARDS = {"train": 4096}
_EXPECTED_TRAIN_ROWS = 102_703
_HUB_VIEWER_SCAN_LIMIT_BYTES = 300_000_000
_HUB_VIEWER_SAFE_MAX_ROWS_PER_SHARD = 32


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
        "merged_image",
        "layers",
        "regions",
        "psd_path",
    }
    assert expected_root_features.issubset(info.features.keys())

    layer_feature = info.features["layers"].feature
    assert layer_feature["label"].__class__.__name__ == "ClassLabel"
    assert "layer_image" in layer_feature
    assert "layer_image_relpath" in layer_feature
    assert "fill_color" in layer_feature
    assert "bbox" in layer_feature


def test_hub_sharding_keeps_viewer_safe_row_counts():
    rows_per_shard = math.ceil(_EXPECTED_TRAIN_ROWS / _HUB_NUM_SHARDS["train"])

    assert rows_per_shard <= _HUB_VIEWER_SAFE_MAX_ROWS_PER_SHARD


def test_readme_point_of_contact_is_explicit(script_dir: str):
    readme = Path(script_dir, "README.md").read_text(encoding="utf-8")
    contact_lines = [
        line
        for line in readme.splitlines()
        if line.startswith("- **Point of Contact:**")
    ]

    assert contact_lines == [
        "- **Point of Contact:** creative-graphic-design maintainers via "
        "https://github.com/creative-graphic-design/huggingface-datasets/issues; "
        "upstream source release maintainer: https://huggingface.co/BruceW91"
    ]


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


def test_compose_merged_image(builder):
    with tempfile.TemporaryDirectory() as tmpdir:
        background_path = os.path.join(tmpdir, "background.png")
        layer_a_path = os.path.join(tmpdir, "layer_a.png")
        layer_b_path = os.path.join(tmpdir, "layer_b.png")

        Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(background_path)
        Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(layer_a_path)
        Image.new("RGBA", (2, 2), (0, 0, 255, 255)).save(layer_b_path)

        merged_image = builder._compose_merged_image(
            background_path,
            [
                {"layer_image": layer_a_path, "bbox": [0, 0, 4, 4]},
                {"layer_image": None, "bbox": [0, 0, 0, 0]},
                {"layer_image": layer_b_path, "bbox": [2, 1, 4, 3]},
            ],
        )

    assert isinstance(merged_image, dict)
    assert merged_image["path"] is None
    assert isinstance(merged_image["bytes"], bytes)

    with Image.open(BytesIO(merged_image["bytes"])) as image:
        pixels = image.convert("RGBA")
        assert pixels.getpixel((0, 0))[:3] == (255, 127, 127)
        assert pixels.getpixel((2, 1)) == (0, 0, 255, 255)
        assert pixels.getpixel((3, 2)) == (0, 0, 255, 255)
        assert pixels.getpixel((3, 3))[:3] == (255, 127, 127)


def test_normalize_label_rejects_unknown(builder):
    with pytest.raises(AssertionError):
        builder._normalize_label("Unknown Label")


def test_generate_examples_keeps_rows_with_missing_background(
    builder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    annotation_path = tmp_path / "annotations.pkl"
    records = [
        (
            "big_poster/poster_metadata/missing_bg.png",
            [
                {
                    "LayerName": "title-layer",
                    "Text": "Poster title",
                    "Bounding Box": [1, 2, 3, 4],
                    "Angle": 0,
                    "psd_size": [10, 20],
                    "StrokeWidth": 0,
                    "Font": "Arial",
                    "FontSize": 12,
                    "Tracking": 0,
                    "Justification": 0,
                    "FillColor": [1, 1, 1, 1],
                    "img": "big_poster/poster_metadata/missing_layer.png",
                    "label": "Title",
                }
            ],
            "big_poster/meta_psd/missing.psd",
            [],
        )
    ]
    with annotation_path.open("wb") as f:
        pickle.dump(records, f)

    monkeypatch.setattr(builder, "_build_image_index", lambda _: ({}, {}))

    examples = list(builder._generate_examples(annotation_path, [tmp_path.as_posix()]))

    assert len(examples) == 1
    key, row = examples[0]
    assert key == 0
    assert row["background_image"] is None
    assert (
        row["background_image_relpath"] == "big_poster/poster_metadata/missing_bg.png"
    )
    assert row["merged_image"] is None
    assert row["layers"][0]["layer_image"] is None
    assert row["layers"][0]["label"] == "Title"


@pytest.mark.skipif(
    condition=os.environ.get("RUN_HEAVY_DATASET_TESTS") != "1",
    reason=(
        "Set RUN_HEAVY_DATASET_TESTS=1 to run full GenPoster100K smoke test. "
        "This test downloads/extracts a very large dataset."
    ),
)
def test_load_dataset(dataset_path: str, repo_id: str):
    os.environ.setdefault("GENPOSTER100K_MAX_EXAMPLES", "64")

    dataset = ds.load_dataset(path=dataset_path, trust_remote_code=True)
    assert isinstance(dataset, ds.DatasetDict)
    assert "train" in dataset
    assert dataset["train"].num_rows > 0
    assert dataset["train"].num_rows <= int(os.environ["GENPOSTER100K_MAX_EXAMPLES"])

    sample = dataset["train"][0]
    assert sample["background_image"] is not None
    assert sample["background_image_relpath"]
    assert sample["merged_image"] is not None
    assert len(sample["regions"]) >= 0
    assert len(sample["layers"]) > 0
    if isinstance(sample["layers"], dict):
        assert isinstance(sample["layers"]["layer_name"][0], str)
        assert isinstance(sample["layers"]["layer_image_relpath"][0], str)
    else:
        assert isinstance(sample["layers"][0]["layer_name"], str)
        assert isinstance(sample["layers"][0]["layer_image_relpath"], str)

    if os.environ.get("RUN_HEAVY_DATASET_TESTS_PUSH"):
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
    if not os.environ.get("RUN_HEAVY_DATASET_TESTS_PUSH"):
        pytest.skip("Set RUN_HEAVY_DATASET_TESTS_PUSH=1 to enable Hub push.")

    readme_path = os.path.join(script_dir, "README.md")

    hf_api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def test_published_hub_parquet_is_viewer_safe(repo_id: str):
    if not os.environ.get("RUN_HUB_DATASET_VALIDATION"):
        pytest.skip("Set RUN_HUB_DATASET_VALIDATION=1 to validate published Hub files.")

    info = HfApi().dataset_info(repo_id, files_metadata=True)
    parquet_files = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.endswith(".parquet")
    )

    assert len(parquet_files) == _HUB_NUM_SHARDS["train"]

    fs = HfFileSystem()
    total_rows = 0
    max_rows_per_file = 0
    max_row_group_size = 0
    for parquet_file in parquet_files:
        with fs.open(f"datasets/{repo_id}/{parquet_file}", "rb") as f:
            metadata = pq.ParquetFile(f).metadata
        total_rows += metadata.num_rows
        max_rows_per_file = max(max_rows_per_file, metadata.num_rows)
        for row_group_index in range(metadata.num_row_groups):
            max_row_group_size = max(
                max_row_group_size,
                metadata.row_group(row_group_index).total_byte_size,
            )

    assert total_rows == _EXPECTED_TRAIN_ROWS
    assert max_rows_per_file <= _HUB_VIEWER_SAFE_MAX_ROWS_PER_SHARD
    assert max_row_group_size < _HUB_VIEWER_SCAN_LIMIT_BYTES
