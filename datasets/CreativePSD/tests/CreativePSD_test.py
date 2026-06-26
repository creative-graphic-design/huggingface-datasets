import importlib.util
import json
import os
import sys
import zipfile
from io import BytesIO
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
    return "CreativePSD"


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
    spec = importlib.util.spec_from_file_location("CreativePSD", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["CreativePSD"] = module
    spec.loader.exec_module(module)
    return module


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_creative_psd_zip(path: Path, poster_id: str, total_layers: int) -> None:
    layer_info = {
        "psd_info": {
            "filename": f"{poster_id}.psd",
            "height": 800,
            "width": 600,
            "resolution": 72,
            "colorMode": "RGB",
            "fill_color": {"red": 255, "green": 255, "blue": 255},
        },
        "total_layers": total_layers,
        "layer_tree": [
            {
                "name": "Title",
                "layerid": 7,
                "kind": "text",
                "text": "Sale",
            }
        ],
    }
    tool_trajectory = [
        {
            "tool_call": "create_document",
            "parameters": {"filename": f"{poster_id}.psd", "width": 600, "height": 800},
        },
        {
            "tool_call": "insert_image",
            "parameters": {"image_path": f"{poster_id}/raw_resource/1_asset.png"},
        },
    ]

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata/1.origin_psd_tree.txt", "origin tree")
        zf.writestr("metadata/2.deleted_psd_tree.txt", "deleted tree")
        zf.writestr("metadata/3.grouped_psd_tree.txt", "grouped tree")
        zf.writestr("metadata/group_child_ids.json", json.dumps({"1": [7]}))
        zf.writestr("metadata/layer_info.json", json.dumps(layer_info))
        zf.writestr("metadata/rendering_id.json", json.dumps({"7": "step.jpg"}))
        zf.writestr("metadata/tool_trajectory.json", json.dumps(tool_trajectory))
        zf.writestr("raw_resource/1_asset.png", _png_bytes((255, 0, 0)))
        zf.writestr("raw_resource/2_asset.png", _png_bytes((0, 255, 0)))
        zf.writestr("rendering_imgs/0_total.jpg", _jpg_bytes((0, 0, 255)))
        zf.writestr("rendering_imgs/1_7_LayerKind.TEXT.jpg", _jpg_bytes((255, 255, 0)))
        zf.writestr(
            "rendering_imgs/render_second_values.json",
            json.dumps({"7": f"{poster_id}/1_7_LayerKind.TEXT.jpg"}),
        )


@pytest.fixture
def tiny_creative_psd_data(tmp_path: Path) -> Path:
    _write_creative_psd_zip(tmp_path / "poster_000001.zip", "poster_000001", 3)
    _write_creative_psd_zip(tmp_path / "poster_000002.zip", "poster_000002", 4)
    return tmp_path


def test_parse_rendering_image_name(dataset_module: ModuleType):
    assert dataset_module._parse_rendering_image_name("rendering_imgs/0_total.jpg") == (
        0,
        0,
        "TOTAL",
    )
    assert dataset_module._parse_rendering_image_name(
        "rendering_imgs/10_70_LayerKind.TEXT.jpg"
    ) == (10, 70, "TEXT")


def test_build_example_includes_all_zip_members(
    tiny_creative_psd_data: Path, dataset_module: ModuleType
):
    zip_path = tiny_creative_psd_data / "poster_000001.zip"

    with zipfile.ZipFile(zip_path) as zf:
        example = dataset_module._build_example(zip_path, zf)
        expected_file_names = set(zf.namelist())

    assert example["id"] == "poster_000001"
    assert example["psd_info"]["filename"] == "poster_000001.psd"
    assert example["total_layers"] == 3
    assert example["origin_psd_tree"] == "origin tree"
    assert len(example["metadata_files"]) == 7
    assert len(example["non_image_files"]) == 8
    assert len(example["raw_resources"]) == 2
    assert len(example["rendering_images"]) == 2
    assert example["final_rendering"]["bytes"]

    all_file_names = {record["filename"] for record in example["all_files"]}
    assert all_file_names == expected_file_names


def test_validate_zip_paths_rejects_incomplete_modelscope_checkout(
    tmp_path: Path, dataset_module: ModuleType
):
    _write_creative_psd_zip(tmp_path / "poster_000001.zip", "poster_000001", 3)
    (tmp_path / ".gitattributes").write_text("*.zip filter=lfs diff=lfs merge=lfs\n")
    (tmp_path / "README.md").write_text("# CreativePSD\n")

    with pytest.raises(ValueError, match="appears incomplete"):
        dataset_module._validate_zip_paths(
            dataset_module._zip_paths_under(tmp_path),
            tmp_path,
        )


def test_validate_zip_paths_rejects_invalid_archives(
    tmp_path: Path, dataset_module: ModuleType
):
    invalid_zip = tmp_path / "poster_000001.zip"
    invalid_zip.write_bytes(b"")

    with pytest.raises(ValueError, match="invalid CreativePSD poster archives"):
        dataset_module._validate_zip_paths([invalid_zip], tmp_path)


def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    tiny_creative_psd_data: Path,
    trust_remote_code: bool = True,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        data_dir=str(tiny_creative_psd_data),
        trust_remote_code=trust_remote_code,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == 2

    first = dataset["train"][0]
    assert first["id"] == "poster_000001"
    assert first["psd_info"]["width"] == 600
    assert first["psd_info"]["height"] == 800
    assert first["total_layers"] == 3
    assert len(first["metadata_files"]) == 7
    assert len(first["non_image_files"]) == 8
    assert len(first["raw_resources"]) == 2
    assert len(first["rendering_images"]) == 2
    assert len(first["all_files"]) == 12
    assert first["final_rendering"].size == (4, 4)
    assert first["raw_resources"][0]["image"].size == (4, 4)
    assert first["rendering_images"][0]["image"].size == (4, 4)
    assert any(record["is_final"] for record in first["rendering_images"])
    assert "create_document" in first["tool_trajectory_json"]

    if os.environ.get("HF_WRITE_TESTS"):
        dataset.push_to_hub(
            repo_id=repo_id,
            max_shard_size="50MB",
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
