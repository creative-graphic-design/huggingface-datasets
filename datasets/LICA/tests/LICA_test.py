import json
import os
from pathlib import Path

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
    return "LICA"


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
def tiny_lica_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "lica-data"
    template_id = "template-001"
    layout_id = "layout-001"

    (data_dir / "images" / template_id).mkdir(parents=True)
    (data_dir / "layouts" / template_id).mkdir(parents=True)
    (data_dir / "annotations" / template_id).mkdir(parents=True)

    image_path = data_dir / "images" / template_id / f"{layout_id}.png"
    Image.new("RGB", (16, 12), color=(240, 120, 40)).save(image_path)

    metadata = (
        "file_name,layout_id,category,template_id,n_template_layouts,"
        "template_layout_index,width,height\n"
        f"images/{template_id}/{layout_id}.png,{layout_id},Posters,{template_id},1,0,16,12\n"
    )
    (data_dir / "metadata.csv").write_text(metadata, encoding="utf-8")

    layout = {
        "components": [
            {
                "type": "TEXT",
                "text": "Sample",
                "left": "1px",
                "top": "2px",
                "width": "10px",
                "height": "4px",
            },
            {
                "type": "IMAGE",
                "src": "https://example.com/sample.png",
                "left": "0px",
                "top": "0px",
                "width": "16px",
                "height": "12px",
            },
        ],
        "background": "rgb(255, 255, 255)",
        "width": "16px",
        "height": "12px",
        "duration": 3,
    }
    (data_dir / "layouts" / template_id / f"{layout_id}.json").write_text(
        json.dumps(layout),
        encoding="utf-8",
    )

    annotation = {
        "description": "A compact poster layout.",
        "aesthetics": "Simple and balanced.",
        "tags": "poster, sample",
        "user_intent": "Create a small poster.",
        "raw": "Description:\nA compact poster layout.",
    }
    (data_dir / "annotations" / template_id / f"{layout_id}.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    template_annotations = {
        template_id: {
            "description": "Template-level description.",
            "aesthetics": "Template-level aesthetics.",
            "tags": "template, poster",
            "user_intent": "Create related poster variants.",
            "raw": "Description:\nTemplate-level description.",
        }
    }
    (data_dir / "annotations" / "template_annotations.json").write_text(
        json.dumps(template_annotations),
        encoding="utf-8",
    )

    return data_dir


def test_load_dataset(
    dataset_path: str,
    tiny_lica_data: Path,
    trust_remote_code: bool = True,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        data_dir=str(tiny_lica_data),
        trust_remote_code=trust_remote_code,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["test"].num_rows == 1

    features = dataset["test"].features
    assert "render_image" in features
    assert "layout_json" in features
    assert "annotation_json" in features
    assert "template_annotation_json" in features

    sample = dataset["test"][0]
    assert sample["layout_id"] == "layout-001"
    assert sample["template_id"] == "template-001"
    assert sample["category"] == "Posters"
    assert sample["n_template_layouts"] == 1
    assert sample["template_layout_index"] == 0
    assert sample["width"] == 16
    assert sample["height"] == 12
    assert sample["render_type"] == "png"
    assert sample["render_image"] is not None
    assert sample["layout_width"] == 16
    assert sample["layout_height"] == 12
    assert sample["layout_background"] == "rgb(255, 255, 255)"
    assert sample["layout_duration"] == 3
    assert sample["n_components"] == 2
    assert sample["component_types"] == ["TEXT", "IMAGE"]
    assert sample["description"] == "A compact poster layout."
    assert sample["template_description"] == "Template-level description."

    layout_json = json.loads(sample["layout_json"])
    annotation_json = json.loads(sample["annotation_json"])
    template_annotation_json = json.loads(sample["template_annotation_json"])
    assert layout_json["components"][0]["type"] == "TEXT"
    assert annotation_json["tags"] == "poster, sample"
    assert template_annotation_json["tags"] == "template, poster"


@pytest.mark.skipif(
    condition=not bool(os.environ.get("LICA_FULL_TESTS")),
    reason="Set LICA_FULL_TESTS=1 to download and validate the full LICA archive.",
)
def test_load_full_dataset(
    dataset_path: str,
    trust_remote_code: bool = True,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        trust_remote_code=trust_remote_code,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["test"].num_rows == 1183


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
