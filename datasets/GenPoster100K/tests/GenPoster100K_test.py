import importlib.util
import os
import sys
import tempfile
from io import BytesIO
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
        "merged_image",
        "layers",
        "regions",
        "psd_path",
    }
    assert expected_root_features.issubset(info.features.keys())
    assert "background_image_relpath" not in info.features

    layer_feature = info.features["layers"].feature
    assert layer_feature["label"].__class__.__name__ == "ClassLabel"
    assert "layer_image" in layer_feature
    assert "layer_image_relpath" not in layer_feature
    assert "fill_color" in layer_feature
    assert "bbox" in layer_feature


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
            [layer_a_path, None, layer_b_path],
        )

    assert isinstance(merged_image, dict)
    assert merged_image["path"] is None
    assert isinstance(merged_image["bytes"], bytes)

    with Image.open(BytesIO(merged_image["bytes"])) as image:
        pixels = image.convert("RGBA")
        assert pixels.getpixel((0, 0)) == (0, 0, 255, 255)
        assert pixels.getpixel((3, 3))[:3] == (255, 127, 127)


def test_normalize_label_rejects_unknown(builder):
    with pytest.raises(AssertionError):
        builder._normalize_label("Unknown Label")


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
    assert sample["merged_image"] is not None
    assert len(sample["regions"]) >= 0
    assert len(sample["layers"]) > 0
    assert isinstance(sample["layers"][0]["layer_name"], str)

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
