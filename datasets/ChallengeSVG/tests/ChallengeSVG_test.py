import importlib.util
import os
from io import BytesIO
from pathlib import Path
from types import ModuleType

import datasets as ds
import pytest
from huggingface_hub import HfApi
from PIL import Image

_EXPECTED_ROWS = 35
_HUB_ID = "creative-graphic-design/ChallengeSVG"
_SIMPLE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><circle cx="0.5" cy="0.5" r="0.5" fill="red"/></svg>'


@pytest.fixture
def script_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset_path(script_dir: Path) -> Path:
    return script_dir / "ChallengeSVG.py"


@pytest.fixture
def dataset_module(dataset_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("ChallengeSVG", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hf_api() -> HfApi:
    return HfApi()


def test_render_svg_creates_transparent_512px_png(dataset_module: ModuleType):
    image = Image.open(BytesIO(dataset_module._render_svg(_SIMPLE_SVG)))

    assert image.size == (512, 512)
    assert "A" in image.getbands()


def test_load_dataset(dataset_path: Path):
    dataset = ds.load_dataset(
        path=str(dataset_path),
        trust_remote_code=True,
        download_mode=ds.DownloadMode.FORCE_REDOWNLOAD,
    )

    assert isinstance(dataset, ds.DatasetDict)
    assert set(dataset) == {"test"}
    test_split = dataset["test"]
    assert test_split.num_rows == _EXPECTED_ROWS
    assert test_split.column_names == ["id", "prompt", "svg", "image"]
    assert len(set(test_split["id"])) == _EXPECTED_ROWS
    for row in test_split:
        assert row["prompt"].strip()
        assert row["svg"].lstrip().startswith("<svg")
        assert row["image"].size == (512, 512)
        assert "A" in row["image"].getbands()


@pytest.mark.skipif(
    os.environ.get("HF_WRITE_TESTS") != "1",
    reason="Set HF_WRITE_TESTS=1 to publish ChallengeSVG to the Hub.",
)
def test_push_dataset_to_hub(dataset_path: Path):
    dataset = ds.load_dataset(path=str(dataset_path), trust_remote_code=True)
    dataset.push_to_hub(repo_id=_HUB_ID, max_shard_size="50MB")


@pytest.mark.skipif(
    os.environ.get("HF_WRITE_TESTS") != "1",
    reason="Set HF_WRITE_TESTS=1 to publish the ChallengeSVG card.",
)
def test_push_readme_to_hub(hf_api: HfApi, script_dir: Path):
    hf_api.upload_file(
        path_or_fileobj=script_dir / "README.md",
        path_in_repo="README.md",
        repo_id=_HUB_ID,
        repo_type="dataset",
    )
