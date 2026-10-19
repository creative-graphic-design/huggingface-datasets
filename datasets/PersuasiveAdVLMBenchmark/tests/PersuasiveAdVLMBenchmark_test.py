import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "PersuasiveAdVLMBenchmark.py"


@pytest.fixture
def dataset_module(dataset_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "PersuasiveAdVLMBenchmark",
        dataset_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["PersuasiveAdVLMBenchmark"] = module
    spec.loader.exec_module(module)
    return module


def test_iter_examples_sorts_identifiers_and_preserves_url_order(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "1/second.jpg": ["https://example.com/b", "https://example.com/a"],
                "0/first.jpg": ["https://example.com/c"],
            }
        ),
        encoding="utf-8",
    )

    examples = list(dataset_module._iter_examples(metadata_path))

    assert examples == [
        (
            "0/first.jpg",
            {
                "id": "0/first.jpg",
                "image_path": "0/first.jpg",
                "image_urls": ["https://example.com/c"],
            },
        ),
        (
            "1/second.jpg",
            {
                "id": "1/second.jpg",
                "image_path": "1/second.jpg",
                "image_urls": ["https://example.com/b", "https://example.com/a"],
            },
        ),
    ]


@pytest.mark.parametrize(
    ("metadata", "error", "message"),
    [
        ([], TypeError, "Expected the released metadata to be a JSON object."),
        (
            {"0/image.jpg": "https://example.com/image.jpg"},
            TypeError,
            "Expected each metadata entry to map an image path to a URL list.",
        ),
        (
            {"0/image.jpg": [1]},
            TypeError,
            "Expected every image URL to be a string for 0/image.jpg.",
        ),
    ],
)
def test_iter_examples_rejects_invalid_metadata(
    tmp_path: Path,
    dataset_module: ModuleType,
    metadata: object,
    error: type[Exception],
    message: str,
):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(error, match=message):
        list(dataset_module._iter_examples(metadata_path))


@pytest.mark.skipif(
    os.environ.get("PERSUASIVE_AD_VLM_RUN_DOWNLOAD_TESTS") != "1",
    reason="Set PERSUASIVE_AD_VLM_RUN_DOWNLOAD_TESTS=1 to download the source JSON.",
)
def test_load_dataset(dataset_path: Path):
    dataset = ds.load_dataset(
        path=str(dataset_path),
        split="test",
        trust_remote_code=True,
    )

    assert isinstance(dataset, ds.Dataset)
    assert dataset.num_rows == 783
    assert set(dataset.features) == {"id", "image_path", "image_urls"}
    assert dataset[0]["id"] == dataset[0]["image_path"]
    assert dataset[0]["image_urls"]
