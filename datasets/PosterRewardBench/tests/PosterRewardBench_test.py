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


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "PosterRewardBench"


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
    spec = importlib.util.spec_from_file_location("PosterRewardBench", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["PosterRewardBench"] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(path)


def test_find_image_root_finds_nested_directory(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    root = tmp_path / "archive" / "nested" / "PRB_basic_images"
    root.mkdir(parents=True)

    assert dataset_module._find_image_root(tmp_path, "PRB_basic_images") == root


def test_find_image_root_raises_for_missing_directory(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    with pytest.raises(FileNotFoundError):
        dataset_module._find_image_root(tmp_path, "PRB_basic_images")


def test_iter_examples_resolves_pair_paths_and_preserves_messages(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    image_root = tmp_path / "PRB_basic_images"
    chosen_path = image_root / "prompt_001_img01_chosen.png"
    rejected_path = image_root / "prompt_001_img02_reject.png"
    _write_image(chosen_path)
    _write_image(rejected_path)

    rows = [
        {
            "messages": [
                {"role": "user", "content": "A festival poster with bold text."},
                {"role": "assistant", "content": ""},
            ],
            "rejected_messages": [
                {"role": "user", "content": "A festival poster with bold text."},
                {"role": "assistant", "content": ""},
            ],
            "images": ["PRB_basic_images/prompt_001_img01_chosen.png"],
            "rejected_images": ["PRB_basic_images/prompt_001_img02_reject.png"],
        }
    ]
    metadata_path = tmp_path / "PRB_basic_relative.json"
    metadata_path.write_text(json.dumps(rows), encoding="utf-8")

    examples = list(dataset_module._iter_examples(metadata_path, image_root, "basic"))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "basic-00000"
    assert example["id"] == "basic-00000"
    assert example["prompt"] == "A festival poster with bold text."
    assert Path(example["chosen_image"]).is_file()
    assert Path(example["rejected_image"]).is_file()
    assert example["chosen_image_path"].endswith("prompt_001_img01_chosen.png")
    assert example["rejected_image_path"].endswith("prompt_001_img02_reject.png")
    assert example["messages"][0] == {
        "role": "user",
        "content": "A festival poster with bold text.",
    }
    assert example["rejected_messages"][1] == {"role": "assistant", "content": ""}


@pytest.mark.skipif(
    condition=os.environ.get("POSTER_REWARD_BENCH_RUN_DOWNLOAD_TESTS") != "1",
    reason=(
        "Set POSTER_REWARD_BENCH_RUN_DOWNLOAD_TESTS=1 to download and load "
        "the full dataset."
    ),
)
@pytest.mark.parametrize(
    argnames=("config_name", "expected_num_train"),
    argvalues=(("basic", 517), ("advanced", 1223)),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    config_name: str,
    expected_num_train: int,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=config_name,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train

    sample = dataset["train"][0]
    assert sample["prompt"]
    assert sample["chosen_image"] is not None
    assert sample["rejected_image"] is not None
    assert sample["messages"][0]["role"] == "user"
    assert sample["rejected_messages"][0]["content"] == sample["prompt"]

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
