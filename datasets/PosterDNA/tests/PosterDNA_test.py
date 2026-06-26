import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import HfApi
from PIL import Image

import datasets as ds

_HUB_MAX_SHARD_SIZE = "50MB"

# Counts reported by the PosterVerse paper appendix for PosterDNA sub-tasks.
_PAPER_EXPECTED_NUM_BLUEPRINT_CREATION = 57_000
_PAPER_EXPECTED_NUM_GRAPHICAL_BACKGROUND_GENERATION = 100_000
_PAPER_EXPECTED_NUM_UNIFIED_LAYOUT_TEXT_RENDERING = 9_000
_PAPER_EXPECTED_NUM_TEST_SET = 1_000

# Counts observed from the official Hugging Face ZIP central directories.
_SOURCE_EXPECTED_NUM_TEST_JSON = 1_000
_SOURCE_EXPECTED_NUM_TEST_BACKGROUNDS = 1_000
_SOURCE_EXPECTED_NUM_TEST_HTML = 1_000


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "PosterDNA"


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
    spec = importlib.util.spec_from_file_location("PosterDNA", dataset_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["PosterDNA"] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(path)


def _write_zip_from_dir(archive_path: Path, source_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir).as_posix())


def _write_script_with_local_urls(
    source_script: Path,
    output_script: Path,
    posterdna_zip: Path,
    test_set_zip: Path,
) -> None:
    script = source_script.read_text(encoding="utf-8")
    script = script.replace(
        "https://huggingface.co/wuhaer/PosterVerse/resolve/main/posterdna.zip",
        posterdna_zip.as_uri(),
    )
    script = script.replace(
        "https://huggingface.co/wuhaer/PosterVerse/resolve/main/test-set.zip",
        test_set_zip.as_uri(),
    )
    output_script.write_text(script, encoding="utf-8")


def test_load_dataset_from_local_posterdna_archive(
    tmp_path: Path,
    dataset_path: str,
):
    archive_root = tmp_path / "archive"
    root = archive_root / "posterdna"
    jsonl_path = root / "Poster_Intention_Analysis_Dataset.jsonl"
    html_path = root / "html" / "abc.html"
    image_path = root / "bg" / "abc.png"
    html_path.parent.mkdir(parents=True)
    row = {
        "id": "abc",
        "prompt": "make a text-dense sale poster",
        "html_file": "html/abc.html",
        "background": "bg/abc.png",
    }
    jsonl_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path.write_text("<main>sale</main>", encoding="utf-8")
    _write_image(image_path)

    posterdna_zip = tmp_path / "posterdna.zip"
    test_set_zip = tmp_path / "test-set.zip"
    _write_zip_from_dir(posterdna_zip, archive_root)
    _write_zip_from_dir(test_set_zip, archive_root)

    local_script = tmp_path / "PosterDNA.py"
    _write_script_with_local_urls(
        Path(dataset_path),
        local_script,
        posterdna_zip,
        test_set_zip,
    )

    dataset = ds.load_dataset(
        path=str(local_script),
        name="posterdna",
        trust_remote_code=True,
    )

    assert isinstance(dataset, ds.DatasetDict)
    assert list(dataset) == ["train"]
    assert dataset["train"].num_rows == 1
    sample = dataset["train"][0]
    assert sample["id"] == "abc"
    assert json.loads(sample["metadata"]) == row
    assert sample["background_image"] is not None
    assert sample["html"] == "<main>sale</main>"


def test_load_dataset_from_local_test_set_archive(
    tmp_path: Path,
    dataset_path: str,
):
    archive_root = tmp_path / "archive"
    root = archive_root / "test-set"
    metadata_path = root / "json" / "design" / "42.json"
    html_path = root / "html" / "design" / "42.html"
    image_path = root / "bg" / "design" / "42.png"
    metadata_path.parent.mkdir(parents=True)
    html_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps({"prompt": "a seasonal poster"}, ensure_ascii=False),
        encoding="utf-8",
    )
    html_path.write_text("<html><body>poster</body></html>", encoding="utf-8")
    _write_image(image_path)

    posterdna_zip = tmp_path / "posterdna.zip"
    test_set_zip = tmp_path / "test-set.zip"
    _write_zip_from_dir(posterdna_zip, archive_root)
    _write_zip_from_dir(test_set_zip, archive_root)

    local_script = tmp_path / "PosterDNA.py"
    _write_script_with_local_urls(
        Path(dataset_path),
        local_script,
        posterdna_zip,
        test_set_zip,
    )

    dataset = ds.load_dataset(
        path=str(local_script),
        name="test_set",
        trust_remote_code=True,
    )

    assert isinstance(dataset, ds.DatasetDict)
    assert list(dataset) == ["test"]
    assert dataset["test"].num_rows == 1
    sample = dataset["test"][0]
    assert sample["id"] == "42"
    assert json.loads(sample["metadata"]) == {"prompt": "a seasonal poster"}
    assert sample["background_image"] is not None
    assert sample["html"] == "<html><body>poster</body></html>"


def test_iter_test_set_examples_pairs_metadata_html_and_image(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    root = tmp_path / "test-set"
    metadata_path = root / "json" / "design" / "42.json"
    html_path = root / "html" / "design" / "42.html"
    image_path = root / "bg" / "design" / "42.png"
    metadata_path.parent.mkdir(parents=True)
    html_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps({"prompt": "a seasonal poster"}, ensure_ascii=False),
        encoding="utf-8",
    )
    html_path.write_text("<html><body>poster</body></html>", encoding="utf-8")
    _write_image(image_path)

    examples = list(dataset_module._iter_test_set_examples(root))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "42"
    assert example["id"] == "42"
    assert json.loads(example["metadata"]) == {"prompt": "a seasonal poster"}
    assert example["metadata_path"] == "json/design/42.json"
    assert Path(example["background_image"]).is_file()
    assert example["background_image_path"] == "bg/design/42.png"
    assert example["html"] == "<html><body>poster</body></html>"
    assert example["html_path"] == "html/design/42.html"


def test_iter_posterdna_examples_resolves_referenced_assets(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    root = tmp_path / "posterdna"
    jsonl_path = root / "Poster_Intention_Analysis_Dataset.jsonl"
    html_path = root / "html" / "abc.html"
    image_path = root / "bg" / "abc.png"
    jsonl_path.parent.mkdir(parents=True)
    html_path.parent.mkdir(parents=True)
    row = {
        "id": "abc",
        "prompt": "make a text-dense sale poster",
        "html_file": "html/abc.html",
        "background": "bg/abc.png",
    }
    jsonl_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path.write_text("<main>sale</main>", encoding="utf-8")
    _write_image(image_path)

    examples = list(dataset_module._iter_posterdna_examples(root))

    assert len(examples) == 1
    key, example = examples[0]
    assert key == "abc"
    assert json.loads(example["metadata"]) == row
    assert example["metadata_path"] == "Poster_Intention_Analysis_Dataset.jsonl"
    assert Path(example["background_image"]).is_file()
    assert example["background_image_path"] == "bg/abc.png"
    assert example["html"] == "<main>sale</main>"
    assert example["html_path"] == "html/abc.html"


def test_extract_zip_raises_for_password_protected_archives(
    tmp_path: Path,
    dataset_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
):
    archive_path = tmp_path / "protected.zip"
    archive_path.write_bytes(b"PK")
    monkeypatch.setattr(dataset_module, "_requires_password", lambda _: True)

    with pytest.raises(RuntimeError, match="POSTERDNA_ZIP_PASSWORD"):
        dataset_module._extract_zip(archive_path, tmp_path / "out", password=None)


def test_extract_zip_extracts_unprotected_archive(
    tmp_path: Path,
    dataset_module: ModuleType,
):
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("posterdna/html/1.html", "<p>ok</p>")

    extracted_dir = dataset_module._extract_zip(
        archive_path,
        tmp_path / "extracted",
        password=None,
    )

    assert (extracted_dir / "posterdna" / "html" / "1.html").read_text(
        encoding="utf-8"
    ) == "<p>ok</p>"
    assert (extracted_dir / ".extracted").is_file()


def test_paper_and_source_reported_counts_are_consistent():
    assert _PAPER_EXPECTED_NUM_TEST_SET == _SOURCE_EXPECTED_NUM_TEST_JSON
    assert _SOURCE_EXPECTED_NUM_TEST_BACKGROUNDS == _PAPER_EXPECTED_NUM_TEST_SET
    assert _SOURCE_EXPECTED_NUM_TEST_HTML == _PAPER_EXPECTED_NUM_TEST_SET
    assert _PAPER_EXPECTED_NUM_BLUEPRINT_CREATION == 57_000
    assert _PAPER_EXPECTED_NUM_GRAPHICAL_BACKGROUND_GENERATION == 100_000
    assert _PAPER_EXPECTED_NUM_UNIFIED_LAYOUT_TEXT_RENDERING == 9_000


@pytest.mark.skipif(
    condition=os.environ.get("POSTERDNA_RUN_DOWNLOAD_TESTS") != "1",
    reason=(
        "Set POSTERDNA_RUN_DOWNLOAD_TESTS=1 and POSTERDNA_ZIP_PASSWORD to "
        "download and load the full password-protected archives."
    ),
)
@pytest.mark.parametrize(
    argnames=("config_name", "split_name", "expected_num_rows"),
    argvalues=(
        ("posterdna", "train", _PAPER_EXPECTED_NUM_GRAPHICAL_BACKGROUND_GENERATION),
        ("test_set", "test", _PAPER_EXPECTED_NUM_TEST_SET),
    ),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    config_name: str,
    split_name: str,
    expected_num_rows: int,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=config_name,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert list(dataset) == [split_name]
    assert dataset[split_name].num_rows == expected_num_rows

    sample = dataset[split_name][0]
    assert sample["id"]
    assert sample["metadata"]

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
