import os

import datasets as ds
import pytest


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "CGLDatasetV2"


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str) -> str:
    return f"{org_name}/CGL-Dataset-v2"


@pytest.mark.parametrize("include_text_features", (True, False))
@pytest.mark.parametrize("decode_rle", (True, False))
def test_load_dataset_builder(
    dataset_path: str,
    include_text_features: bool,
    decode_rle: bool,
):
    builder = ds.load_dataset_builder(
        path=dataset_path,
        decode_rle=decode_rle,
        include_text_features=include_text_features,
        trust_remote_code=True,
    )
    assert {"image", "annotations"} <= set(builder.info.features)


def get_load_kwargs(dataset_path: str, **kwargs):
    load_kwargs = {
        "path": dataset_path,
        "trust_remote_code": True,
        **kwargs,
    }
    if local_archive := os.environ.get("CGL_DATASET_V2_ARCHIVE"):
        load_kwargs["data_dir"] = local_archive
    return load_kwargs


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames="decode_rle",
    argvalues=(
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    argnames="include_text_features",
    argvalues=(
        True,
        False,
    ),
)
def test_load_dataset(
    dataset_path: str,
    include_text_features: bool,
    decode_rle: bool,
    expected_num_train: int = 60548,
    expected_num_test: int = 1035,
):
    dataset = ds.load_dataset(
        **get_load_kwargs(
            dataset_path,
            decode_rle=decode_rle,
            include_text_features=include_text_features,
        ),
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test


def test_load_dataset_with_data_dir(
    dataset_path: str,
    expected_num_train: int = 60548,
    expected_num_test: int = 1035,
):
    local_archive = os.environ.get("CGL_DATASET_V2_ARCHIVE")
    if not local_archive:
        pytest.skip("Set CGL_DATASET_V2_ARCHIVE to test loading a local archive.")

    dataset = ds.load_dataset(
        path=dataset_path,
        data_dir=local_archive,
        decode_rle=False,
        include_text_features=False,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test


def test_push_to_hub(
    repo_id: str,
    dataset_path: str,
):
    if not os.environ.get("HF_WRITE_TESTS"):
        pytest.skip("Set HF_WRITE_TESTS=1 to push to Hugging Face Hub.")

    dataset = ds.load_dataset(
        **get_load_kwargs(
            dataset_path,
            decode_rle=True,
            include_text_features=True,
            rename_category_names=True,
        ),
    )
    assert isinstance(dataset, ds.DatasetDict)

    dataset.push_to_hub(repo_id=repo_id, private=True)
