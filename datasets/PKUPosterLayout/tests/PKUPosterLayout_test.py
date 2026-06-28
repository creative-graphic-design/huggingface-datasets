import os

import datasets as ds
import pytest


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "PKUPosterLayout"


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str) -> str:
    return f"{org_name}/PKU-PosterLayout"


def test_load_dataset_builder(dataset_path: str):
    builder = ds.load_dataset_builder(path=dataset_path, trust_remote_code=True)
    assert {
        "original_poster",
        "inpainted_poster",
        "basnet_saliency_map",
        "pfpn_saliency_map",
        "canvas",
        "annotations",
    } <= set(builder.info.features)


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames=(
        "expected_num_train",
        "expected_num_test",
    ),
    argvalues=((9974, 905),),
)
def test_load_dataset(
    dataset_path: str, expected_num_train: int, expected_num_test, repo_id: str
):
    dataset = ds.load_dataset(
        path=dataset_path,
        token=True,
        trust_remote_code=True,
        # download_mode=ds.DownloadMode.FORCE_REDOWNLOAD,
    )
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test

    # dataset.push_to_hub(repo_id=repo_id, private=True)
