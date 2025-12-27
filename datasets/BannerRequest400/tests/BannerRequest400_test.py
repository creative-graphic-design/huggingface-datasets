import os

import pytest

import datasets as ds


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def dataset_name() -> str:
    return "BannerRequest400"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.mark.parametrize(
    argnames=("config_name", "expected_num_train"),
    argvalues=(
        ("abstract-400", 400),
        ("concrete-5k", 100),
    ),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    config_name: str,
    expected_num_train: int,
    trust_remote_code: bool = True,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=config_name,
        trust_remote_code=trust_remote_code,
    )
    assert isinstance(dataset, ds.DatasetDict)
    breakpoint()

    assert dataset["train"].num_rows == expected_num_train

    # dataset.push_to_hub(repo_id=repo_id, private=True)
