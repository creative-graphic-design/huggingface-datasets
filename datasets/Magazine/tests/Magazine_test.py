import os

import pytest

import datasets as ds


@pytest.fixture
def dataset_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Magazine.py"
    )


def test_load_dataset_builder(dataset_path: str):
    builder = ds.load_dataset_builder(path=dataset_path, trust_remote_code=True)
    assert {"filename", "category", "elements", "images"} <= set(builder.info.features)


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames=("expected_num_dataset",),
    argvalues=((3919,),),
)
def test_load_dataset(dataset_path: str, expected_num_dataset: int):
    dataset = ds.load_dataset(path=dataset_path, token=True, trust_remote_code=True)

    assert dataset["train"].num_rows == expected_num_dataset
