import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PubLayNet.py"
    )


@pytest.mark.parametrize("decode_rle", (False, True))
def test_load_dataset_builder(dataset_path: str, decode_rle: bool):
    builder = ds.load_dataset_builder(
        path=dataset_path,
        decode_rle=decode_rle,
        trust_remote_code=True,
    )
    assert {"image", "annotations"} <= set(builder.info.features)


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames=("decode_rle"),
    argvalues=(False, True),
)
@pytest.mark.parametrize(
    argnames=("expected_num_train", "expected_num_valid", "expected_num_test"),
    argvalues=((335703, 11245, 11405),),
)
def test_load_dataset(
    dataset_path: str,
    decode_rle: bool,
    expected_num_train: int,
    expected_num_valid: int,
    expected_num_test: int,
):
    dataset = ds.load_dataset(
        path=dataset_path, decode_rle=decode_rle, trust_remote_code=True
    )
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["test"].num_rows == expected_num_test
