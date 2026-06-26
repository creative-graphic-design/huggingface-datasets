import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Camera.py"
    )


@pytest.mark.parametrize(
    argnames=("config_name", "expected_features"),
    argvalues=(
        ("without-lp-images", {"asset_id", "kw", "parsed_full_text_annotation"}),
        (
            "with-lp-images",
            {"asset_id", "kw", "parsed_full_text_annotation", "lp_image"},
        ),
    ),
)
def test_load_dataset_builder(
    dataset_path: str, config_name: str, expected_features: set[str]
):
    builder = ds.load_dataset_builder(
        path=dataset_path,
        name=config_name,
        trust_remote_code=True,
    )
    assert expected_features <= set(builder.info.features)


@pytest.mark.skipif(
    bool(os.environ.get("CI", False)),
    reason="Because this test downloads a large data set, we will skip running it on CI.",
)
def test_load_dataset_without_lp_images(
    dataset_path: str,
    expected_train_num_rows: int = 12395,
    expected_val_num_rows: int = 3098,
    expected_test_num_rows: int = 872,
):
    dataset = ds.load_dataset(
        path=dataset_path, name="without-lp-images", trust_remote_code=True
    )

    assert dataset["train"].num_rows == expected_train_num_rows  # type: ignore
    assert dataset["validation"].num_rows == expected_val_num_rows  # type: ignore
    assert dataset["test"].num_rows == expected_test_num_rows  # type: ignore


@pytest.mark.skipif(
    bool(os.environ.get("CI", False)),
    reason="Because this test downloads a large data set, we will skip running it on CI.",
)
def test_load_dataset_with_lp_images(
    dataset_path: str,
    expected_train_num_rows: int = 12395,
    expected_val_num_rows: int = 3098,
    expected_test_num_rows: int = 872,
):
    dataset = ds.load_dataset(
        path=dataset_path, name="with-lp-images", trust_remote_code=True
    )

    assert dataset["train"].num_rows == expected_train_num_rows  # type: ignore
    assert dataset["validation"].num_rows == expected_val_num_rows  # type: ignore
    assert dataset["test"].num_rows == expected_test_num_rows  # type: ignore

    assert "lp_image" in dataset["train"].column_names
