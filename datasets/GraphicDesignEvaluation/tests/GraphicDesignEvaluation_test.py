import os

import pytest

import datasets as ds


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def dataset_name() -> str:
    return "GraphicDesignEvaluation"


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
    argnames="design_principle",
    argvalues=("alignment", "overlap", "whitespace"),
)
@pytest.mark.parametrize(
    argnames="annotation_type",
    argvalues=("gpt", "human"),
)
@pytest.mark.parametrize(
    argnames=("eval_type", "expected_num_train"),
    argvalues=(
        ("absolute", 400),
        ("relative", 300),
    ),
)
def test_load_dataset(
    dataset_path: str,
    repo_id: str,
    eval_type: str,
    annotation_type: str,
    design_principle: str,
    expected_num_train: int,
    trust_remote_code: bool = True,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        eval_type=eval_type,
        annotation_type=annotation_type,
        design_principle=design_principle,
        trust_remote_code=trust_remote_code,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train

    if eval_type == "absolute":
        dataset = dataset.sort(
            column_names=["image", "perturbation"],
        )
    elif eval_type == "relative":
        dataset = dataset.sort(
            column_names=["image", "comparative"],
        )
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    dataset.push_to_hub(
        repo_id=repo_id,
        config_name=f"{eval_type}-{annotation_type}-{design_principle}",
        private=True,
    )
