import os
from enum import StrEnum

import pytest

import datasets as ds


@pytest.fixture
def script_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "DesignBench"


@pytest.fixture
def dataset_path(script_dir: str, dataset_name: str) -> str:
    return os.path.join(script_dir, f"{dataset_name}.py")


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames="framework_name",
    argvalues=(
        "vanilla",
        "angular",
        "react",
        "vue",
    ),
)
@pytest.mark.parametrize(
    argnames="task_name",
    argvalues=(
        # "generation",
        # "edit",
        "repair",
        "compile",
    ),
)
def test_load_dataset(
    dataset_path: str,
    task_name: str,
    framework_name: str,
    repo_id: str,
):
    if task_name == "compile" and framework_name == "vanilla":
        pytest.skip(f"Invalid combination of {task_name=} and {framework_name=}.")

    dataset = ds.load_dataset(
        path=dataset_path,
        task=task_name,
        framework=framework_name,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)

    dataset.push_to_hub(
        repo_id=repo_id,
        config_name=f"{task_name}={framework_name}",
        # private=True,
    )


def test_load_invalid_combination(dataset_path: str):
    with pytest.raises(ValueError):
        ds.load_dataset(
            path=dataset_path,
            task="compile",
            framework="vanilla",
            trust_remote_code=True,
        )
