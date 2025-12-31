# Copyright 2025 Shunsuke Kitada and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script was generated from shunk031/cookiecutter-huggingface-datasets.
#
# TODO: Address all TODOs and remove all explanatory comments
import json
import os
import pathlib
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import List, Literal, Optional, assert_never

import gdown
from datasets.utils.logging import get_logger
from PIL import Image
from tenacity import retry, wait_exponential
from tqdm.auto import tqdm

import datasets as ds

logger = get_logger(__name__)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
TODO: Add BibTeX citation here
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Please input description
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://webpai.github.io/DesignBench/"

# TODO: Add the license for the dataset here if you can find it
_LICENSE = "Please input license information"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "data.zip": "1x0JfWRiJcUHI05oNzx5Lcv1_g9xoh3ET",
    "EditResults.zip": "11boXiSfhwNXLPZCIu4SQcTpiZ3xLbkxk",
    "GenerationResults.zip": "124a7DfG-TNKGJNglKKkPjS8iSqxOSU7P",
    "RepairResults.zip": "1jyiWjUsF6IQFYWeGYCEqE3iXALbuzQMZ",
}


class Task(StrEnum):
    compile = auto()
    edit = auto()
    generation = auto()
    repair = auto()


class Framework(StrEnum):
    vanilla = auto()
    angular = auto()
    react = auto()
    vue = auto()


@dataclass(kw_only=True)
class DesignBenchConfig(ds.BuilderConfig):
    task: Task
    framework: Framework

    def __post_init__(self):
        if isinstance(self.task, str):
            self.task = Task(self.task)

        if isinstance(self.framework, str):
            self.framework = Framework(self.framework)

        if self.task == Task.compile and self.framework == Framework.vanilla:
            raise ValueError(
                f"Invalid combination of {self.task=} and {self.framework=}"
            )

        self.name = f"{self.task}_{self.framework}"


def get_generation_task_config(version: ds.Version) -> List[DesignBenchConfig]:
    return [
        DesignBenchConfig(
            task=Task.generation,
            framework=Framework.vanilla,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.generation,
            framework=Framework.angular,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.generation,
            framework=Framework.react,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.generation,
            framework=Framework.vue,
            version=version,
        ),
    ]


def get_edit_task_config(version: ds.Version) -> List[DesignBenchConfig]:
    return [
        DesignBenchConfig(
            task=Task.edit,
            framework=Framework.vanilla,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.edit,
            framework=Framework.angular,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.edit,
            framework=Framework.react,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.edit,
            framework=Framework.vue,
            version=version,
        ),
    ]


def get_repair_task_config(version: ds.Version) -> List[DesignBenchConfig]:
    return [
        DesignBenchConfig(
            task=Task.repair,
            framework=Framework.vanilla,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.repair,
            framework=Framework.angular,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.repair,
            framework=Framework.react,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.repair,
            framework=Framework.vue,
            version=version,
        ),
    ]


def get_compile_task_config(version: ds.Version) -> List[DesignBenchConfig]:
    return [
        DesignBenchConfig(
            task=Task.compile,
            framework=Framework.angular,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.compile,
            framework=Framework.react,
            version=version,
        ),
        DesignBenchConfig(
            task=Task.compile,
            framework=Framework.vue,
            version=version,
        ),
    ]


def get_generation_task_features() -> ds.Features:
    return ds.Features(
        {
            "screenshot": ds.Image(),
            "html": ds.Value("string"),
            "json": {
                "html": ds.Value("string"),
                "bbox": ds.Value("string"),
                "difficulty": ds.ClassLabel(
                    names=["easy", "medium", "hard"],
                ),
                "framework": ds.Value("string"),
            },
        }
    )


def get_edit_task_features() -> ds.Features:
    return ds.Features(
        {
            "src_screenshot": ds.Image(),
            "dst_screenshot": ds.Image(),
            "json": {
                "prompt": ds.Value("string"),
                "component_jsx": ds.Value("string"),
                "compile": ds.Value("bool"),
                "clarity": ds.ClassLabel(names=["high"]),
                "difficulty": ds.ClassLabel(
                    names=["easy", "medium", "hard"],
                ),
                "operation": ds.Value("string"),
                "score": ds.ClassLabel(names=["excellent", "good", "fair", "poor"]),
                "action_type": ds.Sequence(
                    ds.ClassLabel(
                        names=["", "add", "change", "Change", "Add", "Delete"]
                    )
                ),
                "visual_type": ds.Sequence(
                    ds.ClassLabel(
                        names=[
                            "",
                            "text",
                            "Text",
                            "Component-level",
                            "Position",
                            "Color",
                            "shape",
                            "Shape",
                            "Size",
                            "color",
                            "size",
                        ]
                    )
                ),
                "block_number": ds.Value("int32"),
                "block_ratio": ds.Value("float32"),
                "src_code": ds.Value("string"),
                "dst_code": ds.Value("string"),
                "src_id": ds.Value("string"),
                "dst_id": ds.Value("string"),
                "framework": ds.ClassLabel(
                    names=["vanilla", "angular", "react", "vue"]
                ),
            },
        }
    )


def get_repair_task_features(config: DesignBenchConfig) -> ds.Features:
    if config.framework is Framework.vanilla:
        return ds.Features(
            {
                "original_screenshot": ds.Image(),
                "repaired_screenshot": ds.Image(),
                "mark_screenshot": ds.Image(),
                "original_html": ds.Value("string"),
                "repaired_html": ds.Value("string"),
                "repaired_json": {
                    "display_issue": ds.Value("bool"),
                    "reasoning": ds.Value("string"),
                    "code": ds.Value("string"),
                },
                "json": {
                    "code": ds.Value("string"),
                    "issue": ds.Sequence(
                        ds.ClassLabel(
                            names=[
                                "occlusion",
                                "crowding",
                                "alignment",
                                "contrast",
                                "color and contrast",
                                "overflow",
                                "disorder",
                            ]
                        )
                    ),
                    "type": ds.ClassLabel(
                        names=[
                            "display",
                        ]
                    ),
                },
            }
        )
    elif config.framework is Framework.angular:
        return ds.Features()
    elif config.framework is Framework.react:
        return ds.Features()
    elif config.framework is Framework.vue:
        return ds.Features()
    else:
        assert_never(config.framework)


def get_compile_task_features() -> ds.Features:
    return ds.Features()


def generate_generation_task_examples(target_dirs: List[pathlib.Path]):
    for i, target_dir in enumerate(target_dirs):
        html_path = target_dir / f"{target_dir.stem}.html"
        json_path = target_dir / f"{target_dir.stem}.json"
        screenshot_path = target_dir / f"{target_dir.stem}.png"

        screenshot = Image.open(screenshot_path)

        with json_path.open("r") as rf:
            json_data = json.load(rf)

        html = html_path.read_text()

        yield (
            i,
            {
                "screenshot": screenshot,
                "json": json_data,
                "html": html,
            },
        )


def generate_edit_task_examples(target_dirs: List[pathlib.Path]):
    for i, target_dir in enumerate(target_dirs):
        json_path = target_dir / f"{target_dir.stem}.json"
        with json_path.open("r") as rf:
            json_data = json.load(rf)

        src_id = json_data["src_id"]
        dst_id = json_data["dst_id"]

        src_html_path = target_dir / f"{src_id}.html"
        dst_html_path = target_dir / f"{dst_id}.html"

        src_screenshot_path = target_dir / f"{src_id}.png"
        dst_screenshot_path = target_dir / f"{dst_id}.png"

        src_screenshot = Image.open(src_screenshot_path)
        dst_screenshot = Image.open(dst_screenshot_path)

        src_html = src_html_path.read_text()
        dst_html = dst_html_path.read_text()

        yield (
            i,
            {
                "src_screenshot": src_screenshot,
                "dst_screenshot": dst_screenshot,
                "src_html": src_html,
                "dst_html": dst_html,
                "json": json_data,
            },
        )


def generate_repair_task_examples(
    target_dirs: List[pathlib.Path], config: DesignBenchConfig
):
    def _generate_vanilla_examples(target_dirs: List[pathlib.Path]):
        for i, target_dir in enumerate(target_dirs):
            json_path = target_dir / f"{target_dir.stem}.json"
            with json_path.open("r") as rf:
                json_data = json.load(rf)

            print(json_data["issue"])
            if isinstance(json_data["issue"], str):
                json_data["issue"] = [json_data["issue"]]

            repaired_json_path = target_dir / "repaired.json"
            with repaired_json_path.open("r") as rf:
                repaired_json_data = json.load(rf)

            original_html_path = target_dir / f"{target_dir.stem}.html"
            repaired_html_path = target_dir / "repaired.html"

            screenshot_path = target_dir / f"{target_dir.stem}.png"
            screenshot_mark_path = target_dir / f"{target_dir.stem}_mark.png"
            screenshot_repaired_path = target_dir / "repaired.png"

            yield (
                i,
                {
                    "original_screenshot": Image.open(screenshot_path),
                    "repaired_screenshot": Image.open(screenshot_repaired_path),
                    "mark_screenshot": Image.open(screenshot_mark_path),
                    "original_html": original_html_path.read_text(),
                    "repaired_html": repaired_html_path.read_text(),
                    "repaired_json": repaired_json_data,
                    "json": json_data,
                },
            )

    def _generate_angular_examples(target_dirs: List[pathlib.Path]):
        pass

    def _generate_react_examples(target_dirs: List[pathlib.Path]):
        pass

    def _generate_vue_examples(target_dirs: List[pathlib.Path]):
        pass

    if config.framework is Framework.vanilla:
        yield from _generate_vanilla_examples(target_dirs)
    elif config.framework is Framework.angular:
        yield from _generate_angular_examples(target_dirs)
    elif config.framework is Framework.react:
        yield from _generate_react_examples(target_dirs)
    elif config.framework is Framework.vue:
        yield from _generate_vue_examples(target_dirs)
    else:
        assert_never(config.framework)


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class DesignBenchDataset(ds.GeneratorBasedBuilder):
    """A class for loading DesignBench dataset."""

    config: DesignBenchConfig

    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = DesignBenchConfig
    BUILDER_CONFIGS = (
        get_generation_task_config(version=VERSION)
        + get_edit_task_config(version=VERSION)
        + get_repair_task_config(version=VERSION)
        + get_compile_task_config(version=VERSION)
    )

    def _info(self) -> ds.DatasetInfo:
        if self.config.task is Task.generation:
            features = get_generation_task_features()
        elif self.config.task is Task.edit:
            features = get_edit_task_features()
        elif self.config.task is Task.repair:
            features = get_repair_task_features(config=self.config)
        elif self.config.task is Task.compile:
            features = get_compile_task_features()
        else:
            assert_never(self.config.task)

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        # self._download_files_from_google_drive(dl_manager)

        data_base_dir = dl_manager.extract(pathlib.Path.cwd() / "data.zip")
        assert isinstance(data_base_dir, str)

        data_dir = pathlib.Path(data_base_dir) / "data"
        task_dir = data_dir / self.config.task
        target_dir = task_dir / self.config.framework

        return [
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={"target_dir": target_dir},
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, target_dir: pathlib.Path):
        target_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir()])

        if self.config.task is Task.generation:
            yield from generate_generation_task_examples(target_dirs)

        elif self.config.task is Task.edit:
            yield from generate_edit_task_examples(target_dirs)

        elif self.config.task is Task.repair:
            yield from generate_repair_task_examples(target_dirs, config=self.config)

        elif self.config.task is Task.compile:
            for i, target_dir in enumerate(target_dirs):
                raise NotImplementedError

        else:
            assert_never(self.config.task)
