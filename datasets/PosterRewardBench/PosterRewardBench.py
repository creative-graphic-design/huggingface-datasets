# Copyright 2026 Shunsuke Kitada and the current dataset script contributor.
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
import json
import pathlib
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Iterable, List, assert_never

from datasets.utils.logging import get_logger

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@misc{lai2026posterreward,
  title={PosterReward: Unlocking Accurate Evaluation for High-Quality Graphic Design Generation},
  author={Lai, Jianyu and Chen, Sixiang and Gao, Jialin and Shi, Hengyu and Liu, Zhongying and Zhai, Fuxiang and Luo, Junfeng and Wei, Xiaoming and Wang, Lujia and Zhu, Lei},
  year={2026},
  eprint={2603.29855},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.29855}
}
"""

_DESCRIPTION = """\
PosterRewardBench is the poster preference benchmark introduced with PosterReward. It contains \
Basic and Advanced subsets of generated poster image pairs with a prompt, a preferred image, \
and a rejected image for reward-model evaluation.
"""

_HOMEPAGE = "https://alexlai2860.github.io/PosterReward/"

_LICENSE = "unknown"

_URLS = {
    "basic": {
        "metadata": "https://raw.githubusercontent.com/MeiGen-AI/PosterReward/main/poster_reward_bench/PRB_basic_relative.json",
        "images": "https://huggingface.co/MeiGen-AI/PosterReward_v1/resolve/main/PRB_basic_images.tar.gz",
    },
    "advanced": {
        "metadata": "https://raw.githubusercontent.com/MeiGen-AI/PosterReward/main/poster_reward_bench/PRB_advanced_relative.json",
        "images": "https://huggingface.co/MeiGen-AI/PosterReward_v1/resolve/main/PRB_advanced_images.tar.gz",
    },
}


class PosterRewardBenchType(StrEnum):
    basic = auto()
    advanced = auto()


@dataclass
class PosterRewardBenchConfig(ds.BuilderConfig):
    name: PosterRewardBenchType

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PosterRewardBenchType(self.name)


def _as_path(value: str | pathlib.Path) -> pathlib.Path:
    return value if isinstance(value, pathlib.Path) else pathlib.Path(value)


def _message_features() -> list[dict[str, ds.Value]]:
    return [
        {
            "role": ds.Value("string"),
            "content": ds.Value("string"),
        }
    ]


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        }
        for message in messages
    ]


def _find_image_root(
    extracted_dir: str | pathlib.Path,
    expected_dirname: str,
) -> pathlib.Path:
    root = _as_path(extracted_dir)
    if root.name == expected_dirname and root.is_dir():
        return root

    candidate = root / expected_dirname
    if candidate.is_dir():
        return candidate

    for path in root.rglob(expected_dirname):
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Could not find {expected_dirname}/ under extracted image archive: {root}"
    )


def _resolve_image_path(image_root: pathlib.Path, relative_path: str) -> pathlib.Path:
    path = pathlib.PurePosixPath(relative_path)
    if path.parts and path.parts[0] == image_root.name:
        path = pathlib.PurePosixPath(*path.parts[1:])

    return image_root.joinpath(*path.parts)


def _iter_examples(
    metadata_path: str | pathlib.Path,
    image_root: str | pathlib.Path,
    config_name: str,
) -> Iterable[tuple[str, dict[str, Any]]]:
    image_root = _as_path(image_root)
    with _as_path(metadata_path).open("r", encoding="utf-8") as f:
        rows = json.load(f)

    for index, row in enumerate(rows):
        messages = _normalize_messages(row["messages"])
        rejected_messages = _normalize_messages(row["rejected_messages"])
        chosen_image_path = _resolve_image_path(image_root, row["images"][0])
        rejected_image_path = _resolve_image_path(image_root, row["rejected_images"][0])
        example_id = f"{config_name}-{index:05d}"

        yield (
            example_id,
            {
                "id": example_id,
                "prompt": messages[0]["content"],
                "chosen_image": str(chosen_image_path),
                "rejected_image": str(rejected_image_path),
                "chosen_image_path": str(chosen_image_path),
                "rejected_image_path": str(rejected_image_path),
                "messages": messages,
                "rejected_messages": rejected_messages,
            },
        )


class PosterRewardBench(ds.GeneratorBasedBuilder):
    """A class for loading the PosterRewardBench dataset."""

    config: PosterRewardBenchConfig

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = PosterRewardBenchConfig
    BUILDER_CONFIGS = [
        PosterRewardBenchConfig(
            name=PosterRewardBenchType.basic,
            version=VERSION,
            description=(
                "PosterRewardBench-Basic preference pairs generated by Flux, "
                "Flux-Krea, and SD3.5-L."
            ),
        ),
        PosterRewardBenchConfig(
            name=PosterRewardBenchType.advanced,
            version=VERSION,
            description=(
                "PosterRewardBench-Advanced preference pairs generated by "
                "Seedream-3.0, Seedream-4.0, and Qwen-Image-Lightning."
            ),
        ),
    ]
    DEFAULT_CONFIG_NAME = "basic"

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "id": ds.Value("string"),
                "prompt": ds.Value("string"),
                "chosen_image": ds.Image(),
                "rejected_image": ds.Image(),
                "chosen_image_path": ds.Value("string"),
                "rejected_image_path": ds.Value("string"),
                "messages": _message_features(),
                "rejected_messages": _message_features(),
            }
        )
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
        match self.config.name:
            case PosterRewardBenchType.basic:
                urls = _URLS["basic"]
                image_dirname = "PRB_basic_images"
            case PosterRewardBenchType.advanced:
                urls = _URLS["advanced"]
                image_dirname = "PRB_advanced_images"
            case _:
                assert_never(self.config.name)

        metadata_path = dl_manager.download(urls["metadata"])
        extracted_dir = dl_manager.download_and_extract(urls["images"])
        assert isinstance(metadata_path, str)
        assert isinstance(extracted_dir, str)

        image_root = _find_image_root(extracted_dir, image_dirname)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "image_root": str(image_root),
                    "config_name": str(self.config.name),
                },
            ),
        ]

    def _generate_examples(
        self,
        metadata_path: str,
        image_root: str,
        config_name: str,
    ):
        yield from _iter_examples(metadata_path, image_root, config_name)
