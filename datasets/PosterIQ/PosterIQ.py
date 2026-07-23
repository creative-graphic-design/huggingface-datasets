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
@inproceedings{cvpr2026posteriq,
  title={PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation},
  author={Feng, Yuheng and Zhang, Wen and Duan, Haodong and Zou, Xingxing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
"""

_DESCRIPTION = """\
PosterIQ is a poster design evaluation benchmark with task-level data for poster understanding \
and poster generation. It includes image-grounded understanding tasks and prompt-based \
generation tasks covering typography, layout, composition, OCR, style, and design intention.
"""

_HOMEPAGE = "https://github.com/ArtmeScienceLab/PosterIQ-Benchmark"

_LICENSE = "other"

_VERSION = ds.Version("1.0.0")

_BASE_URL = "https://huggingface.co/datasets/ArtmeScienceLab/PosterIQ"
_RAW_BASE_URL = f"{_BASE_URL}/raw/main"
_DATA_URL = f"{_BASE_URL}/resolve/main/data.zip"


class PosterIQType(StrEnum):
    alignment = auto()
    composition_understanding = auto()
    empty_space = auto()
    font_attributes = auto()
    font_effect = auto()
    font_effect_2 = auto()
    font_matching = auto()
    font_size_ocr = auto()
    hard_ocr = auto()
    intention_understanding = auto()
    layout_comparison = auto()
    layout_generation = auto()
    logo_ocr = auto()
    overall_rating = auto()
    poster_ocr = auto()
    rotation = auto()
    simple_ocr = auto()
    style_understanding = auto()
    text_localization = auto()
    gen_composition = auto()
    gen_dense = auto()
    gen_font = auto()
    gen_intention = auto()
    gen_style = auto()


@dataclass(frozen=True)
class PosterIQTask:
    json_path: str
    count: int
    has_image: bool
    has_gt: bool
    has_subtask: bool = True
    has_original_image: bool = False


_TASKS: dict[PosterIQType, PosterIQTask] = {
    PosterIQType.alignment: PosterIQTask("und_task/alignment.json", 200, True, True),
    PosterIQType.composition_understanding: PosterIQTask(
        "und_task/composition_understanding.json", 117, True, True
    ),
    PosterIQType.empty_space: PosterIQTask(
        "und_task/empty_space.json", 167, True, True
    ),
    PosterIQType.font_attributes: PosterIQTask(
        "und_task/font_attributes.json", 1813, True, True
    ),
    PosterIQType.font_effect: PosterIQTask(
        "und_task/font_effect.json", 450, True, True
    ),
    PosterIQType.font_effect_2: PosterIQTask(
        "und_task/font_effect_2.json", 125, True, True
    ),
    PosterIQType.font_matching: PosterIQTask(
        "und_task/font_matching.json", 400, True, True
    ),
    PosterIQType.font_size_ocr: PosterIQTask(
        "und_task/font_size_ocr.json", 1400, True, True
    ),
    PosterIQType.hard_ocr: PosterIQTask("und_task/hard_ocr.json", 400, True, True),
    PosterIQType.intention_understanding: PosterIQTask(
        "und_task/intention_understanding.json", 202, True, True
    ),
    PosterIQType.layout_comparison: PosterIQTask(
        "und_task/layout_comprison.json", 256, True, True
    ),
    PosterIQType.layout_generation: PosterIQTask(
        "und_task/layout_generation.json", 145, True, True
    ),
    PosterIQType.logo_ocr: PosterIQTask("und_task/logo_ocr.json", 600, True, True),
    PosterIQType.overall_rating: PosterIQTask(
        "und_task/overall_rating.json", 219, True, True
    ),
    PosterIQType.poster_ocr: PosterIQTask(
        "und_task/poster_ocr.json", 205, True, False, True, True
    ),
    PosterIQType.rotation: PosterIQTask("und_task/rotation.json", 205, True, True),
    PosterIQType.simple_ocr: PosterIQTask("und_task/simple_ocr.json", 400, True, True),
    PosterIQType.style_understanding: PosterIQTask(
        "und_task/style_understanding.json", 256, True, True
    ),
    PosterIQType.text_localization: PosterIQTask(
        "und_task/text_localization.json", 205, True, False, True, True
    ),
    PosterIQType.gen_composition: PosterIQTask(
        "gen_task/gen_composition.json", 117, False, True, False
    ),
    PosterIQType.gen_dense: PosterIQTask("gen_task/gen_dense.json", 114, False, True),
    PosterIQType.gen_font: PosterIQTask("gen_task/gen_font.json", 135, False, False),
    PosterIQType.gen_intention: PosterIQTask(
        "gen_task/gen_intention.json", 200, False, True, False
    ),
    PosterIQType.gen_style: PosterIQTask("gen_task/gen_style.json", 256, False, False),
}


@dataclass
class PosterIQConfig(ds.BuilderConfig):
    name: PosterIQType

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PosterIQType(self.name)


_COMMON_KEYS = {
    "task",
    "subtask",
    "name",
    "path",
    "path_original",
    "prompt",
    "gt",
}


def _as_path(value: str | pathlib.Path) -> pathlib.Path:
    return value if isinstance(value, pathlib.Path) else pathlib.Path(value)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_url(task: PosterIQTask) -> str:
    return f"{_RAW_BASE_URL}/{task.json_path}"


def _find_data_root(extracted_dir: str | pathlib.Path) -> pathlib.Path:
    root = _as_path(extracted_dir)
    if root.name == "data" and root.is_dir():
        return root

    candidate = root / "data"
    if candidate.is_dir():
        return candidate

    for path in root.rglob("data"):
        if path.is_dir():
            return path

    raise FileNotFoundError(f"Could not find data/ under extracted archive: {root}")


def _resolve_data_path(
    data_root: str | pathlib.Path, relative_path: str
) -> pathlib.Path:
    path = pathlib.PurePosixPath(_normalize_path(relative_path))
    return _as_path(data_root).joinpath(*path.parts)


def _features_for_task(task: PosterIQTask) -> ds.Features:
    features: dict[str, Any] = {
        "id": ds.Value("string"),
        "task": ds.Value("string"),
        "name": ds.Value("string"),
        "path": ds.Value("string"),
        "prompt": ds.Value("string"),
        "metadata_json": ds.Value("string"),
    }
    if task.has_subtask:
        features["subtask"] = ds.Value("string")
    if task.has_gt:
        features["gt_json"] = ds.Value("string")
    if task.has_image:
        features["image"] = ds.Image()
        features["image_path"] = ds.Value("string")
    if task.has_original_image:
        features["original_image"] = ds.Image()
        features["original_image_path"] = ds.Value("string")

    return ds.Features(features)


def _iter_examples(
    rows: list[dict[str, Any]],
    config_name: PosterIQType,
    task: PosterIQTask,
    data_root: str | pathlib.Path | None = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    for index, row in enumerate(rows):
        normalized_path = _normalize_path(row["path"])
        metadata = {key: value for key, value in row.items() if key not in _COMMON_KEYS}
        example_id = f"{config_name}-{index:05d}"
        example = {
            "id": example_id,
            "task": row["task"],
            "name": row["name"],
            "path": normalized_path,
            "prompt": row["prompt"],
            "metadata_json": _json_dumps(metadata),
        }

        if task.has_subtask:
            example["subtask"] = row["subtask"]

        if task.has_gt:
            example["gt_json"] = _json_dumps(row["gt"])

        if task.has_image:
            assert data_root is not None
            image_path = _resolve_data_path(data_root, row["path"])
            example["image"] = str(image_path)
            example["image_path"] = str(image_path)

        if task.has_original_image:
            assert data_root is not None
            original_image_path = _resolve_data_path(data_root, row["path_original"])
            example["original_image"] = str(original_image_path)
            example["original_image_path"] = str(original_image_path)

        yield example_id, example


class PosterIQ(ds.GeneratorBasedBuilder):
    """A class for loading the PosterIQ dataset."""

    config: PosterIQConfig

    VERSION = _VERSION

    BUILDER_CONFIG_CLASS = PosterIQConfig
    BUILDER_CONFIGS = [
        PosterIQConfig(
            name=config_name,
            version=_VERSION,
            description=f"PosterIQ {config_name.value.replace('_', ' ')} task.",
        )
        for config_name in PosterIQType
    ]
    DEFAULT_CONFIG_NAME = "alignment"

    def _task(self) -> PosterIQTask:
        match self.config.name:
            case PosterIQType.alignment:
                return _TASKS[PosterIQType.alignment]
            case PosterIQType.composition_understanding:
                return _TASKS[PosterIQType.composition_understanding]
            case PosterIQType.empty_space:
                return _TASKS[PosterIQType.empty_space]
            case PosterIQType.font_attributes:
                return _TASKS[PosterIQType.font_attributes]
            case PosterIQType.font_effect:
                return _TASKS[PosterIQType.font_effect]
            case PosterIQType.font_effect_2:
                return _TASKS[PosterIQType.font_effect_2]
            case PosterIQType.font_matching:
                return _TASKS[PosterIQType.font_matching]
            case PosterIQType.font_size_ocr:
                return _TASKS[PosterIQType.font_size_ocr]
            case PosterIQType.hard_ocr:
                return _TASKS[PosterIQType.hard_ocr]
            case PosterIQType.intention_understanding:
                return _TASKS[PosterIQType.intention_understanding]
            case PosterIQType.layout_comparison:
                return _TASKS[PosterIQType.layout_comparison]
            case PosterIQType.layout_generation:
                return _TASKS[PosterIQType.layout_generation]
            case PosterIQType.logo_ocr:
                return _TASKS[PosterIQType.logo_ocr]
            case PosterIQType.overall_rating:
                return _TASKS[PosterIQType.overall_rating]
            case PosterIQType.poster_ocr:
                return _TASKS[PosterIQType.poster_ocr]
            case PosterIQType.rotation:
                return _TASKS[PosterIQType.rotation]
            case PosterIQType.simple_ocr:
                return _TASKS[PosterIQType.simple_ocr]
            case PosterIQType.style_understanding:
                return _TASKS[PosterIQType.style_understanding]
            case PosterIQType.text_localization:
                return _TASKS[PosterIQType.text_localization]
            case PosterIQType.gen_composition:
                return _TASKS[PosterIQType.gen_composition]
            case PosterIQType.gen_dense:
                return _TASKS[PosterIQType.gen_dense]
            case PosterIQType.gen_font:
                return _TASKS[PosterIQType.gen_font]
            case PosterIQType.gen_intention:
                return _TASKS[PosterIQType.gen_intention]
            case PosterIQType.gen_style:
                return _TASKS[PosterIQType.gen_style]
            case _:
                assert_never(self.config.name)

    def _info(self) -> ds.DatasetInfo:
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=_features_for_task(self._task()),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        task = self._task()
        metadata_path = dl_manager.download(_json_url(task))
        assert isinstance(metadata_path, str)

        data_root = None
        if task.has_image:
            extracted_dir = dl_manager.download_and_extract(_DATA_URL)
            assert isinstance(extracted_dir, str)
            data_root = str(_find_data_root(extracted_dir))

        return [
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "config_name": self.config.name,
                    "task": task,
                    "data_root": data_root,
                },
            ),
        ]

    def _generate_examples(
        self,
        metadata_path: str,
        config_name: PosterIQType,
        task: PosterIQTask,
        data_root: str | None,
    ):
        with open(metadata_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        yield from _iter_examples(rows, config_name, task, data_root)
