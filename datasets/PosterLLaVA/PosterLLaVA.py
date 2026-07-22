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
import os
import pathlib
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Iterable, List, assert_never

import gdown
from datasets.utils.logging import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@misc{yang2024posterllava,
  title={PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM},
  author={Yang, Tao and Luo, Yingmin and Qi, Zhongang and Wu, Yang and Shan, Ying and Chen, Chang Wen},
  year={2024},
  eprint={2406.02884},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2406.02884},
  note={Accepted to IEEE Transactions on Multimedia}
}
"""

_DESCRIPTION = """\
PosterLLaVA contains poster layout data released with PosterLLaVA, including \
QB-Poster social-media poster layouts and user-constraint annotations for \
CGL-dataset and PosterLayout poster layout generation.
"""

_HOMEPAGE = "https://github.com/posterllava/PosterLLaVA"
_LICENSE = "cc-by-nc-4.0"

_GOOGLE_DRIVE_IDS = {
    "qb_poster": "1gRHTidpU0nePpjtDQElIVbAts8ziCkVh",
    "user_constrained": "1dlfxTC6QaV3Piyn655TMvTEv7-tCWuWk",
}

_PROMPT_TEMPLATE = """\
<image>
Hello! Could you please help me to place {N} foreground elements over the background image of resolution {resolution} to craft an aesthetically pleasing, harmonious, balanced, and visually appealing {domain_name}?
Finding semantic-meaningful objects or visual foci on the background image at first might help in designing, and you should avoid any unnecessary blocking of them.
Please return the result by completing the following JSON file. Each element's location and size should be represented by a bounding box described as [left, top, right, bottom], and each number is a continuous digit from 0 to 1.
Here is the initial JSON file: {json_data}"""

_QB_POSTER_DOMAIN = "social media promotion poster with qbposter style"


class PosterLLaVAType(StrEnum):
    qb_poster = auto()
    user_constrained = auto()


@dataclass
class PosterLLaVAConfig(ds.BuilderConfig):
    name: PosterLLaVAType

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PosterLLaVAType(self.name)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def download_google_drive_file(file_id: str, output_path: pathlib.Path) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(output_path), quiet=False, resume=True)
    if not output_path.exists():
        raise FileNotFoundError(f"Failed to download Google Drive file: {file_id}")
    return output_path


def _as_path(value: str | pathlib.Path) -> pathlib.Path:
    return value if isinstance(value, pathlib.Path) else pathlib.Path(value)


def _read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _find_qb_poster_root(path: str | pathlib.Path) -> pathlib.Path:
    root = _as_path(path)
    required_dir = "original_poster"
    annotation_names = ("annotations.json", "annotation.json")

    def is_qb_root(candidate: pathlib.Path) -> bool:
        return (candidate / required_dir).is_dir() and any(
            (candidate / name).is_file() for name in annotation_names
        )

    if is_qb_root(root):
        return root

    for relative in (
        pathlib.Path("qbposter/raw"),
        pathlib.Path("data/qbposter/raw"),
        pathlib.Path("raw"),
    ):
        candidate = root / relative
        if is_qb_root(candidate):
            return candidate

    for candidate in root.rglob(required_dir):
        parent = candidate.parent
        if is_qb_root(parent):
            return parent

    raise FileNotFoundError(
        "Could not find QB-Poster raw data. Pass an archive or extracted "
        "directory containing original_poster/ and annotations.json or "
        "annotation.json via data_dir."
    )


def _find_user_constrained_root(path: str | pathlib.Path) -> pathlib.Path:
    root = _as_path(path)
    expected_files = ("cgl_train.json", "cgl_val.json", "posterlayout_train.json")

    def is_uc_root(candidate: pathlib.Path) -> bool:
        return all((candidate / name).is_file() for name in expected_files)

    if is_uc_root(root):
        return root

    candidate = root / "ucposter"
    if is_uc_root(candidate):
        return candidate

    for candidate in root.rglob("ucposter"):
        if candidate.is_dir() and is_uc_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find User-Constrained annotations. Pass an archive or "
        "extracted directory containing ucposter/cgl_train.json, "
        "ucposter/cgl_val.json, and ucposter/posterlayout_train.json via data_dir."
    )


def _annotation_path(qb_root: pathlib.Path) -> pathlib.Path:
    for name in ("annotations.json", "annotation.json"):
        path = qb_root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find QB-Poster annotations under {qb_root}")


def _image_path(qb_root: pathlib.Path, poster_id: str, split: str) -> pathlib.Path:
    image_name = f"{poster_id}.png"
    preferred_dirs = (
        ("inpainted_1d5x", "train"),
        ("inpainted_1x", "validation"),
        ("inpainted_1x", "val"),
        ("original_poster", ""),
    )
    for dirname, split_name in preferred_dirs:
        if split_name and split != split_name:
            continue
        path = qb_root / dirname / image_name
        if path.is_file():
            return path
    return qb_root / "original_poster" / image_name


def _normalize_split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "valid", "validation"}:
        return "validation"
    return split


def _normalize_element(
    element: dict[str, Any],
    canvas_width: int,
    canvas_height: int,
    digits: int = 4,
) -> dict[str, Any]:
    xc = float(element["xc"])
    yc = float(element["yc"])
    width = float(element["width"])
    height = float(element["height"])
    half_width = width // 2
    half_height = height // 2
    left = xc - half_width
    top = yc - half_height
    right = xc + half_width
    bottom = yc + half_height

    return {
        "label": str(element["label"]),
        "x_center": xc,
        "y_center": yc,
        "width": width,
        "height": height,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "box": [
            round(left / canvas_width, digits),
            round(top / canvas_height, digits),
            round(right / canvas_width, digits),
            round(bottom / canvas_height, digits),
        ],
    }


def _qb_prompt(elements: list[dict[str, Any]], width: int, height: int) -> str:
    content_for_prompt = [
        {"label": element["label"], "box": []} for element in elements
    ]
    return _PROMPT_TEMPLATE.format(
        N=len(elements),
        resolution=[width, height],
        domain_name=_QB_POSTER_DOMAIN,
        json_data=_dump_json(content_for_prompt),
    )


def _qb_conversations(
    prompt: str,
    elements: list[dict[str, Any]],
) -> list[dict[str, str]]:
    answer_elements = [
        {"label": element["label"], "box": element["box"]} for element in elements
    ]
    return [
        {"from": "human", "value": prompt},
        {
            "from": "gpt",
            "value": f"Sure! Here is the design results: {_dump_json(answer_elements)}",
        },
    ]


def _iter_qb_poster_examples(
    qb_root: str | pathlib.Path,
    split_name: str,
) -> Iterable[tuple[str, dict[str, Any]]]:
    qb_root = _as_path(qb_root)
    annotations = _read_json(_annotation_path(qb_root))

    for poster_id, poster in sorted(annotations.items()):
        split = _normalize_split_name(str(poster["split"]))
        if split != split_name:
            continue

        width = int(poster["width"])
        height = int(poster["height"])
        elements = [
            _normalize_element(element, width, height)
            for element in poster.get("boxes", [])
        ]
        prompt = _qb_prompt(elements, width, height)
        image_path = _image_path(qb_root, poster_id, split)

        yield (
            poster_id,
            {
                "id": poster_id,
                "image": str(image_path),
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "split": split,
                "elements": elements,
                "prompt": prompt,
                "conversations": _qb_conversations(prompt, elements),
                "raw_annotation": _dump_json(poster),
            },
        )


def _iter_user_constrained_examples(
    uc_root: str | pathlib.Path,
    split_name: str,
) -> Iterable[tuple[str, dict[str, Any]]]:
    uc_root = _as_path(uc_root)
    files = {
        "train": (
            ("cgl", uc_root / "cgl_train.json"),
            ("posterlayout", uc_root / "posterlayout_train.json"),
        ),
        "validation": (("cgl", uc_root / "cgl_val.json"),),
    }

    for source_dataset, path in files[split_name]:
        annotations = _read_json(path)
        for source_id, row in sorted(annotations.items()):
            constraints = [str(item) for item in row.get("user constraints", [])]
            example_id = f"{source_dataset}-{source_id}"
            yield (
                example_id,
                {
                    "id": example_id,
                    "source_dataset": source_dataset,
                    "source_id": str(source_id),
                    "split": split_name,
                    "user_constraints": constraints,
                    "num_constraints": len(constraints),
                    "raw_annotation": _dump_json(row),
                },
            )


class PosterLLaVA(ds.GeneratorBasedBuilder):
    """A class for loading PosterLLaVA dataset releases."""

    config: PosterLLaVAConfig

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = PosterLLaVAConfig
    BUILDER_CONFIGS = [
        PosterLLaVAConfig(
            name=PosterLLaVAType.qb_poster,
            version=VERSION,
            description="QB-Poster poster layout data with images and boxes.",
        ),
        PosterLLaVAConfig(
            name=PosterLLaVAType.user_constrained,
            version=VERSION,
            description=(
                "User-constraint annotations for CGL-dataset and PosterLayout "
                "poster layout generation."
            ),
        ),
    ]
    DEFAULT_CONFIG_NAME = "qb_poster"

    def _info(self) -> ds.DatasetInfo:
        match self.config.name:
            case PosterLLaVAType.qb_poster:
                features = ds.Features(
                    {
                        "id": ds.Value("string"),
                        "image": ds.Image(),
                        "image_path": ds.Value("string"),
                        "width": ds.Value("int32"),
                        "height": ds.Value("int32"),
                        "split": ds.Value("string"),
                        "elements": [
                            {
                                "label": ds.Value("string"),
                                "x_center": ds.Value("float32"),
                                "y_center": ds.Value("float32"),
                                "width": ds.Value("float32"),
                                "height": ds.Value("float32"),
                                "left": ds.Value("float32"),
                                "top": ds.Value("float32"),
                                "right": ds.Value("float32"),
                                "bottom": ds.Value("float32"),
                                "box": [ds.Value("float32")],
                            }
                        ],
                        "prompt": ds.Value("string"),
                        "conversations": [
                            {
                                "from": ds.Value("string"),
                                "value": ds.Value("string"),
                            }
                        ],
                        "raw_annotation": ds.Value("string"),
                    }
                )
            case PosterLLaVAType.user_constrained:
                features = ds.Features(
                    {
                        "id": ds.Value("string"),
                        "source_dataset": ds.Value("string"),
                        "source_id": ds.Value("string"),
                        "split": ds.Value("string"),
                        "user_constraints": [ds.Value("string")],
                        "num_constraints": ds.Value("int32"),
                        "raw_annotation": ds.Value("string"),
                    }
                )
            case _:
                assert_never(self.config.name)

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
        data_root = self._get_data_root(dl_manager)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"data_root": data_root, "split_name": "train"},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"data_root": data_root, "split_name": "validation"},
            ),
        ]

    def _get_data_root(self, dl_manager: ds.DownloadManager) -> pathlib.Path:
        local_data_dir = getattr(self.config, "data_dir", None) or dl_manager.manual_dir
        if local_data_dir:
            local_path = pathlib.Path(os.path.expanduser(local_data_dir))
            if local_path.is_file():
                extracted_path = dl_manager.extract(str(local_path))
                assert isinstance(extracted_path, str)
                return self._find_config_root(extracted_path)
            return self._find_config_root(local_path)

        cache_dir = pathlib.Path(
            dl_manager.download_config.cache_dir or ds.config.DOWNLOADED_DATASETS_PATH
        )
        config_name = str(self.config.name)
        archive_path = cache_dir / "posterllava" / f"{config_name}"
        if not archive_path.exists():
            logger.info("Downloading PosterLLaVA %s data from Google Drive.", config_name)
            download_google_drive_file(_GOOGLE_DRIVE_IDS[config_name], archive_path)

        extracted_path = dl_manager.extract(str(archive_path))
        assert isinstance(extracted_path, str)
        return self._find_config_root(extracted_path)

    def _find_config_root(self, path: str | pathlib.Path) -> pathlib.Path:
        match self.config.name:
            case PosterLLaVAType.qb_poster:
                return _find_qb_poster_root(path)
            case PosterLLaVAType.user_constrained:
                return _find_user_constrained_root(path)
            case _:
                assert_never(self.config.name)

    def _generate_examples(self, data_root: pathlib.Path, split_name: str):
        match self.config.name:
            case PosterLLaVAType.qb_poster:
                yield from _iter_qb_poster_examples(data_root, split_name)
            case PosterLLaVAType.user_constrained:
                yield from _iter_user_constrained_examples(data_root, split_name)
            case _:
                assert_never(self.config.name)
