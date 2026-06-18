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
from typing import Any, List

import gdown
from datasets.utils.logging import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@misc{an2026canvisionlanguagemodelsassess,
  title={Can Vision Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective},
  author={An, Arctanx and Sun, Shizhao and Huang, Danqing and Cheng, Mingxi and Gao, Yan and Li, Ji and Qiao, Yu and Bian, Jiang},
  year={2026},
  eprint={2603.01083},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.01083}
}
"""

_DESCRIPTION = """\
AesEval-Bench is a benchmark for evaluating whether vision-language models can assess \
graphic design aesthetics. It contains perturbed graphic design samples, preview images, \
element-level metadata, and labels for four aesthetic dimensions: layout, typography, \
graphics, and color.
"""

_HOMEPAGE = "https://github.com/arctanxarc/AesEval-Bench"

_LICENSE = "Unknown"

_URLS = {
    "benchmark_data": "1W5ocLYW0U-znD1Aq3C2xg_TLxL80jeiJ",
}

TASKS = {
    "layout": ("balance", "layering", "whitespace", "alignment"),
    "font": ("legibility", "hierarchy"),
    "graphic": ("quality", "relevance"),
    "color": ("harmony", "contrast", "appeal", "psychology"),
}
TASK_KEYS = tuple(
    f"{dimension}-{task}" for dimension, tasks in TASKS.items() for task in tasks
)


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


def _read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _find_benchmark_data_dir(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)

    if path.name == "benchmark_data" and path.is_dir():
        return path

    candidate = path / "benchmark_data"
    if candidate.is_dir():
        return candidate

    for child in path.iterdir():
        candidate = child / "benchmark_data"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find benchmark_data/. Pass either the archive file or an "
        "extracted benchmark_data directory via data_dir."
    )


def _parse_sample_name(sample_name: str) -> tuple[str, int]:
    stem = sample_name.removesuffix("-perturbs_new")
    prefix, sep, suffix = stem.rpartition("_")
    if sep and suffix.isdigit():
        return prefix, int(suffix)
    return stem, -1


def _task_label(task_key: str, gt: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    dimension, task = task_key.split("-", maxsplit=1)
    return {
        "dimension": dimension,
        "task": task,
        "key": task_key,
        "has_issue": bool(gt.get(task_key, [])),
    }


def _gt_annotations(gt: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    annotations = []
    for task_key in TASK_KEYS:
        dimension, task = task_key.split("-", maxsplit=1)
        for item in gt.get(task_key, []):
            element_indices = item["element_index"]
            bbox = item["original"]
            annotations.append(
                {
                    "dimension": dimension,
                    "task": task,
                    "key": task_key,
                    "element_index": int(element_indices[0]),
                    "attribute": item["attribute"],
                    "left": float(bbox["left"]),
                    "top": float(bbox["top"]),
                    "width": float(bbox["width"]),
                    "height": float(bbox["height"]),
                }
            )
    return annotations


def _simplified_elements(simplified_meta_info: dict[str, Any]) -> list[dict[str, Any]]:
    elements = []
    for item in simplified_meta_info["elements"]:
        elements.append(
            {
                "element_index": int(item["element_index"]),
                "type": item["type"],
                "left": float(item["left"]),
                "top": float(item["top"]),
                "width": float(item["width"]),
                "height": float(item["height"]),
                "angle": float(item["angle"]),
                "opacity": float(item["opacity"]),
                "color": item["color"] or [],
                "image_filename": item["image"],
                "text": item["text"],
                "font": item["font"] or "",
                "font_size": None
                if item["font_size"] is None
                else float(item["font_size"]),
                "text_color": item["text_color"] or "",
                "text_align": item["text_align"] or "",
            }
        )
    return elements


def _element_images(sample_dir: pathlib.Path) -> list[dict[str, str]]:
    def image_index(path: pathlib.Path) -> int:
        return int(path.stem)

    image_paths = [path for path in sample_dir.glob("*.png") if path.stem.isdigit()]
    return [
        {"filename": image_path.name, "image": str(image_path)}
        for image_path in sorted(image_paths, key=image_index)
    ]


class AesEvalBenchDataset(ds.GeneratorBasedBuilder):
    """A class for loading AesEval-Bench."""

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIGS = [
        ds.BuilderConfig(version=VERSION),
    ]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "sample_name": ds.Value("string"),
                "source_id": ds.Value("string"),
                "perturbation_id": ds.Value("int32"),
                "canvas_width": ds.Value("int32"),
                "canvas_height": ds.Value("int32"),
                "title": ds.Value("string"),
                "category": ds.Value("int32"),
                "keywords": [ds.Value("string")],
                "industries": [ds.Value("int32")],
                "preview": ds.Image(),
                "preview_highlight": ds.Image(),
                "element_images": [
                    {
                        "filename": ds.Value("string"),
                        "image": ds.Image(),
                    }
                ],
                "elements": [
                    {
                        "element_index": ds.Value("int32"),
                        "type": ds.Value("string"),
                        "left": ds.Value("float32"),
                        "top": ds.Value("float32"),
                        "width": ds.Value("float32"),
                        "height": ds.Value("float32"),
                        "angle": ds.Value("float32"),
                        "opacity": ds.Value("float32"),
                        "color": [ds.Value("string")],
                        "image_filename": ds.Value("string"),
                        "text": ds.Value("string"),
                        "font": ds.Value("string"),
                        "font_size": ds.Value("float32"),
                        "text_color": ds.Value("string"),
                        "text_align": ds.Value("string"),
                    }
                ],
                "task_labels": [
                    {
                        "dimension": ds.Value("string"),
                        "task": ds.Value("string"),
                        "key": ds.Value("string"),
                        "has_issue": ds.Value("bool"),
                    }
                ],
                "gt_annotations": [
                    {
                        "dimension": ds.Value("string"),
                        "task": ds.Value("string"),
                        "key": ds.Value("string"),
                        "element_index": ds.Value("int32"),
                        "attribute": ds.Value("string"),
                        "left": ds.Value("float32"),
                        "top": ds.Value("float32"),
                        "width": ds.Value("float32"),
                        "height": ds.Value("float32"),
                    }
                ],
                "gt_json": ds.Value("string"),
                "changes_json": ds.Value("string"),
                "meta_info_json": ds.Value("string"),
                "simplified_meta_info_json": ds.Value("string"),
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
        data_dir = self._get_data_dir(dl_manager)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            ),
        ]

    def _get_data_dir(self, dl_manager: ds.DownloadManager) -> pathlib.Path:
        local_data_dir = getattr(self.config, "data_dir", None) or dl_manager.manual_dir
        if local_data_dir:
            local_path = pathlib.Path(os.path.expanduser(local_data_dir))
            if local_path.is_file():
                extracted_path = dl_manager.extract(str(local_path))
                assert isinstance(extracted_path, str)
                return _find_benchmark_data_dir(extracted_path)
            return _find_benchmark_data_dir(local_path)

        cache_dir = pathlib.Path(
            dl_manager.download_config.cache_dir or ds.config.DOWNLOADED_DATASETS_PATH
        )
        archive_path = cache_dir / "aeseval_bench" / "benchmark_data.zip"
        if not archive_path.exists():
            logger.info("Downloading AesEval-Bench benchmark data from Google Drive.")
            download_google_drive_file(_URLS["benchmark_data"], archive_path)

        extracted_path = dl_manager.extract(str(archive_path))
        assert isinstance(extracted_path, str)
        return _find_benchmark_data_dir(extracted_path)

    def _generate_examples(self, data_dir: pathlib.Path):
        sample_dirs = sorted(
            d
            for d in data_dir.iterdir()
            if d.is_dir() and "_new" in d.name and (d / "GT.json").exists()
        )

        for key, sample_dir in enumerate(sample_dirs):
            gt = _read_json(sample_dir / "GT.json")
            changes = _read_json(sample_dir / "changes.json")
            meta_info = _read_json(sample_dir / "meta_info.json")
            simplified_meta_info = _read_json(sample_dir / "simplified_meta_info.json")
            source_id, perturbation_id = _parse_sample_name(sample_dir.name)

            yield (
                key,
                {
                    "sample_name": sample_dir.name,
                    "source_id": source_id,
                    "perturbation_id": perturbation_id,
                    "canvas_width": int(simplified_meta_info["canvas_width"]),
                    "canvas_height": int(simplified_meta_info["canvas_height"]),
                    "title": meta_info["title"],
                    "category": int(meta_info["category"]),
                    "keywords": meta_info["keywords"],
                    "industries": meta_info["industries"],
                    "preview": str(sample_dir / "preview.png"),
                    "preview_highlight": str(sample_dir / "preview_highlight.png"),
                    "element_images": _element_images(sample_dir),
                    "elements": _simplified_elements(simplified_meta_info),
                    "task_labels": [
                        _task_label(task_key, gt) for task_key in TASK_KEYS
                    ],
                    "gt_annotations": _gt_annotations(gt),
                    "gt_json": _dump_json(gt),
                    "changes_json": _dump_json(changes),
                    "meta_info_json": _dump_json(meta_info),
                    "simplified_meta_info_json": _dump_json(simplified_meta_info),
                },
            )
