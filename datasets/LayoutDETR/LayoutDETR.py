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
from typing import Any, Iterable, List

import gdown
from datasets.utils.logging import get_logger
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@inproceedings{yu2024layoutdetr,
  title={LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer},
  author={Yu, Ning and Chen, Chia-Chih and Chen, Zeyuan and Meng, Rui and Wu, Gang and Josel, Paul and Niebles, Juan Carlos and Xiong, Caiming and Xu, Ran},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
"""

_DESCRIPTION = """\
LayoutDETR ad banner data contains well-designed advertising banner images, paired \
foreground layout annotations, and background-only inpainted images released with \
LayoutDETR for multimodal layout generation.
"""

_HOMEPAGE = "https://github.com/salesforce/LayoutDETR"

_LICENSE = "apache-2.0"

_URLS = {
    "ads_banner_dataset": "1T09t4dX7zQ7J-8KxtJv1RkyjRNdilD1m",
}

LABEL_NAMES = (
    "header",
    "pre-header",
    "post-header",
    "body text",
    "disclaimer / footnote",
    "button",
    "callout",
    "logo",
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


def _find_dataset_root(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)

    if path.name == "png_json_gt" and path.is_dir():
        return path.parent

    if (path / "png_json_gt").is_dir():
        return path

    for child in path.iterdir():
        if child.is_dir() and (child / "png_json_gt").is_dir():
            return child

    for candidate in path.rglob("png_json_gt"):
        if candidate.is_dir():
            return candidate.parent

    raise FileNotFoundError(
        "Could not find png_json_gt/. Pass the extracted LayoutDETR ad banner "
        "dataset directory or the downloaded archive via data_dir."
    )


def _image_size(path: pathlib.Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _xyxy_to_cxcywh_normalized(
    bbox_xyxy: list[float],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        ((x1 + x2) / 2.0) / float(width),
        ((y1 + y2) / 2.0) / float(height),
        (x2 - x1) / float(width),
        (y2 - y1) / float(height),
    ]


def _has_nonzero_resized_side(box_width: float, box_height: float) -> bool:
    if box_width > box_height:
        return int(float(box_height) / float(box_width) * 256.0) // 2 * 2 != 0
    return int(float(box_width) / float(box_height) * 256.0) // 2 * 2 != 0


def _is_valid_element(element: dict[str, Any], width: int, height: int) -> bool:
    if element.get("label") not in LABEL_NAMES:
        return False
    text = element.get("str")
    if not isinstance(text, str) or len(text) == 0 or len(text) >= 256:
        return False
    bbox = element.get("xyxy_word_fit")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x1 < 0 or y1 < 0 or width < x2 or height < y2:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    box_width = int(x2) - int(x1)
    box_height = int(y2) - int(y1)
    if box_width > 1024 or box_height > 1024:
        return False
    return _has_nonzero_resized_side(box_width, box_height)


def _remove_almost_covered_elements(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid = []
    for index, element in enumerate(elements):
        x1, y1, x2, y2 = [float(value) for value in element["xyxy_word_fit"]]
        is_visible = True
        for other_index, other in enumerate(elements):
            if index == other_index:
                continue
            xx1, yy1, xx2, yy2 = [
                float(value) for value in other["xyxy_word_fit"]
            ]
            x1_max = max(x1, xx1)
            y1_max = max(y1, yy1)
            x2_min = min(x2, xx2)
            y2_min = min(y2, yy2)
            if x1_max < x2_min and y1_max < y2_min:
                overlap_area = (x2_min - x1_max) * (y2_min - y1_max)
                element_area = (x2 - x1) * (y2 - y1)
                if overlap_area / element_area >= 0.95:
                    is_visible = False
                    break
        valid.append(is_visible)
    return [element for index, element in enumerate(elements) if valid[index]]


def _valid_elements(
    annotation: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    elements = [
        element
        for element in annotation
        if _is_valid_element(element, width=width, height=height)
    ]
    return _remove_almost_covered_elements(elements)


def _background_path(root: pathlib.Path, dirname: str, stem: str) -> pathlib.Path | None:
    path = root / dirname / f"{stem}_inpainted.png"
    return path if path.exists() else None


def _split_json_paths(
    json_paths: list[pathlib.Path],
    split: str,
) -> list[pathlib.Path]:
    cutoff = int(len(json_paths) * 0.90)
    if split == "train":
        return json_paths[:cutoff]
    if split == "validation":
        return json_paths[cutoff:]
    raise ValueError(f"Unsupported split: {split}")


def _iter_examples(
    root: str | pathlib.Path,
    split: str,
) -> Iterable[tuple[str, dict[str, Any]]]:
    root = pathlib.Path(root)
    gt_dir = root / "png_json_gt"
    json_paths = _split_json_paths(sorted(gt_dir.glob("*.json")), split)

    for json_path in json_paths:
        stem = json_path.stem
        image_path = gt_dir / f"{stem}.png"
        if not image_path.exists():
            logger.warning("Skipping %s because paired PNG is missing.", json_path)
            continue

        width, height = _image_size(image_path)
        annotation = _read_json(json_path)
        elements = _valid_elements(annotation, width=width, height=height)
        if len(elements) == 0 or len(elements) > 9:
            continue

        background_1x_path = _background_path(
            root, "1x_inpainted_background_png", stem
        )
        background_3x_path = _background_path(
            root, "3x_inpainted_background_png", stem
        )
        example_elements = []
        for element in elements:
            bbox_xyxy = [float(value) for value in element["xyxy_word_fit"]]
            example_elements.append(
                {
                    "text": element["str"],
                    "label": element["label"],
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_cxcywh_normalized": _xyxy_to_cxcywh_normalized(
                        bbox_xyxy, width=width, height=height
                    ),
                }
            )

        yield (
            stem,
            {
                "id": stem,
                "image": str(image_path),
                "image_path": str(image_path),
                "background_1x": None
                if background_1x_path is None
                else str(background_1x_path),
                "background_1x_path": ""
                if background_1x_path is None
                else str(background_1x_path),
                "background_3x": None
                if background_3x_path is None
                else str(background_3x_path),
                "background_3x_path": ""
                if background_3x_path is None
                else str(background_3x_path),
                "width": width,
                "height": height,
                "elements": example_elements,
                "num_elements": len(example_elements),
                "raw_annotation": _dump_json(annotation),
            },
        )


class LayoutDETR(ds.GeneratorBasedBuilder):
    """A class for loading the LayoutDETR ad banner dataset."""

    VERSION = ds.Version("1.0.0")

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "id": ds.Value("string"),
                "image": ds.Image(),
                "image_path": ds.Value("string"),
                "background_1x": ds.Image(),
                "background_1x_path": ds.Value("string"),
                "background_3x": ds.Image(),
                "background_3x_path": ds.Value("string"),
                "width": ds.Value("int32"),
                "height": ds.Value("int32"),
                "elements": [
                    {
                        "text": ds.Value("string"),
                        "label": ds.Value("string"),
                        "bbox_xyxy": [ds.Value("float32")],
                        "bbox_cxcywh_normalized": [ds.Value("float32")],
                    }
                ],
                "num_elements": ds.Value("int32"),
                "raw_annotation": ds.Value("string"),
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
                gen_kwargs={"root": data_dir, "split": "train"},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"root": data_dir, "split": "validation"},
            ),
        ]

    def _get_data_dir(self, dl_manager: ds.DownloadManager) -> pathlib.Path:
        local_data_dir = getattr(self.config, "data_dir", None) or dl_manager.manual_dir
        if local_data_dir:
            local_path = pathlib.Path(os.path.expanduser(local_data_dir))
            if local_path.is_file():
                extracted_path = dl_manager.extract(str(local_path))
                assert isinstance(extracted_path, str)
                return _find_dataset_root(extracted_path)
            return _find_dataset_root(local_path)

        cache_dir = pathlib.Path(
            dl_manager.download_config.cache_dir or ds.config.DOWNLOADED_DATASETS_PATH
        )
        archive_path = cache_dir / "layoutdetr" / "ads_banner_dataset.zip"
        if not archive_path.exists():
            logger.info("Downloading LayoutDETR ad banner data from Google Drive.")
            download_google_drive_file(_URLS["ads_banner_dataset"], archive_path)

        extracted_path = dl_manager.extract(str(archive_path))
        assert isinstance(extracted_path, str)
        return _find_dataset_root(extracted_path)

    def _generate_examples(self, root: pathlib.Path, split: str):
        yield from _iter_examples(root=root, split=split)
