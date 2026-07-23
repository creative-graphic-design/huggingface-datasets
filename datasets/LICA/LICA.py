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
import csv
import json
import os
import pathlib
from typing import Any, List

from datasets.utils.logging import get_logger

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@article{Hirsch2026LICA,
  title   = {LICA: Layered Image Composition Annotations for Graphic Design Research},
  author  = {Hirsch, Elad and Yadav, Shubham and Garg, Mohit and Mehta, Purvanshi},
  journal = {arXiv preprint arXiv:2603.16098},
  year    = {2026}
}
"""

_DESCRIPTION = """\
LICA is a graphic design layout dataset with rendered compositions, component-level layout \
specifications, and natural-language annotations. The released sample contains graphic designs \
grouped by template, with per-layout metadata, rendered PNG or MP4 files, layout JSON, \
per-layout annotations, and template-level annotations.
"""

_HOMEPAGE = "https://github.com/lica-world/lica-dataset"

_LICENSE = "cc-by-4.0"

_URLS = {
    "data_archive": "https://storage.googleapis.com/lica-assets/websites/blog/lica-data.zip",
}

CATEGORY_NAMES = [
    "Art & Design",
    "Brochure",
    "Business Cards",
    "Business Documents",
    "Cards & Invitations",
    "Education",
    "Flyers",
    "Infographics",
    "Instagram Posts",
    "Logo",
    "Menu",
    "Newsletter",
    "Planner & Calendar",
    "Posters",
    "Presentations",
    "Print Products",
    "Resume",
    "Social Media",
    "Videos",
]

COMPONENT_TYPE_NAMES = [
    "GROUP",
    "IMAGE",
    "TEXT",
    "TEXT_NEW",
]

RENDER_TYPE_NAMES = [
    "png",
    "mp4",
]


def _read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _find_lica_data_dir(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)

    if path.name == "lica-data" and path.is_dir():
        return path

    candidate = path / "lica-data"
    if candidate.is_dir():
        return candidate

    if path.is_dir() and (path / "metadata.csv").exists():
        return path

    for child in path.iterdir():
        if child.is_dir() and child.name == "lica-data":
            return child
        if child.is_dir() and (child / "metadata.csv").exists():
            return child

    raise FileNotFoundError(
        "Could not find lica-data/. Pass the extracted lica-data directory or "
        "the lica-data.zip archive via data_dir."
    )


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_px(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.removesuffix("px")
    return int(float(value))


def _annotation_or_empty(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {
            "description": "",
            "aesthetics": "",
            "tags": "",
            "user_intent": "",
            "raw": "",
        }
    data = _read_json(path)
    return {
        "description": data.get("description", ""),
        "aesthetics": data.get("aesthetics", ""),
        "tags": data.get("tags", ""),
        "user_intent": data.get("user_intent", ""),
        "raw": data.get("raw", ""),
    }


def _render_paths(
    data_dir: pathlib.Path,
    row: dict[str, str],
) -> tuple[pathlib.Path, str, str | None]:
    file_name = row.get("file_name", "")
    if file_name:
        render_path = data_dir / file_name
    else:
        base = data_dir / "images" / row["template_id"] / row["layout_id"]
        png_path = base.with_suffix(".png")
        mp4_path = base.with_suffix(".mp4")
        render_path = png_path if png_path.exists() else mp4_path

    suffix = render_path.suffix.lower().removeprefix(".")
    if suffix not in set(RENDER_TYPE_NAMES):
        raise ValueError(f"Unsupported LICA render type: {render_path}")
    render_type = suffix
    render_image = str(render_path) if render_type == "png" else None
    return render_path, render_type, render_image


class LICA(ds.GeneratorBasedBuilder):
    """A class for loading the LICA dataset."""

    VERSION = ds.Version("1.0.0")

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "layout_id": ds.Value("string"),
                "template_id": ds.Value("string"),
                "category": ds.ClassLabel(names=CATEGORY_NAMES),
                "n_template_layouts": ds.Value("int32"),
                "template_layout_index": ds.Value("int32"),
                "width": ds.Value("int32"),
                "height": ds.Value("int32"),
                "file_name": ds.Value("string"),
                "render_type": ds.ClassLabel(names=RENDER_TYPE_NAMES),
                "render_path": ds.Value("string"),
                "render_image": ds.Image(),
                "render_video_path": ds.Value("string"),
                "layout_width": ds.Value("int32"),
                "layout_height": ds.Value("int32"),
                "layout_background": ds.Value("string"),
                "layout_duration": ds.Value("float32"),
                "n_components": ds.Value("int32"),
                "component_types": [ds.ClassLabel(names=COMPONENT_TYPE_NAMES)],
                "layout_json": ds.Value("string"),
                "annotation_json": ds.Value("string"),
                "template_annotation_json": ds.Value("string"),
                "description": ds.Value("string"),
                "aesthetics": ds.Value("string"),
                "tags": ds.Value("string"),
                "user_intent": ds.Value("string"),
                "raw": ds.Value("string"),
                "template_description": ds.Value("string"),
                "template_aesthetics": ds.Value("string"),
                "template_tags": ds.Value("string"),
                "template_user_intent": ds.Value("string"),
                "template_raw": ds.Value("string"),
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
                name=ds.Split.TEST,
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
                return _find_lica_data_dir(extracted_path)
            return _find_lica_data_dir(local_path)

        logger.info("Downloading LICA data archive.")
        archive_path = dl_manager.download_and_extract(_URLS["data_archive"])
        assert isinstance(archive_path, str)
        return _find_lica_data_dir(archive_path)

    def _generate_examples(self, data_dir: pathlib.Path):
        template_annotations_path = (
            data_dir / "annotations" / "template_annotations.json"
        )
        template_annotations = (
            _read_json(template_annotations_path)
            if template_annotations_path.exists()
            else {}
        )

        metadata_path = data_dir / "metadata.csv"
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = sorted(
                reader,
                key=lambda row: (
                    row.get("category", ""),
                    row["template_id"],
                    _parse_optional_int(row.get("template_layout_index")) or 0,
                    row["layout_id"],
                ),
            )

        for key, row in enumerate(rows):
            layout_id = row["layout_id"]
            template_id = row["template_id"]

            layout_path = data_dir / "layouts" / template_id / f"{layout_id}.json"
            annotation_path = (
                data_dir / "annotations" / template_id / f"{layout_id}.json"
            )

            layout = _read_json(layout_path)
            annotation = _annotation_or_empty(annotation_path)
            template_annotation = template_annotations.get(template_id, {})
            template_annotation = {
                "description": template_annotation.get("description", ""),
                "aesthetics": template_annotation.get("aesthetics", ""),
                "tags": template_annotation.get("tags", ""),
                "user_intent": template_annotation.get("user_intent", ""),
                "raw": template_annotation.get("raw", ""),
            }

            components = layout.get("components", [])
            render_path, render_type, render_image = _render_paths(data_dir, row)

            yield (
                key,
                {
                    "layout_id": layout_id,
                    "template_id": template_id,
                    "category": row.get("category", ""),
                    "n_template_layouts": _parse_optional_int(
                        row.get("n_template_layouts")
                    ),
                    "template_layout_index": _parse_optional_int(
                        row.get("template_layout_index")
                    ),
                    "width": _parse_optional_int(row.get("width")),
                    "height": _parse_optional_int(row.get("height")),
                    "file_name": row.get("file_name", ""),
                    "render_type": render_type,
                    "render_path": str(render_path),
                    "render_image": render_image,
                    "render_video_path": str(render_path)
                    if render_type == "mp4"
                    else "",
                    "layout_width": _parse_px(layout.get("width")),
                    "layout_height": _parse_px(layout.get("height")),
                    "layout_background": layout.get("background", ""),
                    "layout_duration": _parse_optional_float(layout.get("duration")),
                    "n_components": len(components),
                    "component_types": [component["type"] for component in components],
                    "layout_json": _dump_json(layout),
                    "annotation_json": _dump_json(annotation),
                    "template_annotation_json": _dump_json(template_annotation),
                    "description": annotation["description"],
                    "aesthetics": annotation["aesthetics"],
                    "tags": annotation["tags"],
                    "user_intent": annotation["user_intent"],
                    "raw": annotation["raw"],
                    "template_description": template_annotation["description"],
                    "template_aesthetics": template_annotation["aesthetics"],
                    "template_tags": template_annotation["tags"],
                    "template_user_intent": template_annotation["user_intent"],
                    "template_raw": template_annotation["raw"],
                },
            )
