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

import gdown
from datasets.utils.logging import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@inproceedings{Chen_2025_CVPR,
  title = {POSTA: A Go-to Framework for Customized Artistic Poster Generation},
  author = {Chen, Haoyu and Xu, Xiaojie and Li, Wenbo and Ren, Jingjing and Ye, Tian and Liu, Songhua and Chen, Ying-Cong and Zhu, Lei and Wang, Xinchao},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
  pages = {28694--28704},
  url = {https://openaccess.thecvf.com/content/CVPR2025/html/Chen_POSTA_A_Go-to_Framework_for_Customized_Artistic_Poster_Generation_CVPR_2025_paper.html}
}
"""

_DESCRIPTION = """\
POSTA-PosterArt is the dataset introduced with POSTA, a framework for customized artistic poster \
generation. It contains PosterArt-Design for poster layout and typography planning, and \
PosterArt-Text for artistic text stylization with pixel-level segmentation masks and captions.
"""

_HOMEPAGE = "https://haoyuchen.com/POSTA"

_LICENSE = "Unknown"

_URLS = {
    # The POSTA website lists PosterArt-Design as 152.3GB, while the public
    # Google Drive folder currently exposes only Part1.zip. This is kept as a
    # mapping because the dataset may have unpublished or future PartN.zip files.
    "design_zips": {
        "Part1.zip": "17WMZsdfC-AxhcvHWmMZq9bW6Dc9CUzDM",
    },
    "text_zip": "1yRk4y4ci3-8vW-ySWVsvevcSK3B_b2N1",
}


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


class POSTAPosterArtType(StrEnum):
    design = auto()
    text = auto()


@dataclass
class POSTAPosterArtConfig(ds.BuilderConfig):
    name: POSTAPosterArtType

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = POSTAPosterArtType(self.name)


def _as_path(value: str | pathlib.Path) -> pathlib.Path:
    return value if isinstance(value, pathlib.Path) else pathlib.Path(value)


def _cache_dir(dl_manager: ds.DownloadManager) -> pathlib.Path:
    return pathlib.Path(
        dl_manager.download_config.cache_dir or ds.config.DOWNLOADED_DATASETS_PATH
    )


def _is_ignored_path(path: pathlib.Path) -> bool:
    return any(part == "__MACOSX" or part.startswith("._") for part in path.parts)


def _find_first_dir(root: pathlib.Path, dirname: str) -> pathlib.Path | None:
    for path in root.rglob(dirname):
        if path.is_dir() and not _is_ignored_path(path):
            return path
    return None


def _flatten_text_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    def visit(layer: dict[str, Any]) -> None:
        if layer.get("kind") == "type" or layer.get("text_content"):
            position = layer.get("position") or {}
            font_info = layer.get("font_info") or {}
            flattened.append(
                {
                    "path": str(layer.get("path") or ""),
                    "name": str(layer.get("name") or ""),
                    "text_content": str(layer.get("text_content") or ""),
                    "visible": bool(layer.get("visible", False)),
                    "kind": str(layer.get("kind") or ""),
                    "opacity": int(layer.get("opacity") or 0),
                    "blend_mode": str(layer.get("blend_mode") or ""),
                    "left": int(position.get("left") or 0),
                    "top": int(position.get("top") or 0),
                    "right": int(position.get("right") or 0),
                    "bottom": int(position.get("bottom") or 0),
                    "width": int(position.get("width") or 0),
                    "height": int(position.get("height") or 0),
                    "center_x": float(position.get("center_x") or 0.0),
                    "center_y": float(position.get("center_y") or 0.0),
                    "font_name": str(font_info.get("font_name") or ""),
                    "font_size": float(font_info.get("font_size") or 0.0),
                    "color_values": [
                        float(value) for value in font_info.get("color_values") or []
                    ],
                    "alignment": str(font_info.get("alignment") or ""),
                    "rotation": float(font_info.get("rotation") or 0.0),
                }
            )

        for child in layer.get("children") or []:
            visit(child)

    for layer in layers:
        visit(layer)

    return flattened


def _iter_text_examples(
    root: str | pathlib.Path,
) -> Iterable[tuple[str, dict[str, Any]]]:
    root = _as_path(root)
    records: dict[str, dict[str, pathlib.Path]] = {}

    for path in root.rglob("*"):
        if not path.is_file() or _is_ignored_path(path) or path.name == ".DS_Store":
            continue

        name = path.name
        key: str
        stem: str
        if name.endswith(".caption"):
            stem = name[: -len(".caption")]
            key = "caption_path"
        elif name.endswith("_mask_img_single.png"):
            stem = name[: -len("_mask_img_single.png")]
            key = "mask_img_single"
        elif name.endswith("_mask.png"):
            stem = name[: -len("_mask.png")]
            key = "mask"
        elif name.endswith(".jpg"):
            stem = name[: -len(".jpg")]
            key = "image"
        else:
            continue

        records.setdefault(stem, {})[key] = path

    required = {"image", "caption_path", "mask", "mask_img_single"}
    for stem in sorted(records):
        record = records[stem]
        if not required <= record.keys():
            continue

        yield (
            stem,
            {
                "id": stem,
                "image": str(record["image"]),
                "caption": record["caption_path"].read_text(encoding="utf-8").strip(),
                "mask": str(record["mask"]),
                "mask_img_single": str(record["mask_img_single"]),
            },
        )


def _iter_design_examples(
    root: str | pathlib.Path,
) -> Iterable[tuple[str, dict[str, Any]]]:
    root = _as_path(root)
    background_dir = _find_first_dir(root, "background")
    poster_dir = _find_first_dir(root, "JPG")
    json_dir = _find_first_dir(root, "json") or _find_first_dir(root, "JSON")
    psd_dir = _find_first_dir(root, "PSD")

    if background_dir is None or poster_dir is None or json_dir is None:
        raise FileNotFoundError(
            "PosterArt-Design must contain background, JPG, and json/JSON directories."
        )

    psd_files = (
        {path.stem: path for path in psd_dir.glob("*.psd") if path.is_file()}
        if psd_dir is not None
        else {}
    )

    for annotation_path in sorted(json_dir.glob("*.json")):
        stem = annotation_path.stem
        example_id = f"{background_dir.parent.name}/{stem}"
        background_path = background_dir / f"{stem}.jpg"
        poster_path = poster_dir / f"{stem}.jpg"
        if not background_path.exists() or not poster_path.exists():
            continue

        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        psd_path = psd_files.get(stem)
        yield (
            stem,
            {
                "id": example_id,
                "background_image": str(background_path),
                "poster_image": str(poster_path),
                "psd_filename": psd_path.name if psd_path is not None else "",
                "annotation": json.dumps(annotation, ensure_ascii=False),
                "text_layers": _flatten_text_layers(annotation.get("layers") or []),
            },
        )


def _text_layer_features() -> dict[str, Any]:
    return {
        "path": ds.Value("string"),
        "name": ds.Value("string"),
        "text_content": ds.Value("string"),
        "visible": ds.Value("bool"),
        "kind": ds.Value("string"),
        "opacity": ds.Value("int32"),
        "blend_mode": ds.Value("string"),
        "left": ds.Value("int32"),
        "top": ds.Value("int32"),
        "right": ds.Value("int32"),
        "bottom": ds.Value("int32"),
        "width": ds.Value("int32"),
        "height": ds.Value("int32"),
        "center_x": ds.Value("float32"),
        "center_y": ds.Value("float32"),
        "font_name": ds.Value("string"),
        "font_size": ds.Value("float32"),
        "color_values": [ds.Value("float32")],
        "alignment": ds.Value("string"),
        "rotation": ds.Value("float32"),
    }


class POSTAPosterArt(ds.GeneratorBasedBuilder):
    """A class for loading the POSTA-PosterArt dataset."""

    config: POSTAPosterArtConfig

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = POSTAPosterArtConfig
    BUILDER_CONFIGS = [
        POSTAPosterArtConfig(
            name=POSTAPosterArtType.text,
            version=VERSION,
            description="PosterArt-Text artistic text stylization and segmentation data.",
        ),
        POSTAPosterArtConfig(
            name=POSTAPosterArtType.design,
            version=VERSION,
            description="PosterArt-Design poster layout and typography planning data.",
        ),
    ]
    DEFAULT_CONFIG_NAME = "text"

    def _info(self) -> ds.DatasetInfo:
        match self.config.name:
            case POSTAPosterArtType.text:
                features = ds.Features(
                    {
                        "id": ds.Value("string"),
                        "image": ds.Image(),
                        "caption": ds.Value("string"),
                        "mask": ds.Image(),
                        "mask_img_single": ds.Image(),
                    }
                )
            case POSTAPosterArtType.design:
                features = ds.Features(
                    {
                        "id": ds.Value("string"),
                        "background_image": ds.Image(),
                        "poster_image": ds.Image(),
                        "psd_filename": ds.Value("string"),
                        "annotation": ds.Value("string"),
                        "text_layers": [_text_layer_features()],
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
        cache_dir = _cache_dir(dl_manager) / "posta-poster-art"

        match self.config.name:
            case POSTAPosterArtType.text:
                archive_path = cache_dir / "PosterArt-Text.zip"
                if not archive_path.exists():
                    logger.info("Downloading PosterArt-Text.zip from Google Drive.")
                    archive_path = download_google_drive_file(
                        file_id=_URLS["text_zip"],
                        output_path=archive_path,
                    )

                data_dir = dl_manager.extract(str(archive_path))
                assert isinstance(data_dir, str)

            case POSTAPosterArtType.design:
                folder_dir = cache_dir / "PosterArt-Design"
                archive_paths: list[pathlib.Path] = []
                for filename, file_id in _URLS["design_zips"].items():
                    archive_path = folder_dir / filename
                    if not archive_path.exists():
                        logger.info("Downloading %s from Google Drive.", filename)
                        archive_path = download_google_drive_file(
                            file_id=file_id,
                            output_path=archive_path,
                        )
                    archive_paths.append(archive_path)

                extracted_dirs = [
                    dl_manager.extract(str(path)) for path in sorted(archive_paths)
                ]
                data_dir = extracted_dirs or [str(folder_dir)]

            case _:
                assert_never(self.config.name)

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            ),
        ]

    def _generate_examples(self, data_dir: str | list[str]):
        match self.config.name:
            case POSTAPosterArtType.text:
                assert isinstance(data_dir, str)
                yield from _iter_text_examples(data_dir)
            case POSTAPosterArtType.design:
                data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
                key = 0
                for root in data_dirs:
                    for _, example in _iter_design_examples(root):
                        yield key, example
                        key += 1
            case _:
                assert_never(self.config.name)
