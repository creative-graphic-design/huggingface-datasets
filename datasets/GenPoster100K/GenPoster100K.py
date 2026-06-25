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
import pathlib
import pickle
import re
from typing import Any, List

from datasets.utils.logging import get_logger
from huggingface_hub import snapshot_download

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@inproceedings{wang2025sega,
  title={SEGA: A Stepwise Evolution Paradigm for Content-Aware Layout Generation with Design Prior},
  author={Wang, Haoran and Zhao, Bo and Wang, Jinghui and Wang, Hanzhang and Yang, Huan and Ji, Wei and Liu, Hao and Xiao, Xinyan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19321--19330},
  year={2025}
}
"""

_DESCRIPTION = """\
GenPoster-100K is a large-scale graphic design dataset containing over 100K PSD-format posters with parseable layer annotations.
Each example includes a rendered background image, PSD source reference, and layer-level metadata such as text content,
bounding boxes, typographic attributes, and color properties. This dataset is designed for content-aware layout generation
and other multimodal graphic design understanding tasks.
"""

_HOMEPAGE = "https://huggingface.co/datasets/BruceW91/GenPoster-100K"

_LICENSE = "cc-by-nc-4.0"

_DATASET_REPO_ID = "BruceW91/GenPoster-100K"
# The source repo provides both:
# - 0503_raw.pkl: 102,736 records with signed HTTP URLs
# - 0503_raw_offline.pkl: 102,703 records with local big_poster/* paths
# For this loader we resolve images from extracted part_*.tar.gz archives, so we
# intentionally use the offline annotation variant and keep path resolution local.
# The 33-record delta is inherited from the upstream offline export.
_ANNOTATION_FILENAME = "0503_raw_offline.pkl"
_ARCHIVE_GLOB = "part_*.tar.gz"
_IMAGE_PATH_MARKER = "big_poster/poster_metadata/"


class GenPoster100K(ds.GeneratorBasedBuilder):
    """GenPoster-100K dataset with layer-level metadata and images."""

    VERSION = ds.Version("1.0.0")

    def _info(self) -> ds.DatasetInfo:
        layer_features = ds.Features(
            {
                "layer_name": ds.Value("string"),
                "text": ds.Value("string"),
                "bbox": ds.Sequence(ds.Value("int32"), length=4),
                "angle": ds.Value("int32"),
                "psd_size": ds.Sequence(ds.Value("int32"), length=2),
                "stroke_width": ds.Value("float32"),
                "font": ds.Value("string"),
                "font_size": ds.Value("float32"),
                "tracking": ds.Value("float32"),
                "justification": ds.Value("int32"),
                "fill_color": ds.Sequence(ds.Value("float32"), length=4),
                "layer_image": ds.Image(),
                "layer_image_relpath": ds.Value("string"),
                "label": ds.Value("string"),
            }
        )

        features = ds.Features(
            {
                "id": ds.Value("int32"),
                "background_image": ds.Image(),
                "background_image_relpath": ds.Value("string"),
                "psd_path": ds.Value("string"),
                "regions": ds.Sequence(ds.Sequence(ds.Value("int32"), length=4)),
                "layers": ds.Sequence(layer_features),
            }
        )

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _snapshot_dataset(self) -> pathlib.Path:
        dataset_path = snapshot_download(
            repo_id=_DATASET_REPO_ID,
            repo_type="dataset",
            allow_patterns=[_ANNOTATION_FILENAME, _ARCHIVE_GLOB],
        )
        return pathlib.Path(dataset_path)

    def _sorted_archives(self, dataset_path: pathlib.Path) -> List[str]:
        archives = list(dataset_path.glob(_ARCHIVE_GLOB))

        def _part_number(path: pathlib.Path) -> int:
            match = re.search(r"part_(\d+)\.tar\.gz$", path.name)
            assert match is not None
            return int(match.group(1))

        archives = sorted(archives, key=_part_number)
        return [str(path) for path in archives]

    def _normalize_relative_image_path(self, image_path: str) -> str:
        cleaned_path = image_path.split("?", maxsplit=1)[0]

        if _IMAGE_PATH_MARKER in cleaned_path:
            image_suffix = cleaned_path.split(_IMAGE_PATH_MARKER, maxsplit=1)[1]
            return f"{_IMAGE_PATH_MARKER}{image_suffix}"

        poster_metadata_marker = "poster_metadata/"
        if poster_metadata_marker in cleaned_path:
            image_name = cleaned_path.split(poster_metadata_marker, maxsplit=1)[1]
            return f"{_IMAGE_PATH_MARKER}{image_name}"

        return cleaned_path.lstrip("/")

    def _normalize_bbox(self, bbox: Any) -> List[int]:
        values = [int(value) for value in list(bbox)[:4]]
        while len(values) < 4:
            values.append(0)
        return values

    def _normalize_size(self, size: Any) -> List[int]:
        values = [int(value) for value in list(size)[:2]]
        while len(values) < 2:
            values.append(0)
        return values

    def _normalize_color(self, color: Any) -> List[float]:
        values = [float(value) for value in list(color)[:4]]
        while len(values) < 4:
            values.append(1.0)
        return values

    def _build_image_index(
        self, extracted_paths: List[str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        image_index: dict[str, str] = {}
        basename_index: dict[str, str] = {}

        for extracted_path in extracted_paths:
            root = pathlib.Path(extracted_path)
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue

                absolute_path = file_path.as_posix()
                basename_index[file_path.name] = absolute_path

                if _IMAGE_PATH_MARKER in absolute_path:
                    image_suffix = absolute_path.split(_IMAGE_PATH_MARKER, maxsplit=1)[
                        1
                    ]
                    image_index[f"{_IMAGE_PATH_MARKER}{image_suffix}"] = absolute_path

        logger.info("Indexed %d images", len(image_index))
        return image_index, basename_index

    def _resolve_image_path(
        self,
        image_path: str,
        image_index: dict[str, str],
        basename_index: dict[str, str],
    ) -> tuple[str | None, str]:
        relative_path = self._normalize_relative_image_path(image_path)
        absolute_path = image_index.get(relative_path)

        if absolute_path is None:
            absolute_path = basename_index.get(pathlib.Path(relative_path).name)

        return absolute_path, relative_path

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        dataset_path = self._snapshot_dataset()
        annotation_path = dataset_path / _ANNOTATION_FILENAME
        archives = self._sorted_archives(dataset_path)

        extracted_archives = dl_manager.extract(archives)
        assert isinstance(extracted_archives, list)

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "annotation_path": annotation_path,
                    "extracted_archives": extracted_archives,
                },
            )
        ]

    def _generate_examples(
        self, annotation_path: pathlib.Path, extracted_archives: List[str]
    ):
        with open(annotation_path, "rb") as f:
            records = pickle.load(f)

        image_index, basename_index = self._build_image_index(extracted_archives)

        for idx, (background_path, layers, psd_path, regions) in enumerate(records):
            background_image_path, background_relative_path = self._resolve_image_path(
                background_path,
                image_index,
                basename_index,
            )

            normalized_layers = []
            for layer in layers:
                layer_image_path, layer_relative_path = self._resolve_image_path(
                    layer.get("img", ""), image_index, basename_index
                )

                normalized_layers.append(
                    {
                        "layer_name": str(layer.get("LayerName", "")),
                        "text": str(layer.get("Text", "")),
                        "bbox": self._normalize_bbox(layer.get("Bounding Box", [])),
                        "angle": int(layer.get("Angle", 0)),
                        "psd_size": self._normalize_size(layer.get("psd_size", [])),
                        "stroke_width": float(layer.get("StrokeWidth", 0.0)),
                        "font": str(layer.get("Font", "")),
                        "font_size": float(layer.get("FontSize", 0.0)),
                        "tracking": float(layer.get("Tracking", 0.0)),
                        "justification": int(layer.get("Justification", 0)),
                        "fill_color": self._normalize_color(layer.get("FillColor", [])),
                        "layer_image": layer_image_path,
                        "layer_image_relpath": layer_relative_path,
                        "label": str(layer.get("label", "")),
                    }
                )

            normalized_regions = [self._normalize_bbox(region) for region in regions]

            yield (
                idx,
                {
                    "id": idx,
                    "background_image": background_image_path,
                    "background_image_relpath": background_relative_path,
                    "psd_path": str(psd_path),
                    "regions": normalized_regions,
                    "layers": normalized_layers,
                },
            )
