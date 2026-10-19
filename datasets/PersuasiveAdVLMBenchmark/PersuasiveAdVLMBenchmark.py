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
from collections.abc import Iterable
from typing import Any, ClassVar

import gdown

import datasets as ds

_CITATION = """\
@InProceedings{Malakouti_2025_WACV,
    author    = {Malakouti, Sina and Aghazadeh, Aysan and Khandelwal, Ashmit and Kovashka, Adriana},
    title     = {Benchmarking VLMs' Reasoning About Persuasive Atypical Images},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4788--4798},
    doi       = {10.1109/wacv61041.2025.00469}
}
"""

_DESCRIPTION = """\
This release contains a 783-image subset of PittAds as relative image identifiers and
reverse-image-search URL lists. It does not include image files or the labels and prompts
needed to reproduce all tasks in the PersuasiveAdVLM paper.
"""

_HOMEPAGE = "https://github.com/aysanaghazadeh/PersuasiveAdVLMBenchmark"
_LICENSE = "unknown"
_GOOGLE_DRIVE_FILE_ID = "1hBV4X0h93oRpB-UvAH0HP_lZtDwLRi6b"
_METADATA_FILENAME = "processed_pittads_wacv_sample.json"


def _cache_dir(dl_manager: ds.DownloadManager) -> pathlib.Path:
    cache_dir = pathlib.Path(
        dl_manager.download_config.cache_dir or ds.config.DOWNLOADED_DATASETS_PATH
    )
    return cache_dir / "persuasive-ad-vlm-benchmark"


def _download_metadata(file_id: str, output_path: pathlib.Path) -> pathlib.Path:
    if output_path.is_file():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = gdown.download(
        id=file_id,
        output=str(output_path),
        quiet=True,
        resume=True,
    )
    if downloaded_path is None or not output_path.is_file():
        raise FileNotFoundError(f"Failed to download metadata file {file_id}")

    return output_path


def _iter_examples(
    metadata_path: str | pathlib.Path,
) -> Iterable[tuple[str, dict[str, Any]]]:
    with pathlib.Path(metadata_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Expected the released metadata to be a JSON object.")
    if not all(isinstance(image_path, str) for image_path in data):
        raise TypeError("Expected each metadata key to be an image path string.")

    for image_path in sorted(data):
        image_urls = data[image_path]
        if not isinstance(image_urls, list):
            raise TypeError(
                "Expected each metadata entry to map an image path to a URL list."
            )
        if not all(isinstance(url, str) for url in image_urls):
            raise TypeError(
                f"Expected every image URL to be a string for {image_path}."
            )

        yield (
            image_path,
            {
                "id": image_path,
                "image_path": image_path,
                "image_urls": image_urls,
            },
        )


class PersuasiveAdVLMBenchmark(ds.GeneratorBasedBuilder):
    """Load the public PittAds image URL metadata released with PersuasiveAdVLM."""

    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS: ClassVar[list[ds.BuilderConfig]] = [
        ds.BuilderConfig(name="default", version=VERSION)
    ]
    DEFAULT_CONFIG_NAME: ClassVar[str] = "default"

    def _info(self) -> ds.DatasetInfo:
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=ds.Features(
                {
                    "id": ds.Value("string"),
                    "image_path": ds.Value("string"),
                    "image_urls": ds.Sequence(ds.Value("string")),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> list[ds.SplitGenerator]:
        metadata_path = _download_metadata(
            _GOOGLE_DRIVE_FILE_ID,
            _cache_dir(dl_manager) / _METADATA_FILENAME,
        )
        return [
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={"metadata_path": str(metadata_path)},
            )
        ]

    def _generate_examples(
        self, metadata_path: str
    ) -> Iterable[tuple[str, dict[str, Any]]]:
        yield from _iter_examples(metadata_path)
