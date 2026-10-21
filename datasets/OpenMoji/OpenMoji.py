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
from pathlib import Path
from typing import Any

import datasets as ds

_OPENMOJI_VERSION = "17.0.0"

_CITATION = (
    """\
@misc{openmoji,
  author = {Groß, Benedikt and Utz, Daniel and The OpenMoji Contributors},
  title = {OpenMoji},
  year = {2026},
  version = {%s},
  doi = {10.5281/zenodo.19704720},
  url = {https://doi.org/10.5281/zenodo.19704720}
}
"""
    % _OPENMOJI_VERSION
)

_DESCRIPTION = f"""\
OpenMoji {_OPENMOJI_VERSION} contains 4,495 Unicode and custom emoji and icon entries, including \
skin-tone variants. Each row includes upstream metadata, color and black SVG source, \
and 618 x 618 PNG images.
"""

_HOMEPAGE = "https://github.com/hfg-gmuend/openmoji"
_LICENSE = "cc-by-sa-4.0"

_URLS = {
    "svg_color": f"https://github.com/hfg-gmuend/openmoji/releases/download/{_OPENMOJI_VERSION}/openmoji-svg-color.zip",
    "svg_black": f"https://github.com/hfg-gmuend/openmoji/releases/download/{_OPENMOJI_VERSION}/openmoji-svg-black.zip",
    "image_color": f"https://github.com/hfg-gmuend/openmoji/releases/download/{_OPENMOJI_VERSION}/openmoji-618x618-color.zip",
    "image_black": f"https://github.com/hfg-gmuend/openmoji/releases/download/{_OPENMOJI_VERSION}/openmoji-618x618-black.zip",
    "metadata": f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/{_OPENMOJI_VERSION}/data/openmoji.json",
}


def _optional_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _split_tags(value: str) -> list[str]:
    return [tag for item in value.split(",") if (tag := item.strip())]


def normalize_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "hexcode": record["hexcode"],
        "emoji": _optional_string(record["emoji"]),
        "group": _optional_string(record["group"]),
        "subgroups": _optional_string(record["subgroups"]),
        "annotation": _optional_string(record["annotation"]),
        "tags": _split_tags(record["tags"]),
        "openmoji_tags": _split_tags(record["openmoji_tags"]),
        "openmoji_author": _optional_string(record["openmoji_author"]),
        "openmoji_date": _optional_string(record["openmoji_date"]),
        "skintone": _optional_string(record["skintone"]),
        "skintone_combination": _optional_string(record["skintone_combination"]),
        "skintone_base_emoji": _optional_string(record["skintone_base_emoji"]),
        "skintone_base_hexcode": _optional_string(record["skintone_base_hexcode"]),
        "unicode": _optional_string(record["unicode"]),
        "order": _optional_int(record["order"]),
    }


class OpenMoji(ds.GeneratorBasedBuilder):
    VERSION = ds.Version(_OPENMOJI_VERSION)

    def _info(self) -> ds.DatasetInfo:
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=ds.Features(
                {
                    "hexcode": ds.Value("string"),
                    "emoji": ds.Value("string"),
                    "group": ds.Value("string"),
                    "subgroups": ds.Value("string"),
                    "annotation": ds.Value("string"),
                    "tags": [ds.Value("string")],
                    "openmoji_tags": [ds.Value("string")],
                    "openmoji_author": ds.Value("string"),
                    "openmoji_date": ds.Value("string"),
                    "skintone": ds.Value("string"),
                    "skintone_combination": ds.Value("string"),
                    "skintone_base_emoji": ds.Value("string"),
                    "skintone_base_hexcode": ds.Value("string"),
                    "unicode": ds.Value("string"),
                    "order": ds.Value("int32"),
                    "svg_color": ds.Value("string"),
                    "svg_black": ds.Value("string"),
                    "image_color": ds.Image(),
                    "image_black": ds.Image(),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> list[ds.SplitGenerator]:
        files = dl_manager.download_and_extract(_URLS)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": files["metadata"],
                    "svg_color_dir": files["svg_color"],
                    "svg_black_dir": files["svg_black"],
                    "image_color_dir": files["image_color"],
                    "image_black_dir": files["image_black"],
                },
            )
        ]

    def _generate_examples(
        self,
        metadata_path: str,
        svg_color_dir: str,
        svg_black_dir: str,
        image_color_dir: str,
        image_black_dir: str,
    ):
        with open(metadata_path, encoding="utf-8") as metadata_file:
            records = json.load(metadata_file)

        for record in records:
            hexcode = record["hexcode"]
            yield (
                hexcode,
                {
                    **normalize_metadata_record(record),
                    "svg_color": (Path(svg_color_dir) / f"{hexcode}.svg").read_text(
                        encoding="utf-8"
                    ),
                    "svg_black": (Path(svg_black_dir) / f"{hexcode}.svg").read_text(
                        encoding="utf-8"
                    ),
                    "image_color": str(Path(image_color_dir) / f"{hexcode}.png"),
                    "image_black": str(Path(image_black_dir) / f"{hexcode}.png"),
                },
            )
