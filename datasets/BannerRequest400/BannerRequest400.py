# Copyright 2024 Shunsuke Kitada and the current dataset script contributor.
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
from typing import List, assert_never

from datasets.utils.logging import get_logger
from PIL import Image

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@misc{wang2025banneragency,
  title={BannerAgency: Advertising Banner Design with Multimodal LLM Agents},
  author={Wang, Heng and Shimose, Yotaro and Takamatsu, Shingo},
  year={2025},
  eprint={2503.11060},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
BannerRequest400 is a multimodal benchmark for evaluating advertising banner generation systems. \
It contains 100 brand logos (in both PNG and SVG formats), 400 diverse abstract banner design \
requests, and 5,200 concrete specifications spanning 13 standard banner dimensions. Each campaign \
includes multiple target audience variations with detailed design specifications including colors, \
text placement, and call-to-action elements.
"""

_HOMEPAGE = "https://github.com/sony/BannerAgency/tree/main/BannerRequest400"

_LICENSE = "MIT"

_URLS = {
    "data_archive": "https://github.com/sony/BannerAgency/archive/refs/heads/main.tar.gz",
}


class BannerRequest400Type(StrEnum):
    abstract_400 = auto()
    concrete_5k = auto()


# Banner dimensions available in concrete requests
BANNER_DIMENSIONS = [
    "300x250",
    "728x90",
    "160x600",
    "300x600",
    "970x250",
    "320x100",
    "468x60",
    "250x250",
    "336x280",
    "120x600",
    "970x90",
    "180x150",
    "300x50",
]


@dataclass
class BannerRequest400Config(ds.BuilderConfig):
    name: BannerRequest400Type


class BannerRequest400Dataset(ds.GeneratorBasedBuilder):
    """A class for loading BannerRequest400 dataset."""

    config: BannerRequest400Config

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = BannerRequest400Config
    BUILDER_CONFIGS = [
        BannerRequest400Config(
            version=VERSION,
            name=BannerRequest400Type.abstract_400,
            description="400 abstract banner design requests with logos",
        ),
        BannerRequest400Config(
            version=VERSION,
            name=BannerRequest400Type.concrete_5k,
            description="100 campaigns with 5,200 concrete specifications across 13 dimensions",
        ),
    ]

    DEFAULT_CONFIG_NAME = "abstract_400"

    def _info(self) -> ds.DatasetInfo:
        match self.config.name:
            case BannerRequest400Type.abstract_400:
                features = ds.Features(
                    {
                        "id": ds.Value("int32"),
                        "banner_request": ds.Value("string"),
                        "logo_png": ds.Image(),
                        "logo_svg": ds.Value("string"),
                    }
                )
            case BannerRequest400Type.concrete_5k:
                # Define pair features for advertising variations
                pair_features = ds.Features(
                    {
                        "target_audience": ds.Value("string"),
                        "primary_purpose": ds.Value("string"),
                        **{
                            f"concrete_request_{dim}": ds.Value("string")
                            for dim in BANNER_DIMENSIONS
                        },
                    }
                )

                features = ds.Features(
                    {
                        "id": ds.Value("int32"),
                        "banner_request": ds.Value("string"),
                        "advertiser": ds.Value("string"),
                        "logo_name": ds.Value("string"),
                        "logo_description": ds.Value("string"),
                        "logo_png": ds.Image(),
                        "logo_svg": ds.Value("string"),
                        "advertising_variations": ds.Features(
                            {
                                "pair_1": pair_features,
                                "pair_2": pair_features,
                                "pair_3": pair_features,
                                "pair_4": pair_features,
                            }
                        ),
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
        # Download and extract the entire repository archive
        archive_path = dl_manager.download_and_extract(_URLS["data_archive"])
        assert isinstance(archive_path, str)

        # Navigate to the BannerRequest400 subdirectory
        base_dir = pathlib.Path(archive_path) / "BannerAgency-main" / "BannerRequest400"

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "base_dir": base_dir,
                },
            ),
        ]

    def _generate_examples(self, base_dir: pathlib.Path):
        # Load logo files
        logos_png_dir = base_dir / "logos_png"
        logos_svg_dir = base_dir / "logos_svg"

        logo_files_png = sorted(logos_png_dir.glob("*.png"))
        logo_files_svg = sorted(logos_svg_dir.glob("*.svg"))

        match self.config.name:
            case BannerRequest400Type.abstract_400:
                # Load abstract_400.jsonl (plain text with quoted strings)
                abstract_path = base_dir / "abstract_400.jsonl"
                with open(abstract_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for idx, line in enumerate(lines):
                    # Remove quotes and newline
                    banner_request = line.strip().strip('"')

                    # Cycle through logos (400 requests / 100 logos = 4 cycles)
                    logo_idx = idx % 100

                    logo_png_path = logo_files_png[logo_idx]
                    logo_svg_path = logo_files_svg[logo_idx]

                    yield (
                        idx,
                        {
                            "id": idx + 1,
                            "banner_request": banner_request,
                            "logo_png": Image.open(logo_png_path),
                            "logo_svg": logo_svg_path.read_text(encoding="utf-8"),
                        },
                    )

            case BannerRequest400Type.concrete_5k:
                # Load concrete_5k.json
                concrete_path = base_dir / "concrete_5k.json"
                with open(concrete_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Build logo lookup dictionaries
                logo_png_map = {p.name: p for p in logo_files_png}
                logo_svg_map = {p.name: p for p in logo_files_svg}

                for entry in data:
                    entry_id = entry["id"]
                    logo_name = entry["logo_name"]

                    # Get logo paths
                    logo_png_path = logo_png_map[logo_name]
                    # SVG uses same name but .svg extension
                    logo_svg_name = logo_name.replace(".png", ".svg")
                    logo_svg_path = logo_svg_map[logo_svg_name]

                    # Build nested advertising_variations structure
                    ad_vars = entry["advertising_variations"]
                    formatted_ad_vars = {}

                    for pair_key in ["pair_1", "pair_2", "pair_3", "pair_4"]:
                        pair_data = ad_vars[pair_key]
                        formatted_pair = {
                            "target_audience": pair_data["target_audience"],
                            "primary_purpose": pair_data["primary_purpose"],
                        }

                        # Add all concrete_request fields
                        for dim in BANNER_DIMENSIONS:
                            field_key = f"concrete_request_{dim}"
                            formatted_pair[field_key] = pair_data.get(field_key, "")

                        formatted_ad_vars[pair_key] = formatted_pair

                    yield (
                        entry_id - 1,
                        {
                            "id": entry["id"],
                            "banner_request": entry["banner_request"],
                            "advertiser": entry["advertiser"],
                            "logo_name": logo_name,
                            "logo_description": entry["logo_description"],
                            "logo_png": Image.open(logo_png_path),
                            "logo_svg": logo_svg_path.read_text(encoding="utf-8"),
                            "advertising_variations": formatted_ad_vars,
                        },
                    )

            case _:
                assert_never(self.config.name)
