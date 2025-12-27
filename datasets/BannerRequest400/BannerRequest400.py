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
#
# This script was generated from shunk031/cookiecutter-huggingface-datasets.
#
# TODO: Address all TODOs and remove all explanatory comments
import json
import os
import pathlib
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Dict, List, assert_never
from urllib.parse import urljoin

import pandas as pd
from datasets.utils.logging import get_logger
from PIL import Image

import datasets as ds

logger = get_logger(__name__)


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{wang2025banneragency,
  title={BannerAgency: Advertising Banner Design with Multimodal LLM Agents},
  author={Wang, Heng and Shimose, Yotaro and Takamatsu, Shingo},
  journal={arXiv preprint arXiv:2503.11060},
  year={2025}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This human-rated dataset is a collection of **700 graphic banner designs** built to study the reliability of graphic design evaluation, focusing on three fundamental design principles: **alignment, overlap, and white space**. The dataset comprises 100 original designs and **600 perturbed samples** artificially generated for evaluation, featuring human annotations collected from 60 participants who provided scores ranging from **1 to 10** for each design."""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/sony/BannerAgency"

# TODO: Add the license for the dataset here if you can find it
_LICENSE = "MIT"

_URLS = {
    "abstract_400": "https://raw.githubusercontent.com/sony/BannerAgency/refs/heads/main/BannerRequest400/abstract_400.jsonl",
    "concrete_5k": "https://raw.githubusercontent.com/sony/BannerAgency/refs/heads/main/BannerRequest400/concrete_5k.json",
}
LOGO_PNG_BASE_URL = "https://raw.githubusercontent.com/sony/BannerAgency/refs/heads/main/BannerRequest400/logos_png/"
LOGO_SVG_BASE_URL = "https://raw.githubusercontent.com/sony/BannerAgency/refs/heads/main/BannerRequest400/logos_svg/"


class BannerRequest400Type(StrEnum):
    abstract_400 = auto()
    concrete_5k = auto()


@dataclass
class BannerRequest400Config(ds.BuilderConfig):
    """BuilderConfig for BannerRequest400."""

    name: BannerRequest400Type


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class BannerRequest400Dataset(ds.GeneratorBasedBuilder):
    """A class for loading BannerRequest400 dataset."""

    config: BannerRequest400Config

    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = BannerRequest400Config
    BUILDER_CONFIGS = [
        BannerRequest400Config(name=BannerRequest400Type.abstract_400, version=VERSION),
        BannerRequest400Config(name=BannerRequest400Type.concrete_5k, version=VERSION),
    ]

    def _info(self) -> ds.DatasetInfo:
        def get_abstract_400_feature() -> ds.Features:
            return ds.Features(
                {
                    "banner_request": ds.Value("string"),
                }
            )

        def get_concrete_5k_feature() -> ds.Features:
            pair_feature = {
                "target_audience": ds.Value("string"),
                "primary_purpose": ds.Value("string"),
                "concrete_request_300x250": ds.Value("string"),
                "concrete_request_200x200": ds.Value("string"),
                "concrete_request_250x250": ds.Value("string"),
                "concrete_request_336x280": ds.Value("string"),
                "concrete_request_970x250": ds.Value("string"),
                "concrete_request_970x90": ds.Value("string"),
                "concrete_request_728x90": ds.Value("string"),
                "concrete_request_468x60": ds.Value("string"),
                "concrete_request_320x50": ds.Value("string"),
                "concrete_request_300x600": ds.Value("string"),
                "concrete_request_160x600": ds.Value("string"),
                "concrete_request_120x600": ds.Value("string"),
                "concrete_request_240x400": ds.Value("string"),
            }
            features = ds.Features(
                {
                    "id": ds.Value("int32"),
                    "banner_request": ds.Value("string"),
                    "advertiser": ds.Value("string"),
                    "logo": ds.Image(),
                    "logo_svg": ds.Value("string"),
                    "logo_description": ds.Value("string"),
                    "advertising_variations": {
                        "pair_1": pair_feature,
                        "pair_2": pair_feature,
                        "pair_3": pair_feature,
                        "pair_4": pair_feature,
                    },
                }
            )
            return features

        if self.config.name is BannerRequest400Type.abstract_400:
            features = get_abstract_400_feature()
        elif self.config.name is BannerRequest400Type.concrete_5k:
            features = get_concrete_5k_feature()
        else:
            assert_never(self.config.name)

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _load_abstract_400(self, filepath: str) -> List[str]:
        with open(filepath, "r") as rf:
            abstract_400 = [line for line in rf]
            # remove newline characters
            abstract_400 = [line.strip() for line in abstract_400]
            # remove starting and ending double quotes
            abstract_400 = [line[1:-1] for line in abstract_400]
        assert len(abstract_400) == 400
        return abstract_400

    def _load_concrete_5k(self, filepath: str) -> pd.DataFrame:
        with open(filepath, "r") as rf:
            concrete_5k = json.load(rf)
        assert len(concrete_5k) == 100
        return concrete_5k

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        files = dl_manager.download_and_extract(_URLS)

        abstract_400 = self._load_abstract_400(files["abstract_400"])
        concrete_5k = self._load_concrete_5k(files["concrete_5k"])

        for example in concrete_5k:
            logo_name = example["logo_name"]

            logo_png_url = urljoin(LOGO_PNG_BASE_URL, logo_name)
            logo_png_path = dl_manager.download(logo_png_url)
            assert isinstance(logo_png_path, str)
            example["logo_png_path"] = logo_png_path

            logo_svg_url = urljoin(LOGO_SVG_BASE_URL, logo_name.replace(".png", ".svg"))
            logo_svg_path = dl_manager.download(logo_svg_url)
            assert isinstance(logo_svg_path, str)
            example["logo_svg_path"] = logo_svg_path

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "abstract_400": abstract_400,
                    "concrete_5k": concrete_5k,
                },
            ),
        ]

    def _generate_abstract_400_examples(
        self, abstract_400: List[str], concrete_5k: List[Dict[str, Any]]
    ):
        for i, concrete_example in enumerate(concrete_5k):
            logo_name = concrete_example["logo_name"]
            logo_name, logo_ext = os.path.splitext(logo_name)
            for j in range(0, len(abstract_400), 14):
                for abstract_request in abstract_400[j : j + 14]:
                    replace_logo_name = (
                        f"../local-server/images/{logo_name}_logo_cropped{logo_ext}"
                    )
                    abstract_request = abstract_request.replace(
                        replace_logo_name, "{{logo_path}}"
                    )

                    print()
                    print(f"{replace_logo_name=}")
                    print(f"{abstract_request=}")

                    if "local-server" in abstract_request:
                        breakpoint()

                    yield i, {"banner_request": abstract_request}

    def _generate_concrete_5k_examples(self, concrete_5k: List[Dict[str, Any]]):
        for i, example in enumerate(concrete_5k):
            logo_png = Image.open(example.pop("logo_png_path"))
            logo_svg = pathlib.Path(example.pop("logo_svg_path")).read_text()

            example["logo_png"] = logo_png
            example["logo_svg"] = logo_svg

            yield i, example

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
        self, abstract_400: List[str], concrete_5k: List[Dict[str, Any]]
    ):
        if self.config.name is BannerRequest400Type.abstract_400:
            yield from self._generate_abstract_400_examples(
                abstract_400=abstract_400, concrete_5k=concrete_5k
            )
        elif self.config.name is BannerRequest400Type.concrete_5k:
            yield from self._generate_concrete_5k_examples(
                concrete_5k=concrete_5k,
            )
        else:
            assert_never(self.config.name)
