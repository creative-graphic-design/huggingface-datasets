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

import pathlib
from typing import List, Union

from datasets.utils.logging import get_logger
from huggingface_hub import snapshot_download
from PIL import Image

import datasets as ds

logger = get_logger(__name__)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
TODO: Add BibTeX citation here
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\

"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha"

# TODO: Add the license for the dataset here if you can find it
_LICENSE = "apache-2.0"


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class ObjectRemovalAlphaDataset(ds.GeneratorBasedBuilder):
    """A class for loading ObjectRemovalAlpha dataset."""

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIGS = [
        ds.BuilderConfig(version=VERSION),
    ]

    ORIGINAL_DATASET_NAME = "lrzjason/ObjectRemovalAlpha"

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "text": ds.Value("string"),
                "ground_truth_image": ds.Image(),
                "factual_image": ds.Image(),
                "mask_image": ds.Image(),
            }
        )

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, *args, **kwargs) -> List[ds.SplitGenerator]:
        dataset_path = snapshot_download(
            repo_id=self.ORIGINAL_DATASET_NAME, repo_type="dataset"
        )

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "dataset_path": dataset_path,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, dataset_path: Union[str, pathlib.Path]):
        if isinstance(dataset_path, str):
            dataset_path = pathlib.Path(dataset_path)

        txt_files = [f for f in dataset_path.iterdir() if f.suffix == ".txt"]
        indices = set([int(txt_file.stem.split("_")[0]) for txt_file in txt_files])

        for i in indices:
            # G, F, and M represent ground truth, factual, and mask, respectively.
            # ref. https://github.com/lrzjason/T2ITrainer/blob/main/doc/flux_fill.md#-training-dataset-requirements

            txt_g = (dataset_path / f"{i}_G.txt").read_text()
            txt_f = (dataset_path / f"{i}_F.txt").read_text()
            txt_m = (dataset_path / f"{i}_M.txt").read_text()
            assert txt_g == txt_f == txt_m

            img_g = Image.open(dataset_path / f"{i}_G.png")
            img_f = Image.open(dataset_path / f"{i}_F.png")
            img_m = Image.open(dataset_path / f"{i}_M.png")

            data = {
                "text": txt_g,
                "ground_truth_image": img_g,
                "factual_image": img_f,
                "mask_image": img_m,
            }
            yield i, data
