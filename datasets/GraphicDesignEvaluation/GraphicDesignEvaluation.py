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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import pandas as pd
from datasets.utils.logging import get_logger
from PIL import Image

import datasets as ds

logger = get_logger(__name__)


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{haraguchi2024can,
  title={Can GPTs Evaluate Graphic Design Based on Design Principles?},
  author={Haraguchi, Daichi and Inoue, Naoto and Shimoda, Wataru and Mitani, Hayato and Uchida, Seiichi and Yamaguchi, Kota},
  booktitle={SIGGRAPH Asia 2024 Technical Communications},
  pages={1--4},
  year={2024}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This human-rated dataset is a collection of **700 graphic banner designs** built to study the reliability of graphic design evaluation, focusing on three fundamental design principles: **alignment, overlap, and white space**. The dataset comprises 100 original designs and **600 perturbed samples** artificially generated for evaluation, featuring human annotations collected from 60 participants who provided scores ranging from **1 to 10** for each design."""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/CyberAgentAILab/Graphic-design-evaluation"

# TODO: Add the license for the dataset here if you can find it
_LICENSE = "apache-2.0"

_URLS = {
    "images": "https://github.com/CyberAgentAILab/Graphic-design-evaluation/raw/refs/heads/main/GPT_eval_data/images.zip",
    "absolute": {
        "gpt": {
            "alignment": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/gpt_eval/gpt_abs_alignment.csv",
            "overlap": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/gpt_eval/gpt_abs_overlap.csv",
            "whitespace": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/gpt_eval/gpt_abs_whitespace.csv",
        },
        "human": {
            "alignment": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/human_eval/human_abs_alignment.csv",
            "overlap": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/human_eval/human_abs_overlap.csv",
            "whitespace": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/human_eval/human_abs_whitespace.csv",
        },
    },
    "relative": {
        "gpt": {
            "alignment": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/gpt_eval/gpt_comp_alignment.csv",
            "overlap": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/gpt_eval/gpt_comp_overlap.csv",
            "whitespace": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/gpt_eval/gpt_comp_whitespace.csv",
        },
        "human": {
            "alignment": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/human_eval/human_comp_alignment.csv",
            "overlap": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/human_eval/human_comp_overlap.csv",
            "whitespace": "https://raw.githubusercontent.com/CyberAgentAILab/Graphic-design-evaluation/refs/heads/main/GPT_eval_data/human_eval/human_comp_whitespace.csv",
        },
    },
}


DesignPrinciple = Literal["alignment", "overlap", "whitespace"]

EvalType = Literal["absolute", "relative"]
AnnotationType = Literal["gpt", "human"]


@dataclass(kw_only=True)
class GraphicDesignEvaluationConfig(ds.BuilderConfig):
    eval_type: EvalType
    annotation_type: AnnotationType
    design_principle: DesignPrinciple

    def __post_init__(self) -> None:
        self.name = f"{self.eval_type}-{self.annotation_type}-{self.design_principle}"


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class GraphicDesignEvaluationDataset(ds.GeneratorBasedBuilder):
    """A class for loading GraphicDesignEvaluation dataset."""

    config: GraphicDesignEvaluationConfig

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = GraphicDesignEvaluationConfig
    BUILDER_CONFIGS = [
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="absolute",
            annotation_type="gpt",
            design_principle="alignment",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="absolute",
            annotation_type="gpt",
            design_principle="overlap",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="absolute",
            annotation_type="gpt",
            design_principle="whitespace",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="absolute",
            annotation_type="human",
            design_principle="alignment",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="absolute",
            annotation_type="human",
            design_principle="overlap",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="absolute",
            annotation_type="human",
            design_principle="whitespace",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="relative",
            annotation_type="gpt",
            design_principle="alignment",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="relative",
            annotation_type="gpt",
            design_principle="overlap",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="relative",
            annotation_type="gpt",
            design_principle="whitespace",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="relative",
            annotation_type="human",
            design_principle="alignment",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="relative",
            annotation_type="human",
            design_principle="overlap",
        ),
        GraphicDesignEvaluationConfig(
            version=VERSION,
            eval_type="relative",
            annotation_type="human",
            design_principle="whitespace",
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        if self.config.eval_type == "absolute":
            features = ds.Features(
                {
                    "image_id": ds.Value("string"),
                    "image": ds.Image(),
                    "perturbation": ds.ClassLabel(
                        names=["none", "small", "medium", "large"],
                    ),
                    "scores": ds.Sequence(ds.Value("int32")),
                    "avg": ds.Value("float32"),
                }
            )
        elif self.config.eval_type == "relative":
            features = ds.Features(
                {
                    "image_id": ds.Value("string"),
                    "image": ds.Image(),
                    "comparative": ds.ClassLabel(
                        names=["small", "medium", "large"],
                    ),
                    "avg": ds.Value("string"),
                    "scores": ds.Sequence(ds.Value("string")),
                }
            )

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _preprocess_absolute_evaluation_df(
        self,
        csv_path: str,
        score_columns: Sequence[str] = ("0", "1", "2", "3", "4"),
    ) -> pd.DataFrame:
        score_columns = list(score_columns)
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=["id", "perturbation"])

        def convert_to_scores(xs):
            if xs.isnull().any():
                logger.warning(f"Found NaN values in scores for {self.config=}")
                xs = xs.dropna()  # Drop NaN values
            return list(xs)

        # Combine the target columns into a list and store it in a new column "scores"
        df["scores"] = df[score_columns].apply(convert_to_scores, axis=1)

        # If the first column is "Unnamed: 0", include it in the columns to be removed
        if "Unnamed: 0" in df.columns:
            score_columns = ["Unnamed: 0"] + score_columns

        # Remove the target columns
        df = df.drop(columns=score_columns)

        return df

    def _preprocess_relative_evaluation_df(
        self,
        csv_path: str,
        target_columns: Sequence[str] = (
            "better_design_0",
            "better_design_1",
            "better_design_2",
            "better_design_3",
            "better_design_4",
        ),
    ) -> pd.DataFrame:
        target_columns = list(target_columns)

        df = pd.read_csv(csv_path)
        df = df.sort_values(by=["id", "comparative"])

        # Combine the target columns into a list and store it in a new column "scores"
        df["scores"] = df[target_columns].apply(list, axis=1)

        # If the first column is "Unnamed: 0", include it in the columns to be removed
        if "Unnamed: 0" in df.columns:
            target_columns = ["Unnamed: 0"] + target_columns

        # Remove the target columns
        df = df.drop(columns=target_columns)

        return df

    def load_evaluation_df(self, csv_path: str) -> pd.DataFrame:
        if self.config.eval_type == "absolute":
            return self._preprocess_absolute_evaluation_df(csv_path=csv_path)
        elif self.config.eval_type == "relative":
            return self._preprocess_relative_evaluation_df(csv_path=csv_path)
        else:
            raise ValueError(f"Unknown eval_type: {self.config.eval_type}")

    def load_image_paths(
        self, images_dir: pathlib.Path
    ) -> Dict[str, Dict[str, pathlib.Path]]:
        def get_glob_pattern(design_principle: DesignPrinciple) -> str:
            if design_principle == "alignment":
                return "**/lefttop_*/*.png"
            elif design_principle in ("overlap", "whitespace"):
                return "**/size_*/*.png"
            else:
                raise ValueError(f"Unknown design_principle: {design_principle}")

        # left-top perturbations for "alignment" and
        # size-based perturbations for "overlap" and "whitespace"
        glob_pat = get_glob_pattern(self.config.design_principle)

        image_paths = sorted(
            # original images
            list(images_dir.glob("**/org/*.png"))
            # perturbed images for the target design principle
            + list(images_dir.glob(glob_pat)),
            key=lambda p: (p.stem, p.parent.stem),
        )

        image_paths_dict: Dict[str, Dict[str, pathlib.Path]] = defaultdict(dict)

        for image_path in image_paths:
            dirname = image_path.parent.name
            if dirname == "org":
                perturbation_element, perturbation_size = "none", "none"
            else:
                perturbation_element, perturbation_size = dirname.split("_")

                if self.config.design_principle == "alignment":
                    assert perturbation_element == "lefttop"
                elif self.config.design_principle in ("overlap", "whitespace"):
                    assert perturbation_element == "size"
                else:
                    raise ValueError(
                        f"Unknown design_principle: {self.config.design_principle}"
                    )

            image_paths_dict[image_path.stem][perturbation_size] = image_path

        return image_paths_dict

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        # Prepare CSV URL
        eval_type_csv_dict = _URLS[self.config.eval_type]
        ann_type_csv_dict = eval_type_csv_dict[self.config.annotation_type]
        csv_url = ann_type_csv_dict[self.config.design_principle]

        # Load evaluation DataFrame from the URL
        csv_path = dl_manager.download(csv_url)
        assert isinstance(csv_path, str)
        df_eval = self.load_evaluation_df(csv_path=csv_path)

        # Load target images
        images_base_dir = dl_manager.download_and_extract(_URLS["images"])
        assert isinstance(images_base_dir, str)
        image_paths_dict = self.load_image_paths(
            images_dir=pathlib.Path(images_base_dir) / "images"
        )

        # Return the split generators
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "image_paths_dict": image_paths_dict,
                    "df_eval": df_eval,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
        self,
        image_paths_dict: Dict[str, Dict[str, pathlib.Path]],
        df_eval: pd.DataFrame,
    ):
        def get_perturbation_column_name() -> str:
            if self.config.eval_type == "absolute":
                return "perturbation"
            elif self.config.eval_type == "relative":
                return "comparative"
            else:
                raise ValueError(f"Unknown eval_type: {self.config.eval_type}")

        for i in range(len(df_eval)):
            row = df_eval.iloc[i].to_dict()
            row["image_id"] = row.pop("id")  # rename "id" to "image_id"
            image_id = row["image_id"]

            perturbation_column = get_perturbation_column_name()
            perturbation = row[perturbation_column]

            image_path = image_paths_dict[image_id][perturbation]

            image = Image.open(image_path)
            row["image"] = image

            yield i, row
