# Copyright 2025 Shunsuke Kitada and the current dataset script contributor.
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
from typing import List

from datasets.utils.logging import get_logger

import datasets as ds

logger = get_logger(__name__)

_CITATION = r"""\
@article{zhao2018modeling,
  title={Modeling Fonts in Context: Font Prediction on Web Designs},
  author={Zhao, Nanxuan and Cao, Ying and Lau, Rynson W.H.},
  journal={Computer Graphics Forum},
  volume={37},
  number={7},
  year={2018},
  publisher={The Eurographics Association and John Wiley \& Sons Ltd.}
}
"""

_DESCRIPTION = """\
CTXFont (Context Font) dataset is a collection of 1,065 professional web designs from awwwards.com with annotations for 4,893 text elements. \
Each text element is annotated with font properties (font face, color, size), HTML tags, and bounding boxes. \
The dataset contains 492 unique font faces and is designed for studying font prediction in the context of web design. \
Web designs were captured at 768×1366 resolution and include design tags describing their characteristics.
"""

_HOMEPAGE = "https://github.com/nanxuanzhao/CTXFont-dataset"

# License information is not explicitly stated in the repository
# Citation is recommended when using the dataset
_LICENSE = "Unknown"

# URLs to the raw dataset files on GitHub
_URLS = {
    "annotations": "https://raw.githubusercontent.com/nanxuanzhao/CTXFont-dataset/master/CTXFont_dataset-PG18/Annotations/annotations.mat",
    "screenshots": "https://github.com/nanxuanzhao/CTXFont-dataset/archive/refs/heads/master.zip",
}


class CTXFont(ds.GeneratorBasedBuilder):
    """CTXFont dataset with font property annotations for web design text elements."""

    VERSION = ds.Version("1.0.0")

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                # Design information
                "design_name": ds.Value("string"),
                "design_image": ds.Image(),
                "design_url": ds.Value("string"),
                "awwward_url": ds.Value("string"),
                "design_tags": ds.Sequence(ds.Value("uint8"), length=54),
                # Text element properties
                "text_content": ds.Value("string"),
                "html_tags": ds.Sequence(ds.Value("uint8"), length=10),
                # Font properties
                "font_face": ds.Value("string"),
                "font_size": ds.Value("float32"),
                "font_color_r": ds.Value("uint8"),
                "font_color_g": ds.Value("uint8"),
                "font_color_b": ds.Value("uint8"),
                "font_color_a": ds.Value("uint8"),
                "font_face_embedding": ds.Sequence(ds.Value("float32"), length=40),
                # Element position and size
                "center_x": ds.Value("uint16"),
                "center_y": ds.Value("uint16"),
                "width": ds.Value("float32"),
                "height": ds.Value("uint16"),
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
        # Download annotations.mat file and screenshots
        annotations_path = dl_manager.download(_URLS["annotations"])
        screenshots_archive = dl_manager.download_and_extract(_URLS["screenshots"])

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "annotations_path": annotations_path,
                    "screenshots_dir": screenshots_archive,
                    "split": "train",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={
                    "annotations_path": annotations_path,
                    "screenshots_dir": screenshots_archive,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, annotations_path, screenshots_dir, split):
        import os

        import scipy.io

        # Load the .mat file
        mat_data = scipy.io.loadmat(annotations_path)

        # Construct path to screenshots directory
        screenshots_path = os.path.join(
            screenshots_dir,
            "CTXFont-dataset-master",
            "CTXFont_dataset-PG18",
            "Screenshots",
        )

        # Get split-specific design names
        if split == "train":
            split_design_names = set(
                [name[0][0] for name in mat_data["selected_design_name_train"]]
            )
        else:  # test
            split_design_names = set(
                [name[0][0] for name in mat_data["selected_design_name_test"]]
            )

        # Iterate through all examples
        idx = 0
        for i in range(len(mat_data["selected_design_name"])):
            design_name = mat_data["selected_design_name"][i, 0][0]

            # Filter by split
            if design_name not in split_design_names:
                continue

            # Construct image path
            image_path = os.path.join(screenshots_path, design_name)

            yield (
                idx,
                {
                    # Design information
                    "design_name": design_name,
                    "design_image": image_path,
                    "design_url": mat_data["selected_design_url"][i, 0][0],
                    "awwward_url": mat_data["selected_awwward_url"][i, 0][0],
                    "design_tags": mat_data["selected_design_tags"][i].tolist(),
                    # Text element properties
                    "text_content": mat_data["selected_element_text_content"][i, 0][0],
                    "html_tags": mat_data["selected_element_html_tags"][i].tolist(),
                    # Font properties
                    "font_face": mat_data["selected_element_font_face"][i, 0][0],
                    "font_size": float(mat_data["selected_element_font_size"][i, 0]),
                    "font_color_r": int(mat_data["selected_element_color"][i, 0]),
                    "font_color_g": int(mat_data["selected_element_color"][i, 1]),
                    "font_color_b": int(mat_data["selected_element_color"][i, 2]),
                    "font_color_a": int(mat_data["selected_element_color"][i, 3]),
                    "font_face_embedding": mat_data[
                        "selected_element_font_face_embedding"
                    ][i].tolist(),
                    # Element position and size
                    "center_x": int(mat_data["selected_element_center_position"][i, 0]),
                    "center_y": int(mat_data["selected_element_center_position"][i, 1]),
                    "width": float(mat_data["selected_element_width"][i, 0]),
                    "height": int(mat_data["selected_element_height"][i, 0]),
                },
            )
            idx += 1
