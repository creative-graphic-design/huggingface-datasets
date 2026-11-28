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
from pathlib import Path

import datasets as ds
import resvg_py

_HOMEPAGE = "https://matanlevy.com/LiveSVG/"
_LICENSE = "cc-by-nc-4.0"
_URL = "https://matanlevy.com/LiveSVG/ChallengeSVG.zip"
_CITATION = """\
@article{levy2026livesvg,
  title={LiveSVG: Zero-Shot SVG Animation via Video Generation},
  author={Levy, Matan and Margolin, Ran and Cavia, Bar and Samuel, Dvir and Pritch, Yael and Peleg, Shmuel and Acha, Alex Rav and Shamir, Ariel and Lischinski, Dani},
  journal={arXiv preprint arXiv:2605.30174},
  year={2026}
}
"""
_DESCRIPTION = """\
ChallengeSVG is an evaluation-only collection of 35 SVG illustrations paired with English motion prompts. Each row includes the original SVG source and a 512 x 512 PNG rendering with transparency.
"""


def _render_svg(svg: str) -> bytes:
    return resvg_py.svg_to_bytes(
        svg_string=svg,
        width=512,
        height=512,
        background=None,
    )


class ChallengeSVG(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")

    def _info(self) -> ds.DatasetInfo:
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=ds.Features(
                {
                    "id": ds.Value("string"),
                    "prompt": ds.Value("string"),
                    "svg": ds.Value("string"),
                    "image": ds.Image(),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> list[ds.SplitGenerator]:
        extracted_path = dl_manager.download_and_extract(_URL)
        return [
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={"data_dir": str(Path(extracted_path) / "ChallengeSVG")},
            )
        ]

    def _generate_examples(self, data_dir: str):
        for svg_path in sorted(Path(data_dir).glob("*/*.svg")):
            item_id = svg_path.parent.name
            svg = svg_path.read_text(encoding="utf-8")
            prompt = svg_path.with_suffix(".txt").read_text(encoding="utf-8").strip()
            yield (
                item_id,
                {
                    "id": item_id,
                    "prompt": prompt,
                    "svg": svg,
                    "image": {"bytes": _render_svg(svg), "path": None},
                },
            )
