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
from typing import Iterable

import pandas as pd

import datasets as ds

_CITATION = """\
@article{lin2023designbench,
  title={DEsignBench: Exploring and Benchmarking DALL-E 3 for Imagining Visual Design},
  author={Kevin Lin and Zhengyuan Yang and Linjie Li and Jianfeng Wang and Lijuan Wang},
  journal={arXiv preprint arXiv:2310.15144},
  year={2023}
}
"""

_DESCRIPTION = """\
DEsignBench Prompts contains visual design text-to-image prompts from DEsignBench, \
including original user inputs, expanded prompts, and target aspect ratios.
"""

_HOMEPAGE = "https://design-bench.github.io/"
_LICENSE = "unknown"
_URLS = {
    "prompts": "https://design-bench.github.io/DesignBench_Prompts.tsv",
}


def _iter_prompt_rows(tsv_path: str) -> Iterable[tuple[int, dict[str, str]]]:
    df = pd.read_csv(tsv_path, sep="\t").fillna("")
    for index, row in df.iterrows():
        yield (
            index,
            {
                "id": str(row["demo_imgname"]),
                "demo_imgname": str(row["demo_imgname"]),
                "userinput": str(row["userinput"]),
                "expandprompt": str(row["expandprompt"]),
                "aspectratio": str(row["aspectratio"]),
            },
        )


class DEsignBenchPrompts(ds.GeneratorBasedBuilder):
    """A class for loading the DEsignBench prompt dataset."""

    VERSION = ds.Version("1.0.0")

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "id": ds.Value("string"),
                "demo_imgname": ds.Value("string"),
                "userinput": ds.Value("string"),
                "expandprompt": ds.Value("string"),
                "aspectratio": ds.Value("string"),
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
    ) -> list[ds.SplitGenerator]:
        tsv_path = dl_manager.download(_URLS["prompts"])
        return [
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={"tsv_path": tsv_path},
            ),
        ]

    def _generate_examples(self, tsv_path: str):
        yield from _iter_prompt_rows(tsv_path)
