from typing import Final, List

import datasets as ds
import pyarrow.parquet as pq

_CITATION = """\
@article{xiao2024desigen,
  title={Desigen: A Pipeline for Controllable Design Template Generation},
  author={Xiao, Shishi and Wang, Yufei and Zhou, Rui and Hao, Haohan and Chen, Kai and Chen, Xi and Wei, Zhongyu},
  journal={arXiv preprint arXiv:2403.09093},
  year={2024}
}
"""

_DESCRIPTION = """\
Desigen web design data with advertisement banner backgrounds, text prompts, and layout metadata.
"""

_HOMEPAGE = "https://whaohan.github.io/desigen/"
_LICENSE = "unknown"
_BASE_URL = (
    "https://huggingface.co/datasets/creative-graphic-design/Desigen/resolve/main"
)

_DATA_URLS = {
    "train": [
        f"{_BASE_URL}/data/train-{shard_id:05d}-of-00029.parquet"
        for shard_id in range(29)
    ],
    "validation": [f"{_BASE_URL}/data/validation-00000-of-00001.parquet"],
}

LAYOUT_CLASS_LABELS: Final[List[str]] = [
    "background",
    "button",
    "email",
    "image",
    "link-button",
    "number",
    "password",
    "radio",
    "range",
    "search",
    "select",
    "static-text",
    "submit",
    "tel",
    "text",
    "textarea",
]


class DesigenDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [ds.BuilderConfig(version=VERSION, description=_DESCRIPTION)]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "image": ds.Image(),
                "prompt": ds.Value("string"),
                "region": ds.Sequence(ds.Sequence(ds.Value("int64"), length=4)),
                "description": ds.Value("string"),
                "elements": ds.Sequence(
                    {
                        "position": ds.Sequence(ds.Value("int64"), length=4),
                        "text": ds.Value("string"),
                        "type": ds.ClassLabel(
                            num_classes=len(LAYOUT_CLASS_LABELS),
                            names=LAYOUT_CLASS_LABELS,
                        ),
                    }
                ),
                "size": ds.Sequence(ds.Value("int64"), length=2),
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
        data_files = dl_manager.download(_DATA_URLS)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"parquet_paths": data_files["train"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"parquet_paths": data_files["validation"]},
            ),
        ]

    def _generate_examples(self, parquet_paths: List[str]):
        idx = 0
        for parquet_path in parquet_paths:
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    yield idx, row
                    idx += 1


Desigen = DesigenDataset
