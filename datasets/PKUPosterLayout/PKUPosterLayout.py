import ast
import os
import pathlib
import zipfile
from typing import List, Optional, TypedDict, Union, cast

import datasets as ds
import gdown
import pandas as pd
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger(__name__)

_DESCRIPTION = (
    "A New Dataset and Benchmark for Content-aware Visual-Textual Presentation Layout"
)

_CITATION = """\
@inproceedings{hsu2023posterlayout,
  title={PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}
"""

_HOMEPAGE = "http://59.108.48.34/tiki/PosterLayout/"

_LICENSE = "Images in PKU PosterLayout are distributed under the CC BY-SA 4.0 license."


class TrainPoster(TypedDict):
    original: str
    inpainted: str


class TestPoster(TypedDict):
    canvas: str


class SaliencyMaps(TypedDict):
    pfpn: str
    basnet: str


class TrainDataset(TypedDict):
    poster: TrainPoster
    saliency_maps: SaliencyMaps


class TestDataset(TypedDict):
    poster: TestPoster
    saliency_maps: SaliencyMaps


class Annotation(TypedDict):
    train: str


class DatasetUrls(TypedDict):
    train: TrainDataset
    test: TestDataset
    annotation: Annotation


class ArchiveFiles(TypedDict):
    train: TrainDataset
    test: TestDataset


_GOOGLE_DRIVE_FILE_IDS: ArchiveFiles = {
    "train": {
        "poster": {
            "original": "1u9LwWodBogUbgfNh6fUcScLK1Dgg3wjP",
            "inpainted": "1EJnAfqv5oIWj5f3MZ6N2Ee_8o_o54tKf",
        },
        "saliency_maps": {
            "pfpn": "1EnZCtzt10ZPgkQqFwFYG6XGI7E0pc2vT",
            "basnet": "1E3kRSv_oOtKQFu7xru1jmq11_7rXLyU6",
        },
    },
    "test": {
        "poster": {
            "canvas": "1hcXueYYh2iY5XLtyTZFsXUZsI5JwFnaT",
        },
        "saliency_maps": {
            "pfpn": "1FDRU-2FFZHK2IZe83Py469MCAydVRKzU",
            "basnet": "1rSsIvoPfkj1s9W2wMq2jSIFnZuw4iEo7",
        },
    },
}

_ANNOTATION_URLS: Annotation = {
    "train": "https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout-private/resolve/main/annotations/train_csv_9973.csv",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def download_google_drive_file(file_id: str, output_path: pathlib.Path) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verify = os.environ.get("GDOWN_VERIFY", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    gdown.download(
        id=file_id,
        output=str(output_path),
        quiet=False,
        resume=True,
        verify=verify,
    )
    if not output_path.exists():
        raise FileNotFoundError(f"Failed to download Google Drive file: {file_id}")
    return output_path


def file_sorter(f: pathlib.Path) -> int:
    idx, *_ = f.stem.split("_")
    return int(idx)


def load_image(file_path: pathlib.Path) -> PilImage:
    logger.info(f"Load from {file_path}")
    return Image.open(file_path)


def get_original_poster_files(base_dir: str) -> List[pathlib.Path]:
    poster_dir = pathlib.Path(base_dir) / "original_poster"
    return sorted(poster_dir.iterdir(), key=lambda f: int(f.stem))


def get_inpainted_poster_files(base_dir: str) -> List[pathlib.Path]:
    inpainted_dir = pathlib.Path(base_dir) / "inpainted_poster"
    return sorted(inpainted_dir.iterdir(), key=file_sorter)


def get_basnet_map_files(base_dir: str) -> List[pathlib.Path]:
    basnet_map_dir = pathlib.Path(base_dir) / "saliencymaps_basnet"
    return sorted(basnet_map_dir.iterdir(), key=file_sorter)


def get_pfpn_map_files(base_dir: str) -> List[pathlib.Path]:
    pfpn_map_dir = pathlib.Path(base_dir) / "saliencymaps_pfpn"
    return sorted(pfpn_map_dir.iterdir(), key=file_sorter)


def get_canvas_files(base_dir: str) -> List[pathlib.Path]:
    canvas_dir = pathlib.Path(base_dir) / "image_canvas"
    return sorted(canvas_dir.iterdir(), key=lambda f: int(f.stem))


class PKUPosterLayout(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [ds.BuilderConfig(version=VERSION)]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "original_poster": ds.Image(),
                "inpainted_poster": ds.Image(),
                "basnet_saliency_map": ds.Image(),
                "pfpn_saliency_map": ds.Image(),
                "canvas": ds.Image(),
                "annotations": ds.Sequence(
                    {
                        "poster_path": ds.Value("string"),
                        "total_elem": ds.Value("int32"),
                        "cls_elem": ds.ClassLabel(
                            num_classes=4, names=["text", "logo", "underlay", "INVALID"]
                        ),
                        "box_elem": ds.Sequence(ds.Value("int32")),
                    }
                ),
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        file_paths = self._download_and_extract(dl_manager)

        tng_files = file_paths["train"]  # type: ignore
        tst_files = file_paths["test"]  # type: ignore
        ann_file = file_paths["annotation"]  # type: ignore

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "poster": tng_files["poster"],
                    "saliency_maps": tng_files["saliency_maps"],
                    "annotation": ann_file["train"],
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={
                    "poster": tst_files["poster"],
                    "saliency_maps": tst_files["saliency_maps"],
                },
            ),
        ]

    def _download_and_extract(self, dl_manager: ds.DownloadManager) -> DatasetUrls:
        cache_dir = pathlib.Path(
            dl_manager.download_config.cache_dir or ds.config.DOWNLOADED_DATASETS_PATH
        )
        archive_paths = self._download_archive_files(cache_dir)
        extracted_paths = dl_manager.extract(
            {
                "train": archive_paths["train"],
                "test": archive_paths["test"],
            }
        )
        assert isinstance(extracted_paths, dict)
        extracted_paths["annotation"] = {
            "train": dl_manager.download(_ANNOTATION_URLS["train"])
        }
        return extracted_paths  # type: ignore[return-value]

    def _download_archive_files(self, cache_dir: pathlib.Path) -> ArchiveFiles:
        output_dir = cache_dir / "pku_poster_layout"
        archive_paths: ArchiveFiles = {
            "train": {
                "poster": {
                    "original": str(
                        self._download_archive_file(
                            _GOOGLE_DRIVE_FILE_IDS["train"]["poster"]["original"],
                            output_dir / "train" / "original_poster.zip",
                        )
                    ),
                    "inpainted": str(
                        self._download_archive_file(
                            _GOOGLE_DRIVE_FILE_IDS["train"]["poster"]["inpainted"],
                            output_dir / "train" / "inpainted_poster.zip",
                        )
                    ),
                },
                "saliency_maps": {
                    "pfpn": str(
                        self._download_archive_file(
                            _GOOGLE_DRIVE_FILE_IDS["train"]["saliency_maps"]["pfpn"],
                            output_dir / "train" / "saliencymaps_pfpn.zip",
                        )
                    ),
                    "basnet": str(
                        self._download_archive_file(
                            _GOOGLE_DRIVE_FILE_IDS["train"]["saliency_maps"]["basnet"],
                            output_dir / "train" / "saliencymaps_basnet.zip",
                        )
                    ),
                },
            },
            "test": {
                "poster": {
                    "canvas": str(
                        self._download_archive_file(
                            _GOOGLE_DRIVE_FILE_IDS["test"]["poster"]["canvas"],
                            output_dir / "test" / "image_canvas.zip",
                        )
                    ),
                },
                "saliency_maps": {
                    "pfpn": str(
                        self._download_archive_file(
                            _GOOGLE_DRIVE_FILE_IDS["test"]["saliency_maps"]["pfpn"],
                            output_dir / "test" / "saliencymaps_pfpn.zip",
                        )
                    ),
                    "basnet": str(
                        self._download_archive_file(
                            _GOOGLE_DRIVE_FILE_IDS["test"]["saliency_maps"]["basnet"],
                            output_dir / "test" / "saliencymaps_basnet.zip",
                        )
                    ),
                },
            },
        }

        return archive_paths

    def _download_archive_file(
        self, file_id: str, output_path: pathlib.Path
    ) -> pathlib.Path:
        if output_path.exists() and zipfile.is_zipfile(output_path):
            return output_path
        logger.info(f"Downloading PKU PosterLayout archive to {output_path}.")
        archive_path = download_google_drive_file(file_id, output_path)
        if not zipfile.is_zipfile(archive_path):
            raise zipfile.BadZipFile(
                f"Downloaded file is not a valid zip: {archive_path}"
            )
        return archive_path

    def _generate_train_examples(
        self,
        poster: TrainPoster,
        saliency_maps: SaliencyMaps,
        annotation: Optional[str],
    ):
        if annotation:
            ann_df = pd.read_csv(annotation)

            ann_df = ann_df.assign(
                # Convert string to list
                box_elem=ann_df["box_elem"].apply(ast.literal_eval),
                # Since PKU's label is 1-indexed, we need to convert it to 0-indexed
                cls_elem=ann_df["cls_elem"] - 1,
            )
            ann_df = ann_df.assign(
                cls_elem=ann_df["cls_elem"].replace(
                    #
                    # Convert class index to class name.
                    #
                    # The index = -1 produced by the conversion from 1-indexed to 0-indexed
                    # is treated here as an INVALID class.
                    #
                    {-1: "INVALID", 0: "text", 1: "logo", 2: "underlay"}
                )
            )
        else:
            ann_df = None

        poster_files = get_original_poster_files(base_dir=poster["original"])
        inpainted_files = get_inpainted_poster_files(base_dir=poster["inpainted"])

        basnet_map_files = get_basnet_map_files(base_dir=saliency_maps["basnet"])
        pfpn_map_files = get_pfpn_map_files(base_dir=saliency_maps["pfpn"])

        assert (
            len(poster_files)
            == len(inpainted_files)
            == len(basnet_map_files)
            == len(pfpn_map_files)
        )

        it = zip(poster_files, inpainted_files, basnet_map_files, pfpn_map_files)
        for i, (
            original_poster_path,
            inpainted_poster_path,
            basnet_map_path,
            pfpn_map_path,
        ) in enumerate(it):
            poster_path = f"train/{original_poster_path.name}"
            annotations = (
                ann_df[ann_df["poster_path"] == poster_path].to_dict(orient="records")
                if ann_df is not None
                else []
            )

            yield (
                i,
                {
                    "original_poster": load_image(original_poster_path),
                    "inpainted_poster": load_image(inpainted_poster_path),
                    "basnet_saliency_map": load_image(basnet_map_path),
                    "pfpn_saliency_map": load_image(pfpn_map_path),
                    "canvas": None,
                    "annotations": annotations,
                },
            )

    def _generate_test_examples(self, poster: TestPoster, saliency_maps: SaliencyMaps):
        canvas_files = get_canvas_files(base_dir=poster["canvas"])

        basnet_map_files = get_basnet_map_files(base_dir=saliency_maps["basnet"])
        pfpn_map_files = get_pfpn_map_files(base_dir=saliency_maps["pfpn"])

        assert len(canvas_files) == len(basnet_map_files) == len(pfpn_map_files)
        it = zip(canvas_files, basnet_map_files, pfpn_map_files)
        for i, (canvas_path, basnet_map_path, pfpn_map_path) in enumerate(it):
            yield (
                i,
                {
                    "original_poster": None,
                    "inpainted_poster": None,
                    "basnet_saliency_map": load_image(basnet_map_path),
                    "pfpn_saliency_map": load_image(pfpn_map_path),
                    "canvas": load_image(canvas_path),
                    "annotations": None,
                },
            )

    def _generate_examples(
        self,
        poster: Union[TrainPoster, TestPoster],
        saliency_maps: SaliencyMaps,
        annotation: Optional[str] = None,
    ):
        if "original" in poster and "inpainted" in poster:
            yield from self._generate_train_examples(
                poster=cast(TrainPoster, poster),
                saliency_maps=saliency_maps,
                annotation=annotation,
            )
        elif "canvas" in poster:
            yield from self._generate_test_examples(
                poster=cast(TestPoster, poster),
                saliency_maps=saliency_maps,
            )
        else:
            raise ValueError("Invalid dataset")


PosterLayoutDataset = PKUPosterLayout
