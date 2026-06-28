import json
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import datasets as ds
import pandas as pd
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage

logger = get_logger(__name__)

JsonDict = Dict[str, Any]


_DESCRIPTION = """\
PosterErase is a new dataset, which contains 60K high-resolution posters with texts and is more challenging for the text erasing task.
"""

_CITATION = """
@inproceedings{jiang2022self,
  title={Self-supervised text erasing with controllable image synthesis},
  author={Jiang, Gangwei and Wang, Shiyao and Ge, Tiezheng and Jiang, Yuning and Wei, Ying and Lian, Defu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1973--1983},
  year={2022}
}
"""

_HOMEPAGE = "https://github.com/alimama-creative/Self-supervised-Text-Erasing"

_LICENSE = """\
The upstream source has shown conflicting CC BY-SA 4.0 and CC BY-NC-SA 4.0
license indicators. This packaging uses the more restrictive CC BY-NC-SA 4.0
metadata and users should verify the current upstream terms before use.
"""

_URL_BASE = (
    "https://huggingface.co/datasets/shunk031/PosterErase-private/resolve/main/{}"
)
_ZIP_FILES = [f"erase_{i}.zip" for i in range(1, 7)]
_URLS = [_URL_BASE.format(zip_file) for zip_file in _ZIP_FILES]


def load_image(file_path: pathlib.Path) -> PilImage:
    return Image.open(file_path)


@dataclass
class ColorData(object):
    c1: Optional[int]
    c2: Optional[int]
    c3: Optional[int]

    @classmethod
    def from_string(cls, s: str) -> "ColorData":
        assert isinstance(s, str)
        cs = s.split(",")
        if len(cs) == 3:
            return ColorData(*list(map(lambda s: int(s), cs)))
        elif len(cs) == 1:
            return ColorData(*[None] * 3)
        else:
            raise ValueError(f"Invalid value: {cs}")


@dataclass
class TextData(object):
    x: int
    y: int
    cs: List[ColorData]

    @classmethod
    def from_text_tuple(cls, text_tuple: Tuple[int, int, List[str]]) -> "TextData":
        x, y, cs = text_tuple
        assert isinstance(x, int) and isinstance(y, int)
        return cls(x=x, y=y, cs=[ColorData.from_string(c) for c in cs])


@dataclass
class ObjectData(object):
    text: Optional[str]
    size: Optional[int]
    direction: Optional[int]

    @classmethod
    def from_string(cls, s: str) -> "ObjectData":
        assert isinstance(s, str)
        ss = s.split(",")
        if len(ss) == 3:
            return cls(text=ss[0], size=int(ss[1]), direction=int(ss[2]))
        elif len(ss) == 1:
            return cls(*[None] * 3)
        else:
            raise ValueError(f"Invalid value: {ss}")


@dataclass
class PlaceData(object):
    objs: List[ObjectData]
    texts: List[List[TextData]]

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "PlaceData":
        objs = [
            ObjectData.from_string(s) for s in json_dict["obj"].strip(";").split(";")
        ]
        texts = [
            [TextData.from_text_tuple(tt) for tt in tts] for tts in json_dict["text"]
        ]
        return cls(objs=objs, texts=texts)


@dataclass
class MaskData(object):
    x1: Optional[int]
    x2: Optional[int]
    y1: Optional[int]
    y2: Optional[int]

    @classmethod
    def from_string(cls, s: str) -> "MaskData":
        assert isinstance(s, str)
        ss = s.split(",")

        if len(ss) == 4:
            return cls(*list(map(lambda s: int(s), ss)))
        elif len(ss) == 1:
            return cls(*[None] * 4)
        else:
            raise ValueError(f"Invalid value: {ss}")


@dataclass
class Annotation(object):
    masks: List[MaskData]
    place: Optional[PlaceData]

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "Annotation":
        masks = [
            MaskData.from_string(s) for s in json_dict["mask"].strip(";").split(";")
        ]

        place_json = json_dict.get("place")
        place = (
            PlaceData.from_dict(json_dict["place"]) if place_json is not None else None
        )
        return cls(masks=masks, place=place)


@dataclass
class EraseData(object):
    number: int
    path: str
    annotation: Annotation

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "EraseData":
        number = int(json_dict["number"])
        path = json_dict["path"]
        annotation = Annotation.from_dict(json_dict["json"])
        return cls(number=number, path=path, annotation=annotation)


def _load_annotation(file_path: pathlib.Path, columns: List[str]) -> pd.DataFrame:
    df = pd.read_csv(file_path, delimiter="\t", names=columns)
    df["json"] = df["json"].apply(json.loads)
    return df


def _load_tng_annotation(file_path: pathlib.Path) -> pd.DataFrame:
    return _load_annotation(file_path=file_path, columns=["number", "path", "json"])


def _load_val_annotation(file_path: pathlib.Path) -> pd.DataFrame:
    return _load_annotation(
        file_path=file_path, columns=["number", "path", "json", "gt_path"]
    )


def _load_tst_annotation(file_path: pathlib.Path) -> pd.DataFrame:
    return _load_val_annotation(file_path=file_path)


class PosterErase(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [ds.BuilderConfig(version=VERSION, description=_DESCRIPTION)]

    @property
    def _manual_download_instructions(self) -> str:
        return (
            "To use PosterErase dataset, you need to download the dataset "
            "via [Alibaba Cloud](https://tianchi.aliyun.com/dataset/134810)."
        )

    def _info(self) -> ds.DatasetInfo:
        masks = ds.Sequence(
            {
                "x1": ds.Value("int32"),
                "x2": ds.Value("int32"),
                "y1": ds.Value("int32"),
                "y2": ds.Value("int32"),
            }
        )
        objs = ds.Sequence(
            {
                "text": ds.Value("string"),
                "size": ds.Value("int32"),
                "direction": ds.Value("int8"),
            }
        )
        color = {
            "c1": ds.Value("int32"),
            "c2": ds.Value("int32"),
            "c3": ds.Value("int32"),
        }
        text_feature = {
            "x": ds.Value("int32"),
            "y": ds.Value("int32"),
            "cs": ds.Sequence(color),
        }
        texts = ds.Sequence(ds.Sequence(text_feature))
        place = {"objs": objs, "texts": texts}
        annotation = {"masks": masks, "place": place}
        features = ds.Features(
            {
                "number": ds.Value("int32"),
                "path": ds.Value("string"),
                "image": ds.Image(),
                "gt_image": ds.Image(),
                "annotation": annotation,
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _download_from_hf(self, dl_manager: ds.DownloadManager) -> List[str]:
        return dl_manager.download_and_extract(_URLS)

    def _download_from_local(self, dl_manager: ds.DownloadManager) -> List[str]:
        assert dl_manager.manual_dir is not None, dl_manager.manual_dir
        dir_path = os.path.expanduser(dl_manager.manual_dir)

        if not os.path.exists(dir_path):
            raise FileNotFoundError(
                "Make sure you have downloaded and placed the PosterErase dataset correctly. "
                'Furthermore, you shoud check that a manual dir via `datasets.load_dataset("creative-graphic-design/PosterErase", data_dir=...)` '
                "that include zip files from the downloaded files. "
                f"Manual downloaded instructions: {self._manual_download_instructions}"
            )

        return dl_manager.extract(
            path_or_paths=[os.path.join(dir_path, zip_file) for zip_file in _ZIP_FILES]
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        base_dir_paths = (
            self._download_from_hf(dl_manager)
            if dl_manager.download_config.token
            else self._download_from_local(dl_manager)
        )
        dir_paths = [pathlib.Path(dir_path) for dir_path in base_dir_paths]
        dir_paths = [
            dir_path / f"erase_{i + 1}" for i, dir_path in enumerate(dir_paths)
        ]
        dir_path, *sub_dir_paths = dir_paths

        tng_df = _load_tng_annotation(dir_path / "train.txt")
        val_df = _load_val_annotation(dir_path / "ps_valid.txt")
        tst_df = _load_tst_annotation(dir_path / "ps_test.txt")

        tng_image_files = {
            f"{f.parent.name}/{f.name}": f for f in dir_path.glob("train/*.png")
        }
        val_image_files = {
            f"{f.parent.name}/{f.name}": f for f in dir_path.glob("valid/*.png")
        }
        val_gt_image_files = {
            f"{f.parent.name}/{f.name}": f for f in dir_path.glob("valid/*_gt.png")
        }
        tst_image_files = {
            f"{f.parent.name}/{f.name}": f for f in dir_path.glob("test/*.png")
        }
        tst_gt_image_files = {
            f"{f.parent.name}/{f.name}": f for f in dir_path.glob("test/*_gt.png")
        }
        for sub_dir_path in sub_dir_paths:
            tng_image_files.update(
                {
                    f"{f.parent.name}/{f.name}": f
                    for f in sub_dir_path.glob("train/*.png")
                }
            )
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "annotation_df": tng_df,
                    "image_files": tng_image_files,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={
                    "annotation_df": val_df,
                    "image_files": val_image_files,
                    "gt_image_files": val_gt_image_files,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={
                    "annotation_df": tst_df,
                    "image_files": tst_image_files,
                    "gt_image_files": tst_gt_image_files,
                },
            ),
        ]

    def _generate_examples(
        self,
        annotation_df: pd.DataFrame,
        image_files: Dict[str, pathlib.Path],
        gt_image_files: Optional[Dict[str, pathlib.Path]] = None,
    ):
        ann_dicts = annotation_df.to_dict(orient="records")
        for i, ann_dict in enumerate(ann_dicts):
            image_path = image_files[ann_dict["path"]]
            image = load_image(image_path)
            erase_data = EraseData.from_dict(json_dict=ann_dict)

            example = asdict(erase_data)
            example["image"] = image

            if gt_image_files is not None and "gt_path" in ann_dict:
                gt_image_path = gt_image_files[ann_dict["gt_path"]]
                gt_image = load_image(gt_image_path)
                example["gt_image"] = gt_image
            else:
                example["gt_image"] = None

            yield i, example


PosterEraseDataset = PosterErase
