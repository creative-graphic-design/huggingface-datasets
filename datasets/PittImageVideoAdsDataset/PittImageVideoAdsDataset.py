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
import csv
import json
import pathlib
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Iterable, List, assert_never
from urllib.parse import quote

import datasets as ds

_CITATION = """\
@inproceedings{hussain2017automatic,
  title={Automatic Understanding of Image and Video Advertisements},
  author={Hussain, Zaeem and Zhang, Mingda and Zhang, Xiaozhong and Ye, Keren and Thomas, Christopher and Agha, Zuha and Ong, Nathan and Kovashka, Adriana},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1705--1715},
  year={2017}
}
"""

_DESCRIPTION = """\
PittImageVideoAdsDataset contains image and video advertisement annotations released with \
Automatic Understanding of Image and Video Advertisements. It includes image-ad URLs and \
annotations for topics, sentiments, slogans, persuasive strategies, symbolic references, and \
action/reason Q/A, plus YouTube video IDs with raw and majority-vote annotations for topics, \
sentiments, language, effectiveness, excitement, funniness, and action/reason Q/A.
"""

_HOMEPAGE = "https://people.cs.pitt.edu/~kovashka/ads/"

_LICENSE = "unknown"

_ORIGINAL_IMAGE_BASE = "https://people.cs.pitt.edu/~mzhang/image_ads"

_URLS = {
    "image_ads": {
        "annotations": "https://people.cs.pitt.edu/~kovashka/ads/annotations_images.zip",
        "images": [
            f"https://storage.googleapis.com/ads-dataset/subfolder-{index}.zip"
            for index in range(11)
        ],
    },
    "video_ads": "https://people.cs.pitt.edu/~kovashka/ads/annotations_videos.zip",
}


class PittImageVideoAdsDatasetType(StrEnum):
    image_ads = auto()
    video_ads = auto()


@dataclass
class PittImageVideoAdsDatasetConfig(ds.BuilderConfig):
    name: PittImageVideoAdsDatasetType

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PittImageVideoAdsDatasetType(self.name)


def _as_path(value: str | pathlib.Path) -> pathlib.Path:
    return value if isinstance(value, pathlib.Path) else pathlib.Path(value)


def _load_json(path: str | pathlib.Path) -> dict[str, Any]:
    with _as_path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def _load_video_ids(path: str | pathlib.Path) -> list[str]:
    with _as_path(path).open("r", encoding="utf-8") as f:
        return [
            row[0].strip().strip("'").strip('"')
            for row in csv.reader(f)
            if row and row[0].strip()
        ]


def _image_url(source_path: str) -> str:
    return f"{_ORIGINAL_IMAGE_BASE}/{quote(source_path, safe='/')}"


def _image_annotation_paths(extracted_dir: str | pathlib.Path) -> dict[str, str]:
    image_dir = _as_path(extracted_dir) / "image"
    return {
        "qa_action": str(image_dir / "QA_Action.json"),
        "qa_reason": str(image_dir / "QA_Reason.json"),
        "qa_combined_action_reason": str(image_dir / "QA_Combined_Action_Reason.json"),
        "sentiments": str(image_dir / "Sentiments.json"),
        "slogans": str(image_dir / "Slogans.json"),
        "strategies": str(image_dir / "Strategies.json"),
        "symbols": str(image_dir / "Symbols.json"),
        "topics": str(image_dir / "Topics.json"),
    }


def _video_annotation_paths(extracted_dir: str | pathlib.Path) -> dict[str, str]:
    video_dir = _as_path(extracted_dir) / "video"
    raw_dir = video_dir / "raw_result"
    clean_dir = video_dir / "cleaned_result"
    return {
        "video_ids": str(video_dir / "final_video_id_list.csv"),
        "raw_qa_action": str(raw_dir / "video_QA_Action_raw.json"),
        "raw_qa_reason": str(raw_dir / "video_QA_Reason_raw.json"),
        "raw_topics": str(raw_dir / "video_Topics_raw.json"),
        "raw_sentiments": str(raw_dir / "video_Sentiments_raw.json"),
        "raw_funny": str(raw_dir / "video_Funny_raw.json"),
        "raw_exciting": str(raw_dir / "video_Exciting_raw.json"),
        "raw_language": str(raw_dir / "video_Language_raw.json"),
        "clean_topic": str(clean_dir / "video_Topics_clean.json"),
        "clean_sentiment": str(clean_dir / "video_Sentiments_clean.json"),
        "clean_funny": str(clean_dir / "video_Funny_clean.json"),
        "clean_exciting": str(clean_dir / "video_Exciting_clean.json"),
        "clean_language": str(clean_dir / "video_Language_clean.json"),
        "clean_effective": str(clean_dir / "video_Effective_clean.json"),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _nested_string_list(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    return [_string_list(items) for items in value]


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _symbols(value: Any) -> list[dict[str, float | str]]:
    if not isinstance(value, list):
        return []

    symbols = []
    for item in value:
        if not isinstance(item, list) or len(item) < 5:
            continue
        symbols.append(
            {
                "x1": float(item[0]),
                "y1": float(item[1]),
                "x2": float(item[2]),
                "y2": float(item[3]),
                "label": str(item[4]),
            }
        )
    return symbols


def _resolve_local_image_path(
    image_roots: dict[str, str],
    source_path: str,
) -> str:
    subfolder = pathlib.PurePosixPath(source_path).parts[0]
    return str(_as_path(image_roots[subfolder]).joinpath(*source_path.split("/")))


def _iter_image_examples(
    paths: dict[str, str],
    image_roots: dict[str, str],
) -> Iterable[tuple[str, dict[str, Any]]]:
    qa_action = _load_json(paths["qa_action"])
    qa_reason = _load_json(paths["qa_reason"])
    qa_combined_action_reason = _load_json(paths["qa_combined_action_reason"])
    sentiments = _load_json(paths["sentiments"])
    slogans = _load_json(paths["slogans"])
    strategies = _load_json(paths["strategies"])
    symbols = _load_json(paths["symbols"])
    topics = _load_json(paths["topics"])

    source_paths = sorted(
        set().union(
            qa_action,
            qa_reason,
            qa_combined_action_reason,
            sentiments,
            slogans,
            strategies,
            symbols,
            topics,
        )
    )

    for index, source_path in enumerate(source_paths):
        yield (
            f"image-{index:05d}",
            {
                "id": f"image-{index:05d}",
                "source_path": source_path,
                "image": _resolve_local_image_path(image_roots, source_path),
                "image_url": _image_url(source_path),
                "qa_action": _string_list(qa_action.get(source_path, [])),
                "qa_reason": _string_list(qa_reason.get(source_path, [])),
                "qa_combined_action_reason": _string_list(
                    qa_combined_action_reason.get(source_path, [])
                ),
                "slogans": _string_list(slogans.get(source_path, [])),
                "topics": _string_list(topics.get(source_path, [])),
                "sentiments": _nested_string_list(sentiments.get(source_path, [])),
                "strategies": _nested_string_list(strategies.get(source_path, [])),
                "symbols": _symbols(symbols.get(source_path, [])),
            },
        )


def _int_or_default(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    return int(value)


def _float_or_default(value: Any, default: float = -1.0) -> float:
    if value is None:
        return default
    return float(value)


def _iter_video_examples(paths: dict[str, str]) -> Iterable[tuple[str, dict[str, Any]]]:
    video_ids = _load_video_ids(paths["video_ids"])
    raw_qa_action = _load_json(paths["raw_qa_action"])
    raw_qa_reason = _load_json(paths["raw_qa_reason"])
    raw_topics = _load_json(paths["raw_topics"])
    raw_sentiments = _load_json(paths["raw_sentiments"])
    raw_funny = _load_json(paths["raw_funny"])
    raw_exciting = _load_json(paths["raw_exciting"])
    raw_language = _load_json(paths["raw_language"])
    clean_topic = _load_json(paths["clean_topic"])
    clean_sentiment = _load_json(paths["clean_sentiment"])
    clean_funny = _load_json(paths["clean_funny"])
    clean_exciting = _load_json(paths["clean_exciting"])
    clean_language = _load_json(paths["clean_language"])
    clean_effective = _load_json(paths["clean_effective"])

    for index, video_id in enumerate(video_ids):
        yield (
            f"video-{index:05d}",
            {
                "id": f"video-{index:05d}",
                "youtube_id": video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "raw_qa_action": _string_list(raw_qa_action.get(video_id, [])),
                "raw_qa_reason": _string_list(raw_qa_reason.get(video_id, [])),
                "raw_topics": _string_list(raw_topics.get(video_id, [])),
                "raw_sentiments": _string_list(raw_sentiments.get(video_id, [])),
                "raw_funny": _float_list(raw_funny.get(video_id, [])),
                "raw_exciting": _float_list(raw_exciting.get(video_id, [])),
                "raw_language": _string_list(raw_language.get(video_id, [])),
                "clean_topic": _int_or_default(clean_topic.get(video_id)),
                "clean_sentiment": _int_or_default(clean_sentiment.get(video_id)),
                "clean_funny": _float_or_default(clean_funny.get(video_id)),
                "clean_exciting": _float_or_default(clean_exciting.get(video_id)),
                "clean_language": str(clean_language.get(video_id, "")),
                "clean_effective": _float_or_default(clean_effective.get(video_id)),
            },
        )


class PittImageVideoAdsDataset(ds.GeneratorBasedBuilder):
    """A class for loading the Pitt Image and Video Ads Dataset."""

    config: PittImageVideoAdsDatasetConfig

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = PittImageVideoAdsDatasetConfig
    BUILDER_CONFIGS = [
        PittImageVideoAdsDatasetConfig(
            name=PittImageVideoAdsDatasetType.image_ads,
            version=VERSION,
            description="Image advertisement annotations with source image URLs.",
        ),
        PittImageVideoAdsDatasetConfig(
            name=PittImageVideoAdsDatasetType.video_ads,
            version=VERSION,
            description="Video advertisement YouTube IDs with raw and cleaned annotations.",
        ),
    ]
    DEFAULT_CONFIG_NAME = "image_ads"

    def _info(self) -> ds.DatasetInfo:
        match self.config.name:
            case PittImageVideoAdsDatasetType.image_ads:
                features = ds.Features(
                    {
                        "id": ds.Value("string"),
                        "source_path": ds.Value("string"),
                        "image": ds.Image(),
                        "image_url": ds.Value("string"),
                        "qa_action": [ds.Value("string")],
                        "qa_reason": [ds.Value("string")],
                        "qa_combined_action_reason": [ds.Value("string")],
                        "slogans": [ds.Value("string")],
                        "topics": [ds.Value("string")],
                        "sentiments": [[ds.Value("string")]],
                        "strategies": [[ds.Value("string")]],
                        "symbols": [
                            {
                                "x1": ds.Value("float32"),
                                "y1": ds.Value("float32"),
                                "x2": ds.Value("float32"),
                                "y2": ds.Value("float32"),
                                "label": ds.Value("string"),
                            }
                        ],
                    }
                )
            case PittImageVideoAdsDatasetType.video_ads:
                features = ds.Features(
                    {
                        "id": ds.Value("string"),
                        "youtube_id": ds.Value("string"),
                        "youtube_url": ds.Value("string"),
                        "raw_qa_action": [ds.Value("string")],
                        "raw_qa_reason": [ds.Value("string")],
                        "raw_topics": [ds.Value("string")],
                        "raw_sentiments": [ds.Value("string")],
                        "raw_funny": [ds.Value("float32")],
                        "raw_exciting": [ds.Value("float32")],
                        "raw_language": [ds.Value("string")],
                        "clean_topic": ds.Value("int32"),
                        "clean_sentiment": ds.Value("int32"),
                        "clean_funny": ds.Value("float32"),
                        "clean_exciting": ds.Value("float32"),
                        "clean_language": ds.Value("string"),
                        "clean_effective": ds.Value("float32"),
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
        match self.config.name:
            case PittImageVideoAdsDatasetType.image_ads:
                urls = _URLS["image_ads"]
                assert isinstance(urls, dict)
                paths = _image_annotation_paths(
                    dl_manager.download_and_extract(urls["annotations"])
                )
                image_dirs = dl_manager.download_and_extract(urls["images"])
                assert isinstance(image_dirs, list)
                gen_kwargs = {
                    "paths": paths,
                    "image_roots": {
                        str(index): image_dir
                        for index, image_dir in enumerate(image_dirs)
                    },
                }
            case PittImageVideoAdsDatasetType.video_ads:
                url = _URLS["video_ads"]
                paths = _video_annotation_paths(dl_manager.download_and_extract(url))
                gen_kwargs = {"paths": paths}
            case _:
                assert_never(self.config.name)

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs=gen_kwargs,
            )
        ]

    def _generate_examples(
        self,
        paths: dict[str, str],
        image_roots: dict[str, str] | None = None,
    ):
        match self.config.name:
            case PittImageVideoAdsDatasetType.image_ads:
                assert image_roots is not None
                yield from _iter_image_examples(paths, image_roots)
            case PittImageVideoAdsDatasetType.video_ads:
                yield from _iter_video_examples(paths)
            case _:
                assert_never(self.config.name)
