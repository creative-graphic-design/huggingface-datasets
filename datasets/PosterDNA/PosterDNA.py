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
import json
import os
import pathlib
import zipfile
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Iterable, List, assert_never

from datasets.utils.logging import get_logger

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@inproceedings{liu2026posterverse,
  title={PosterVerse: A Full-Workflow Framework for Commercial-Grade Poster Generation with HTML-Based Scalable Typography},
  author={Liu, Junle and Zhang, Peirong and Zhang, Yuyi and Yan, Pengyu and Zhou, Hui and Zhou, Xinyue and Guo, Fengjun and Jin, Lianwen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={9},
  pages={7197--7205},
  year={2026},
  doi={10.1609/aaai.v40i9.37656},
  url={https://doi.org/10.1609/aaai.v40i9.37656}
}
"""

_DESCRIPTION = """\
PosterDNA is the poster dataset released with PosterVerse. It contains commercial-grade, \
text-dense poster data with background images, HTML-based layout and typography specifications, \
poster intention metadata, and a held-out test set for poster generation research.
"""

_HOMEPAGE = "https://github.com/wuhaer/PosterVerse"

_LICENSE = "cc-by-nc-nd-4.0"

_URLS = {
    "posterdna": "https://huggingface.co/wuhaer/PosterVerse/resolve/main/posterdna.zip",
    "test_set": "https://huggingface.co/wuhaer/PosterVerse/resolve/main/test-set.zip",
}

_ZIP_PASSWORD_ENV = "POSTERDNA_ZIP_PASSWORD"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class PosterDNAType(StrEnum):
    posterdna = auto()
    test_set = auto()


@dataclass
class PosterDNAConfig(ds.BuilderConfig):
    name: PosterDNAType

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PosterDNAType(self.name)


def _as_path(value: str | pathlib.Path) -> pathlib.Path:
    return value if isinstance(value, pathlib.Path) else pathlib.Path(value)


def _requires_password(zip_path: str | pathlib.Path) -> bool:
    with zipfile.ZipFile(zip_path) as archive:
        return any(info.flag_bits & 0x1 for info in archive.infolist())


def _extract_zip(
    zip_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    password: str | None,
) -> pathlib.Path:
    zip_path = _as_path(zip_path)
    output_dir = _as_path(output_dir)
    marker_path = output_dir / ".extracted"

    if marker_path.exists():
        return output_dir

    if _requires_password(zip_path) and not password:
        raise RuntimeError(
            f"{zip_path.name} is password-protected. Apply for PosterDNA access "
            "through the upstream project, then set "
            f"{_ZIP_PASSWORD_ENV}=<decompression password> before loading."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pwd = password.encode("utf-8") if password else None
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir, pwd=pwd)

    marker_path.write_text("ok\n", encoding="utf-8")
    return output_dir


def _find_dataset_root(extracted_dir: str | pathlib.Path, expected_name: str) -> pathlib.Path:
    root = _as_path(extracted_dir)
    if root.name == expected_name and root.is_dir():
        return root

    direct = root / expected_name
    if direct.is_dir():
        return direct

    for path in root.rglob(expected_name):
        if path.is_dir():
            return path

    return root


def _relative_path(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _index_files(root: str | pathlib.Path) -> dict[str, dict[str, pathlib.Path]]:
    root = _as_path(root)
    index: dict[str, dict[str, pathlib.Path]] = {
        "image": {},
        "html": {},
        "json": {},
        "jsonl": {},
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS:
            index["image"].setdefault(path.stem, path)
        elif suffix == ".html":
            index["html"].setdefault(path.stem, path)
        elif suffix == ".json":
            index["json"].setdefault(path.stem, path)
        elif suffix == ".jsonl":
            index["jsonl"].setdefault(path.stem, path)

    return index


def _read_text(path: pathlib.Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _resolve_referenced_asset(
    row: dict[str, Any],
    assets: dict[str, pathlib.Path],
    suffixes: set[str],
) -> pathlib.Path | None:
    for value in _iter_strings(row):
        path = pathlib.PurePosixPath(value)
        if path.suffix.lower() not in suffixes:
            continue
        if path.stem in assets:
            return assets[path.stem]
    return None


def _row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "poster_id", "image_id", "uid", "name", "filename"):
        value = row.get(key)
        if value not in (None, ""):
            return pathlib.PurePosixPath(str(value)).stem
    return f"{fallback:06d}"


def _iter_test_set_examples(root: str | pathlib.Path) -> Iterable[tuple[str, dict[str, Any]]]:
    root = _as_path(root)
    assets = _index_files(root)

    for metadata_path in sorted(assets["json"].values()):
        row = json.loads(metadata_path.read_text(encoding="utf-8"))
        example_id = metadata_path.stem
        image_path = assets["image"].get(example_id)
        html_path = assets["html"].get(example_id)

        yield (
            example_id,
            {
                "id": example_id,
                "metadata": _json_dumps(row),
                "metadata_path": _relative_path(metadata_path, root),
                "background_image": str(image_path) if image_path else None,
                "background_image_path": _relative_path(image_path, root)
                if image_path
                else "",
                "html": _read_text(html_path),
                "html_path": _relative_path(html_path, root) if html_path else "",
            },
        )


def _iter_posterdna_examples(root: str | pathlib.Path) -> Iterable[tuple[str, dict[str, Any]]]:
    root = _as_path(root)
    assets = _index_files(root)
    jsonl_path = next(iter(sorted(assets["jsonl"].values())), None)

    if jsonl_path is None:
        for index, html_path in enumerate(sorted(assets["html"].values())):
            example_id = html_path.stem
            image_path = assets["image"].get(example_id)
            yield (
                example_id,
                {
                    "id": example_id,
                    "metadata": _json_dumps({"index": index}),
                    "metadata_path": "",
                    "background_image": str(image_path) if image_path else None,
                    "background_image_path": _relative_path(image_path, root)
                    if image_path
                    else "",
                    "html": _read_text(html_path),
                    "html_path": _relative_path(html_path, root),
                },
            )
        return

    with jsonl_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            example_id = _row_id(row, index)
            image_path = (
                _resolve_referenced_asset(row, assets["image"], _IMAGE_EXTENSIONS)
                or assets["image"].get(example_id)
            )
            html_path = (
                _resolve_referenced_asset(row, assets["html"], {".html"})
                or assets["html"].get(example_id)
            )

            yield (
                example_id,
                {
                    "id": example_id,
                    "metadata": _json_dumps(row),
                    "metadata_path": _relative_path(jsonl_path, root),
                    "background_image": str(image_path) if image_path else None,
                    "background_image_path": _relative_path(image_path, root)
                    if image_path
                    else "",
                    "html": _read_text(html_path),
                    "html_path": _relative_path(html_path, root) if html_path else "",
                },
            )


class PosterDNA(ds.GeneratorBasedBuilder):
    """A class for loading the PosterDNA dataset."""

    config: PosterDNAConfig

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = PosterDNAConfig
    BUILDER_CONFIGS = [
        PosterDNAConfig(
            name=PosterDNAType.posterdna,
            version=VERSION,
            description=(
                "PosterDNA training archive with poster intention metadata, "
                "background assets, and HTML layout specifications."
            ),
        ),
        PosterDNAConfig(
            name=PosterDNAType.test_set,
            version=VERSION,
            description=(
                "PosterDNA held-out test-set archive with JSON metadata, "
                "background assets, and HTML layout specifications."
            ),
        ),
    ]
    DEFAULT_CONFIG_NAME = "posterdna"

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "id": ds.Value("string"),
                "metadata": ds.Value("string"),
                "metadata_path": ds.Value("string"),
                "background_image": ds.Image(),
                "background_image_path": ds.Value("string"),
                "html": ds.Value("string"),
                "html_path": ds.Value("string"),
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
        match self.config.name:
            case PosterDNAType.posterdna:
                archive_url = _URLS["posterdna"]
                root_name = "posterdna"
                split = ds.Split.TRAIN
            case PosterDNAType.test_set:
                archive_url = _URLS["test_set"]
                root_name = "test-set"
                split = ds.Split.TEST
            case _:
                assert_never(self.config.name)

        archive_path = pathlib.Path(dl_manager.download(archive_url))
        extracted_dir = _extract_zip(
            archive_path,
            archive_path.with_suffix(".extracted"),
            os.environ.get(_ZIP_PASSWORD_ENV),
        )
        data_root = _find_dataset_root(extracted_dir, root_name)

        return [
            ds.SplitGenerator(
                name=split,
                gen_kwargs={"data_root": str(data_root)},
            ),
        ]

    def _generate_examples(self, data_root: str):
        match self.config.name:
            case PosterDNAType.posterdna:
                yield from _iter_posterdna_examples(data_root)
            case PosterDNAType.test_set:
                yield from _iter_test_set_examples(data_root)
            case _:
                assert_never(self.config.name)
