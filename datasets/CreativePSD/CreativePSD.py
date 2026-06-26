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
from typing import Any, List

from datasets.utils.logging import get_logger

import datasets as ds

logger = get_logger(__name__)

_CITATION = """\
@inproceedings{shuai2026psdesigner,
  title={PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow},
  author={Shuai, Xincheng and Tang, Song and Huang, Yutong and Ding, Henghui and Tao, Dacheng},
  booktitle={CVPR},
  year={2026},
}
"""

_DESCRIPTION = """\
CreativePSD is a graphic design dataset released with PSDesigner. It contains PSD-derived \
poster design archives with layer metadata, PSD tree text, tool-call trajectories, source image \
resources, and stepwise rendered images that trace the creative workflow.
"""

_HOMEPAGE = "https://modelscope.cn/datasets/song322/CreativePSD"

_LICENSE = "CC-BY-NC-4.0"

_DATASET_ID = "song322/CreativePSD"
_EXPECTED_MODELSCOPE_POSTER_ZIP_COUNT = 7978
_DEFAULT_LOCAL_DIR = pathlib.Path(
    "/root/ghq/www.modelscope.cn/datasets/song322/CreativePSD"
)
_MODEL_SCOPE_CACHE_CANDIDATES = [
    pathlib.Path("~/.cache/modelscope/hub/datasets/song322/CreativePSD"),
    pathlib.Path("~/.cache/modelscope/datasets/song322/CreativePSD"),
    pathlib.Path("~/modelscope/datasets/song322/CreativePSD"),
]

_KNOWN_METADATA_MEMBERS = {
    "metadata/1.origin_psd_tree.txt": "origin_psd_tree",
    "metadata/2.deleted_psd_tree.txt": "deleted_psd_tree",
    "metadata/3.grouped_psd_tree.txt": "grouped_psd_tree",
    "metadata/group_child_ids.json": "group_child_ids_json",
    "metadata/layer_info.json": "layer_info_json",
    "metadata/rendering_id.json": "rendering_id_json",
    "metadata/tool_trajectory.json": "tool_trajectory_json",
    "rendering_imgs/render_second_values.json": "render_second_values_json",
}


def _expand_path(path: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.expanduser(str(path))).resolve()


def _zip_paths_under(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file() and path.name.startswith("poster_") and path.suffix == ".zip":
        return [path]
    if not path.is_dir():
        return []
    return sorted(path.rglob("poster_*.zip"))


def _find_first_zip_dir(path: pathlib.Path) -> pathlib.Path | None:
    zip_paths = _zip_paths_under(path)
    if not zip_paths:
        return None
    return zip_paths[0].parent


def _is_modelscope_checkout(path: pathlib.Path) -> bool:
    if path.is_file():
        path = path.parent
    return (path / ".gitattributes").exists() and (path / "README.md").exists()


def _validate_zip_paths(zip_paths: list[pathlib.Path], root_path: pathlib.Path) -> None:
    invalid_zip_paths = [
        path
        for path in zip_paths
        if path.stat().st_size == 0 or not zipfile.is_zipfile(path)
    ]
    if invalid_zip_paths:
        invalid_preview = ", ".join(path.name for path in invalid_zip_paths[:10])
        raise ValueError(
            "Found invalid CreativePSD poster archives: "
            f"{invalid_preview}. Re-download these files before loading."
        )

    if (
        _is_modelscope_checkout(root_path)
        and len(zip_paths) != _EXPECTED_MODELSCOPE_POSTER_ZIP_COUNT
    ):
        raise ValueError(
            "The CreativePSD ModelScope checkout appears incomplete: found "
            f"{len(zip_paths)} poster_*.zip files, expected "
            f"{_EXPECTED_MODELSCOPE_POSTER_ZIP_COUNT}. Re-run the ModelScope "
            "download or pass a complete data_dir."
        )


def _ensure_modelscope_download() -> pathlib.Path | None:
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        logger.warning(
            "modelscope is not installed. Pass data_dir pointing to CreativePSD "
            "poster_*.zip files, or install the modelscope package."
        )
        return None

    logger.info("Loading CreativePSD with MsDataset.load(%r).", _DATASET_ID)
    MsDataset.load(_DATASET_ID)

    for candidate in _MODEL_SCOPE_CACHE_CANDIDATES:
        found = _find_first_zip_dir(_expand_path(candidate))
        if found is not None:
            return found
    return None


def _resolve_zip_paths(
    dl_manager: ds.DownloadManager, data_dir: str | None
) -> list[pathlib.Path]:
    if data_dir:
        local_path = _expand_path(data_dir)
        zip_paths = _zip_paths_under(local_path)
        if not zip_paths and zipfile.is_zipfile(local_path):
            zip_paths = [local_path]
        if zip_paths:
            _validate_zip_paths(zip_paths, local_path)
            return zip_paths
        raise FileNotFoundError(
            f"No poster_*.zip files found under data_dir={local_path}"
        )

    if dl_manager.manual_dir:
        manual_path = _expand_path(dl_manager.manual_dir)
        zip_paths = _zip_paths_under(manual_path)
        if zip_paths:
            _validate_zip_paths(zip_paths, manual_path)
            return zip_paths

    if _DEFAULT_LOCAL_DIR.exists():
        zip_paths = _zip_paths_under(_DEFAULT_LOCAL_DIR)
        if zip_paths:
            _validate_zip_paths(zip_paths, _DEFAULT_LOCAL_DIR)
            return zip_paths

    modelscope_dir = _ensure_modelscope_download()
    if modelscope_dir is not None:
        zip_paths = _zip_paths_under(modelscope_dir)
        if zip_paths:
            _validate_zip_paths(zip_paths, modelscope_dir)
            return zip_paths

    raise FileNotFoundError(
        "Could not find CreativePSD poster_*.zip files. Download the dataset with "
        "`from modelscope.msdatasets import MsDataset; MsDataset.load('song322/CreativePSD')` "
        "or pass data_dir to the directory that contains poster_*.zip files."
    )


def _member_category(member: str) -> str:
    if member.startswith("metadata/"):
        return "metadata"
    if member.startswith("raw_resource/"):
        return "raw_resource"
    if member.startswith("rendering_imgs/"):
        return "rendering_imgs"
    return "other"


def _read_member_bytes(zf: zipfile.ZipFile, member: str) -> bytes:
    with zf.open(member) as f:
        return f.read()


def _read_optional_text(zf: zipfile.ZipFile, member: str) -> str:
    if member not in zf.namelist():
        return ""
    return _read_member_bytes(zf, member).decode("utf-8", errors="replace")


def _read_optional_json(zf: zipfile.ZipFile, member: str) -> Any:
    text = _read_optional_text(zf, member)
    if not text:
        return {}
    return json.loads(text)


def _safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _color_info(value: dict[str, Any] | None) -> dict[str, int]:
    value = value or {}
    return {
        "red": _safe_int(value.get("red")),
        "green": _safe_int(value.get("green")),
        "blue": _safe_int(value.get("blue")),
    }


def _image_from_zip(
    zf: zipfile.ZipFile, member: str, archive_filename: str
) -> dict[str, bytes | str]:
    return {
        "bytes": _read_member_bytes(zf, member),
        "path": f"{archive_filename}/{member}",
    }


def _parse_rendering_image_name(filename: str) -> tuple[int, int, str]:
    name = pathlib.PurePosixPath(filename).name
    if name == "0_total.jpg":
        return 0, 0, "TOTAL"

    stem = pathlib.PurePosixPath(name).stem
    parts = stem.split("_", 2)
    if len(parts) != 3:
        return 0, 0, ""

    step_index = int(parts[0])
    layer_id = int(parts[1])
    layer_kind = parts[2].removeprefix("LayerKind.")
    return step_index, layer_id, layer_kind


def _all_file_records(zf: zipfile.ZipFile) -> list[dict[str, Any]]:
    records = []
    for info in sorted(zf.infolist(), key=lambda value: value.filename):
        if info.is_dir():
            continue
        records.append(
            {
                "filename": info.filename,
                "category": _member_category(info.filename),
                "file_size_bytes": info.file_size,
            }
        )
    return records


def _non_image_file_records(zf: zipfile.ZipFile) -> list[dict[str, Any]]:
    records = []
    for info in sorted(zf.infolist(), key=lambda value: value.filename):
        member = info.filename
        if info.is_dir() or member.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        content = _read_member_bytes(zf, member)
        records.append(
            {
                "filename": member,
                "category": _member_category(member),
                "file_size_bytes": info.file_size,
                "text": content.decode("utf-8", errors="replace"),
                "bytes": content,
            }
        )
    return records


def _metadata_file_records(zf: zipfile.ZipFile) -> list[dict[str, Any]]:
    records = []
    for info in sorted(zf.infolist(), key=lambda value: value.filename):
        member = info.filename
        if info.is_dir() or not member.startswith("metadata/"):
            continue
        content = _read_member_bytes(zf, member)
        records.append(
            {
                "filename": member,
                "file_size_bytes": info.file_size,
                "text": content.decode("utf-8", errors="replace"),
            }
        )
    return records


def _raw_resource_records(
    zf: zipfile.ZipFile, archive_filename: str
) -> list[dict[str, Any]]:
    records = []
    for info in sorted(zf.infolist(), key=lambda value: value.filename):
        member = info.filename
        if info.is_dir() or not member.startswith("raw_resource/"):
            continue
        if not member.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        records.append(
            {
                "filename": member,
                "file_size_bytes": info.file_size,
                "image": _image_from_zip(zf, member, archive_filename),
            }
        )
    return records


def _rendering_image_records(
    zf: zipfile.ZipFile, archive_filename: str
) -> list[dict[str, Any]]:
    records = []
    for info in sorted(zf.infolist(), key=lambda value: value.filename):
        member = info.filename
        if info.is_dir() or not member.startswith("rendering_imgs/"):
            continue
        if not member.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        step_index, layer_id, layer_kind = _parse_rendering_image_name(member)
        records.append(
            {
                "filename": member,
                "file_size_bytes": info.file_size,
                "step_index": step_index,
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "is_final": pathlib.PurePosixPath(member).name == "0_total.jpg",
                "image": _image_from_zip(zf, member, archive_filename),
            }
        )
    return records


def _build_example(zip_path: pathlib.Path, zf: zipfile.ZipFile) -> dict[str, Any]:
    archive_filename = zip_path.name
    poster_id = zip_path.stem
    layer_info = _read_optional_json(zf, "metadata/layer_info.json")
    psd_info = layer_info.get("psd_info") or {}

    example = {
        "id": poster_id,
        "archive_filename": archive_filename,
        "archive_path": str(zip_path),
        "archive_size_bytes": zip_path.stat().st_size,
        "psd_info": {
            "filename": str(psd_info.get("filename") or ""),
            "width": _safe_int(psd_info.get("width")),
            "height": _safe_int(psd_info.get("height")),
            "resolution": _safe_int(psd_info.get("resolution")),
            "color_mode": str(
                psd_info.get("colorMode") or psd_info.get("color_mode") or ""
            ),
            "fill_color": _color_info(psd_info.get("fill_color")),
        },
        "total_layers": _safe_int(layer_info.get("total_layers")),
        "metadata_files": _metadata_file_records(zf),
        "non_image_files": _non_image_file_records(zf),
        "raw_resources": _raw_resource_records(zf, archive_filename),
        "rendering_images": _rendering_image_records(zf, archive_filename),
        "all_files": _all_file_records(zf),
    }

    for member, feature_name in _KNOWN_METADATA_MEMBERS.items():
        example[feature_name] = _read_optional_text(zf, member)

    final_member = "rendering_imgs/0_total.jpg"
    example["final_rendering"] = (
        _image_from_zip(zf, final_member, archive_filename)
        if final_member in zf.namelist()
        else None
    )
    return example


class CreativePSD(ds.GeneratorBasedBuilder):
    """A class for loading CreativePSD dataset."""

    VERSION = ds.Version("1.0.0")
    DEFAULT_WRITER_BATCH_SIZE = 1

    @property
    def _manual_download_instructions(self) -> str:
        return (
            "Download CreativePSD from ModelScope with "
            "`from modelscope.msdatasets import MsDataset; "
            "MsDataset.load('song322/CreativePSD')`, then pass the directory "
            "containing poster_*.zip files via data_dir."
        )

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "id": ds.Value("string"),
                "archive_filename": ds.Value("string"),
                "archive_path": ds.Value("string"),
                "archive_size_bytes": ds.Value("int64"),
                "psd_info": {
                    "filename": ds.Value("string"),
                    "width": ds.Value("int32"),
                    "height": ds.Value("int32"),
                    "resolution": ds.Value("int32"),
                    "color_mode": ds.Value("string"),
                    "fill_color": {
                        "red": ds.Value("int32"),
                        "green": ds.Value("int32"),
                        "blue": ds.Value("int32"),
                    },
                },
                "total_layers": ds.Value("int32"),
                "origin_psd_tree": ds.Value("string"),
                "deleted_psd_tree": ds.Value("string"),
                "grouped_psd_tree": ds.Value("string"),
                "group_child_ids_json": ds.Value("string"),
                "layer_info_json": ds.Value("string"),
                "rendering_id_json": ds.Value("string"),
                "tool_trajectory_json": ds.Value("string"),
                "render_second_values_json": ds.Value("string"),
                "metadata_files": [
                    {
                        "filename": ds.Value("string"),
                        "file_size_bytes": ds.Value("int64"),
                        "text": ds.Value("string"),
                    }
                ],
                "non_image_files": [
                    {
                        "filename": ds.Value("string"),
                        "category": ds.Value("string"),
                        "file_size_bytes": ds.Value("int64"),
                        "text": ds.Value("string"),
                        "bytes": ds.Value("binary"),
                    }
                ],
                "raw_resources": [
                    {
                        "filename": ds.Value("string"),
                        "file_size_bytes": ds.Value("int64"),
                        "image": ds.Image(),
                    }
                ],
                "rendering_images": [
                    {
                        "filename": ds.Value("string"),
                        "file_size_bytes": ds.Value("int64"),
                        "step_index": ds.Value("int32"),
                        "layer_id": ds.Value("int32"),
                        "layer_kind": ds.Value("string"),
                        "is_final": ds.Value("bool"),
                        "image": ds.Image(),
                    }
                ],
                "final_rendering": ds.Image(),
                "all_files": [
                    {
                        "filename": ds.Value("string"),
                        "category": ds.Value("string"),
                        "file_size_bytes": ds.Value("int64"),
                    }
                ],
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
        data_dir = getattr(self.config, "data_dir", None)
        zip_paths = _resolve_zip_paths(dl_manager, data_dir)
        logger.info("Found %d CreativePSD poster archives.", len(zip_paths))
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"zip_paths": zip_paths},
            ),
        ]

    def _generate_examples(self, zip_paths: list[pathlib.Path]):
        for key, zip_path in enumerate(sorted(zip_paths)):
            with zipfile.ZipFile(zip_path) as zf:
                yield key, _build_example(zip_path, zf)
