---
language:
  - zh
license: cc-by-nc-4.0
pretty_name: CreativePSD
tags:
  - graphic-design
  - psd
  - creative-ai
  - tool-use
  - multimodal
  - modelscope
annotations_creators:
  - machine-generated
language_creators:
  - found
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - image-to-text
  - text-to-image
task_ids: []
---

# Dataset Card for CreativePSD

## Dataset Description

- **Homepage:** https://modelscope.cn/datasets/song322/CreativePSD
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/CreativePSD
- **Paper (Preprint):** https://arxiv.org/abs/2603.25738
- **Paper (Conference/Journal Name):** [More Information Needed]
- **Point of Contact:** [More Information Needed]

### Dataset Summary

CreativePSD is the PSD-derived graphic design dataset released with PSDesigner. Each example is a poster archive containing PSD tree text, structured layer metadata, tool-call trajectories, source image resources, and stepwise rendered images.

This loader keeps the contents of each `poster_*.zip` archive: all metadata text/JSON files, all `raw_resource` images, all `rendering_imgs` images, and a manifest of every member in the archive.

### Languages

The dataset contains Chinese text in poster designs and metadata fields.

## Dataset Structure

### Data Instances

Each row corresponds to one `poster_*.zip` archive:

```json
{
  "id": "poster_000175",
  "archive_filename": "poster_000175.zip",
  "psd_info": {
    "filename": "poster_000175.psd",
    "width": 800,
    "height": 800,
    "resolution": 72,
    "color_mode": "RGB"
  },
  "total_layers": 18,
  "origin_psd_tree": "...",
  "tool_trajectory_json": "[...]",
  "raw_resources": [{"filename": "raw_resource/1_asset.png", "image": "<image>"}],
  "rendering_images": [{"filename": "rendering_imgs/0_total.jpg", "image": "<image>"}],
  "final_rendering": "<image>"
}
```

### Data Fields

- `id`: Poster archive stem.
- `archive_filename`, `archive_path`, `archive_size_bytes`: Source archive metadata.
- `psd_info`: PSD filename, canvas size, resolution, color mode, and fill color.
- `total_layers`: Number of PSD layers from `metadata/layer_info.json`.
- `origin_psd_tree`, `deleted_psd_tree`, `grouped_psd_tree`: PSD tree text files.
- `group_child_ids_json`, `layer_info_json`, `rendering_id_json`, `tool_trajectory_json`, `render_second_values_json`: Original JSON files serialized as strings.
- `metadata_files`: All files under `metadata/` as text.
- `non_image_files`: Every non-image file in the archive as text and bytes.
- `raw_resources`: All source images under `raw_resource/`.
- `rendering_images`: All images under `rendering_imgs/`, including `0_total.jpg` and intermediate layer renders.
- `final_rendering`: `rendering_imgs/0_total.jpg` when present.
- `all_files`: Manifest of every file member in the zip.

### Data Splits

The loader exposes a single `train` split. The row count depends on the ModelScope files available locally or in cache. The local copy used during implementation contained 3,283 `poster_*.zip` files.

## Usage

Download the dataset with ModelScope:

```python
from modelscope.msdatasets import MsDataset

MsDataset.load("song322/CreativePSD")
```

Then load from the directory containing `poster_*.zip` files:

```python
import datasets as ds

dataset = ds.load_dataset(
    "datasets/CreativePSD/CreativePSD.py",
    data_dir="/root/ghq/www.modelscope.cn/datasets/song322/CreativePSD",
    trust_remote_code=True,
)
```

## Considerations for Using the Data

The ModelScope dataset card states that the dataset is for non-commercial research use only. Users should review the original dataset terms before redistribution or model training.

## Additional Information

### Licensing Information

The source dataset card lists `CC-BY-NC-4.0`.

### Citation Information

```bibtex
@misc{shuai2026psdesigner,
  title={PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow},
  author={Shuai, Xincheng and Tang, Song and Huang, Yutong and Ding, Henghui and Tao, Dacheng},
  year={2026},
  eprint={2603.25738},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

### Contributions

Thanks to the PSDesigner and CreativePSD authors for creating and releasing this dataset.
