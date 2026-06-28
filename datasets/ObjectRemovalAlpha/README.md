---
annotations_creators:
  - found
language:
  - en
language_creators:
  - found
license: apache-2.0
pretty_name: ObjectRemovalAlpha
size_categories:
  - n<1K
source_datasets:
  - lrzjason/ObjectRemovalAlpha
tags:
  - object-removal
  - image-inpainting
  - image-editing
  - text-to-image
task_categories:
  - image-to-image
---

# Dataset Card for ObjectRemovalAlpha

## Dataset Description

- **Homepage:** https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/ObjectRemovalAlpha
- **Original Dataset:** https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/ObjectRemovalAlpha
- **Paper:** Not available in the original dataset release.

### Dataset Summary

ObjectRemovalAlpha is a small object-removal training dataset reformatted from `lrzjason/ObjectRemovalAlpha`. Each example contains a shared text prompt, a ground-truth image, a factual/source image, and a mask image.

The loader groups the original `_G`, `_F`, and `_M` files by numeric ID and exposes them as one row per object-removal example.

### Supported Tasks and Leaderboards

This dataset can be used for image editing, object removal, image inpainting, and mask-conditioned image generation experiments. No public leaderboard is bundled with the source dataset.

### Languages

The text prompts are in English.

## Dataset Structure

### Data Instances

Load the dataset with:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/ObjectRemovalAlpha")
```

```json
{
  "text": "object-removal prompt",
  "ground_truth_image": "<image>",
  "factual_image": "<image>",
  "mask_image": "<image>"
}
```

### Data Fields

- `text`: Prompt text shared by the original `_G`, `_F`, and `_M` files.
- `ground_truth_image`: Target image after removal.
- `factual_image`: Source image before removal.
- `mask_image`: Mask image indicating the removal region.

### Data Splits

| Split | Examples |
| --- | ---: |
| train | 20 |

## Dataset Creation

### Source Data

The data is loaded from the upstream Hugging Face dataset `lrzjason/ObjectRemovalAlpha` using `snapshot_download`.

### Licensing Information

The upstream dataset is marked Apache-2.0 on Hugging Face Hub.

### Citation Information

No dataset-specific paper or BibTeX citation was found in the upstream dataset release.

### Contributions

This Hugging Face dataset implementation was created by the creative-graphic-design organization to expose `lrzjason/ObjectRemovalAlpha` through a typed dataset loader.
