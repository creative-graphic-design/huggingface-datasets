---
annotations_creators:
  - machine-generated
language:
  - zh
language_creators:
  - found
license:
  - cc-by-nc-sa-4.0
multilinguality:
  - monolingual
pretty_name: PosterErase
size_categories:
  - 10K<n<100K
source_datasets:
  - original
tags:
  - graphic-design
  - poster
  - text-erasing
  - image-inpainting
task_categories:
  - image-to-image
task_ids: []
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
      - split: test
        path: data/test-*
---

# Dataset Card for PosterErase

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://github.com/alimama-creative/Self-supervised-Text-Erasing
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PosterErase
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/PosterErase
- **Original Data:** https://tianchi.aliyun.com/dataset/134810
- **Paper (arXiv):** https://arxiv.org/abs/2204.12743
- **Paper (ACM MM 2022):** https://doi.org/10.1145/3503161.3547905

### Dataset Summary

PosterErase is a poster text-erasing dataset released with *Self-supervised Text Erasing with Controllable Image Synthesis*. It contains high-resolution poster images with text regions and structured annotations for text-erasing research.

This Hugging Face version exposes the original train, validation, and test splits as parquet files. The validation and test splits include ground-truth erased poster images; the training split contains source posters and annotations only.

### Supported Tasks and Leaderboards

PosterErase is intended for image-to-image text erasing and inpainting on graphic design posters. A model receives a poster image and text-region annotations, then predicts a poster with the text removed while preserving the surrounding visual design.

No public leaderboard is bundled with this Hugging Face dataset. Use the upstream paper and repository for the original training and evaluation protocol.

### Languages

Poster text is primarily Chinese (`zh`).

## Dataset Structure

### Data Instances

Each row contains a poster image, the original relative image path, and parsed annotation fields. Validation and test rows also include `gt_image`.

```json
{
  "number": 0,
  "path": "train/000000.png",
  "image": "<image>",
  "gt_image": null,
  "annotation": {
    "masks": [
      {
        "x1": 0,
        "x2": 0,
        "y1": 0,
        "y2": 0
      }
    ],
    "place": {
      "objs": [
        {
          "text": "...",
          "size": 0,
          "direction": 0
        }
      ],
      "texts": [
        [
          {
            "x": 0,
            "y": 0,
            "cs": [
              {
                "c1": 0,
                "c2": 0,
                "c3": 0
              }
            ]
          }
        ]
      ]
    }
  }
}
```

### Data Fields

- `number` (`int32`): Original numeric example identifier from the annotation file.
- `path` (`string`): Original relative path of the source poster image.
- `image` (`Image`): Source poster image with text.
- `gt_image` (`Image`, nullable): Ground-truth erased poster image. This field is populated for validation and test rows and is `null` for training rows.
- `annotation.masks` (`list`): Text mask bounding boxes with `x1`, `x2`, `y1`, and `y2` integer coordinates.
- `annotation.place.objs` (`list`): Parsed text object metadata with `text`, `size`, and `direction`.
- `annotation.place.texts` (`list`): Parsed text placement and color metadata. Each text item contains `x`, `y`, and `cs`; each color has `c1`, `c2`, and `c3` integer channels.

### Data Splits

| Split | Rows | Ground-truth erased images |
| --- | ---: | ---: |
| train | 58,114 | 0 |
| validation | 148 | 148 |
| test | 146 | 146 |

## Dataset Creation

### Source Data

The original data was released by alimama-creative for the PosterErase text-erasing task. The original distribution uses six zip files named `erase_1.zip` through `erase_6.zip`.

### Annotations

The loader parses the upstream tab-separated annotation files:

- `train.txt` for the training split.
- `ps_valid.txt` for the validation split.
- `ps_test.txt` for the test split.

The validation and test annotation files include a `gt_path` column that points to the ground-truth erased image.

### Personal and Sensitive Information

The dataset contains poster images and rendered text from the upstream release. The dataset card does not identify personal information in the annotations, but posters may contain names, brands, faces, or culturally specific visual/textual content.

## Considerations for Using the Data

### Social Impact of Dataset

PosterErase can support better text erasing and design editing systems for poster images. It should be used with care when editing copyrighted, branded, or identity-bearing poster content.

### Discussion of Biases

The dataset is centered on Chinese poster designs from the upstream release. Models evaluated on this dataset may not generalize to other writing systems, design cultures, poster genres, or typography styles.

### Other Known Limitations

The training split does not include ground-truth erased images in this loader. The original task relies on the paper's self-supervised setup for training and uses validation/test ground truth for evaluation.

## Additional Information

### Dataset Curators

The original dataset was created by alimama-creative. This Hugging Face packaging is maintained by the creative-graphic-design project.

### Licensing Information

The upstream Tianchi page has shown conflicting license indicators: the page text has stated CC BY-SA 4.0, while the page license selector has appeared to indicate a non-commercial ShareAlike Creative Commons license. This dataset card uses the more restrictive `cc-by-nc-sa-4.0` metadata and users should verify the current upstream terms before redistribution or commercial use.

### Citation Information

```bibtex
@inproceedings{jiang2022self,
  title={Self-supervised text erasing with controllable image synthesis},
  author={Jiang, Gangwei and Wang, Shiyao and Ge, Tiezheng and Jiang, Yuning and Wei, Ying and Lian, Defu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1973--1983},
  year={2022}
}
```

### Contributions

Thanks to [alimama-creative](https://github.com/alimama-creative) for creating the original dataset.
