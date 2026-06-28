---
language:
  - en
license: unknown
pretty_name: CTXFont
tags:
  - design
  - typography
  - font-prediction
  - web-design
  - graphic-design
  - context-aware
annotations_creators:
  - machine-generated
language_creators:
  - found
size_categories:
  - 1K<n<10K
source_datasets:
  - original
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: test
        path: data/test-*
---

# Dataset Card for CTXFont

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://github.com/nanxuanzhao/CTXFont-dataset
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/CTXFont
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/CTXFont
- **Paper (Pacific Graphics 2018):** https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13576

### Dataset Summary

CTXFont is a context-aware font prediction dataset released with *Modeling Fonts in Context: Font Prediction on Web Designs*. The original dataset contains 1,065 professional web designs, 4,893 text elements, and 492 unique font faces, with annotations for font face, color, size, element geometry, HTML tags, design tags, and learned font embeddings.

### Supported Tasks and Leaderboards

The dataset supports font property prediction, context-aware typography recommendation, and web design analysis. No public leaderboard is bundled with this Hugging Face dataset.

### Languages

Most text content and metadata are in English (`en`).

## Dataset Structure

### Data Fields

Rows contain design-level information (`design_name`, `design_image`, `design_url`, `awwward_url`, `design_tags`) and text-element information (`text_content`, `html_tags`, `font_face`, `font_size`, RGBA color channels, `font_face_embedding`, `center_x`, `center_y`, `width`, and `height`).

### Data Splits

| Split | Rows |
| --- | ---: |
| train | 4,268 |
| test | 625 |

## Dataset Creation

The original dataset was created from awwwards.com web designs and HTML/CSS-derived font annotations to study font prediction in visual context.

## Considerations for Using the Data

The data reflects professional web designs from the collection period and may not cover all languages, accessibility requirements, or contemporary typography practices.

## Additional Information

### Licensing Information

The dataset license is listed as unknown in the local loader metadata.

### Citation Information

```bibtex
@article{zhao2018modeling,
  title={Modeling Fonts in Context: Font Prediction on Web Designs},
  author={Zhao, Nanxuan and Cao, Ying and Lau, Rynson W. H.},
  journal={Computer Graphics Forum},
  year={2018}
}
```
