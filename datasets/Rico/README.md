---
annotations_creators:
  - machine-generated
language:
  - en
language_creators:
  - found
license:
  - unknown
pretty_name: Rico
size_categories:
  - 10K<n<100K
source_datasets:
  - original
tags:
  - mobile-ui
  - user-interface
  - view-hierarchy
  - screenshot
task_categories:
  - image-to-text
  - object-detection
task_ids: []
configs:
  - config_name: default
    data_files:
      - split: metadata
        path: data/metadata-*
  - config_name: ui-screenshots-and-hierarchies-with-semantic-annotations
    data_files:
      - split: train
        path: ui-screenshots-and-hierarchies-with-semantic-annotations/train-*
      - split: validation
        path: ui-screenshots-and-hierarchies-with-semantic-annotations/validation-*
      - split: test
        path: ui-screenshots-and-hierarchies-with-semantic-annotations/test-*
  - config_name: ui-screenshots-and-view-hierarchies
    data_files:
      - split: train
        path: ui-screenshots-and-view-hierarchies/train-*
      - split: validation
        path: ui-screenshots-and-view-hierarchies/validation-*
      - split: test
        path: ui-screenshots-and-view-hierarchies/test-*
---

# Dataset Card for Rico

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** http://www.interactionmining.org/rico.html
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/Rico
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/Rico
- **Paper (UIST 2017):** https://dl.acm.org/doi/10.1145/3126594.3126651

### Dataset Summary

Rico is a mobile app UI dataset for building data-driven design applications. The original dataset mines Android apps at runtime and exposes visual, textual, structural, and interactive design properties from more than 9.3k apps across 27 categories and more than 66k unique UI screens. This packaging provides metadata, screenshots, view hierarchies, and semantic annotations as separate configs.

### Supported Tasks and Leaderboards

The dataset supports mobile UI understanding, screen hierarchy modeling, semantic element detection, and screenshot-conditioned interface analysis. No public leaderboard is bundled with this Hugging Face dataset.

### Languages

Metadata and UI text are primarily English (`en`), though app screenshots may contain other languages.

## Dataset Structure

### Data Fields

- `default`: app and trace metadata.
- `ui-screenshots-and-hierarchies-with-semantic-annotations`: semantic hierarchy fields including `ancestors`, `klass`, `bounds`, `clickable`, `children`, and `screenshot`.
- `ui-screenshots-and-view-hierarchies`: screenshot and Android view hierarchy metadata.

### Data Splits

| Config | Split | Rows |
| --- | --- | ---: |
| default | metadata | 66,261 |
| semantic annotations | train | 56,322 |
| semantic annotations | validation | 3,314 |
| semantic annotations | test | 6,625 |
| view hierarchies | train | 56,322 |
| view hierarchies | validation | 3,314 |
| view hierarchies | test | 6,625 |

## Dataset Creation

Rico was collected from mobile app interaction traces and UI screenshots to support data-driven UI research.

## Considerations for Using the Data

The dataset contains mobile app screenshots and UI metadata. It may include app-specific text, brands, and interface content from the collection period.

## Additional Information

### Licensing Information

The dataset license is not specified in the local loader metadata. Users should verify the upstream terms before redistribution or commercial use.

### Citation Information

```bibtex
@inproceedings{deka2017rico,
  title={Rico: A mobile app dataset for building data-driven design applications},
  author={Deka, Biplab and Huang, Zifeng and Franzen, Chad and Hibschman, Joshua and Afergan, Daniel and Li, Yang and Nichols, Jeffrey and Kumar, Ranjitha},
  booktitle={Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology},
  pages={845--854},
  year={2017}
}
```
