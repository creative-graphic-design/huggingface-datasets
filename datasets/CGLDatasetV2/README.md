---
annotations_creators:
  - crowdsourced
language:
  - zh
language_creators:
  - found
license:
  - unknown
multilinguality:
  - monolingual
pretty_name: CGL-Dataset v2
size_categories:
  - 10K<n<100K
source_datasets:
  - CGL-Dataset
tags:
  - graphic-design
  - poster
  - layout-generation
task_categories:
  - image-to-image
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: test
        path: data/test-*
  - config_name: ralf-style
    data_files:
      - split: train
        path: ralf-style/train-*
      - split: validation
        path: ralf-style/validation-*
      - split: test
        path: ralf-style/test-*
      - split: no_annotation
        path: ralf-style/no_annotation-*
---

# Dataset Card for CGL-Dataset v2

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://github.com/liuan0803/RADM
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/CGLDatasetV2
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset-v2
- **Paper (arXiv):** https://arxiv.org/abs/2306.09086
- **Paper (CIKM 2023):** https://dl.acm.org/doi/10.1145/3583780.3615028

### Dataset Summary

CGL-Dataset v2 is an advertising-poster layout dataset released with *Relation-Aware Diffusion Model for Controllable Poster Layout Generation*. The paper argues that poster layouts should account for both visual-textual relationships and geometry relationships between elements. This version extends CGL-Dataset with richer element annotations, text annotations, and text features for controllable poster layout generation.

### Supported Tasks and Leaderboards

The dataset supports poster layout generation, layout understanding, and relation-aware controllable generation. No public leaderboard is bundled with this Hugging Face dataset.

### Languages

Poster text and annotations are primarily Chinese (`zh`).

## Dataset Structure

### Data Instances

CGL-Dataset v2 provides `default` and `ralf-style` configurations. Load one configuration by passing its name:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/CGL-Dataset-v2", name="default")
```

### Data Fields

The `default` config contains poster images and COCO-style instance annotations with optional `text_annotations` and `text_features`.

The `ralf-style` config exposes the same data in a layout-generation format with original posters, inpainted posters, saliency maps, and annotations.

### Data Splits

| Config | Split | Rows |
| --- | --- | ---: |
| default | train | 60,548 |
| default | test | 1,035 |
| ralf-style | train | 48,438 |
| ralf-style | validation | 6,055 |
| ralf-style | test | 6,055 |
| ralf-style | no_annotation | 1,035 |

## Dataset Creation

The original release was created for relation-aware diffusion research on controllable poster layout generation. Posters are annotated with visual elements such as logos, text, underlays, embellishments, and highlighted text, and the dataset supports generation under user constraints.

## Considerations for Using the Data

The dataset is focused on advertising posters and Chinese e-commerce-style visual content. Models trained on it may inherit the visual conventions and category distribution of the source data.

## Additional Information

### Licensing Information

The dataset license is not specified in the local loader metadata. Users should verify the upstream terms before redistribution or commercial use.

### Citation Information

```bibtex
@inproceedings{li2023relation,
  title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
  author={Li, Fengheng and Liu, An and Feng, Wei and Zhu, Honghe and Li, Yaoyu and Zhang, Zheng and Lv, Jingjing and Zhu, Xin and Shen, Junjie and Lin, Zhangang},
  booktitle={Proceedings of the 32nd ACM international conference on information & knowledge management},
  pages={1249--1258},
  year={2023}
}
```

### Contributions

Thanks to [liuan0803](https://github.com/liuan0803) for creating the original dataset.
