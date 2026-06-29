---
annotations_creators:
  - crowdsourced
language:
  - en
language_creators:
  - found
license:
  - apache-2.0
pretty_name: GraphicDesignEvaluation
size_categories:
  - n<1K
source_datasets:
  - original
tags:
  - graphic-design-evaluation
  - design-principles
  - human-annotation
task_categories:
  - image-to-text
configs:
  - config_name: absolute-gpt-alignment
    data_files:
      - split: train
        path: absolute-gpt-alignment/train-*
  - config_name: absolute-gpt-overlap
    data_files:
      - split: train
        path: absolute-gpt-overlap/train-*
  - config_name: absolute-gpt-whitespace
    data_files:
      - split: train
        path: absolute-gpt-whitespace/train-*
  - config_name: absolute-human-alignment
    data_files:
      - split: train
        path: absolute-human-alignment/train-*
  - config_name: absolute-human-overlap
    data_files:
      - split: train
        path: absolute-human-overlap/train-*
  - config_name: absolute-human-whitespace
    data_files:
      - split: train
        path: absolute-human-whitespace/train-*
  - config_name: relative-gpt-alignment
    data_files:
      - split: train
        path: relative-gpt-alignment/train-*
  - config_name: relative-gpt-overlap
    data_files:
      - split: train
        path: relative-gpt-overlap/train-*
  - config_name: relative-gpt-whitespace
    data_files:
      - split: train
        path: relative-gpt-whitespace/train-*
  - config_name: relative-human-alignment
    data_files:
      - split: train
        path: relative-human-alignment/train-*
  - config_name: relative-human-overlap
    data_files:
      - split: train
        path: relative-human-overlap/train-*
  - config_name: relative-human-whitespace
    data_files:
      - split: train
        path: relative-human-whitespace/train-*
---

# Dataset Card for GraphicDesignEvaluation

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://cyberagentailab.github.io/Graphic-design-evaluation/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/GraphicDesignEvaluation
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/GraphicDesignEvaluation
- **Paper (arXiv):** https://arxiv.org/abs/2410.08885
- **Paper (SIGGRAPH Asia 2024):** https://doi.org/10.1145/3681758.3698010

### Dataset Summary

GraphicDesignEvaluation is a human-rated benchmark released with *Can GPTs Evaluate Graphic Design Based on Design Principles?*. The paper compares GPT-based evaluation and heuristic metrics against human ratings for three representative design principles: alignment, overlap, and white space. The dataset contains graphic banner designs curated from an online service, perturbed low-quality variants, and human annotations collected from 60 subjects.

### Supported Tasks and Leaderboards

The dataset supports graphic design quality evaluation, human/model score correlation analysis, and design-principle-specific assessment. No public leaderboard is bundled with this Hugging Face dataset.

### Languages

Annotations and evaluation descriptions are in English (`en`).

## Dataset Structure

### Data Instances

Configs are named as `{eval_type}-{annotation_type}-{design_principle}`. Use `absolute` or `relative`, `gpt` or `human`, and `alignment`, `overlap`, or `whitespace`. Load one configuration by passing its name:

```python
import datasets as ds

dataset = ds.load_dataset(
    "creative-graphic-design/GraphicDesignEvaluation",
    name="absolute-gpt-alignment",
)
```

### Data Fields

Absolute configs contain `image_id`, `image`, `perturbation`, `scores`, and `avg`. Relative configs contain `image_id`, `image`, `comparative`, `scores`, and `avg`.

### Data Splits

All configs expose a single `train` split. Absolute configs have 400 rows each; relative configs have 300 rows each.

## Dataset Creation

The dataset was created to study whether GPT-based evaluators can assess graphic design quality according to core design principles and how those scores compare with human annotations.

## Considerations for Using the Data

The dataset is small and principle-specific. It should be used as an evaluation resource rather than a complete measure of graphic design quality.

## Additional Information

### Licensing Information

The local loader lists the dataset license as Apache 2.0.

### Citation Information

```bibtex
@inproceedings{haraguchi2024can,
  title={Can GPTs Evaluate Graphic Design Based on Design Principles?},
  author={Haraguchi, Daichi and Inoue, Naoto and Shimoda, Wataru and Mitani, Hayato and Uchida, Seiichi and Yamaguchi, Kota},
  booktitle={SIGGRAPH Asia 2024 Technical Communications},
  pages={1--4},
  year={2024}
}
```
