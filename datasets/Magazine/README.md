---
annotations_creators:
  - machine-generated
language:
  - en
language_creators:
  - found
license:
  - unknown
multilinguality:
  - monolingual
pretty_name: Magazine
size_categories:
  - 1K<n<10K
source_datasets:
  - original
tags:
  - graphic-design
  - layout
  - content-aware
task_categories:
  - image-to-image
  - text-to-image
task_ids: []
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
---

# Dataset Card for Magazine

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://xtqiao.com/projects/content_aware_layout/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/Magazine
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/Magazine
- **Paper (SIGGRAPH 2019):** https://dl.acm.org/doi/10.1145/3306346.3322971

### Dataset Summary

Magazine is a magazine layout dataset released with *Content-aware Generative Modeling of Graphic Design Layouts*. The paper studies graphic layout generation conditioned on visual and textual content and introduces a large-scale magazine layout dataset with fine-grained layout annotations and keyword labels.

### Supported Tasks and Leaderboards

The dataset supports content-aware layout generation, graphic layout modeling, and image-conditioned design generation. No public leaderboard is bundled with this Hugging Face packaging.

### Languages

The source magazine pages and keywords are primarily English (`en`).

## Dataset Structure

### Data Fields

Each row contains `filename`, `category`, `size`, `elements`, `keywords`, and `images`.

### Data Splits

| Config | Split | Rows |
| --- | --- | ---: |
| default | train | 3,919 |

## Dataset Creation

The original release includes magazine images and fine-grained layout annotations for content-aware layout modeling.

## Considerations for Using the Data

The dataset represents magazine layouts from the upstream collection and may not cover all editorial styles, languages, or publication domains.

## Additional Information

### Licensing Information

The dataset license is not specified in the local loader metadata. Users should verify the upstream terms before redistribution or commercial use.

### Citation Information

```bibtex
@article{zheng2019content,
  title={Content-aware generative modeling of graphic design layouts},
  author={Zheng, Xinru and Qiao, Xiaotian and Cao, Ying and Lau, Rynson W. H.},
  journal={ACM Transactions on Graphics},
  volume={38},
  number={4},
  year={2019}
}
```

### Contributions

Thanks to the authors of the original Magazine layout dataset.
