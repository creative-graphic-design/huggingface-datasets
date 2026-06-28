---
annotations_creators:
  - crowdsourced
language:
  - zh
language_creators:
  - found
license:
  - cc-by-nc-sa-4.0
multilinguality:
  - monolingual
pretty_name: CGL-Dataset
size_categories:
  - 10K<n<100K
source_datasets:
  - original
tags:
  - graphic-design
  - poster
  - layout-generation
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

# Dataset Card for CGL-Dataset

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://github.com/minzhouGithub/CGL-GAN
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/CGLDataset
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset
- **Paper (arXiv):** https://arxiv.org/abs/2205.00303
- **Paper (IJCAI 2022):** https://www.ijcai.org/proceedings/2022/692

### Dataset Summary

CGL-Dataset is a poster layout dataset released with *Composition-aware Graphic Layout GAN for Visual-Textual Presentation Designs*. The paper studies layout generation for a given image, emphasizing that both global semantics and spatial image composition affect where graphic elements should be placed. The original dataset contains 60,548 advertising posters with annotated layout information.

### Supported Tasks and Leaderboards

The dataset supports poster layout generation and layout-conditioned graphic design modeling. No public leaderboard is bundled with this Hugging Face dataset.

### Languages

Poster text is primarily Chinese (`zh`).

## Dataset Structure

### Data Instances

CGL-Dataset provides `default` and `ralf-style` configurations. Load one configuration by passing its name:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/CGL-Dataset", name="default")
```

### Data Fields

The `default` config contains `image_id`, `file_name`, `width`, `height`, `image`, and COCO-style `annotations`.

The `ralf-style` config provides original posters, inpainted posters, saliency maps, and annotations for layout-generation pipelines.

### Data Splits

| Config | Split | Rows |
| --- | --- | ---: |
| default | train | 54,546 |
| default | validation | 6,002 |
| default | test | 1,000 |
| ralf-style | train | 48,438 |
| ralf-style | validation | 6,055 |
| ralf-style | test | 6,055 |
| ralf-style | no_annotation | 1,000 |

## Dataset Creation

The dataset was created for Composition-aware Graphic Layout GAN research. It provides visual element categories and positions for poster layout generation, enabling models to synthesize text and decorative layouts conditioned on image content rather than using template-only rules.

## Considerations for Using the Data

The data focuses on advertising poster layouts and may reflect the visual conventions of the source domain.

## Additional Information

### Licensing Information

The dataset card uses the CC BY-NC-SA 4.0 metadata from the local loader.

### Citation Information

```bibtex
@inproceedings{ijcai2022p692,
  title     = {Composition-aware Graphic Layout GAN for Visual-Textual Presentation Designs},
  author    = {Zhou, Min and Xu, Chenchen and Ma, Ye and Ge, Tiezheng and Jiang, Yuning and Xu, Weiwei},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {4995--5001},
  year      = {2022},
  doi       = {10.24963/ijcai.2022/692},
  url       = {https://doi.org/10.24963/ijcai.2022/692}
}
```

### Contributions

Thanks to [minzhouGithub](https://github.com/minzhouGithub) for creating the original dataset.
