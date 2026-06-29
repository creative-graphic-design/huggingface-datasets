---
annotations_creators:
  - expert-generated
language:
  - zh
language_creators:
  - found
license:
  - cc-by-sa-4.0
pretty_name: PKU-PosterLayout
size_categories:
  - 10K<n<100K
source_datasets:
  - extended|PosterErase
tags:
  - layout-generation
  - graphic-design
  - poster
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

# Dataset Card for PKU-PosterLayout

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** http://59.108.48.34/tiki/PosterLayout/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PKUPosterLayout
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout
- **Paper (arXiv):** https://arxiv.org/abs/2303.15937
- **Paper (CVPR 2023):** https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html

### Dataset Summary

PKU-PosterLayout is a content-aware visual-textual poster layout benchmark released with *PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout*. The paper defines the task as arranging predefined text, logo, and underlay elements on a non-empty poster canvas while considering both inter-element and inter-layer relationships. The original benchmark contains 9,974 poster-layout pairs and 905 non-empty canvas images.

This Hugging Face release exposes the repository loader output as parquet files. It includes the original loader-style `default` config and a `ralf-style` config for downstream layout-generation workflows.

### Supported Tasks and Leaderboards

The dataset supports poster layout generation and layout-conditioned image editing. No public leaderboard is bundled with this Hugging Face packaging.

### Languages

Poster text is primarily Chinese (`zh`).

## Dataset Structure

### Data Instances

PKU-PosterLayout provides `default` and `ralf-style` configurations. Load one configuration by passing its name:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/PKU-PosterLayout", name="default")
```

### Data Fields

The `default` config contains `original_poster`, `inpainted_poster`, `basnet_saliency_map`, `pfpn_saliency_map`, `canvas`, and `annotations`.

The `ralf-style` config contains `image_id`, `original_poster`, `inpainted_poster`, `canvas`, `saliency_map`, `saliency_map_sub`, and `annotations`.

### Data Splits

| Config | Split | Rows |
| --- | --- | ---: |
| default | train | 9,974 |
| default | test | 905 |
| ralf-style | train | 7,972 |
| ralf-style | validation | 996 |
| ralf-style | test | 997 |
| ralf-style | no_annotation | 905 |

## Dataset Creation

The original dataset extends PosterErase with layout annotations for content-aware poster layout generation. The annotations describe visual-textual elements and their positions on poster canvases.

## Considerations for Using the Data

The dataset is focused on Chinese poster layouts and may not represent other languages, writing systems, or design cultures. Images and poster text remain subject to the original dataset terms.

## Additional Information

### Licensing Information

Images in PKU-PosterLayout are distributed under the CC BY-SA 4.0 license according to the local loader metadata.

### Citation Information

```bibtex
@inproceedings{hsu2023posterlayout,
  title={PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}
```

### Contributions

Thanks to [PKU-ICST-MIPL](https://github.com/PKU-ICST-MIPL) for creating the original dataset.
