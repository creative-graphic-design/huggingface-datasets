---
annotations_creators:
  - machine-generated
language:
  - en
language_creators:
  - found
license:
  - cdla-permissive-1.0
pretty_name: PubLayNet
size_categories:
  - 100K<n<1M
source_datasets:
  - original
tags:
  - document-layout-analysis
  - object-detection
  - segmentation
task_categories:
  - object-detection
  - image-segmentation
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

# Dataset Card for PubLayNet

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://developer.ibm.com/exchanges/data/all/publaynet/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PubLayNet
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/PubLayNet
- **Paper (arXiv):** https://arxiv.org/abs/1908.07836
- **Paper (ICDAR 2019):** https://ieeexplore.ieee.org/document/8977963

### Dataset Summary

PubLayNet is a large document layout analysis dataset built by automatically matching XML representations and PDF content from more than one million PubMed Central Open Access articles. It contains more than 360,000 document images with COCO-style annotations for common layout elements such as text, title, list, table, and figure regions.

### Supported Tasks and Leaderboards

The dataset supports document layout object detection and segmentation. No leaderboard is bundled with this Hugging Face packaging.

### Languages

Document content is primarily English (`en`), but the task is visual document layout analysis.

## Dataset Structure

### Data Instances

Load the dataset with:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/PubLayNet")
```

### Data Fields

Rows contain `image_id`, `file_name`, `width`, `height`, `image`, and COCO-style `annotations`.

### Data Splits

| Split | Rows |
| --- | ---: |
| train | 335,703 |
| validation | 11,245 |
| test | 11,405 |

## Dataset Creation

PubLayNet was created from automatically parsed document layouts and released for large-scale document layout analysis.

## Considerations for Using the Data

The dataset is document-centric and may not represent all document domains or non-English layout conventions.

## Additional Information

### Licensing Information

This dataset card uses the CDLA Permissive 1.0 license metadata from the local loader.

### Citation Information

```bibtex
@inproceedings{zhong2019publaynet,
  title={Publaynet: largest dataset ever for document layout analysis},
  author={Zhong, Xu and Tang, Jianbin and Yepes, Antonio Jimeno},
  booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
  pages={1015--1022},
  year={2019}
}
```
