---
annotations_creators:
- machine-generated
language:
- zh
language_creators:
- found
license:
- cc-by-sa-4.0
multilinguality:
- monolingual
pretty_name: PosterErase
size_categories: []
source_datasets:
- original
tags:
- graphic design
task_categories:
- other
task_ids: []
---

# Dataset Card for PosterErase


## Table of Contents
- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://github.com/alimama-creative/Self-supervised-Text-Erasing
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PosterErase
- **Paper (Preprint):** https://arxiv.org/abs/2204.12743
- **Paper (ACMMM2022):** https://dl.acm.org/doi/abs/10.1145/3503161.3547905

### Dataset Summary

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

The language data in PKU-PosterLayout is in Chinese (BCP-47 zh).

## Dataset Structure

### Data Instances

To use PosterErase dataset, you need to download the dataset via [Alibaba Cloud](https://tianchi.aliyun.com/dataset/134810).
Then place the downloaded files in the following structure and specify its path.

```
/path/to/datasets
├── erase_1.zip
├── erase_2.zip
├── erase_3.zip
├── erase_4.zip
├── erase_5.zip
└── erase_6.zip
```

```python
import datasets as ds

dataset = ds.load_dataset(
    path="creative-graphic-design/PosterErase",
    data_dir="/path/to/datasets/",
)
```

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

[More Information Needed]

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

[More Information Needed]

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

You can find the following statement in [the license section](https://tianchi.aliyun.com/dataset/134810#license) of t[he dataset distribution location](https://tianchi.aliyun.com/dataset/134810).

> The dataset is distributed under the CC BY-SA 4.0 license.

However, the license setting on that page appears to be set to [CC-BY-SA-NC 4.0](http://creativecommons.org/licenses/by-sa/4.0/?spm=a2c22.12282016.0.0.7abc5a92qnyxdR).

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

Thanks to [alimama-creative](https://github.com/alimama-creative) for creating this dataset.
