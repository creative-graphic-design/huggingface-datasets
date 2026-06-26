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
size_categories: []
source_datasets:
- original
tags:
- graphic-design
- layout-generation
- poster-generation
task_categories:
- other
task_ids: []
---

# Dataset Card for CGL-Dataset


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

- **Homepage:** https://github.com/minzhouGithub/CGL-GAN
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/CGLDataset
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset
- **Paper (Preprint):** https://arxiv.org/abs/2205.00303
- **Paper (IJCAI2022):** https://www.ijcai.org/proceedings/2022/692

### Dataset Summary

The CGL-Dataset is a dataset used for the task of automatic graphic layout design for advertising posters. It contains 61,548 samples and is provided by Alibaba Group.

### Supported Tasks and Leaderboards

The task is to generate high-quality graphic layouts for advertising posters based on clean product images and their visual contents. The training set and validation set are collections of 60,548 e-commerce advertising posters, with manual annotations of the categories and positions of elements (such as logos, texts, backgrounds, and embellishments on the posters). Note that the validation set also consists of posters, not clean product images. The test set contains 1,000 clean product images without graphic elements such as logos or texts, consistent with real application data.

### Languages

[More Information Needed]


## Dataset Structure

### Data Instances

[More Information Needed]


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

[More Information Needed]


### Citation Information

```bibtex
@inproceedings{ijcai2022p692,
  title     = {Composition-aware Graphic Layout GAN for Visual-Textual Presentation Designs},
  author    = {Zhou, Min and Xu, Chenchen and Ma, Ye and Ge, Tiezheng and Jiang, Yuning and Xu, Weiwei},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4995--5001},
  year      = {2022},
  month     = {7},
  note      = {AI and Arts},
  doi       = {10.24963/ijcai.2022/692},
  url       = {https://doi.org/10.24963/ijcai.2022/692},
}
```

### Contributions

Thanks to [@minzhouGithub](https://github.com/minzhouGithub) for adding this dataset.
