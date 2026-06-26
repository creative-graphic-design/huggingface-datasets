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
size_categories: []
source_datasets:
- CGL-Dataset
tags:
- graphic design
task_categories:
- other
task_ids: []
---

# Dataset Card for CGL-Dataset-v2


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

- **Homepage:** https://github.com/liuan0803/RADM
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/CGLDatasetV2
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset-v2
- **Paper (Preprint):** https://arxiv.org/abs/2306.09086
- **Paper (CIKM'23):** https://dl.acm.org/doi/10.1145/3583780.3615028

### Dataset Summary

CGL-Dataset V2 is a dataset for the task of automatic graphic layout design of advertising posters, containing 60,548 training samples and 1035 testing samples. It is an extension of CGL-Dataset.

### Supported Tasks and Leaderboards

[More Information Needed]


### Languages

The language data in CGL-Dataset v2 is in Chinese ([BCP-47 zh](https://www.rfc-editor.org/info/bcp47)).

## Dataset Structure

### Data Instances


```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/CGL-Dataset-v2")
```


### Data Fields

[More Information Needed]


### Data Splits

[More Information Needed]


## Dataset Creation

### Curation Rationale

The CGL-Dataset V2 was curated to address the limitations of previous datasets and to support the development of advanced models for automatic poster layout generation. By incorporating text content annotations and creating clean background images, the dataset enables the generation of high-quality, visually balanced, and informative poster layouts. This dataset is a significant contribution to the field, facilitating research and development in automatic graphic design.

### Source Data

[More Information Needed]


- Poster Images: The dataset contains a large collection of poster images specifically designed for advertising purposes. These images are annotated with various graphic elements such as logos, texts, underlays, and embellishments.
- Textual Content: The textual content primarily focuses on promotional slogans and descriptions relevant to the e-commerce field. This content is crucial for studying the influence of text on poster layout design.
- Element Annotations: Each poster image is annotated with detailed information about the graphic elements, including their categories and coordinates. This helps in understanding the spatial relationships between different elements on the poster.

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
@inproceedings{li2023relation,
  title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
  author={Li, Fengheng and Liu, An and Feng, Wei and Zhu, Honghe and Li, Yaoyu and Zhang, Zheng and Lv, Jingjing and Zhu, Xin and Shen, Junjie and Lin, Zhangang},
  booktitle={Proceedings of the 32nd ACM international conference on information & knowledge management},
  pages={1249--1258},
  year={2023}
}
```

### Contributions

Thanks to [@liuan0803](https://github.com/liuan0803) for creating this dataset.
