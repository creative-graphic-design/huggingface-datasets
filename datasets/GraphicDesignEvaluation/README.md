---
annotations_creators:
- crowdsourced
language:
- en
language_creators:
- found
license:
- apache-2.0
multilinguality: []
pretty_name: GraphicDesignEvaluation
size_categories:
- n<1K
source_datasets:
- original
tags:
- graphic design evaluation
- design principles
- large multimodal model
- gpt-based evaluation
- human annotation
- human-rated dataset
task_categories: []
task_ids: []
--- 


# Dataset Card for GraphicDesignEvaluation

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

- **Homepage:** https://cyberagentailab.github.io/Graphic-design-evaluation/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/GraphicDesignEvaluation
- **Paper (Preprint):** https://arxiv.org/abs/2410.08885
- **Paper (SIGGRAPH Asia'24):** https://doi.org/10.1145/3681758.3698010

### Dataset Summary

The GraphicDesignEvaluation dataset evaluates whether large multimodal models (LMMs), such as GPT-4o, can assess the quality of graphic designs according to core design principles—specifically alignment, overlap, and white space.

It contains 700 banner and poster designs (100 original and 600 perturbed), collected from VistaCreate, each rated by 60 human annotators.
Each image has associated human scores (1–10 scale) and GPT-based scores for the three principles, enabling the study of correlations between human judgment, heuristic metrics, and LMM-based evaluation.

The dataset was created to benchmark the ability of GPT-based evaluators to perform reliable aesthetic judgment in visual communication design.

### Supported Tasks and Leaderboards

[More Information Needed]


### Languages

The dataset is in English (en), as both prompts and annotations are written in English. All participants and model instructions use English-language descriptions of design principles and rating guidelines.

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
@inproceedings{haraguchi2024can,
  title={Can GPTs Evaluate Graphic Design Based on Design Principles?},
  author={Haraguchi, Daichi and Inoue, Naoto and Shimoda, Wataru and Mitani, Hayato and Uchida, Seiichi and Yamaguchi, Kota},
  booktitle={SIGGRAPH Asia 2024 Technical Communications},
  pages={1--4},
  year={2024}
}
```

### Contributions

Thanks to [@DaichiHaraguchi](https://github.com/DaichiHaraguchi) for adding this dataset.
