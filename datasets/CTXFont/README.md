---
# Core fields
language:
  - en
license: unknown
pretty_name: CTXFont
tags:
  - design
  - typography
  - font-prediction
  - web-design
  - graphic-design
  - context-aware

# Recommended fields
annotations_creators:
  - machine-generated
language_creators:
  - found
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - other
task_ids: []
---

# Dataset Card for CTXFont

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

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

- **Homepage:** https://github.com/nanxuanzhao/CTXFont-dataset
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/CTXFont
- **Paper (Preprint):** <!-- No arXiv preprint available -->
- **Paper (Pacific Graphics 2018):** https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13576
- **Leaderboard:** N/A
- **Point of Contact:** Nanxuan Zhao (contact information not publicly available)

### Dataset Summary

CTXFont (Context Font) is a dataset for studying font selection in the context of web design. It contains 1,065 professional web designs from awwwards.com with annotations for 4,893 text elements. Each text element is annotated with font properties including font face, color (RGBA), and size, along with contextual information such as HTML tags, design tags, and element positioning. The dataset includes 492 unique font faces and provides 40-dimensional font face embeddings learned using an autoencoder. Web design screenshots are included at 768×1366 resolution.

The dataset was created for the task of predicting font properties (face, color, size) that match a given web design context, enabling automatic font selection systems that consider the visual and semantic context of the design.

The dataset is split into training (4,268 examples) and test (625 examples) sets.

### Supported Tasks and Leaderboards

- **Font Property Prediction**: The dataset can be used to train models that predict font properties (font face, color, size) for text elements in web designs based on visual and semantic context. The original paper uses a multi-task deep neural network with adversarial learning.
- **Font Face Prediction**: Predict which font face best matches a given design context
- **Font Color Prediction**: Predict RGB color values for text that fits the design
- **Font Size Prediction**: Predict appropriate font size for text elements

No public leaderboard is currently available for this dataset.

### Languages

The text content in the dataset is primarily in English, though the dataset focuses on visual and typographic properties rather than language modeling. Design tags and HTML tags are also in English.

## Dataset Structure

### Data Instances

A typical example from the dataset:

```python
{
  'design_name': '1003_2.png',
  'design_image': <PIL.Image.Image image mode=RGB size=768x1366>,
  'design_url': 'http://example.com/design',
  'awwward_url': 'https://www.awwwards.com/sites/...',
  'design_tags': [1, 0, 0, ...],  # 54-dimensional binary vector
  'text_content': 'WE ARE A CREATIVE DIGITAL AGENCY',
  'html_tags': [0, 0, 1, ...],  # 10-dimensional binary vector
  'font_face': 'Roboto',
  'font_size': 12.0,
  'font_color_r': 210,
  'font_color_g': 175,
  'font_color_b': 146,
  'font_color_a': 255,
  'font_face_embedding': [0.123, -0.456, ...],  # 40-dimensional embedding
  'center_x': 113,
  'center_y': 200,
  'width': 220.0,
  'height': 44
}
```

### Data Fields

- `design_name` (string): Filename of the web design screenshot (e.g., "1003_2.png")
- `design_image` (image): Screenshot of the web design at 768×1366 resolution (PNG format)
- `design_url` (string): URL of the original website
- `awwward_url` (string): URL on awwwards.com
- `design_tags` (sequence of uint8): 54-dimensional binary vector representing design characteristics (e.g., "colorful", "minimalist")
- `text_content` (string): The actual text content of the element
- `html_tags` (sequence of uint8): 10-dimensional binary vector representing HTML tag (e.g., h1, p, a)
- `font_face` (string): Name of the font face used
- `font_size` (float32): Font size in pixels
- `font_color_r` (uint8): Red channel of font color (0-255)
- `font_color_g` (uint8): Green channel of font color (0-255)
- `font_color_b` (uint8): Blue channel of font color (0-255)
- `font_color_a` (uint8): Alpha channel of font color (0-255)
- `font_face_embedding` (sequence of float32): 40-dimensional embedding of the font face learned via autoencoder
- `center_x` (uint16): X-coordinate of the element's center position
- `center_y` (uint16): Y-coordinate of the element's center position
- `width` (float32): Width of the text element in pixels
- `height` (uint16): Height of the text element in pixels

### Data Splits

The dataset is split into two sets:

|          | train | test |
| -------- | ----: | ---: |
| Examples | 4,268 |  625 |

The split is based on unique web designs, ensuring that all text elements from the same design appear in the same split.

## Dataset Creation

### Curation Rationale

The dataset was created to enable research on context-aware font selection for web design. Traditional font selection tools model fonts in isolation without considering the visual and semantic context where they are used. This dataset enables the development of systems that can automatically suggest fonts that match the style, mood, and purpose of a given web design.

### Source Data

The source data consists of professional web designs from awwwards.com, a platform where web designers submit their work for peer review and recognition.

#### Initial Data Collection and Normalization

The authors collected 1,065 web designs from awwwards.com, capturing screenshots at 768×1366 resolution (the most common screen resolution at the time). They automatically extracted font properties and text element information by parsing HTML source files. The dataset includes:

1. Screenshots of web designs
2. Annotations extracted from HTML/CSS: font face, size, color, position, HTML tags
3. Design tags provided by designers to describe the design characteristics
4. Font face embeddings learned using an autoencoder trained on 35,364 TrueType fonts

Not all fonts shown on webpages could be captured, as some may be embedded in images.

#### Who are the source language producers?

The source content was created by professional web designers who submitted their work to awwwards.com. These designers represent the global web design community and created the designs for various clients and purposes.

### Annotations

The annotations consist of font properties and contextual information for text elements on web designs.

#### Annotation process

The annotations were automatically extracted from HTML and CSS source files of the web designs. For each text element visible on a webpage, the following were extracted:

- Font properties (face, color, size) from CSS
- HTML tag enclosing the text
- Position and bounding box from rendered layout
- Design tags were provided by the designers themselves when submitting to awwwards.com

Font face embeddings were computed using a separately trained autoencoder network.

#### Who are the annotators?

The annotations are machine-generated from HTML/CSS parsing. The design tags were provided by the original web designers who created the designs.

### Personal and Sensitive Information

[More Information Needed]

<!-- State whether the dataset uses identity categories and, if so, how the information is used. Describe where this information comes from (i.e. self-reporting, collecting from profiles, inferring, etc.). See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender. State whether the data is linked to individuals and whether those individuals can be identified in the dataset, either directly or indirectly (i.e., in combination with other data).

State whether the dataset contains other data that might be considered sensitive (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history).

If efforts were made to anonymize the data, describe the anonymization process. -->

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

<!-- Please discuss some of the ways you believe the use of this dataset will impact society.

The statement should include both positive outlooks, such as outlining how technologies developed through its use may improve people's lives, and discuss the accompanying risks. These risks may range from making important decisions more opaque to people who are affected by the technology, to reinforcing existing harmful biases (whose specifics should be discussed in the next section), among other considerations.

Also describe in this section if the proposed dataset contains a low-resource or under-represented language. If this is the case or if this task has any impact on underserved communities, please elaborate here. -->

### Discussion of Biases

[More Information Needed]

<!-- Provide descriptions of specific biases that are likely to be reflected in the data, and state whether any steps were taken to reduce their impact.

For Wikipedia text, see for example [Dinan et al 2020 on biases in Wikipedia (esp. Table 1)](https://arxiv.org/abs/2005.00614), or [Blodgett et al 2020](https://www.aclweb.org/anthology/2020.acl-main.485/) for a more general discussion of the topic.

If analyses have been run quantifying these biases, please add brief summaries and links to the studies here. -->

### Other Known Limitations

[More Information Needed]

<!-- If studies of the datasets have outlined other limitations of the dataset, such as annotation artifacts, please outline and cite them here. -->

## Additional Information

### Dataset Curators

[More Information Needed]

<!-- List the people involved in collecting the dataset and their affiliation(s). If funding information is known, include it here. -->

### Licensing Information

[More Information Needed]

<!-- Provide the license and link to the license webpage if available. -->

### Citation Information

```bibtex
@article{zhao2018modeling,
  title={Modeling Fonts in Context: Font Prediction on Web Designs},
  author={Zhao, Nanxuan and Cao, Ying and Lau, Rynson W.H.},
  journal={Computer Graphics Forum},
  volume={37},
  number={7},
  year={2018},
  publisher={The Eurographics Association and John Wiley \& Sons Ltd.}
}
```

### Contributions

Thanks to [@nanxuanzhao](https://github.com/nanxuanzhao) for adding this dataset.
