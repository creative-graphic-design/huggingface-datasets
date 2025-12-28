---
annotations_creators:
  - machine-generated
language:
  - en
language_creators:
  - machine-generated
license:
  - mit
multilinguality: []
pretty_name: BannerRequest400
size_categories:
  - n<1K
source_datasets:
  - original
tags:
  - banner generation
  - advertising
  - multimodal
  - graphic design
  - llm agents
task_categories:
  - image-to-text
  - text-to-image
task_ids: []
---

# Dataset Card for BannerRequest400

## Table of Contents

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
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://github.com/sony/BannerAgency
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/BannerRequest400
- **Paper (EMNLP'25):** https://aclanthology.org/2025.emnlp-main.214/
- **Paper (arXiv):** https://arxiv.org/abs/2503.11060

### Dataset Summary

BannerRequest400 is the first multimodal benchmark specifically designed to evaluate advertising banner generation systems. It addresses gaps in existing design datasets by providing both visual and textual inputs.

The dataset pairs **100 unique brand logos** (provided in both PNG and SVG formats) with **400 diverse banner design requests** (textual modality). The logos were synthetically generated using Claude 3.5 Sonnet, then refined by experts to ensure authentic aesthetics and avoid bias.

Researchers leveraged GPT-4o to expand each design intention into four distinct banner requests targeting different audience-purpose combinations. This systematic expansion yielded 400 diverse specifications, further extended across **13 standard banner dimensions** for a comprehensive evaluation set of **5,200 multimodal banner specifications**.

The benchmark enables the first rigorous evaluation of banner generation approaches across diverse design requests, multiple contexts (different audience segments and campaign purposes), and industry-standard display sizes.

### Supported Tasks and Leaderboards

This dataset can be used for:

- **Banner Generation**: Generate advertising banners from textual design requests and brand logos
- **Multimodal Design Evaluation**: Evaluate the quality of generated banners against detailed specifications
- **Design Specification Understanding**: Train models to understand structured design requirements

### Languages

The dataset is in English (en). All banner requests, target audience descriptions, and design specifications are written in English.

## Dataset Structure

### Data Instances

The dataset provides two configurations:

**Abstract Configuration** (400 examples):

```json
{
  "id": 1,
  "banner_request": "Design a banner ad image of size 300x250 for a discussion on the ethical issues surrounding artificial intelligence...",
  "logo_png": <PIL.Image>,
  "logo_svg": "<svg>...</svg>"
}
```

**Concrete Configuration** (100 examples):

```json
{
  "id": 1,
  "banner_request": "Design a banner ad image for a discussion on the ethical issues surrounding artificial intelligence",
  "advertiser": "ETHIC AI",
  "logo_name": "001_ethicai.png",
  "logo_description": "A shield-shaped logo with geometric diamond pattern...",
  "logo_png": <PIL.Image>,
  "logo_svg": "<svg>...</svg>",
  "advertising_variations": {
    "pair_1": {
      "target_audience": "Tech enthusiasts and AI developers",
      "primary_purpose": "Event registration and networking",
      "concrete_request_300x250": "Design a banner ad image...",
      "concrete_request_728x90": "Design a banner ad image...",
      ...
    },
    "pair_2": {...},
    "pair_3": {...},
    "pair_4": {...}
  }
}
```

### Data Fields

**Abstract Configuration:**

- `id` (int32): Unique identifier (1-400)
- `banner_request` (string): Text description of the banner design request
- `logo_png` (Image): Brand logo in PNG format (PIL Image object)
- `logo_svg` (string): Brand logo in SVG format (XML string)

**Concrete Configuration:**

- `id` (int32): Unique campaign identifier (1-100)
- `banner_request` (string): Base banner design request
- `advertiser` (string): Brand/advertiser name
- `logo_name` (string): Logo filename
- `logo_description` (string): Detailed visual description of the logo
- `logo_png` (Image): Brand logo in PNG format
- `logo_svg` (string): Brand logo in SVG format
- `advertising_variations` (nested structure):
  - `pair_1` through `pair_4`: Each contains:
    - `target_audience` (string): Description of target demographic
    - `primary_purpose` (string): Marketing goal/call-to-action
    - `concrete_request_<dimension>` (string): Design request for specific banner size
      - Available dimensions: 300x250, 728x90, 160x600, 300x600, 970x250, 320x100, 468x60, 250x250, 336x280, 120x600, 970x90, 180x150, 300x50

### Data Splits

The dataset contains only a training split:

| Configuration | Train |
| ------------- | ----: |
| abstract      |   400 |
| concrete      |   100 |

The abstract configuration provides 400 individual banner requests with logos (cycling through 100 logos 4 times).
The concrete configuration provides 100 campaigns, each with 4 audience variations × 13 banner dimensions = 5,200 total specifications.

## Dataset Creation

### Curation Rationale

BannerRequest400 was created to fill a critical gap in banner generation evaluation. Existing datasets for graphic design focus primarily on single-image generation without considering the multimodal nature of advertising design, which requires integrating brand logos with textual specifications.

The dataset enables:

1. Evaluation of multimodal LLM agents in advertising banner generation
2. Systematic assessment across diverse audience segments and campaign purposes
3. Standardized testing across industry-relevant banner dimensions
4. Research into design intention understanding and visual communication

### Source Data

The dataset origins trace to 100 design intentions from the DESIGNERINTENTION dataset. These were:

1. Synthetically expanded by GPT-4o into 400 diverse banner requests (4 variations per intention)
2. Paired with 100 brand logos generated by Claude 3.5 Sonnet
3. Logos refined by experts to ensure authentic aesthetics
4. Extended across 13 standard banner dimensions (300x250, 728x90, etc.)
5. Enhanced with logo-specific visual characteristics in the design specifications

## Additional Information

### Dataset Curators

Created by researchers at Sony as part of the BannerAgency project. The dataset was curated through a combination of AI-generated content (logos via Claude 3.5 Sonnet, requests via GPT-4o) and expert refinement.

### Licensing Information

This dataset is released under the MIT License. See the [original repository](https://github.com/sony/BannerAgency) for full license details.

### Citation Information

```bibtex
@misc{wang2025banneragency,
  title={BannerAgency: Advertising Banner Design with Multimodal LLM Agents},
  author={Wang, Heng and Shimose, Yotaro and Takamatsu, Shingo},
  year={2025},
  eprint={2503.11060},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

### Contributions

This Hugging Face dataset implementation was created by the creative-graphic-design organization to make BannerRequest400 more accessible to the research community.
