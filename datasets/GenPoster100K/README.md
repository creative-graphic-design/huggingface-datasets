---
language:
  - en
license: cc-by-nc-4.0
pretty_name: GenPoster100K
tags:
  - graphic-design
  - layout-generation
  - content-aware-layout-generation
  - multimodal
  - poster-design
annotations_creators:
  - machine-generated
language_creators:
  - found
size_categories:
  - 100K<n<1M
source_datasets:
  - original
task_categories:
  - text-to-image
  - image-to-text
task_ids: []
---

# Dataset Card for GenPoster100K

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Table of Contents

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

- **Homepage:** https://huggingface.co/datasets/BruceW91/GenPoster-100K
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/GenPoster100K
- **Paper (Preprint):** https://arxiv.org/abs/2510.15749
- **Paper (Conference):** https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SEGA_A_Stepwise_Evolution_Paradigm_for_Content-Aware_Layout_Generation_with_ICCV_2025_paper.html
- **Leaderboard:** No official public leaderboard URL is documented for GenPoster-100K.
- **Point of Contact:** Original dataset/project authors (paper) and maintainers of the source HF dataset (`BruceW91/GenPoster-100K`).

### Dataset Summary

GenPoster-100K is a large-scale dataset for content-aware graphic layout generation introduced in the SEGA paper.
The paper describes it as a high-quality poster dataset with layer-parseable source materials and rich metadata.

This repository provides a Hugging Face `datasets` loader implementation that reads the source release (`BruceW91/GenPoster-100K`) and exposes normalized examples with:

- poster background image (`background_image`)
- background/layer composite image (`merged_image`)
- PSD reference path (`psd_path`)
- region boxes (`regions`)
- layer-level annotations (`layers`) including text, bbox, typography, color, and per-layer rendered image

Implementation note: this loader uses `0503_raw_offline.pkl` and `part_*.tar.gz`, yielding 102,703 rows in `train`.

### Supported Tasks and Leaderboards

- `other:content-aware-layout-generation`: Generate or refine poster element layouts conditioned on background imagery and textual element metadata.
- `text-to-image`: Can be used in poster design workflows where textual content and structured attributes guide generated visual composition.
- `image-to-text`: Can support structured extraction/understanding tasks over design layers and poster metadata.

No official leaderboard URL specific to GenPoster-100K is currently provided in the public source materials.

### Languages

- Frontmatter language is set to English (`en`) because the released examples predominantly contain English poster text.
- The dataset may include additional languages in real-world templates, but a full language distribution is not documented in the source materials.

## Dataset Structure

### Data Instances

Each example includes image assets and structured layer metadata.

```json
{
  "id": 0,
  "background_image": "<image>",
  "merged_image": "<image>",
  "psd_path": "big_poster/meta_psd/3841272.psd",
  "regions": [[1656, 481, 2545, 855]],
  "layers": [
    {
      "layer_name": "&#wText&#wTitle",
      "text": "Super price!",
      "bbox": [1754, 573, 2423, 689],
      "angle": 0,
      "psd_size": [3508, 2480],
      "stroke_width": 0.0,
      "font": "Aftaserif",
      "font_size": 113.29,
      "tracking": 0.0,
      "justification": 1,
      "fill_color": [0.0, 0.0, 0.0, 1.0],
      "layer_image": "<image>",
      "label": "Calls to Action"
    }
  ]
}
```

### Data Fields

- `id` (`int32`): Example identifier assigned by loader order.
- `background_image` (`Image`): Rendered background image for the poster.
- `merged_image` (`Image`): Background image composited with available rendered layer images.
- `psd_path` (`string`): Relative PSD path recorded in annotations.
- `regions` (`Sequence[Sequence[int32]]`): Region boxes as `[x1, y1, x2, y2]`.
- `layers` (`Sequence[struct]`): Layer-level annotations.
  - `layer_name` (`string`)
  - `text` (`string`)
  - `bbox` (`Sequence[int32]`, length=4)
  - `angle` (`int32`)
  - `psd_size` (`Sequence[int32]`, length=2)
  - `stroke_width` (`float32`)
  - `font` (`string`)
  - `font_size` (`float32`)
  - `tracking` (`float32`)
  - `justification` (`int32`)
  - `fill_color` (`Sequence[float32]`, length=4)
  - `layer_image` (`Image`)
  - `label` (`ClassLabel`): one of `Bodytext`, `Calls to Action`, `Date`, `Detailed items`, `Location`, `Menu Items`, `Name`, `Others`, `Phone number`, `Social Media`, `Subtitle`, `Title`, or `Website`

### Data Splits

This implementation exposes a single `train` split from the upstream release.

| Split | Rows |
| --- | ---: |
| train | 102,703 |

Notes:

- The paper reports 105,456 posters in the broader GenPoster-100K corpus.
- The HF source release used by this loader provides `0503_raw_offline.pkl` with 102,703 records.
- The original source data is `train`-only, and this loader preserves that split.

## Dataset Creation

### Curation Rationale

According to the SEGA paper, GenPoster-100K was introduced to improve data quality and scale for content-aware layout generation.
The paper highlights limitations in earlier datasets (for example, artifacts from inpainted backgrounds and less structured metadata) and positions GenPoster-100K as a higher-fidelity, large-scale alternative with rich component-level information.

### Source Data

The source release includes poster metadata archives (`part_*.tar.gz`) and annotation pickle files.
A disclaimer in the source dataset indicates copyright belongs to the original owner (Freepik) and commercial use may require additional permission.

#### Initial Data Collection and Normalization

From public materials:

- The paper describes data built from layer-parseable source materials with hierarchical metadata.
- The source HF dataset release distributes image assets split across 79 archive parts (`part_0.tar.gz` ... `part_78.tar.gz`).

In this loader implementation:

- Annotation source is `0503_raw_offline.pkl`.
- URL query strings are normalized/removed and image paths are resolved against extracted archive contents.
- BBox/size/color fields are normalized to fixed lengths and numeric dtypes.

#### Who are the source language producers?

The textual content appears to originate from poster templates designed by content creators in the source design corpus.
No demographic metadata for these creators is provided in the public release.

### Annotations

The release provides machine-readable layer metadata per poster example, including geometry and typography attributes.
The dataset card metadata marks annotation creation as machine-generated.

#### Annotation process

Public paper/source materials indicate hierarchical metadata extracted from PSD-parseable design sources.
The implementation-level annotation fields include:

- text and layer names
- bounding boxes and angle
- typography attributes (font, size, tracking, justification, stroke)
- RGBA fill color
- region boxes
- references to rendered background/layer images

Detailed internal annotation tooling and QA workflow are not fully specified in the public documents.

#### Who are the annotators?

Annotations are primarily machine-generated from source design assets.
Named individual annotators are not documented.

### Personal and Sensitive Information

The dataset is composed of poster design assets and textual elements.
It is not released as a personal-data dataset, but real-world template text may include names, brands, or contact-like strings depending on source content.
Users should perform downstream filtering/redaction if their use case requires stricter privacy constraints.

## Considerations for Using the Data

### Social Impact of Dataset

Potential positive impact:

- Enables research on automated graphic design and multimodal layout understanding.
- Supports reproducible benchmarking for content-aware layout generation.

Potential risks:

- May be used to generate misleading or low-quality promotional content at scale.
- Could be misused for style imitation or copyright-sensitive commercial outputs.

### Discussion of Biases

Likely biases include:

- Domain/style bias toward stock-template aesthetics and marketing layouts.
- Language/domain imbalance (predominantly English and commercial poster styles).
- Visual-cultural bias inherited from source platforms and curation choices.

No public bias audit report specific to this release is currently documented.

### Other Known Limitations

- The upstream source is train-only; this loader follows that source format, while our Hub publication additionally provides an `8:1:1` split.
- Paper-level total (105,456) and loader-level total (102,703) differ due to release artifacts used by this loader.
- `psd_path` is recorded as metadata path, but only a limited subset of raw PSD files is present in the public source release.
- License constraints limit direct commercial usage without additional permission from the copyright owner.

## Additional Information

### Dataset Curators

- Original dataset/paper authors: Wang et al. (SEGA, ICCV 2025).
- Source HF release contributor: [@BruceW91](https://huggingface.co/BruceW91).
- This repository contains a community-maintained Hugging Face dataset loader implementation.

### Licensing Information

- Dataset card license: `cc-by-nc-4.0`
- Source disclaimer indicates use is intended for academic purposes, and commercial use may require prior permission from the copyright owner (Freepik): https://www.freepik.com/

Always verify license compatibility with your intended use before redistribution or deployment.

### Citation Information

If your implementation is based on this dataset, please cite the original paper and the Hugging Face dataset implementation:

```bibtex
@inproceedings{wang2025sega,
  title={SEGA: A Stepwise Evolution Paradigm for Content-Aware Layout Generation with Design Prior},
  author={Wang, Haoran and Zhao, Bo and Wang, Jinghui and Wang, Hanzhang and Yang, Huan and Ji, Wei and Liu, Hao and Xiao, Xinyan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19321--19330},
  year={2025}
}
```

```bibtex
@misc{genposter100kdataset,
  title = {GenPoster100K dataset},
  author = {{Creative Graphic Design Lab} and Kitada, Shunsuke},
  howpublished = {Hugging Face dataset},
  year = {2025},
  note = {URL: https://huggingface.co/datasets/creative-graphic-design/GenPoster100K},
}
```

### Contributions

Thanks to the original GenPoster-100K authors and [@BruceW91](https://huggingface.co/BruceW91) for releasing the source dataset.
This Hugging Face dataset implementation was created for the `creative-graphic-design/huggingface-datasets` monorepo.
