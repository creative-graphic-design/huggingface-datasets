---
language:
  - en
license: unknown
pretty_name: LayoutDETR
tags:
  - graphic-design
  - advertising
  - banner
  - layout-generation
  - object-detection
annotations_creators:
  - crowdsourced
  - machine-generated
language_creators:
  - found
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - object-detection
  - image-to-image
---

# Dataset Card for LayoutDETR

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
  - [Annotations](#annotations)
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

- **Homepage:** https://github.com/salesforce/LayoutDETR
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/LayoutDETR
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/LayoutDETR
- **Original Code and Data:** https://github.com/salesforce/LayoutDETR
- **Project Page:** https://ningyu1991.github.io/projects/LayoutDETR.html
- **Paper (ECCV 2024 / arXiv):** https://arxiv.org/abs/2212.09877
- **Leaderboard:** Not available in the original release.
- **Point of Contact:** https://github.com/creative-graphic-design/huggingface-datasets/issues

### Dataset Summary

LayoutDETR is the ad banner dataset released with *LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer*. The upstream release contains 7,672 well-designed ad banner samples. Each sample pairs a rendered banner image with foreground element annotations and corresponding inpainted background-only images.

The raw release contains three directories:

- `png_json_gt/`: banner PNG images and same-stem JSON layout annotations.
- `1x_inpainted_background_png/`: background-only images for inference and evaluation.
- `3x_inpainted_background_png/`: background-only images with extra random inpainting for training.

This Hugging Face loader exposes the raw assets and annotations. It does not reproduce LayoutDETR's preprocessed training ZIP format with cropped patches and masks.

### Supported Tasks and Leaderboards

The dataset supports content-aware graphic layout generation, where a model predicts reasonable foreground element boxes conditioned on an advertising background image and foreground text/category content. The released annotations can also be inspected as object-detection-style boxes over rendered banner images.

No active public leaderboard is bundled with this Hugging Face dataset. For exact reproduction of the original training and evaluation pipeline, use the upstream LayoutDETR repository.

### Languages

OCR text content is primarily English (`en`) advertising copy, although the upstream release does not provide a formal language distribution.

## Dataset Structure

### Data Instances

Load a locally downloaded copy by passing either the extracted dataset root or the archive path:

```python
import datasets as ds

dataset = ds.load_dataset(
    "creative-graphic-design/LayoutDETR",
    data_dir="/path/to/ads_banner_dataset",
)
```

Each row corresponds to one valid sample from the sorted raw JSON files.

```json
{
  "id": "example_stem",
  "image": "<image>",
  "image_path": ".../png_json_gt/example_stem.png",
  "background_1x": "<image or null>",
  "background_1x_path": ".../1x_inpainted_background_png/example_stem_inpainted.png",
  "background_3x": "<image or null>",
  "background_3x_path": ".../3x_inpainted_background_png/example_stem_inpainted.png",
  "width": 1024,
  "height": 1024,
  "elements": [
    {
      "text": "SHOP NOW",
      "label": "button",
      "bbox_xyxy": [120.0, 830.0, 420.0, 910.0],
      "bbox_cxcywh_normalized": [0.2637, 0.8496, 0.2930, 0.0781]
    }
  ],
  "num_elements": 1,
  "raw_annotation": "[...]"
}
```

### Data Fields

- `id` (`string`): Source file stem shared by the image and JSON annotation.
- `image` (`Image`): Well-designed ad banner image from `png_json_gt/`.
- `image_path` (`string`): Local path to the banner image.
- `background_1x` (`Image`): Optional background-only image from `1x_inpainted_background_png/`.
- `background_1x_path` (`string`): Local path to the 1x inpainted background image, or an empty string when absent.
- `background_3x` (`Image`): Optional background-only image from `3x_inpainted_background_png/`.
- `background_3x_path` (`string`): Local path to the 3x inpainted background image, or an empty string when absent.
- `width` (`int32`): Banner image width in pixels.
- `height` (`int32`): Banner image height in pixels.
- `elements` (`list`): Valid foreground elements after applying the filtering logic used by the upstream preprocessing script.
- `elements.text` (`string`): OCR text content from the source JSON `str` field.
- `elements.label` (`string`): Manually annotated element category. Categories include `header`, `pre-header`, `post-header`, `body text`, `disclaimer / footnote`, `button`, `callout`, and `logo`.
- `elements.bbox_xyxy` (`list<float32>`): Pixel-edge bounding box `[x1, y1, x2, y2]` from `xyxy_word_fit`.
- `elements.bbox_cxcywh_normalized` (`list<float32>`): Normalized `[center_x, center_y, width, height]` box computed from `bbox_xyxy`.
- `num_elements` (`int32`): Number of valid foreground elements exposed for the sample.
- `raw_annotation` (`string`): JSON string containing the full source annotation list before loader filtering.

### Data Splits

The raw upstream release is not pre-split. This loader deterministically sorts JSON files in `png_json_gt/` and applies the same 9:1 split rule used by the upstream preprocessing script: the first 90% are `train`, and the remaining 10% are `validation`.

The upstream README reports 7,672 raw samples before loader validity filtering. This loader first applies the upstream validity filters, then applies the 9:1 split to the remaining valid examples, matching `dataset_tool.py`.

| Split | Rows |
| --- | ---: |
| `train` | 90% of valid examples |
| `validation` | Remaining 10% of valid examples |

Rows with zero valid foreground elements or more than nine valid foreground elements are skipped before splitting, matching the upstream preprocessing behavior.

## Dataset Creation

### Curation Rationale

The dataset was created to evaluate and train multimodal layout generation systems for advertising banners, where a model must place foreground text and logo elements on a given background image.

### Source Data

The upstream README states that part of the source images are filtered from the Pitt Image Ads Dataset and the rest are crawled from Google image search using retailer-brand keywords. The released archive is hosted on Google Drive from the LayoutDETR repository.

### Annotations

The `xyxy_word_fit` boxes and text strings were detected with Salesforce Einstein OCR according to the upstream README. Element categories were manually annotated through Amazon Mechanical Turk. This loader treats `xyxy_word_fit` as `[x1, y1, x2, y2]` pixel edges, matching the upstream `dataset_tool.py` implementation.

### Personal and Sensitive Information

The dataset contains advertising banners and OCR text. It may include brand names, product names, rendered people, or other content present in source advertisements. The dataset card does not identify personal information annotations in the upstream release.

## Considerations for Using the Data

### Social Impact of Dataset

LayoutDETR can support more controllable advertising design generation and evaluation. Users should account for the commercial-advertising context and avoid treating automated layout generation as a substitute for review by designers, brand owners, and legal or accessibility reviewers.

### Discussion of Biases

The dataset reflects the source advertisements, retailer keyword searches, OCR system, and crowdsourced category annotations used by the original authors. Models trained on it may inherit visual style, brand, product, and language biases from those sources.

### Other Known Limitations

The release is large, about 14.7GB according to the upstream README, so normal tests do not download it. The raw archive is not pre-split; this loader's train/validation split is deterministic and mirrors the upstream preprocessing intent. The loader does not expose cropped foreground patches, masks, or LayoutDETR-specific ZIP metadata.

## Additional Information

### Dataset Curators

The dataset was created by the LayoutDETR authors at Salesforce Research. This Hugging Face dataset loader was added in the `creative-graphic-design/huggingface-datasets` repository.

### Licensing Information

The upstream repository code is licensed under Apache License Version 2.0, copyright Salesforce 2023. A separate dataset-content license was not confirmed in the upstream release. Because the source images are partly from the Pitt Image Ads Dataset and partly crawled from Google image search, source image and content rights may follow their original sources. This dataset card therefore marks the dataset content license as `unknown`.

### Citation Information

```bibtex
@inproceedings{yu2024layoutdetr,
  title={LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer},
  author={Yu, Ning and Chen, Chia-Chih and Chen, Zeyuan and Meng, Rui and Wu, Gang and Josel, Paul and Niebles, Juan Carlos and Xiong, Caiming and Xu, Ran},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### Contributions

Thanks to [Salesforce Research](https://github.com/salesforce) and the LayoutDETR authors for creating and releasing the dataset and code.
