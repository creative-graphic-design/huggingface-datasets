---
language:
  - en
license: unknown
pretty_name: AesEval-Bench
tags:
  - graphic-design
  - aesthetics
  - vision-language-models
  - benchmark
  - visual-evaluation
  - crello
annotations_creators:
  - expert-generated
language_creators:
  - found
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - image-to-text
  - visual-question-answering
task_ids: []
---

# Dataset Card for AesEval-Bench

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

- **Homepage:** https://github.com/arctanxarc/AesEval-Bench
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/AesEvalBench
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/AesEvalBench
- **Paper (ICLR 2026 / OpenReview):** https://openreview.net/forum?id=QGv6QwDA4z
- **Paper (arXiv):** https://arxiv.org/abs/2603.01083
- **Data:** https://drive.google.com/file/d/1W5ocLYW0U-znD1Aq3C2xg_TLxL80jeiJ/view?usp=sharing
- **Leaderboard:** Not available in the original release.
- **Point of Contact:** https://github.com/arctanxarc/AesEval-Bench/issues

### Dataset Summary

AesEval-Bench is a benchmark from the paper *Can Vision-Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective*. It evaluates whether vision-language models can assess the aesthetic quality of graphic designs in a way that is comparable to human judgment.

The benchmark frames aesthetic assessment as question answering over a design image, optionally with design metadata such as layout, font, or color information in JSON format. The original benchmark defines three quantifiable tasks:

- **Aesthetic judgment:** decide whether the design has an aesthetic issue for a given indicator.
- **Region selection:** choose the problematic region from candidate regions.
- **Precise localization:** predict the bounding box of the problematic region, or `None` when no issue is present.

The aesthetic taxonomy covers four dimensions and twelve indicators:

| Dimension | Loader value | Indicators |
| --- | --- | --- |
| Layout | `layout` | `balance`, `layering`, `whitespace`, `alignment` |
| Typography / font | `font` | `legibility`, `hierarchy` |
| Graphics | `graphic` | `quality`, `relevance` |
| Color | `color` | `harmony`, `contrast`, `appeal`, `psychology` |

This Hugging Face loader exposes the released benchmark archive as a single `test` split. Each row corresponds to one perturbed design sample and includes preview images, highlighted previews, element-level metadata, binary labels for the twelve indicators, positive issue annotations, and the original source JSON files serialized as strings.

The paper reports 1,200 sampled Crello designs and 4,500 base question-answer pairs. The public Google Drive archive currently contains 1,198 sample directories that can be loaded, so this dataset exposes 1,198 rows.

### Supported Tasks and Leaderboards

This dataset is intended for evaluation, not model training. It can support the following tasks from the original benchmark:

- **Aesthetic judgment:** use `preview`, optional metadata fields, and one indicator from `task_labels` to predict whether an issue exists. The original paper evaluates this task with accuracy.
- **Region selection:** use the original `GT.json`, element boxes, and candidate regions generated by the original evaluation code to select a problematic region. The original paper evaluates this task with accuracy.
- **Precise localization:** predict a problematic region as a bounding box, or predict `None` if no issue is present. The original paper evaluates `None` cases with accuracy and positive bounding-box cases with intersection over union (IoU).

No active public leaderboard is bundled with this Hugging Face dataset. For exact reproduction of the original prompt formats, candidate choices, and output parsing, use the upstream evaluation code.

### Languages

The metadata text and task descriptions are in English (`en`). Some preview images contain rendered text from the original graphic design templates. Category and industry fields are numeric identifiers inherited from the source design metadata.

## Dataset Structure

### Data Instances

Each row corresponds to one `{sample_id}-perturbs_new` directory from the released `benchmark_data/` archive.

```json
{
  "sample_name": "5888a55a95a7a863ddcc1d1d_3-perturbs_new",
  "source_id": "5888a55a95a7a863ddcc1d1d",
  "perturbation_id": 3,
  "canvas_width": 1200,
  "canvas_height": 600,
  "title": "Citation about volunteer work",
  "category": 12,
  "keywords": ["invitation", "invitation", "background", "sign", "symbol"],
  "industries": [],
  "preview": "<image>",
  "preview_highlight": "<image>",
  "element_images": [
    {
      "filename": "0.png",
      "image": "<image>"
    }
  ],
  "elements": [
    {
      "element_index": 0,
      "type": "shape",
      "left": -44.0,
      "top": -19.0,
      "width": 1284.0,
      "height": 722.0,
      "angle": 0.0,
      "opacity": 1.0,
      "color": ["rgba(209, 211, 212, 1)"],
      "image_filename": "0.png",
      "text": "",
      "font": "",
      "font_size": null,
      "text_color": "",
      "text_align": ""
    }
  ],
  "task_labels": [
    {
      "dimension": "layout",
      "task": "balance",
      "key": "layout-balance",
      "has_issue": false
    }
  ],
  "gt_annotations": [
    {
      "dimension": "layout",
      "task": "whitespace",
      "key": "layout-whitespace",
      "element_index": 1,
      "attribute": "Image",
      "left": 70.0,
      "top": 72.0,
      "width": 1054.0,
      "height": 451.0
    }
  ],
  "gt_json": "{\"layout-balance\": []}",
  "changes_json": "{\"...\": \"...\"}",
  "meta_info_json": "{\"...\": \"...\"}",
  "simplified_meta_info_json": "{\"...\": \"...\"}"
}
```

### Data Fields

- `sample_name` (`string`): Directory name in the original benchmark release.
- `source_id` (`string`): Base design identifier parsed from `sample_name`.
- `perturbation_id` (`int32`): Perturbation index parsed from `sample_name`; `-1` when the name does not include a numeric perturbation suffix.
- `canvas_width`, `canvas_height` (`int32`): Canvas size from `simplified_meta_info.json`.
- `title` (`string`): Design title from `meta_info.json`.
- `category` (`int32`): Numeric category identifier from the source metadata.
- `keywords` (`list[string]`): Keyword metadata from the source design.
- `industries` (`list[int32]`): Numeric industry identifiers from the source metadata.
- `preview` (`Image`): Perturbed design image.
- `preview_highlight` (`Image`): Perturbed design image with highlighted regions.
- `element_images` (`list`): Per-element PNG assets with:
  - `filename` (`string`): Element asset filename from the sample directory.
  - `image` (`Image`): Element-level PNG image.
- `elements` (`list`): Simplified element metadata with element index, type, position, size, rotation angle, opacity, color values, image filename, text content, font, font size, text color, and text alignment.
- `task_labels` (`list`): One binary label for each of the twelve aesthetic indicators:
  - `dimension` (`string`): One of `layout`, `font`, `graphic`, or `color`.
  - `task` (`string`): Indicator name within the dimension.
  - `key` (`string`): Source label key such as `layout-balance` or `color-harmony`.
  - `has_issue` (`bool`): Whether the source ground truth contains an issue annotation for this indicator.
- `gt_annotations` (`list`): Positive issue annotations derived from `GT.json`. Each item contains the dimension, task, label key, affected element index, changed attribute, and bounding-box coordinates.
- `gt_json` (`string`): Original `GT.json` serialized as JSON text.
- `changes_json` (`string`): Original `changes.json` serialized as JSON text.
- `meta_info_json` (`string`): Original `meta_info.json` serialized as JSON text.
- `simplified_meta_info_json` (`string`): Original `simplified_meta_info.json` serialized as JSON text.

### Data Splits

This dataset contains one evaluation split:

| Split | Rows |
| --- | ---: |
| test | 1,198 |

The split is named `test` because AesEval-Bench is an evaluation benchmark. The original release does not provide train or validation splits in this archive.

## Dataset Creation

### Curation Rationale

AesEval-Bench was created to provide a systematic and reproducible way to evaluate aesthetic assessment in graphic design. The paper argues that prior aesthetics benchmarks are limited by narrow design principles, coarse evaluation protocols, a lack of systematic VLM comparison, and limited training data for model improvement. AesEval-Bench addresses this by combining a multidimensional taxonomy with choice and bounding-box style evaluation tasks.

### Source Data

The benchmark uses the Crello dataset as its source of professional-quality graphic designs. Crello provides rendered design images, element-level metadata in JSON format, and separated design layers. The AesEval-Bench release packages perturbed samples under a `benchmark_data/` directory.

Expected extracted layout:

```text
benchmark_data/
`-- {sample_id}-perturbs_new/
    |-- preview.png
    |-- preview_highlight.png
    |-- simplified_meta_info.json
    |-- GT.json
    |-- changes.json
    |-- 0.png, 1.png, ...
    `-- meta_info.json
```

This loader downloads the archive with `gdown` when no local `data_dir` is provided. For local testing, pass either the downloaded ZIP file or an extracted `benchmark_data/` directory through `data_dir`.

### Initial Data Collection and Normalization

The paper describes sampling designs from the Crello test split and introducing controlled perturbations to create realistic examples that may or may not degrade aesthetic quality. Perturbations include operations such as repositioning elements, changing fonts, and adjusting colors. Each base design undergoes one to three random perturbations, and the modified JSON metadata is rendered back into a design image by recombining it with the source layers.

The Hugging Face loader does not re-run the perturbation pipeline. It reads the already released `benchmark_data/` archive, scans sample directories whose names contain `_new` and include `GT.json`, and normalizes the source files into typed Hugging Face `datasets` features.

### Who are the source language producers?

The source graphic designs come from Crello templates. The paper does not provide demographic information about the creators of those source designs. Metadata text, rendered design text, and template titles should be treated as found content from the source dataset and benchmark release.

### Annotations

The benchmark includes human-verified aesthetic issue labels. The released `GT.json` files store labels using keys such as `layout-balance`, `font-legibility`, and `color-harmony`.

### Annotation process

The paper describes a human-in-the-loop review process. Annotators were shown a design image and a focal indicator description, then asked whether the design contains the corresponding aesthetic flaw. Annotators received a tutorial with examples of well-designed and flawed cases, and final binary labels were derived by majority voting across multiple annotators.

The benchmark then generated task answers from the human labels and perturbation metadata:

- For aesthetic judgment, flawed designs map to `yes` and non-flawed designs map to `no`.
- For region selection, flawed designs use the perturbed element box as the correct choice among candidates, while non-flawed designs use `None`.
- For precise localization, flawed designs use the perturbed element bounding box and non-flawed designs use `None`.

### Who are the annotators?

The paper states that human annotators performed the benchmark review and that professional designers were consulted when refining the aesthetic taxonomy. It does not specify annotator demographics, compensation, or selection criteria in the public paper text.

### Personal and Sensitive Information

This dataset is a graphic design benchmark and does not intentionally provide identity labels or personal profiles. Some source template images may include people, brand-like design text, or event-style content because the source data consists of graphic design templates. Users should inspect samples for their own deployment context before using the data in public-facing systems.

## Considerations for Using the Data

### Social Impact of Dataset

AesEval-Bench can help make evaluation of graphic-design aesthetic assessment more transparent and reproducible. It is useful for diagnosing whether VLMs can reason over design principles such as hierarchy, alignment, legibility, contrast, and relevance.

At the same time, aesthetic quality is partly subjective and culturally situated. Models trained or selected only against this benchmark may overfit to the benchmark taxonomy, the source template distribution, or the perturbation pipeline. Use this dataset as one evaluation signal rather than as a complete definition of design quality.

### Discussion of Biases

The benchmark is derived from Crello graphic design templates and from a taxonomy refined through literature review and consultation with professional designers. This makes the benchmark practical for design-aesthetics evaluation, but it may also reflect the visual styles, languages, categories, and cultural assumptions present in those sources.

The perturbation pipeline intentionally injects controlled flaws. Those synthetic perturbations are useful for quantifiable evaluation, but they may not capture all naturally occurring design failures or all forms of aesthetic preference across audiences and cultures.

### Other Known Limitations

- The paper reports 1,200 sampled designs, while the public Google Drive archive used by this loader currently contains 1,198 loadable sample directories. This Hugging Face dataset exposes the released archive as-is.
- The original paper reports 4,500 base question-answer pairs and instantiates tasks across twelve indicators. This loader exposes one row per released perturbed design sample, with labels and source JSON that can be used to reconstruct task prompts.
- The loader does not bundle the original model prompts, region-choice candidates, or evaluator outputs. Use the upstream repository for exact benchmark reproduction.
- The benchmark focuses on twelve selected indicators. It is not a complete coverage of all possible graphic design aesthetics, accessibility issues, brand constraints, or cultural preferences.
- License terms for the released data are not declared in the upstream repository at the time this dataset card was updated.

## Additional Information

### Dataset Curators

AesEval-Bench was created by Arctanx An, Shizhao Sun, Danqing Huang, Mingxi Cheng, Yan Gao, Ji Li, Yu Qiao, and Jiang Bian. The paper lists affiliations including Peking University, Microsoft Research Asia, Microsoft, and Central South University.

### Licensing Information

The upstream AesEval-Bench repository does not declare a dataset license in its README or include a `LICENSE` file at the time this dataset card was updated. The license is therefore marked as `unknown`.

### Citation Information

```bibtex
@misc{an2026canvisionlanguagemodelsassess,
  title={Can Vision Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective},
  author={An, Arctanx and Sun, Shizhao and Huang, Danqing and Cheng, Mingxi and Gao, Yan and Li, Ji and Qiao, Yu and Bian, Jiang},
  year={2026},
  eprint={2603.01083},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.01083}
}
```

### Contributions

Thanks to [@arctanxarc](https://github.com/arctanxarc) and the AesEval-Bench authors for creating the original benchmark and public data release. This Hugging Face dataset implementation was created by the creative-graphic-design organization to make AesEval-Bench easier to load with the `datasets` library.
