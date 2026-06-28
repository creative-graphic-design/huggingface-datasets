---
language:
  - en
license: other
license_name: non-commercial-research-license
license_link: https://huggingface.co/datasets/ArtmeScienceLab/PosterIQ/blob/main/LICENSE
pretty_name: PosterIQ
tags:
  - poster
  - graphic-design
  - visual-design
  - typography
  - layout
  - image-generation
annotations_creators:
  - expert-generated
  - machine-generated
language_creators:
  - machine-generated
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - image-to-text
  - text-to-image
task_ids: []
configs:
  - config_name: alignment
    data_files:
      - split: test
        path: alignment/test-*
  - config_name: composition_understanding
    data_files:
      - split: test
        path: composition_understanding/test-*
  - config_name: empty_space
    data_files:
      - split: test
        path: empty_space/test-*
  - config_name: font_attributes
    data_files:
      - split: test
        path: font_attributes/test-*
  - config_name: font_effect
    data_files:
      - split: test
        path: font_effect/test-*
  - config_name: font_effect_2
    data_files:
      - split: test
        path: font_effect_2/test-*
  - config_name: font_matching
    data_files:
      - split: test
        path: font_matching/test-*
  - config_name: font_size_ocr
    data_files:
      - split: test
        path: font_size_ocr/test-*
  - config_name: hard_ocr
    data_files:
      - split: test
        path: hard_ocr/test-*
  - config_name: intention_understanding
    data_files:
      - split: test
        path: intention_understanding/test-*
  - config_name: layout_comparison
    data_files:
      - split: test
        path: layout_comparison/test-*
  - config_name: layout_generation
    data_files:
      - split: test
        path: layout_generation/test-*
  - config_name: logo_ocr
    data_files:
      - split: test
        path: logo_ocr/test-*
  - config_name: overall_rating
    data_files:
      - split: test
        path: overall_rating/test-*
  - config_name: poster_ocr
    data_files:
      - split: test
        path: poster_ocr/test-*
  - config_name: rotation
    data_files:
      - split: test
        path: rotation/test-*
  - config_name: simple_ocr
    data_files:
      - split: test
        path: simple_ocr/test-*
  - config_name: style_understanding
    data_files:
      - split: test
        path: style_understanding/test-*
  - config_name: text_localization
    data_files:
      - split: test
        path: text_localization/test-*
  - config_name: gen_composition
    data_files:
      - split: test
        path: gen_composition/test-*
  - config_name: gen_dense
    data_files:
      - split: test
        path: gen_dense/test-*
  - config_name: gen_font
    data_files:
      - split: test
        path: gen_font/test-*
  - config_name: gen_intention
    data_files:
      - split: test
        path: gen_intention/test-*
  - config_name: gen_style
    data_files:
      - split: test
        path: gen_style/test-*
---

# Dataset Card for PosterIQ

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://github.com/ArtmeScienceLab/PosterIQ-Benchmark
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PosterIQ
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/PosterIQ
- **Original Data:** https://huggingface.co/datasets/ArtmeScienceLab/PosterIQ
- **Original Code:** https://github.com/ArtmeScienceLab/PosterIQ-Benchmark
- **Paper (CVPR 2026 / arXiv):** https://arxiv.org/abs/2603.24078
- **Leaderboard:** Not available in the original release.
- **Point of Contact:** https://github.com/ArtmeScienceLab/PosterIQ-Benchmark/issues

### Dataset Summary

PosterIQ is the poster design benchmark released with *PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation*. It contains task-level evaluation data for poster understanding and poster generation from a design perspective, including typography, layout, OCR, composition, style, empty-space use, and design intention.

This Hugging Face loader exposes the upstream release as 24 task-level configurations. Understanding configurations include the released poster image from `data.zip`. Generation configurations contain prompts, target criteria, and task metadata; their `path` values identify intended generation output paths and do not point to released source images.

### Supported Tasks and Leaderboards

PosterIQ is intended for evaluating vision-language and image-generation systems on poster design tasks. Understanding tasks pair an input poster image with a prompt and answer metadata. Generation tasks provide poster-generation prompts and evaluation targets or attributes for generated outputs.

No active public leaderboard is bundled with this Hugging Face dataset. For exact reproduction of the original evaluation scripts, use the upstream PosterIQ-Benchmark repository.

### Languages

Prompts and annotations are primarily in English (`en`).

## Dataset Structure

### Data Instances

PosterIQ exposes one configuration per task. Examples include `alignment`, `composition_understanding`, `font_attributes`, `layout_generation`, and `gen_style`; see the configuration list in the dataset card metadata for all 24 names. Load one configuration by passing its name:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/PosterIQ", name="alignment")
```

An understanding row contains task metadata, prompt text, JSON-encoded answer metadata, and an input image:

```json
{
  "id": "alignment-00000",
  "task": "alignment",
  "subtask": "",
  "name": "000_30_center_.png",
  "path": "alignment/000_30_center_.png",
  "prompt": "Please observe the text alignment...",
  "gt_json": "[\"center-aligned\"]",
  "metadata_json": "{\"alignment\": [\"center-aligned\"]}",
  "image": "<image>",
  "image_path": ".../data/alignment/000_30_center_.png"
}
```

A generation row contains the prompt and metadata only:

```json
{
  "id": "gen_dense-00000",
  "task": "poster dense",
  "subtask": "",
  "name": "1000.jpg",
  "path": "dense/1000.jpg",
  "prompt": "NBA Pinnacle Night...",
  "gt_json": "[\"Aspect ratio 2:3...\"]",
  "metadata_json": "{\"theme\": \"NBA\", \"elements\": [[\"Kevin Durant\", \"Klay Thompson\"]]}"
}
```

### Data Fields

- `id` (`string`): Stable row identifier generated as `{config}-{index:05d}`.
- `task` (`string`): Upstream task name.
- `subtask` (`string`, configs where present): Upstream subtask label.
- `name` (`string`): Upstream file name.
- `path` (`string`): Upstream relative path, normalized to POSIX separators.
- `prompt` (`string`): Model prompt for the task.
- `gt_json` (`string`, only configs with `gt`): JSON-encoded upstream ground-truth field.
- `metadata_json` (`string`): JSON-encoded source fields not represented as standard columns.
- `image` (`Image`, understanding configs): Input poster image.
- `image_path` (`string`, understanding configs): Local resolved image path.
- `original_image` (`Image`, `poster_ocr` and `text_localization`): Original-resolution poster image.
- `original_image_path` (`string`, `poster_ocr` and `text_localization`): Local resolved original image path.

### Data Splits

All configurations expose a single `test` split because PosterIQ is an evaluation dataset and the upstream release does not provide training partitions.

| Config | Split | Rows | Image |
| --- | --- | ---: | --- |
| `alignment` | test | 200 | yes |
| `composition_understanding` | test | 117 | yes |
| `empty_space` | test | 167 | yes |
| `font_attributes` | test | 1,813 | yes |
| `font_effect` | test | 450 | yes |
| `font_effect_2` | test | 125 | yes |
| `font_matching` | test | 400 | yes |
| `font_size_ocr` | test | 1,400 | yes |
| `hard_ocr` | test | 400 | yes |
| `intention_understanding` | test | 202 | yes |
| `layout_comparison` | test | 256 | yes |
| `layout_generation` | test | 145 | yes |
| `logo_ocr` | test | 600 | yes |
| `overall_rating` | test | 219 | yes |
| `poster_ocr` | test | 205 | yes |
| `rotation` | test | 205 | yes |
| `simple_ocr` | test | 400 | yes |
| `style_understanding` | test | 256 | yes |
| `text_localization` | test | 205 | yes |
| `gen_composition` | test | 117 | no |
| `gen_dense` | test | 114 | no |
| `gen_font` | test | 135 | no |
| `gen_intention` | test | 200 | no |
| `gen_style` | test | 256 | no |

## Dataset Creation

### Curation Rationale

PosterIQ was created to evaluate poster understanding and generation from a design perspective rather than relying only on general visual recognition or generic image-generation criteria.

### Source Data

The upstream Hugging Face dataset provides task JSON files under `und_task/` and `gen_task/`. The `data.zip` archive contains the 7,765 images referenced by the understanding tasks. Generation tasks provide prompts and metadata for evaluating generated poster outputs.

### Annotations

Rows contain task-specific ground truth or target metadata from the upstream release. This loader keeps task-specific fields in `metadata_json` to avoid forcing heterogeneous task schemas into a single lossy structure.

### Personal and Sensitive Information

The dataset consists of poster images, poster-generation prompts, and design-task annotations. The dataset card does not identify personal information in the released benchmark. Posters may include names, brands, events, or culturally specific text as part of graphic design examples.

## Considerations for Using the Data

### Social Impact of Dataset

PosterIQ can support more design-aware evaluation of poster understanding and generation systems, especially typography, OCR, layout, composition, and style control.

### Discussion of Biases

Poster design conventions, language use, typography, and style labels reflect the upstream data creation process and may not cover all cultures, domains, accessibility needs, or professional design contexts equally.

### Other Known Limitations

Generation configurations do not include generated images in the upstream release. They provide prompts and evaluation metadata for systems that generate their own outputs.

## Additional Information

### Dataset Curators

The original PosterIQ dataset was created by Yuheng Feng, Wen Zhang, Haodong Duan, and Xingxing Zou.

### Licensing Information

The upstream Hugging Face dataset card declares `non-commercial-research-license` with `license: other`. The upstream `LICENSE` file is empty at the time this loader was created, so users should consult the original dataset page and repository before redistribution or commercial use.

### Citation Information

```bibtex
@inproceedings{cvpr2026posteriq,
  title={PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation},
  author={Feng, Yuheng and Zhang, Wen and Duan, Haodong and Zou, Xingxing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

### Contributions

Thanks to [@ArtmeScienceLab](https://github.com/ArtmeScienceLab) for creating this dataset.
