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
annotations_creators:
  - expert-generated
  - machine-generated
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

## Dataset Description

- **Homepage:** https://github.com/arctanxarc/AesEval-Bench
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/AesEvalBench
- **Paper (ICLR 2026):** https://arxiv.org/abs/2603.01083
- **Data:** https://drive.google.com/file/d/1W5ocLYW0U-znD1Aq3C2xg_TLxL80jeiJ/view?usp=sharing

### Dataset Summary

AesEval-Bench is a benchmark for evaluating whether vision-language models can assess graphic design aesthetics. The benchmark contains perturbed graphic design samples with preview images, highlighted previews, element-level metadata, and labels for aesthetic issues.

The benchmark spans four dimensions and twelve indicators:

| Dimension | Indicators |
| --- | --- |
| Layout | balance, layering, whitespace, alignment |
| Typography | legibility, hierarchy |
| Graphics | quality, relevance |
| Color | harmony, contrast, appeal, psychology |

The original evaluation code supports binary aesthetic judgment, problematic-region choice, and bounding-box localization. This Hugging Face loader exposes the benchmark samples and labels in a single `train` split.

### Languages

The metadata text is in English.

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
  "preview": "<image>",
  "preview_highlight": "<image>",
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
      "dimension": "graphic",
      "task": "quality",
      "key": "graphic-quality",
      "element_index": 5,
      "attribute": "Shape",
      "left": 458.0,
      "top": 410.0,
      "width": 282.0,
      "height": 65.0
    }
  ]
}
```

### Data Fields

- `sample_name`: Directory name in the original benchmark release.
- `source_id`: Base design identifier parsed from `sample_name`.
- `perturbation_id`: Perturbation index parsed from `sample_name`.
- `canvas_width`, `canvas_height`: Canvas size.
- `title`, `category`, `keywords`, `industries`: Design metadata from `meta_info.json`.
- `preview`: Perturbed preview image.
- `preview_highlight`: Preview image with highlighted regions.
- `element_images`: Variable-length list of per-element PNG assets.
- `elements`: Simplified element metadata from `simplified_meta_info.json`.
- `task_labels`: One binary label for each of the 12 aesthetic indicators.
- `gt_annotations`: Positive issue annotations with element index, attribute, and bounding box.
- `gt_json`: Original `GT.json` serialized as JSON text.
- `changes_json`: Original `changes.json` serialized as JSON text.
- `meta_info_json`: Original `meta_info.json` serialized as JSON text.
- `simplified_meta_info_json`: Original `simplified_meta_info.json` serialized as JSON text.

### Data Splits

| Split | Rows |
| --- | ---: |
| train | 1,198 |

## Dataset Creation

### Source Data

The original repository distributes the benchmark as a Google Drive archive. The expected extracted layout is:

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

## Additional Information

### Licensing Information

The source repository did not declare a license at the time this dataset loader was added, so the dataset license is marked as `unknown`.

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

Thanks to [@arctanxarc](https://github.com/arctanxarc) and the AesEval-Bench authors for creating this dataset.
