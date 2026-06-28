---
language:
  - en
license: cc-by-4.0
pretty_name: LICA
tags:
  - graphic-design
  - design-template
  - layout-data
  - layout-generation
  - design-composition
  - multi-layer
  - design-evaluation
annotations_creators:
  - machine-generated
language_creators:
  - machine-generated
size_categories:
  - 1K<n<10K
source_datasets:
  - original
---

# Dataset Card for LICA

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://github.com/lica-world/lica-dataset
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/LICA
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/LICA
- **Paper (arXiv):** https://arxiv.org/abs/2603.16098
- **Data:** https://storage.googleapis.com/lica-assets/websites/blog/lica-data.zip
- **Point of Contact:** https://github.com/lica-world/lica-dataset/issues

### Dataset Summary

LICA (Layered Image Composition Annotations for Graphic Design Research) is a dataset of graphic design layouts with rendered compositions, component-level layout specifications, and natural-language annotations. The public sample groups layouts by template and includes per-layout metadata, rendered PNG or MP4 files, layout JSON, per-layout annotations, and template-level annotations.

The dataset is intended for research on structured layout generation, design composition understanding, template-consistent variation, design editing, and design quality evaluation. This loader exposes the released `lica-data.zip` archive as a single `test` split.

### Supported Tasks and Leaderboards

The dataset can support evaluation and benchmarking for graphic design research tasks:

- **Structured layout generation:** generate component-level design layouts from text intent, metadata, or template context.
- **Template-consistent variation:** model how layouts vary within a shared template group.
- **Design composition analysis:** inspect component types, rendered images, and natural-language annotations for design understanding.
- **Design quality evaluation:** use the visual render and annotation text as inputs for learned or prompted design assessment.

No active public leaderboard is bundled with this Hugging Face dataset.

### Languages

The annotation text is in English (`en`). Rendered designs may contain text inside images, inherited from the original graphic design templates.

## Dataset Structure

### Data Instances

Load the dataset with:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/LICA")
```

Each row corresponds to one layout from `metadata.csv`.

```json
{
  "layout_id": "gT6aha6BT385mIz8ddgx",
  "template_id": "00863b23-aa9e-4572-b124-01749574e893",
  "category": "Art & Design",
  "n_template_layouts": 1,
  "template_layout_index": 0,
  "width": 940,
  "height": 788,
  "file_name": "images/00863b23-aa9e-4572-b124-01749574e893/gT6aha6BT385mIz8ddgx.png",
  "render_type": "png",
  "render_image": "<image>",
  "render_video_path": "",
  "layout_width": 940,
  "layout_height": 788,
  "layout_background": "rgb(252, 252, 252)",
  "layout_duration": 3.0,
  "n_components": 20,
  "component_types": ["GROUP", "IMAGE", "TEXT"],
  "description": "A natural-language description of the rendered design.",
  "aesthetics": "Notes on composition, visual hierarchy, typography, and color.",
  "tags": "comma, separated, tags",
  "user_intent": "Inferred purpose of the design."
}
```

### Data Fields

- `layout_id`: Unique layout ID, matching filenames under `layouts/`, `images/`, and `annotations/`.
- `template_id`: Template UUID shared by related layout variants.
- `category`: Design category as a class label, such as presentations, posters, flyers, or social media.
- `n_template_layouts`: Number of layouts in the template group.
- `template_layout_index`: Zero-based position within the template group.
- `width`, `height`: Rendered canvas dimensions in pixels from `metadata.csv`.
- `file_name`: Relative rendered media path when provided by the source metadata.
- `render_type`: Rendered media type class label, currently `png` or `mp4`.
- `render_path`: Local cached path to the rendered media file.
- `render_image`: Decoded rendered image for PNG layouts.
- `render_video_path`: Local cached path for MP4 layouts. This field is empty for PNG rows.
- `layout_width`, `layout_height`: Canvas dimensions parsed from the layout JSON.
- `layout_background`: Canvas background from the layout JSON.
- `layout_duration`: Optional slide duration in seconds.
- `n_components`: Number of components in the layout JSON.
- `component_types`: Ordered component type class labels from the layout JSON: `GROUP`, `IMAGE`, `TEXT`, and `TEXT_NEW`.
- `layout_json`: Full layout JSON serialized as a string.
- `annotation_json`: Full per-layout annotation JSON serialized as a string.
- `template_annotation_json`: Matching template-level annotation JSON serialized as a string.
- `description`, `aesthetics`, `tags`, `user_intent`, `raw`: Per-layout natural-language annotation fields.
- `template_description`, `template_aesthetics`, `template_tags`, `template_user_intent`, `template_raw`: Template-level annotation fields.

### Data Splits

| Split | Rows |
| --- | ---: |
| `test` | 1,148 |

## Dataset Creation

### Source Data

The source release is the public LICA sample archive from Lica World. It contains:

```text
lica-data/
├── metadata.csv
├── layouts/
├── images/
└── annotations/
```

Layouts are represented as editable component hierarchies with CSS-like positioning, typography, color, transform, and media fields. Rendered files are stored as PNG images or MP4 videos.

### Annotations

Each layout has natural-language fields for description, aesthetics, tags, user intent, and a concatenated raw text. Template-level annotations use the same fields and describe the design theme shared across all layouts in the template group.

### Personal and Sensitive Information

The upstream dataset card states that the dataset consists of professionally designed graphic layout templates and does not contain personal, sensitive, or private information.

## Considerations for Using the Data

### Biases and Limitations

Design quality and aesthetics are context-dependent. Models trained or evaluated on this sample may reflect the distribution of categories, templates, annotation style, and visual conventions in the released LICA sample rather than universal design preferences.

This loader exposes the public sample, not the full LICA corpus described in the paper.

## Additional Information

### Dataset Curators

The dataset was released by Lica World.

### Licensing Information

The dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

### Citation Information

```bibtex
@article{Hirsch2026LICA,
  title   = {LICA: Layered Image Composition Annotations for Graphic Design Research},
  author  = {Hirsch, Elad and Yadav, Shubham and Garg, Mohit and Mehta, Purvanshi},
  journal = {arXiv preprint arXiv:2603.16098},
  year    = {2026}
}
```

### Contributions

Thanks to [Lica World](https://github.com/lica-world) and the LICA authors for creating this dataset.
