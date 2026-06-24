---
annotations_creators:
  - expert-generated
  - machine-generated
language:
  - en
language_creators:
  - machine-generated
license: unknown
pretty_name: POSTA-PosterArt
size_categories:
  - 1K<n<10K
source_datasets:
  - original
tags:
  - posta
  - posterart
  - poster-generation
  - graphic-design
  - layout-generation
  - typography
  - text-segmentation
task_categories:
  - image-to-text
  - text-to-image
  - image-segmentation
task_ids: []
---

# Dataset Card for POSTA-PosterArt

## Dataset Description

- **Homepage:** https://haoyuchen.com/POSTA
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/POSTAPosterArt
- **Paper (arXiv):** https://arxiv.org/abs/2503.14908

### Dataset Summary

POSTA-PosterArt is the dataset introduced with POSTA, a framework for customized artistic poster generation. It contains two subsets:

- **PosterArt-Design**: poster backgrounds with professional layout and typography annotations extracted from PSD files.
- **PosterArt-Text**: poster title regions with artistic text captions, masks, and single-region mask images for text stylization and segmentation.

The dataset supports research on controllable poster generation, layout planning, typography prediction, artistic text stylization, and text segmentation in graphic design.

### Languages

The captions and text descriptions are in English.

## Dataset Structure

### Configurations

The dataset provides two configurations:

- `text` (default): PosterArt-Text examples.
- `design`: PosterArt-Design examples.

### Data Instances

`text` example:

```json
{
  "id": "2785",
  "image": "<image path>",
  "caption": "artistic text style description",
  "mask": "<segmentation mask path>",
  "mask_img_single": "<single text-region mask image path>"
}
```

`design` example:

```json
{
  "id": "Part1/0002",
  "background_image": "<background image path>",
  "poster_image": "<final poster image path>",
  "psd_filename": "0002.psd",
  "annotation": "{... raw PSD JSON annotation ...}",
  "text_layers": [
    {
      "path": "Title",
      "name": "Title",
      "text_content": "POSTA",
      "font_name": "PlayfairDisplay",
      "font_size": 45.5,
      "alignment": "center",
      "rotation": 0.0
    }
  ]
}
```

### Data Fields

`text` fields:

- `id` (string): Shared file stem for the grouped example.
- `image` (Image): Source poster or background image.
- `caption` (string): Artistic text style caption.
- `mask` (Image): Pixel-level text segmentation mask.
- `mask_img_single` (Image): Single text-region mask image.

`design` fields:

- `id` (string): Shared file stem for a poster design.
- `background_image` (Image): Poster background without layout text.
- `poster_image` (Image): Final poster image with complete layout.
- `psd_filename` (string): Original PSD filename when available.
- `annotation` (string): Raw JSON annotation extracted from the PSD structure.
- `text_layers` (sequence): Flattened text layers with layer path, text content, bbox, font, color, alignment, and rotation attributes.

### Data Splits

Both configurations expose a single `train` split.

| Configuration | Train |
| ------------- | ----: |
| text          |  2218 |
| design        |   353 |

`design` is distributed through Google Drive. The POSTA project page lists PosterArt-Design as 152.3GB, while the currently downloadable `Part1.zip` used by this loader is 18.2GB compressed and about 24.8GB uncompressed.

## Dataset Creation

### Source Data

PosterArt was created for the POSTA framework. PosterArt-Design contains artistic poster backgrounds with layout and typography information crafted by professional designers. PosterArt-Text contains artistic title text regions, captions generated with vision-language models, and manually segmented title masks.

### Annotations

PosterArt-Design annotations include PSD-derived layer structure, text positions, font information, color values, alignment, and rotation. PosterArt-Text annotations include captions and pixel-level text segmentation masks.

## Considerations for Using the Data

### Known Limitations

The dataset is large. Loading the `design` configuration downloads multi-GB Google Drive files. CI and lightweight tests should use helper-level tests or opt-in full download tests.

The license was not clearly identified from the public project page or Drive files during this implementation, so the dataset card records it as `unknown`.

## Additional Information

### Citation Information

```bibtex
@misc{chen2025posta,
  title={POSTA: A Go-to Framework for Customized Artistic Poster Generation},
  author={Chen, Haoyu and others},
  year={2025},
  eprint={2503.14908},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

### Contributions

Thanks to the POSTA authors for creating and releasing the PosterArt dataset. This Hugging Face dataset implementation was created by the creative-graphic-design organization to make POSTA-PosterArt easier to load for research workflows.
