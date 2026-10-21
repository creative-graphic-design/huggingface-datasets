---
annotations_creators:
  - found
language:
  - en
language_creators:
  - found
license: cc-by-sa-4.0
pretty_name: OpenMoji
size_categories:
  - 1K<n<10K
source_datasets:
  - original
tags:
  - svg
  - emoji
  - icon
  - openmoji
task_categories:
  - text-to-image
  - image-to-text
---

# Dataset Card for OpenMoji

## Table of Contents

- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Source Data](#source-data)
- [Considerations for Using the Data](#considerations-for-using-the-data)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://openmoji.org/
- **Original Repository:** https://github.com/hfg-gmuend/openmoji
- **Release:** https://github.com/hfg-gmuend/openmoji/releases/tag/17.0.0
- **Point of Contact:** https://github.com/creative-graphic-design/huggingface-datasets/issues
- **Original Source Contact:** https://github.com/hfg-gmuend/openmoji/issues

### Dataset Summary

OpenMoji 17.0.0 contains 4,495 Unicode and custom emoji and icon entries, including extras-openmoji, extras-unicode, and skin-tone variants. Each row pairs upstream metadata with color and black SVG source text and 618 × 618 PNG images. SVG generation and understanding are primary uses; the PNG images are included alongside each SVG.

## Dataset Structure

### Data Instances

The dataset has one configuration and one `train` split. Load the source files through the local loader:

```python
import datasets as ds

dataset = ds.load_dataset(
    "datasets/OpenMoji/OpenMoji.py",
    trust_remote_code=True,
)
```

Each row represents one entry in the release metadata, matched to its SVG and PNG files by `hexcode`.

```json
{
  "hexcode": "1F600",
  "emoji": "😀",
  "group": "smileys-emotion",
  "annotation": "grinning face",
  "tags": ["grinning", "happy", "smile"],
  "svg_color": "<svg ...>",
  "svg_black": "<svg ...>",
  "image_color": "<Image: 618 × 618 PNG>",
  "image_black": "<Image: 618 × 618 PNG>"
}
```

### Data Fields

| Field                   | Type           | Description                                                        |
| ----------------------- | -------------- | ------------------------------------------------------------------ |
| `hexcode`               | string         | Unique OpenMoji code used to join metadata and assets.             |
| `emoji`                 | string or null | Emoji character, or null when the release value is empty.          |
| `group`                 | string         | Upstream emoji group.                                              |
| `subgroups`             | string         | Upstream subgroup.                                                 |
| `annotation`            | string         | Upstream short description.                                        |
| `tags`                  | list[string]   | Comma-separated upstream tags, trimmed and empty items removed.    |
| `openmoji_tags`         | list[string]   | Comma-separated OpenMoji tags, trimmed and empty items removed.    |
| `openmoji_author`       | string         | OpenMoji artwork author.                                           |
| `openmoji_date`         | string         | Upstream ISO date string.                                          |
| `skintone`              | string or null | Skin-tone value, or null when empty.                               |
| `skintone_combination`  | string or null | Skin-tone combination, or null when empty.                         |
| `skintone_base_emoji`   | string or null | Base emoji for a skin-tone variant, or null when empty.            |
| `skintone_base_hexcode` | string or null | Base hexcode for a skin-tone variant, or null when empty.          |
| `unicode`               | string or null | Upstream Unicode value normalized to a string, or null when empty. |
| `order`                 | int32 or null  | Upstream display order, or null when empty.                        |
| `svg_color`             | string         | Raw UTF-8 color SVG source.                                        |
| `svg_black`             | string         | Raw UTF-8 black SVG source.                                        |
| `image_color`           | Image          | Color artwork as a 618 × 618 PNG.                                  |
| `image_black`           | Image          | Black artwork as a 618 × 618 PNG.                                  |

### Data Splits

The pinned release provides no train, validation, or test partition. The loader exposes all 4,495 release entries in a single `train` split without filtering.

| Split   |  Rows |
| ------- | ----: |
| `train` | 4,495 |

## Dataset Creation

### Source Data

The loader downloads the four SVG and 618 × 618 PNG archives and `data/openmoji.json` from upstream tag `17.0.0`. It joins each JSON record to the flat `<HEXCODE>.svg` and `<HEXCODE>.png` archive members using `hexcode`. The release includes standard Unicode emoji, extras-openmoji, extras-unicode, and skin-tone variants.

No rows are filtered. Empty upstream scalar strings become null; comma-separated tag values become trimmed lists with empty items removed. Numeric or string Unicode values are converted to strings, and order values are converted to int32.

## Considerations for Using the Data

OpenMoji primarily provides SVG vector artwork for generation and understanding tasks; the matching raster images support image-based use. The metadata includes English annotations and tags. Skin-tone variants should be interpreted as variants of the same base emoji where the corresponding base fields are populated.

## Additional Information

### Licensing Information

OpenMoji graphics are licensed under CC BY-SA 4.0. The upstream project requests this attribution:

> All emojis designed by [OpenMoji](https://openmoji.org/) – the open-source emoji and icon project. License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/#)

The upstream repository's code is licensed under LGPL-3.0; this dataset contains the graphics and metadata, not the upstream code.

### Citation Information

The citation follows upstream `CITATION.cff`; the version-specific Zenodo record identifies release 17.0.0.

```bibtex
@misc{openmoji,
  author = {Groß, Benedikt and Utz, Daniel and The OpenMoji Contributors},
  title = {OpenMoji},
  year = {2026},
  version = {17.0.0},
  doi = {10.5281/zenodo.19704720},
  url = {https://doi.org/10.5281/zenodo.19704720}
}
```

### Contributions

Thanks to [OpenMoji](https://github.com/hfg-gmuend/openmoji) and its contributors for creating and releasing the artwork and metadata.
