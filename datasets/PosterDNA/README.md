---
language:
  - zh
  - en
license: cc-by-nc-nd-4.0
pretty_name: PosterDNA
tags:
  - poster-generation
  - graphic-design
  - layout-generation
  - typography
  - html
  - text-to-image
annotations_creators:
  - machine-generated
language_creators:
  - machine-generated
size_categories:
  - 10K<n<100K
source_datasets:
  - original
task_categories:
  - text-to-image
  - image-to-text
configs:
  - config_name: posterdna
    data_files:
      - split: train
        path: posterdna/train-*
  - config_name: test_set
    data_files:
      - split: test
        path: test_set/test-*
---

# Dataset Card for PosterDNA

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

- **Homepage:** https://github.com/wuhaer/PosterVerse
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PosterDNA
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/PosterDNA
- **Original Code and Data:** https://github.com/wuhaer/PosterVerse
- **Original Hugging Face Release:** https://huggingface.co/wuhaer/PosterVerse
- **Paper (AAAI 2026):** https://doi.org/10.1609/aaai.v40i9.37656
- **Paper (arXiv):** https://arxiv.org/abs/2601.03993
- **Leaderboard:** Not available in the original release.
- **Point of Contact:** https://github.com/wuhaer/PosterVerse/issues

### Dataset Summary

PosterDNA is the dataset released with *PosterVerse: A Full-Workflow Framework for Commercial-Grade Poster Generation with HTML-Based Scalable Typography*. It contains text-dense commercial poster data with background images, fine-grained HTML-based layout and typography specifications, poster intention metadata, and a held-out test set.

This Hugging Face loader exposes two configurations:

- `posterdna`: the main PosterDNA archive, loaded as a `train` split from `posterdna.zip`.
- `test_set`: the held-out PosterDNA test archive, loaded as a `test` split from `test-set.zip`.

The upstream ZIP files are password-protected. Apply for access through the official PosterVerse project and set `POSTERDNA_ZIP_PASSWORD` before loading the full data.

### Supported Tasks and Leaderboards

PosterDNA supports poster generation research across blueprint creation, graphical background generation, and unified layout-text rendering. The released assets can be used to train or evaluate poster generation systems that condition on textual intention metadata and produce poster images with dense, accurate typography and layout.

No active public leaderboard is bundled with this Hugging Face dataset. For exact model training and inference workflows, use the upstream PosterVerse repository.

### Languages

The dataset contains Chinese (`zh`) and English (`en`) poster text, prompts, and metadata.

## Dataset Structure

### Data Instances

PosterDNA provides `posterdna` and `test_set` configurations. The dataset is not mirrored to the Hugging Face Hub, so load the local loader and pass the configuration name:

```python
import datasets as ds

dataset = ds.load_dataset(
    "datasets/PosterDNA/PosterDNA.py",
    name="posterdna",
    trust_remote_code=True,
)
```

Each row exposes raw metadata as a JSON string plus resolved local asset paths when the corresponding background image or HTML specification can be matched.

```json
{
  "id": "42",
  "metadata": "{\"prompt\": \"a seasonal sale poster\"}",
  "metadata_path": "json/design/42.json",
  "background_image": "<image>",
  "background_image_path": "bg/design/42.png",
  "html": "<html>...</html>",
  "html_path": "html/design/42.html"
}
```

### Data Fields

- `id` (`string`): Stable row identifier derived from metadata identifiers, filenames, or the row index.
- `metadata` (`string`): Raw JSON metadata serialized with `ensure_ascii=False`.
- `metadata_path` (`string`): Relative path to the metadata file inside the extracted archive.
- `background_image` (`Image`): Resolved poster background image when available.
- `background_image_path` (`string`): Relative path to the resolved background image.
- `html` (`string`): HTML layout and typography specification when available.
- `html_path` (`string`): Relative path to the resolved HTML file.

### Data Splits

| Config | Split | Rows |
| --- | --- | ---: |
| `posterdna` | train | 100,000 |
| `test_set` | test | 1,000 |

The paper reports 57,000 samples for blueprint creation, 100,000 samples for graphical background generation, 9,000 samples for unified layout-text rendering, and 1,000 held-out test examples. The official `test-set.zip` central directory contains 1,000 JSON metadata files, 1,000 background images, and 1,000 HTML files.

## Dataset Creation

### Curation Rationale

PosterDNA was created to support full-workflow commercial-grade poster generation, including intention analysis, background generation, and text-dense layout rendering with scalable HTML typography.

### Source Data

The upstream PosterVerse release hosts `posterdna.zip` and `test-set.zip` in the `wuhaer/PosterVerse` Hugging Face repository. The official GitHub repository states that researchers must apply for authorization and will receive the decompression password after approval.

### Annotations

The dataset contains machine-generated or curated poster intention metadata, HTML specifications, and poster assets released by the PosterVerse authors.

### Personal and Sensitive Information

The dataset consists of poster generation data and design assets. The dataset card does not identify personal information in the released archives. Because posters may depict people, products, events, or brands, users should inspect the source data for their intended use case.

## Considerations for Using the Data

### Social Impact of Dataset

PosterDNA can support better poster generation systems with stronger layout, typography, and text rendering. It should be used in accordance with the upstream non-commercial research authorization.

### Discussion of Biases

The data reflects the design sources, text distributions, style choices, and curation process used by the PosterVerse authors. Models trained on this dataset may reproduce visual styles, commercial design conventions, and language distributions present in the source release.

### Other Known Limitations

The upstream ZIP files are password-protected and require prior authorization. This loader cannot decrypt the archives unless `POSTERDNA_ZIP_PASSWORD` is set. The loader stores heterogeneous upstream metadata as raw JSON strings because the released metadata schemas differ between the main archive and test-set archive.

## Additional Information

### Dataset Curators

The dataset was created by the PosterVerse authors. This Hugging Face dataset loader was added in the `creative-graphic-design/huggingface-datasets` repository.

### Licensing Information

The upstream repository and Hugging Face model card mark PosterVerse and PosterDNA as CC BY-NC-ND 4.0 for non-commercial research purposes.

### Citation Information

```bibtex
@inproceedings{liu2026posterverse,
  title={PosterVerse: A Full-Workflow Framework for Commercial-Grade Poster Generation with HTML-Based Scalable Typography},
  author={Liu, Junle and Zhang, Peirong and Zhang, Yuyi and Yan, Pengyu and Zhou, Hui and Zhou, Xinyue and Guo, Fengjun and Jin, Lianwen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={9},
  pages={7197--7205},
  year={2026},
  doi={10.1609/aaai.v40i9.37656},
  url={https://doi.org/10.1609/aaai.v40i9.37656}
}
```

### Contributions

Thanks to [@wuhaer](https://github.com/wuhaer) and the PosterVerse authors for creating and releasing PosterDNA.
