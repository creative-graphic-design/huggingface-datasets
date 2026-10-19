---
license: unknown
pretty_name: PersuasiveAdVLMBenchmark
tags:
  - advertising
  - image-retrieval
  - visual-reasoning
  - vision-language
  - atypicality
annotations_creators:
  - found
size_categories:
  - n<1K
source_datasets:
  - original
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/test-*
---

# Dataset Card for PersuasiveAdVLMBenchmark

## Dataset Description

- **Homepage:** https://github.com/aysanaghazadeh/PersuasiveAdVLMBenchmark
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PersuasiveAdVLMBenchmark
- **Paper (WACV 2025):** https://doi.org/10.1109/wacv61041.2025.00469
- **Paper (arXiv):** https://arxiv.org/abs/2409.10719
- **Data:** https://drive.google.com/drive/folders/1sEUUxfGHIrsFF2CU7VJzavPC3LC0jeAM
- **Point of Contact:** https://github.com/creative-graphic-design/huggingface-datasets/issues
- **Original Source Contact:** https://github.com/aysanaghazadeh/PersuasiveAdVLMBenchmark/issues

### Dataset Summary

This release maps 783 relative PittAds image identifiers to 10,450 image URL references (10,208 unique URLs) curated through reverse image search. It is a metadata subset associated with _Benchmarking VLMs' Reasoning About Persuasive Atypical Images_.

The paper introduces multi-label atypicality classification, atypicality statement retrieval, and atypical object recognition. The released JSON contains image identifiers and URL lists only; it does not include image files, task prompts, or ground-truth labels, so it cannot reproduce the full benchmark by itself.

### Supported Tasks and Leaderboards

The paper evaluates vision-language model reasoning about persuasive atypical images using three tasks: multi-label atypicality classification, atypicality statement retrieval, and atypical object recognition. This loader exposes the released image URL metadata, which can support image retrieval workflows but is insufficient on its own to evaluate those tasks. No leaderboard is included.

### Languages

The released metadata contains image paths and URLs, not language annotations. The languages of the linked web pages are not specified.

## Dataset Structure

### Data Instances

Load the published dataset from the Hub with:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/PersuasiveAdVLMBenchmark")
```

An instance contains an image identifier and its associated URL list:

```json
{
  "id": "0/10000.jpg",
  "image_path": "0/10000.jpg",
  "image_urls": ["<image URL>", "<image URL>"]
}
```

`image_path` is the relative identifier used as a key in the upstream JSON. The URL lists vary in length from 1 to 82 entries.

### Data Fields

- `id`: Stable identifier copied from the upstream JSON key.
- `image_path`: Relative PittAds image identifier from the upstream JSON. The image itself is not included.
- `image_urls`: URLs associated with that image identifier by the upstream reverse-image-search curation.

### Data Splits

The upstream release is a single JSON file with 783 image identifiers and no named train, validation, or test partition. This implementation exposes the file as a `test` split because it accompanies an evaluation benchmark; that split name is a loader convention, not a split assignment documented by the upstream authors.

| Split  | Image identifiers | URL references |
| :----- | ----------------: | -------------: |
| `test` |               783 |         10,450 |

## Dataset Creation

### Curation Rationale

The paper studies how vision-language models reason about atypical imagery in persuasive advertisements. The upstream README describes this released subset as PittAds image metadata with multiple URLs per image, collected through reverse image search.

### Source Data

The upstream Google Drive folder contains `processed_pittads_wacv_sample.json`. Its top-level keys are relative image paths and each value is a list of associated URLs. The upstream README says the file covers only a subset of PittAds and cautions that URLs may be invalid or irrelevant; users should verify them before relying on them.

The upstream repository's MIT license covers its code. The README and Google Drive release do not specify a license for the dataset metadata or linked images, so this dataset card uses `license: unknown`.

#### Initial Data Collection and Normalization

The upstream authors state that they used reverse image search to collect multiple URLs for each image and attempted to remove invalid, irrelevant, or unwanted URLs. The README does not describe the search provider, filtering criteria, or annotation process in further detail.

#### Who are the source language producers?

This release contains URLs and image identifiers rather than authored text. The source and language of the linked pages vary and are not described.

### Annotations

The URL lists are curated metadata, not ground-truth labels for the paper's three evaluation tasks. The upstream release does not provide task prompts or atypicality labels in this JSON file.

### Personal and Sensitive Information

The release contains links to third-party web pages. Those pages and their contents are outside this dataset card's control and may change or become unavailable.

## Considerations for Using the Data

### Social Impact of Dataset

The metadata can help locate advertisement images for research on visual persuasion and atypicality. Because it contains URL references rather than preserved image files, retrieval results can change over time and may not reproduce the original authors' data exactly.

### Discussion of Biases

The upstream README identifies this release as a subset of PittAds but does not state the sampling procedure or provide demographics for people or groups that may appear in linked advertisements. The subset should not be assumed to represent advertisements broadly.

### Other Known Limitations

- The dataset does not include image bytes, benchmark prompts, or task labels.
- Upstream authors warn that some URLs may be invalid or irrelevant; verify each URL before use.
- The upstream release does not specify a license for the metadata or linked image content.
- The `test` split name is assigned by this loader and is not documented as an official source split.

## Additional Information

### Dataset Curators

Thanks to [@aysanaghazadeh](https://github.com/aysanaghazadeh) and the PersuasiveAdVLM authors for releasing the project and data metadata.

### Citation

```bibtex
@InProceedings{Malakouti_2025_WACV,
    author    = {Malakouti, Sina and Aghazadeh, Aysan and Khandelwal, Ashmit and Kovashka, Adriana},
    title     = {Benchmarking VLMs' Reasoning About Persuasive Atypical Images},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4788--4798},
    doi       = {10.1109/wacv61041.2025.00469}
}
```
