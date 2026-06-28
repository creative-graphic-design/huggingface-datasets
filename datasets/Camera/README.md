---
annotations_creators:
- crowdsourced
language:
- ja
language_creators:
- found
license:
- cc-by-nc-sa-4.0
multilinguality:
- monolingual
pretty_name: CAMERA
size_categories: []
source_datasets:
- original
tags: []
task_categories:
- text-generation
task_ids: []
---

# Dataset Card for CAMERA 📷


## Table of Contents
- [Table of Contents](#table-of-contents)
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

- **Homepage:** https://github.com/CyberAgentAILab/camera
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/Camera
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/CAMERA
- **Paper (NLP2023):** https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H11-4.pdf

### Dataset Summary

From [the official README.md](https://github.com/CyberAgentAILab/camera#camera-dataset):

> CAMERA (CyberAgent Multimodal Evaluation for Ad Text GeneRAtion) is the Japanese ad text generation dataset. We hope that our dataset will be useful in research for realizing more advanced ad text generation models.

### Supported Tasks and Leaderboards

[More Information Needed]

#### Supported Tasks

[More Information Needed]

#### Leaderboard

[More Information Needed]

### Languages

The language data in CAMERA is in Japanese ([BCP-47 ja-JP](https://www.rfc-editor.org/info/bcp47)).

## Dataset Structure

### Data Instances

When loading a specific configuration, users has to append a version dependent suffix:

#### without-lp-images

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/CAMERA", name="without-lp-images")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation'],
#         num_rows: 12395
#     })
#     validation: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation'],
#         num_rows: 3098
#     })
#     test: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation'],
#         num_rows: 869
#     })
# })
```

An example of the CAMERA (w/o LP images) dataset looks as follows:

```json
{
    "asset_id": 13861,
    "kw": "仙台 ホテル",
    "lp_meta_description": "仙台のホテルや旅館をお探しなら楽天トラベルへ！楽天ポイントが使えて、貯まって、とってもお得な宿泊予約サイトです。さらに割引クーポンも使える！国内ツアー・航空券・レンタカー・バス予約も！",
    "title_org": "仙台市のホテル",
    "title_ne1": "",
    "title_ne2": "",
    "title_ne3": "",
    "domain": "",
    "parsed_full_text_annotation": {
        "text": [
            "trivago",
            "Oops...AccessDenied 可",
            "Youarenotallowedtoviewthispage!Ifyouthinkthisisanerror,pleasecontacttrivago.",
            "Errorcode:0.3c99e86e.1672026945.25ba640YourIP:240d:1a:4d8:2800:b9b0:ea86:2087:d141AffectedURL:https://www.trivago.jp/ja/odr/%E8%BB%92", "%E4%BB%99%E5%8F%B0-%E5%9B%BD%E5%86%85?search=20072325",
            "Backtotrivago"
        ],
        "xmax": [
            653,
            838,
            765,
            773,
            815,
            649
        ],
        "xmin": [
            547,
            357,
            433,
            420,
            378,
            550
        ],
        "ymax": [
            47,
            390,
            475,
            558,
            598,
            663
        ],
        "ymin": [
            18,
            198,
            439,
            504,
            566,
            651
        ]
    }
}
```

#### with-lp-images

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/CAMERA", name="with-lp-images")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation', 'lp_image'],
#         num_rows: 12395
#     })
#     validation: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation', 'lp_image'],
#         num_rows: 3098
#     })
#     test: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation', 'lp_image'],
#         num_rows: 869
#     })
# })
```

An example of the CAMERA (w/ LP images) dataset looks as follows:

```json
{
    "asset_id": 13861,
    "kw": "仙台 ホテル",
    "lp_meta_description": "仙台のホテルや旅館をお探しなら楽天トラベルへ！楽天ポイントが使えて、貯まって、とってもお得な宿泊予約サイトです。さらに割引クーポンも使える！国内ツアー・航空券・レンタカー・バス予約も！",
    "title_org": "仙台市のホテル",
    "title_ne1": "",
    "title_ne2": "",
    "title_ne3": "",
    "domain": "",
    "parsed_full_text_annotation": {
        "text": [
            "trivago",
            "Oops...AccessDenied 可",
            "Youarenotallowedtoviewthispage!Ifyouthinkthisisanerror,pleasecontacttrivago.",
            "Errorcode:0.3c99e86e.1672026945.25ba640YourIP:240d:1a:4d8:2800:b9b0:ea86:2087:d141AffectedURL:https://www.trivago.jp/ja/odr/%E8%BB%92", "%E4%BB%99%E5%8F%B0-%E5%9B%BD%E5%86%85?search=20072325",
            "Backtotrivago"
        ],
        "xmax": [
            653,
            838,
            765,
            773,
            815,
            649
        ],
        "xmin": [
            547,
            357,
            433,
            420,
            378,
            550
        ],
        "ymax": [
            47,
            390,
            475,
            558,
            598,
            663
        ],
        "ymin": [
            18,
            198,
            439,
            504,
            566,
            651
        ]
    },
    "lp_image": <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1200x680 at 0x7F8513446B20>
}
```

### Data Fields

#### without-lp-images

- `asset_id`: ids (associated with LP images)
- `kw`: search keyword
- `lp_meta_description`: meta description extracted from LP (i.e., LP Text)
- `title_org`: ad text (original gold reference)
- `title_ne{1-3}`: ad text (additonal gold references for multi-reference evaluation)
- `domain`: industry domain (HR, EC, Fin, Edu) for industry-wise evaluation
- `parsed_full_text_annotation`: OCR results for LP images

#### with-lp-images

- `asset_id`: ids (associated with LP images)
- `kw`: search keyword
- `lp_meta_description`: meta description extracted from LP (i.e., LP Text)
- `title_org`: ad text (original gold reference)
- `title_ne{1-3}`: ad text (additional gold references for multi-reference evaluation)
- `domain`: industry domain (HR, EC, Fin, Edu) for industry-wise evaluation
- `parsed_full_text_annotation`: OCR results for LP images
- `lp_image`: Landing page (LP) image

### Data Splits

From [the official paper](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H11-4.pdf):

| Split | # of data | # of reference ad text | industry domain label |
|-------|----------:|-----------------------:|:---------------------:|
| Train | 12,395    | 1                      | -                     |
| Valid | 3,098     | 1                      | -                     |
| Test  | 869       | 4                      | ✔                     |

The public Google Cloud Storage archives referenced by the original loader currently return 404, so these split sizes follow the official paper rather than a fresh download.

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

[More Information Needed]

### Dataset Curators

[More Information Needed]

### Licensing Information

> This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

### Citation Information

```bibtex
@inproceedings{mita-et-al:nlp2023,
    author =    "三田 雅人 and 村上 聡一朗 and 張 培楠",
    title =	    "広告文生成タスクの規定とベンチマーク構築",
    booktitle = "言語処理学会 第 29 回年次大会",
    year =      2023,
}
```

### Contributions

Thanks to [Masato Mita](https://github.com/chemicaltree), [Soichiro Murakami](https://github.com/ichiroex), and [Peinan Zhang](https://github.com/peinan) for creating this dataset.
