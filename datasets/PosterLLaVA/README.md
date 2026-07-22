---
language:
  - en
license: cc-by-nc-4.0
pretty_name: PosterLLaVA
tags:
  - poster
  - graphic-design
  - layout-generation
  - visual-design
  - multimodal
annotations_creators:
  - machine-generated
language_creators:
  - machine-generated
size_categories:
  - 1K<n<10K
  - 10K<n<100K
source_datasets:
  - original
  - CGLDataset
  - PKUPosterLayout
task_categories:
  - image-to-text
  - text-to-image
configs:
  - config_name: qb_poster
    data_files:
      - split: train
        path: qb_poster/train-*
      - split: validation
        path: qb_poster/validation-*
  - config_name: user_constrained
    data_files:
      - split: train
        path: user_constrained/train-*
      - split: validation
        path: user_constrained/validation-*
---

# Dataset Card for PosterLLaVA

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://github.com/posterllava/PosterLLaVA
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/PosterLLaVA
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/PosterLLaVA
- **Original Code and Data:** https://github.com/posterllava/PosterLLaVA
- **Paper (IEEE TMM / arXiv):** https://arxiv.org/abs/2406.02884
- **Demo:** https://huggingface.co/spaces/posterllava/PosterLLaVA
- **Point of Contact:** https://github.com/posterllava/PosterLLaVA/issues

### Dataset Summary

PosterLLaVA contains poster layout data released with *PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM*. This loader exposes two configurations:

- `qb_poster`: QB-Poster social-media poster layout data with poster images, canvas size, foreground element labels, pixel boxes, normalized boxes, and LLaVA-style instruction conversations.
- `user_constrained`: user-constraint annotations for CGL-dataset and PosterLayout poster layout generation. The upstream release contains annotation text only; images and original bounding box annotations must be obtained from CGL-dataset and PosterLayout.

The upstream model README states that PosterLLaVA was fine-tuned with 7k banner layouts, 60k commercial poster layouts from CGL-dataset and PosterLayout with text constraints, and 4k social-media poster layouts from QB-Poster.

### Supported Tasks and Leaderboards

The dataset supports poster layout generation and multimodal instruction tuning for layout placement. For QB-Poster, a model can condition on an image and a prompt that lists foreground element labels, then generate normalized `[left, top, right, bottom]` boxes. For User-Constrained, a model can condition layout generation on natural-language constraints such as relative placement, alignment, overlap, and ordering.

No active public leaderboard is bundled with this Hugging Face dataset. The upstream repository notes that evaluation code is planned separately.

### Languages

Prompts and user constraints are in English (`en`).

## Dataset Structure

### Data Instances

Load QB-Poster:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/PosterLLaVA", name="qb_poster")
```

A QB-Poster row contains a poster image path, normalized layout elements, the generated prompt, and the instruction-answer conversation:

```json
{
  "id": "poster_id",
  "image": "<image>",
  "image_path": ".../inpainted_1d5x/poster_id.png",
  "width": 800,
  "height": 600,
  "split": "train",
  "elements": [
    {
      "label": "text",
      "x_center": 400.0,
      "y_center": 300.0,
      "width": 200.0,
      "height": 100.0,
      "left": 300.0,
      "top": 250.0,
      "right": 500.0,
      "bottom": 350.0,
      "box": [0.375, 0.4167, 0.625, 0.5833]
    }
  ],
  "prompt": "<image>\\nHello! Could you please help me...",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "Sure! Here is the design results: [...]"}
  ],
  "raw_annotation": "{...}"
}
```

Load User-Constrained:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/PosterLLaVA", name="user_constrained")
```

A User-Constrained row contains a source dataset identifier and natural-language constraints:

```json
{
  "id": "cgl-O1CN01HnK3zH1HoH7oxbsE5_!!3409010804-0-alimamazszw",
  "source_dataset": "cgl",
  "source_id": "O1CN01HnK3zH1HoH7oxbsE5_!!3409010804-0-alimamazszw",
  "split": "train",
  "user_constraints": [
    "text_0 needs to be at the bottom of the background image.",
    "text_0 needs to be centered horizontally in the background image."
  ],
  "num_constraints": 2,
  "raw_annotation": "{...}"
}
```

### Data Fields

`qb_poster` fields:

- `id` (`string`): Poster identifier from the upstream annotation key.
- `image` (`Image`): Poster/background image resolved from the extracted archive. The loader prefers `inpainted_1d5x` for train, `inpainted_1x` for validation, and falls back to `original_poster`.
- `image_path` (`string`): Local resolved image path.
- `width`, `height` (`int32`): Poster canvas size in pixels.
- `split` (`string`): Normalized split name, either `train` or `validation`.
- `elements` (`list`): Foreground elements with label, center coordinates, pixel box edges, and normalized `[left, top, right, bottom]` box.
- `prompt` (`string`): LLaVA-style human instruction generated with the upstream prompt template.
- `conversations` (`list`): Human/GPT instruction-answer pair generated from the annotation.
- `raw_annotation` (`string`): JSON-encoded upstream annotation object.

`user_constrained` fields:

- `id` (`string`): Stable row identifier generated as `{source_dataset}-{source_id}`.
- `source_dataset` (`string`): `cgl` or `posterlayout`.
- `source_id` (`string`): Identifier from the upstream annotation file.
- `split` (`string`): `train` or `validation`.
- `user_constraints` (`list`): Natural-language user constraints.
- `num_constraints` (`int32`): Number of constraints in the row.
- `raw_annotation` (`string`): JSON-encoded upstream annotation object.

### Data Splits

QB-Poster preserves the `split` field from the upstream annotation file. The upstream README describes QB-Poster as 4k social-media poster layouts, but this loader does not hard-code a train/validation count because the full raw archive was not mirrored in this repository.

| Config | Split | Rows |
| --- | --- | ---: |
| `qb_poster` | train | From upstream annotation |
| `qb_poster` | validation | From upstream annotation |
| `user_constrained` | train | 64,519 |
| `user_constrained` | validation | 6,002 |

The User-Constrained counts come from the public Google Drive archive files inspected for this loader: `cgl_train.json` has 54,546 rows, `posterlayout_train.json` has 9,973 rows, and `cgl_val.json` has 6,002 rows.

### Local Data and Downloads

The loader downloads the upstream Google Drive archives with `gdown` when no local data path is supplied. For reproducible local work, pass either a downloaded archive file or an extracted directory through `data_dir`.

Expected QB-Poster extracted layout:

```text
data/qbposter/raw/
  original_poster/
  saliency_map/
  inpainted_1x/
  inpainted_1d5x/
  annotations.json
```

The upstream README also provides preprocessed QB-Poster assets separately:

- Training inpainted backgrounds: Google Drive id `1lfq-OY7yrsNl59v8sgSi6vYklpySDHED`
- Evaluation inpainted backgrounds: Google Drive id `1YVrvFT_jVkTodSZsbmOyPDyQV6fVjigp`
- Saliency maps: Google Drive id `1I8sXNX7QHfHlocI_23EN6Xl9pYQdGEoC`

Expected User-Constrained extracted layout:

```text
ucposter/
  cgl_train.json
  cgl_val.json
  posterlayout_train.json
```

Run the full download test explicitly:

```shell
POSTER_LLAVA_RUN_DOWNLOAD_TESTS=1 uv run pytest -vsx datasets/PosterLLaVA/tests/PosterLLaVA_test.py::test_load_dataset
```

## Dataset Creation

### Curation Rationale

PosterLLaVA was created to unify multiple poster layout generation settings under a multimodal layout generator. QB-Poster targets social-media poster layouts, while User-Constrained augments commercial poster layout generation with natural-language layout constraints.

### Source Data

QB-Poster raw data is released through Google Drive and contains original poster images and JSON annotations. The upstream README notes that inpainting and saliency detection are needed to obtain the background images and saliency maps used by the paper; the loader uses available inpainted directories when present and otherwise falls back to original poster images.

User-Constrained is released through Google Drive and contains only user-constraint annotation files. Use CGL-dataset and PosterLayout to obtain the corresponding poster images and original bounding box annotations.

### Annotations

QB-Poster annotations contain canvas dimensions and element boxes with center coordinates, width, height, and label. This loader converts each element to pixel box edges and normalized `[left, top, right, bottom]` boxes, then recreates the instruction-answer conversations used by the upstream preprocessing script.

User-Constrained annotations contain natural-language constraints for CGL-dataset and PosterLayout examples.

### Personal and Sensitive Information

The public dataset card and upstream repository do not identify personal information in the released annotations. Poster imagery may contain commercial-style designs, rendered products, logos, or text because the dataset targets poster layout generation.

## Considerations for Using the Data

### Social Impact of Dataset

The dataset is intended for scientific research on poster layout generation and multimodal design systems. Generated layouts may inherit biases or commercial design assumptions from the source poster datasets and the language constraints.

### Discussion of Biases

Poster layouts and constraints may emphasize commercial advertising aesthetics and the design conventions represented in CGL-dataset, PosterLayout, and QB-Poster. Models trained on this data may generalize less well to other design domains, languages, accessibility needs, or cultural layout conventions.

### Other Known Limitations

QB-Poster uses image assets hosted outside this repository on Google Drive. User-Constrained does not include poster images or original bounding boxes; it must be joined with CGL-dataset or PosterLayout for full multimodal layout-generation experiments.

## Additional Information

### Dataset Curators

The original PosterLLaVA dataset and code were released by the PosterLLaVA authors. This Hugging Face loader and dataset card are maintained by the Creative Graphic Design Datasets contributors.

### Licensing Information

The upstream PosterLLaVA repository is licensed under Creative Commons Attribution-NonCommercial 4.0 International (`cc-by-nc-4.0`). The upstream README states that the proposed dataset is authorized for scientific research and not for commercial use without authorization.

### Citation Information

```bibtex
@misc{yang2024posterllava,
  title={PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM},
  author={Yang, Tao and Luo, Yingmin and Qi, Zhongang and Wu, Yang and Shan, Ying and Chen, Chang Wen},
  year={2024},
  eprint={2406.02884},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2406.02884},
  note={Accepted to IEEE Transactions on Multimedia}
}
```

### Contributions

Contributions are welcome through pull requests to the Creative Graphic Design Datasets repository.
