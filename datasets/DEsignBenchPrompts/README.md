---
language:
  - multilingual
license: unknown
pretty_name: DEsignBench Prompts
tags:
  - visual-design
  - graphic-design
  - text-to-image
  - prompts
  - design-benchmark
annotations_creators:
  - found
language_creators:
  - found
size_categories:
  - n<1K
source_datasets:
  - original
task_categories:
  - text-to-image
task_ids: []
configs:
  - config_name: default
    data_files:
      - split: test
        path: test-*
---

# Dataset Card for DEsignBench Prompts

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://design-bench.github.io/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/DEsignBenchPrompts
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/DEsignBench-Prompts
- **Original Code:** https://github.com/microsoft/design-bench
- **Paper (arXiv):** https://arxiv.org/abs/2310.15144
- **Point of Contact:** https://github.com/microsoft/design-bench/issues

### Dataset Summary

DEsignBench Prompts contains the prompt table released with *DEsignBench: Exploring and Benchmarking DALL-E 3 for Imagining Visual Design*. It provides visual design text-to-image prompts with original user inputs, expanded prompts, and requested aspect ratios.

This Hugging Face loader covers the prompt TSV only. It does not include the DEsignBench gallery images or generated model outputs.

### Supported Tasks and Leaderboards

The dataset is intended for text-to-image prompt evaluation in visual design scenarios. Models can use `userinput` or `expandprompt` as generation prompts and evaluate generated images against the design intent described by each row.

No active public leaderboard is bundled with this Hugging Face dataset.

### Languages

The prompts are primarily written in English, with some prompts or expanded prompt text containing multilingual rendered text examples.

## Dataset Structure

### Data Instances

Load the dataset with:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/DEsignBench-Prompts")
```

```json
{
  "id": "text_0_0",
  "demo_imgname": "text_0_0",
  "userinput": "a graffiti art of the text \"free the pink\" on a wall",
  "expandprompt": "Photo of a smooth stone wall with the graffiti art 'free the pink' painted in a gradient from pink to blue. Surrounding the text are intricate patterns and a silhouette of a city skyline at the base.",
  "aspectratio": "wide"
}
```

### Data Fields

- `id` (`string`): Stable row identifier, copied from `demo_imgname`.
- `demo_imgname` (`string`): Original DEsignBench demo image name.
- `userinput` (`string`): Original user prompt.
- `expandprompt` (`string`): Expanded text-to-image prompt.
- `aspectratio` (`string`): Requested aspect ratio. The source values are `wide`, `square`, and `tall`.

### Data Splits

The dataset exposes one benchmark split:

| split | rows |
|------:|-----:|
| test | 215 |

The source aspect-ratio counts are:

| aspectratio | rows |
|------------:|-----:|
| wide | 173 |
| square | 36 |
| tall | 6 |

## Dataset Creation

### Source Data

The prompt TSV is distributed by the DEsignBench project page. DEsignBench was created as a text-to-image benchmark for visual design scenarios and includes categories for design technical capability and design application scenarios.

### Annotations

This loader exposes the original prompt metadata from the released TSV. It does not add new annotations.

### Personal and Sensitive Information

The dataset contains natural-language prompts for generated visual design tasks. It may include names of places, objects, events, brands, or text that a generated image should render. The dataset card does not identify personal information in the released prompt table.

## Considerations for Using the Data

### Social Impact and Limitations

The prompt set reflects the benchmark authors' visual design categories and prompt-writing process. Models evaluated with these prompts may overfit to the represented prompt distribution, aspect ratios, and design scenarios.

The gallery images released separately by the original project follow the terms of the image generation models that produced them. This Hugging Face dataset contains prompts only.

## Additional Information

### Licensing Information

The source Hugging Face dataset and project release do not declare a prompt dataset license, so this dataset is marked as `unknown`.

### Citation Information

```bibtex
@article{lin2023designbench,
  title={DEsignBench: Exploring and Benchmarking DALL-E 3 for Imagining Visual Design},
  author={Kevin Lin and Zhengyuan Yang and Linjie Li and Jianfeng Wang and Lijuan Wang},
  journal={arXiv preprint arXiv:2310.15144},
  year={2023}
}
```

### Contributions

Thanks to Kevin Lin, Zhengyuan Yang, Linjie Li, Jianfeng Wang, and Lijuan Wang for creating DEsignBench.
