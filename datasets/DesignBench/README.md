---
language:
  - en
license: unknown
pretty_name: DesignBench
tags:
  - web-design
  - html
  - css
  - design-generation
  - design-editing
  - design-repair
annotations_creators:
  - found
language_creators:
  - found
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - text-generation
  - image-to-text
task_ids: []
configs:
  - config_name: "edit=angular"
    data_files:
      - split: test
        path: edit=angular/test-*
  - config_name: "edit=react"
    data_files:
      - split: test
        path: edit=react/test-*
  - config_name: "edit=vanilla"
    data_files:
      - split: test
        path: edit=vanilla/test-*
  - config_name: "edit=vue"
    data_files:
      - split: test
        path: edit=vue/test-*
  - config_name: "generation=angular"
    data_files:
      - split: test
        path: generation=angular/test-*
  - config_name: "generation=react"
    data_files:
      - split: test
        path: generation=react/test-*
  - config_name: "generation=vanilla"
    data_files:
      - split: test
        path: generation=vanilla/test-*
  - config_name: "generation=vue"
    data_files:
      - split: test
        path: generation=vue/test-*
  - config_name: "repair=vanilla"
    data_files:
      - split: test
        path: repair=vanilla/test-*
---

# Dataset Card for DesignBench

[![CI](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets/actions/workflows/push_to_hub.yaml)

## Dataset Description

- **Homepage:** https://webpai.github.io/DesignBench/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/DesignBench
- **Hugging Face Dataset:** https://huggingface.co/datasets/creative-graphic-design/DesignBench
- **Paper (arXiv):** https://arxiv.org/abs/2506.06251

### Dataset Summary

DesignBench is a multi-framework, multi-task benchmark for evaluating MLLM-based front-end engineering. The paper targets limitations of prior UI code generation benchmarks by covering React, Vue, Angular, and vanilla HTML/CSS, and by evaluating generation, edit, and repair workflows. The full benchmark contains 900 webpage samples spanning multiple topics, edit types, and issue categories.

### Supported Tasks and Leaderboards

The dataset supports front-end code generation, design editing, and repair evaluation from visual and textual design inputs. No public leaderboard is bundled with this Hugging Face packaging.

### Languages

Prompts, code, and metadata are primarily English (`en`).

## Dataset Structure

### Data Fields

Generation configs contain `screenshot`, `html`, and `json`. Edit and repair configs contain source/target screenshots and structured JSON metadata for the requested operation.

### Data Splits

All configs expose a single `test` split.

| Config | Rows |
| --- | ---: |
| edit=angular | 66 |
| edit=react | 108 |
| edit=vanilla | 80 |
| edit=vue | 105 |
| generation=angular | 83 |
| generation=react | 109 |
| generation=vanilla | 120 |
| generation=vue | 118 |
| repair=vanilla | 28 |

## Dataset Creation

The benchmark was released to evaluate front-end code generation and design transformation systems across different frameworks and task types.

## Considerations for Using the Data

DesignBench is an evaluation benchmark and should not be treated as a complete proxy for production front-end quality, accessibility, or maintainability.

## Additional Information

### Licensing Information

The dataset license is listed as unknown in the local loader metadata.

### Citation Information

```bibtex
@misc{xiao2025designbench,
  title={DesignBench: A Comprehensive Benchmark for MLLM-based Front-end Code Generation},
  author={Jingyu Xiao and Ming Wang and Man Ho Lam and Yuxuan Wan and Junliang Liu and Yintong Huo and Michael R. Lyu},
  year={2025},
  eprint={2506.06251},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2506.06251}
}
```
