---
license: unknown
task_categories:
  - image-to-text
  - text-to-image
pretty_name: Desigen
size_categories:
  - 10K<n<100K
---

# Dataset Card for Desigen

## Dataset Description

- **Homepage:** https://whaohan.github.io/desigen/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/Desigen
- **Paper:** https://arxiv.org/abs/2403.09093

### Dataset Summary

Desigen contains web advertisement design data with background images, content prompts, layout element annotations, and design canvas sizes. This loader reads the parquet shards hosted at `creative-graphic-design/Desigen`.

## Dataset Structure

### Data Fields

- `image`: Background or rendered advertisement image.
- `prompt`: Text prompt associated with the background image.
- `region`: Image region boxes.
- `description`: Design content description.
- `elements`: Layout elements with bounding boxes, text, and element type labels.
- `size`: Design canvas size.

### Data Splits

| split      | examples |
|------------|---------:|
| train      |    36322 |
| validation |      999 |

## Citation

```bibtex
@article{xiao2024desigen,
  title={Desigen: A Pipeline for Controllable Design Template Generation},
  author={Xiao, Shishi and Wang, Yufei and Zhou, Rui and Hao, Haohan and Chen, Kai and Chen, Xi and Wei, Zhongyu},
  journal={arXiv preprint arXiv:2403.09093},
  year={2024}
}
```
