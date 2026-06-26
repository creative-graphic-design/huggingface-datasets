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
- **Paper (Preprint):** https://arxiv.org/abs/2403.09093
- **Paper (CVPR2024):** https://doi.org/10.1109/CVPR52733.2024.01209

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
@inproceedings{Weng_2024_CVPR,
  title = {Desigen: A Pipeline for Controllable Design Template Generation},
  author = {Weng, Haohan and Huang, Danqing and Qiao, Yu and Hu, Zheng and Lin, Chin-Yew and Zhang, Tong and Chen, C. L. Philip},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
  pages = {12721--12732},
  doi = {10.1109/CVPR52733.2024.01209},
  url = {https://openaccess.thecvf.com/content/CVPR2024/html/Weng_Desigen_A_Pipeline_for_Controllable_Design_Template_Generation_CVPR_2024_paper.html}
}
```
