---
language:
  - en
license: cc-by-nc-4.0
pretty_name: ChallengeSVG
tags:
  - svg
  - animation
  - evaluation
  - motion-prompts
annotations_creators:
  - found
language_creators:
  - found
size_categories:
  - n<1K
source_datasets:
  - xingxm/SVGX-Core-250k
task_categories:
  - text-to-video
  - image-to-video
dataset_info:
  features:
  - name: id
    dtype: string
  - name: prompt
    dtype: string
  - name: svg
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: test
    num_bytes: 1631632.0
    num_examples: 35
  download_size: 1486930
  dataset_size: 1631632.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# Dataset Card for ChallengeSVG

## Dataset Description

- **Homepage:** https://matanlevy.com/LiveSVG/
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets/tree/main/datasets/ChallengeSVG
- **Paper:** https://arxiv.org/abs/2605.30174
- **Venue:** SIGGRAPH Asia 2026, as stated on the project page
- **Source archive:** https://matanlevy.com/LiveSVG/ChallengeSVG.zip
  ChallengeSVG is an evaluation-only stress test for SVG animation. It pairs 35 SVG illustrations with short English motion prompts and was curated from [SVGX-Core-250k](https://huggingface.co/datasets/xingxm/SVGX-Core-250k).

## Dataset Structure

### Data Instances

Load the published dataset from the Hub:

```python
import datasets as ds

dataset = ds.load_dataset("creative-graphic-design/ChallengeSVG")
```

The single `test` split contains 35 rows. Each row has an `id` (folder name), a `prompt` (trimmed text-file line), the raw UTF-8 `svg` source, and an `image` preview rendered as a transparent 512 x 512 PNG.

```json
{
  "id": "bowling",
  "prompt": "A bowling ball striking the pins",
  "svg": "<svg ...>",
  "image": "<image>"
}
```

### Rebuild from the original source

Load the original source archive through the repository loader:

```python
import datasets as ds

dataset = ds.load_dataset(
    "datasets/ChallengeSVG/ChallengeSVG.py",
    trust_remote_code=True,
)
```

### Data Splits

The `test` split is for evaluation only. The dataset has no training split, reference videos, or benchmark scores.

## Dataset Creation

The LiveSVG paper describes ChallengeSVG as a compact evaluation stress test curated from SVGX-Core-250k. The source archive contains 70 files: an SVG and one-line prompt for each of 35 IDs. Each ID matches its folder name and SVG file stem.

The loader downloads [ChallengeSVG.zip](https://matanlevy.com/LiveSVG/ChallengeSVG.zip). Its HTTP `Last-Modified` date is 2026-10-23 and its SHA-256 is `a71644e71d689ab5056d4bdd660cc09b92f589f1c53633a89b08e4302f7adf09`. SVG text is preserved; prompts are stripped of surrounding whitespace. The image preview is rendered at load time with `resvg_py`.

## Considerations for Using the Data

The 35 items form a small curated stress test, not a representative sample of SVG styles, topics, or motion descriptions. The data is intended for evaluation, not training. Results should not be generalized to all SVG animation tasks.

## Additional Information

### Licensing Information

ChallengeSVG states no license. Its SVGs come from [SVGX-Core-250k](https://huggingface.co/datasets/xingxm/SVGX-Core-250k), which is CC BY-NC 4.0 and requires attribution. Some original emoji sets may have more permissive licenses, but these were not verified item by item. This dataset therefore follows the source collection's CC BY-NC 4.0 terms; attribution is required and commercial use is prohibited.

### Citation Information

Please cite LiveSVG and SVGX-Core-250k. SVGX-Core-250k also requests citations for [LLM4SVG](https://arxiv.org/abs/2412.11102) and [SVGFusion](https://arxiv.org/abs/2412.10437).

```bibtex
@article{levy2026livesvg,
  title={LiveSVG: Zero-Shot SVG Animation via Video Generation},
  author={Levy, Matan and Margolin, Ran and Cavia, Bar and Samuel, Dvir and Pritch, Yael and Peleg, Shmuel and Acha, Alex Rav and Shamir, Ariel and Lischinski, Dani},
  journal={arXiv preprint arXiv:2605.30174},
  year={2026}
}
```

The project page states that the paper was accepted to SIGGRAPH Asia 2026. No ACM Digital Library record was found; the citation above follows the supplied arXiv BibTeX.

- **Point of Contact:** https://github.com/creative-graphic-design/huggingface-datasets/issues
- **Upstream contact:** No public contact address is listed on the project page.

### Contributions

Thanks to the LiveSVG and SVGX-Core-250k authors for releasing the source data. This repository provides the Hugging Face loader.
