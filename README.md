<p align="center">
  <img src=".github/teaser.png" alt="Creative Graphic Design Datasets teaser" width="100%">
</p>

# 🤗 Creative Graphic Design Datasets

A collection of Hugging Face dataset loaders and dataset cards for graphic design research. This repository makes datasets for design generation, layout understanding, typography, editing, and aesthetic evaluation easier to find, load, and cite.

## Datasets

- **[AesEvalBench](datasets/AesEvalBench/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2603.01083-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.01083) [![Paper](https://img.shields.io/badge/Paper-ICLR%2726-blue?logo=doi&logoColor=white)](https://openreview.net/forum?id=QGv6QwDA4z) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/arctanxarc/AesEval-Bench) [![HF Hub](https://img.shields.io/badge/HF%20Hub-AesEvalBench-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/AesEvalBench)
  - Graphic design samples with aesthetic ratings and vision-language model judgments.
    - ➡️ Input: Perturbed graphic design preview, optional element metadata, and aesthetic indicator.
    - ⬅️ Output: Issue labels, problematic-region choices, and bounding boxes.
- **[BannerRequest400](datasets/BannerRequest400/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2503.11060-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.11060) [![Paper](https://img.shields.io/badge/Paper-EMNLP%2725-blue?logo=doi&logoColor=white)](https://doi.org/10.18653/v1/2025.emnlp-main.214) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/sony/BannerAgency/tree/main/BannerRequest400) [![HF Hub](https://img.shields.io/badge/HF%20Hub-BannerRequest400-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/BannerRequest400)
  - Advertising banner requests, brand logos, and multimodal design instructions.
    - ➡️ Input: Brand logo plus abstract or concrete English banner request.
    - ⬅️ Output: Expected banner design matching size and campaign context.
- **[Camera](datasets/Camera/)**
  - ![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-lightgrey?logo=arxiv&logoColor=white) [![Paper](https://img.shields.io/badge/Paper-NLP%2723-blue?logo=doi&logoColor=white)](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H11-4.pdf) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/CyberAgentAILab/camera) [![HF Hub](https://img.shields.io/badge/HF%20Hub-CAMERA-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CAMERA)
  - Japanese advertising landing-page images, metadata, and ad copy references.
    - ➡️ Input: Japanese keyword, landing-page text/OCR, domain, and optional LP screenshot.
    - ⬅️ Output: Japanese ad headline/title references.
- **[CGLDataset](datasets/CGLDataset/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2205.00303-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2205.00303) [![Paper](https://img.shields.io/badge/Paper-IJCAI%2722-blue?logo=doi&logoColor=white)](https://doi.org/10.24963/ijcai.2022/692) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/minzhouGithub/CGL-GAN) [![HF Hub](https://img.shields.io/badge/HF%20Hub-CGL--Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)
  - Advertising poster images, background assets, and layout annotations.
    - ➡️ Input: Advertising poster or inpainted background image.
    - ⬅️ Output: COCO-style element categories and bounding boxes for poster layout.
- **[CGLDatasetV2](datasets/CGLDatasetV2/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2306.09086-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2306.09086) [![Paper](https://img.shields.io/badge/Paper-CIKM%2723-blue?logo=doi&logoColor=white)](https://doi.org/10.1145/3583780.3615028) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/liuan0803/RADM) [![HF Hub](https://img.shields.io/badge/HF%20Hub-CGL--Dataset--v2-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset-v2)
  - Poster background images with text annotations and layout metadata.
    - ➡️ Input: Poster/background image with text annotations or text features.
    - ⬅️ Output: Element categories, boxes, masks, and text-aware layout annotations.
- **[CreativePSD](datasets/CreativePSD/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2603.25738-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.25738) [![Paper](https://img.shields.io/badge/Paper-CVPR%2726-blue?logo=doi&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2026/html/Shuai_PSDesigner_Automated_Graphic_Design_with_a_Human-Like_Creative_Workflow_CVPR_2026_paper.html) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](https://modelscope.cn/datasets/song322/CreativePSD) [![HF Hub](https://img.shields.io/badge/HF%20Hub-CreativePSD-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CreativePSD)
  - PSD-derived graphic design data with layer structures, tool-call trajectories, source assets, and stepwise rendered images.
    - ➡️ Input: PSD-derived poster archive with layer metadata, source assets, and tool-call trajectory.
    - ⬅️ Output: PSD tree text, rendered poster images, final rendering, and archive file manifest.
- **[CTXFont](datasets/CTXFont/)**
  - ![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-lightgrey?logo=arxiv&logoColor=white) [![Paper](https://img.shields.io/badge/Paper-Pacific%20Graphics%2718-blue?logo=doi&logoColor=white)](https://doi.org/10.1111/cgf.13576) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/nanxuanzhao/CTXFont-dataset) [![HF Hub](https://img.shields.io/badge/HF%20Hub-CTXFont-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/CTXFont)
  - Web design screenshots, text elements, font properties, and contextual metadata.
    - ➡️ Input: Web screenshot, text element text/box, HTML tags, and design context.
    - ⬅️ Output: Text element font face, color, and size.
- **[DesignBench](datasets/DesignBench/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2506.06251-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.06251) [![Paper](https://img.shields.io/badge/Paper-arXiv-blue?logo=doi&logoColor=white)](https://doi.org/10.48550/arXiv.2506.06251) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=githubpages&logoColor=white)](https://webpai.github.io/DesignBench/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-DesignBench-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/DesignBench)
  - Web design prompts, HTML/CSS code, editing cases, repair cases, and compilation metadata.
    - ➡️ Input: Webpage screenshot/code plus task metadata for generation, edit, or repair.
    - ⬅️ Output: HTML/CSS code, edited target page, or repaired page/code.
- **[DEsignBenchPrompts](datasets/DEsignBenchPrompts/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2310.15144-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2310.15144) [![Paper](https://img.shields.io/badge/Paper-arXiv-blue?logo=doi&logoColor=white)](https://doi.org/10.48550/arXiv.2310.15144) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=githubpages&logoColor=white)](https://design-bench.github.io/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-DEsignBench--Prompts-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/DEsignBench-Prompts)
  - Visual design text-to-image prompts with original user inputs, expanded prompts, and aspect ratios.
    - ➡️ Input: User or expanded visual-design prompt, plus requested aspect ratio.
    - ⬅️ Output: Expected generated design image; this loader does not include ground-truth images.
- **[Desigen](datasets/Desigen/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2403.09093-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2403.09093) [![Paper](https://img.shields.io/badge/Paper-CVPR%2724-blue?logo=doi&logoColor=white)](https://doi.org/10.1109/CVPR52733.2024.01209) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](https://whaohan.github.io/desigen/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-Desigen-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Desigen)
  - Web advertisement design data with background images, text prompts, and layout annotations.
    - ➡️ Input: Advertisement background image, prompt, regions, descriptions, and canvas size.
    - ⬅️ Output: Layout elements with boxes, text, and element-type labels.
- **[GraphicDesignEvaluation](datasets/GraphicDesignEvaluation/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2410.08885-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.08885) [![Paper](https://img.shields.io/badge/Paper-SIGGRAPH%20Asia%2724-blue?logo=doi&logoColor=white)](https://doi.org/10.1145/3681758.3698010) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=githubpages&logoColor=white)](https://cyberagentailab.github.io/Graphic-design-evaluation/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-GraphicDesignEvaluation-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/GraphicDesignEvaluation)
  - Graphic design samples with alignment, overlap, and white-space quality scores.
    - ➡️ Input: Graphic banner image, perturbation/comparison setting, evaluator type, and design principle.
    - ⬅️ Output: Absolute quality scores or relative preference labels.
- **[GenPoster100K](datasets/GenPoster100K/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2510.15749-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.15749) [![Paper](https://img.shields.io/badge/Paper-ICCV%2725-blue?logo=doi&logoColor=white)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SEGA_A_Stepwise_Evolution_Paradigm_for_Content-Aware_Layout_Generation_with_ICCV_2025_paper.html) [![Original](https://img.shields.io/badge/Original-HF%20Hub-0F766E?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/BruceW91/GenPoster-100K) [![HF Hub](https://img.shields.io/badge/HF%20Hub-GenPoster100K-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/GenPoster100K)
  - Poster layout data with rendered backgrounds, composited images, PSD references, regions, and layer-level typography and color metadata.
    - ➡️ Input: Poster background image, PSD path, regions, and per-layer rendered images plus text/typography metadata.
    - ⬅️ Output: Composited poster image and structured layer annotations with boxes, colors, labels, and typography attributes.
- **[LayoutDETR](datasets/LayoutDETR/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2212.09877-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2212.09877) [![Paper](https://img.shields.io/badge/Paper-ECCV%2724-blue?logo=doi&logoColor=white)](https://arxiv.org/abs/2212.09877) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/salesforce/LayoutDETR) [![HF Hub](https://img.shields.io/badge/HF%20Hub-LayoutDETR-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/LayoutDETR)
  - Advertising banner images, foreground layout annotations, and inpainted background assets.
    - ➡️ Input: Ad banner image or inpainted background image plus foreground text/category annotations.
    - ⬅️ Output: Foreground element labels and bounding boxes in pixel and normalized formats.
- **[LICA](datasets/LICA/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2603.16098-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.16098) [![Paper](https://img.shields.io/badge/Paper-arXiv-blue?logo=doi&logoColor=white)](https://doi.org/10.48550/arXiv.2603.16098) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/lica-world/lica-dataset) [![HF Hub](https://img.shields.io/badge/HF%20Hub-LICA-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/LICA)
  - Rendered graphic design layouts, component-level specifications, and natural-language annotations.
    - ➡️ Input: Rendered designs, template/category metadata, and natural-language design annotations.
    - ⬅️ Output: Component-level layout JSON, template annotations, and design/aesthetic descriptions.
- **[Magazine](datasets/Magazine/)**
  - ![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-lightgrey?logo=arxiv&logoColor=white) [![Paper](https://img.shields.io/badge/Paper-SIGGRAPH%2719-blue?logo=doi&logoColor=white)](https://doi.org/10.1145/3306346.3322971) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](https://xtqiao.com/projects/content_aware_layout/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-Magazine-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Magazine)
  - Magazine layout data with fine-grained layout annotations and keyword labels.
    - ➡️ Input: Magazine page images, category labels, and text keywords.
    - ⬅️ Output: Polygon layouts for text, images, headlines, and overlay elements.
- **[ObjectRemovalAlpha](datasets/ObjectRemovalAlpha/)**
  - ![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-lightgrey?logo=arxiv&logoColor=white) ![Paper](https://img.shields.io/badge/Paper-not%20found-lightgrey) [![Original](https://img.shields.io/badge/Original-HF%20Hub-0F766E?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha) [![HF Hub](https://img.shields.io/badge/HF%20Hub-ObjectRemovalAlpha-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/ObjectRemovalAlpha)
  - Paired images, object-removal targets, and image inpainting references.
    - ➡️ Input: Source image, object-removal prompt, and removal mask.
    - ⬅️ Output: Ground-truth image after object removal.
- **[PKUPosterLayout](datasets/PKUPosterLayout/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2303.15937-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2303.15937) [![Paper](https://img.shields.io/badge/Paper-CVPR%2723-blue?logo=doi&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](http://59.108.48.34/tiki/PosterLayout/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-PKU--PosterLayout-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout)
  - Poster images, text elements, saliency maps, and visual-textual layout annotations.
    - ➡️ Input: Non-empty poster canvas or inpainted poster with saliency maps.
    - ⬅️ Output: Text, logo, and underlay bounding boxes.
- **[PittImageVideoAdsDataset](datasets/PittImageVideoAdsDataset/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-1707.03067-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/1707.03067) [![Paper](https://img.shields.io/badge/Paper-CVPR%2717-blue?logo=doi&logoColor=white)](https://openaccess.thecvf.com/content_cvpr_2017/html/Hussain_Automatic_Understanding_of_CVPR_2017_paper.html) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](https://people.cs.pitt.edu/~kovashka/ads/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-PittImageVideoAdsDataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PittImageVideoAdsDataset)
  - Image and video advertisement annotations with topics, sentiments, slogans, persuasive strategies, symbolic references, and action/reason Q/A.
    - ➡️ Input: Advertisement image or YouTube video ID with raw annotation responses.
    - ⬅️ Output: Topics, sentiments, slogans, persuasive strategies, symbolic references, and action/reason Q/A.
- **[PosterDNA 🔐](datasets/PosterDNA/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2601.03993-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.03993) [![Paper](https://img.shields.io/badge/Paper-AAAI%2726-blue?logo=doi&logoColor=white)](https://doi.org/10.1609/aaai.v40i9.37656) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/wuhaer/PosterVerse) ![HF Hub](https://img.shields.io/badge/HF%20Hub-not%20uploaded-lightgrey?logo=huggingface&logoColor=white)
  - Commercial-grade, text-dense poster images with HTML layout specifications, typography metadata, poster intention data, and a held-out test set; password-protected ZIPs under CC BY-NC-ND 4.0 are not mirrored to our Hugging Face Hub, and the loader is provided for reference.
    - ➡️ Input: Poster intention metadata, prompts, and design requirements.
    - ⬅️ Output: Background image plus HTML layout/typography specification.
- **[PosterIQ](datasets/PosterIQ/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2603.24078-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.24078) [![Paper](https://img.shields.io/badge/Paper-CVPR%2726-blue?logo=doi&logoColor=white)](https://arxiv.org/abs/2603.24078) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/ArtmeScienceLab/PosterIQ-Benchmark) [![HF Hub](https://img.shields.io/badge/HF%20Hub-PosterIQ-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PosterIQ)
  - Poster understanding images, generation prompts, and design-task metadata for typography, layout, OCR, composition, style, and design intention.
    - ➡️ Input: Poster image plus task prompt, or generation prompt only.
    - ⬅️ Output: Task answer metadata, ratings, OCR/localization labels, or generation criteria.
- **[PosterRewardBench](datasets/PosterRewardBench/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2603.29855-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.29855) [![Paper](https://img.shields.io/badge/Paper-CVPR%2726-blue?logo=doi&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2026/html/Lai_PosterReward_Unlocking_Accurate_Evaluation_for_High-Quality_Graphic_Design_Generation_CVPR_2026_paper.html) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/MeiGen-AI/PosterReward) [![HF Hub](https://img.shields.io/badge/HF%20Hub-PosterRewardBench-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PosterRewardBench)
  - Poster prompts with Basic and Advanced chosen/rejected image preference pairs.
    - ➡️ Input: Poster prompt with two generated candidate images.
    - ⬅️ Output: Pairwise preference: chosen higher-quality poster versus rejected poster.
- **[POSTA-PosterArt](datasets/POSTAPosterArt/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2503.14908-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.14908) [![Paper](https://img.shields.io/badge/Paper-CVPR%2725-blue?logo=doi&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_POSTA_A_Go-to_Framework_for_Customized_Artistic_Poster_Generation_CVPR_2025_paper.html) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](https://haoyuchen.com/POSTA) [![HF Hub](https://img.shields.io/badge/HF%20Hub-POSTAPosterArt-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/POSTAPosterArt)
  - Artistic poster images with layout, typography, stylized text, and segmentation annotations.
    - ➡️ Input: Poster backgrounds and title-region images with captions.
    - ⬅️ Output: Final posters, typography/layout annotations, and text segmentation masks.
- **[PosterErase](datasets/PosterErase/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-2204.12743-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2204.12743) [![Paper](https://img.shields.io/badge/Paper-ACM%20MM%2722-blue?logo=doi&logoColor=white)](https://doi.org/10.1145/3503161.3547905) [![Original](https://img.shields.io/badge/Original-GitHub-0F766E?logo=github&logoColor=white)](https://github.com/alimama-creative/Self-supervised-Text-Erasing) [![HF Hub](https://img.shields.io/badge/HF%20Hub-PosterErase-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PosterErase)
  - Poster images, text masks, and clean targets for text removal.
    - ➡️ Input: Text-containing poster image with text masks and placement annotations.
    - ⬅️ Output: Text-erased poster image; train split lacks ground truth.
- **[PubLayNet](datasets/PubLayNet/)**
  - [![arXiv](https://img.shields.io/badge/arXiv-1908.07836-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/1908.07836) [![Paper](https://img.shields.io/badge/Paper-ICDAR%2719-blue?logo=doi&logoColor=white)](https://doi.org/10.1109/ICDAR.2019.00166) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](https://developer.ibm.com/exchanges/data/all/publaynet/) [![HF Hub](https://img.shields.io/badge/HF%20Hub-PubLayNet-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
  - Scientific document page images with COCO-style layout annotations.
    - ➡️ Input: Scientific document page image.
    - ⬅️ Output: COCO-style boxes/segmentations for text, title, list, table, and figure regions.
- **[Rico](datasets/Rico/)**
  - ![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-lightgrey?logo=arxiv&logoColor=white) [![Paper](https://img.shields.io/badge/Paper-UIST%2717-blue?logo=doi&logoColor=white)](https://doi.org/10.1145/3126594.3126651) [![Original](https://img.shields.io/badge/Original-Project%20Page-0F766E?logo=homepage&logoColor=white)](http://www.interactionmining.org/rico.html) [![HF Hub](https://img.shields.io/badge/HF%20Hub-Rico-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
  - Mobile app screenshots, view hierarchies, UI layout vectors, and semantic annotations.
    - ➡️ Input: Mobile app screenshot, metadata, and Android view hierarchy.
    - ⬅️ Output: Semantic labels, hierarchies, layout vectors, or app metadata.

## Maintainer Notes

For agent-assisted maintenance, name the relevant skill and include the source links needed for the task.

Use `create-dataset` to add a dataset. Include the dataset name when known, plus paper, project, upstream dataset, archive, or data-file links.

```text
# Codex
$create-dataset Add a dataset from <paper URL>. The project page is <project URL>, and the source data appears to be available from <dataset or archive URL>.

# Claude Code
/create-dataset Add a dataset from <paper URL>. The project page is <project URL>, and the source data appears to be available from <dataset or archive URL>.
```

Use `publish-dataset` to verify or publish an existing `datasets/<DatasetName>` loader on the Hugging Face Hub.

For dataset card fixes or Hub README updates, follow `docs/dataset-card-maintenance.md`:

```shell
uv run pytest -q tests/test_dataset_cards.py
```

## License

This repository is licensed under the [Apache License 2.0](LICENSE). Dataset contents may be subject to the terms of their original sources; see each dataset card and original source for details.
