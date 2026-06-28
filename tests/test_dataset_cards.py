from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_dataset_cards_to_hub import (  # noqa: E402
    DATASET_CARD_REPOS,
    merge_remote_frontmatter,
)

EXPECTED_HUB_REPOS = {
    "AesEvalBench": "creative-graphic-design/AesEvalBench",
    "BannerRequest400": "creative-graphic-design/BannerRequest400",
    "Camera": "creative-graphic-design/CAMERA",
    "CGLDataset": "creative-graphic-design/CGL-Dataset",
    "CGLDatasetV2": "creative-graphic-design/CGL-Dataset-v2",
    "CTXFont": "creative-graphic-design/CTXFont",
    "DesignBench": "creative-graphic-design/DesignBench",
    "Desigen": "creative-graphic-design/Desigen",
    "GraphicDesignEvaluation": "creative-graphic-design/GraphicDesignEvaluation",
    "LICA": "creative-graphic-design/LICA",
    "Magazine": "creative-graphic-design/Magazine",
    "ObjectRemovalAlpha": "creative-graphic-design/ObjectRemovalAlpha",
    "PKUPosterLayout": "creative-graphic-design/PKU-PosterLayout",
    "POSTAPosterArt": "creative-graphic-design/POSTAPosterArt",
    "PosterErase": "creative-graphic-design/PosterErase",
    "PosterIQ": "creative-graphic-design/PosterIQ",
    "PosterRewardBench": "creative-graphic-design/PosterRewardBench",
    "PubLayNet": "creative-graphic-design/PubLayNet",
    "Rico": "creative-graphic-design/Rico",
}

EXPECTED_ROOT_README_IO_DATASETS = [
    "AesEvalBench",
    "BannerRequest400",
    "Camera",
    "CGLDataset",
    "CGLDatasetV2",
    "CTXFont",
    "DesignBench",
    "DEsignBenchPrompts",
    "Desigen",
    "GraphicDesignEvaluation",
    "LICA",
    "Magazine",
    "ObjectRemovalAlpha",
    "PKUPosterLayout",
    "PosterDNA",
    "PosterIQ",
    "PosterRewardBench",
    "POSTA-PosterArt",
    "PosterErase",
    "PubLayNet",
    "Rico",
]

EXPECTED_CARD_LINKS = {
    "Camera": "https://huggingface.co/datasets/creative-graphic-design/CAMERA",
    "CGLDataset": "https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset",
    "CGLDatasetV2": "https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset-v2",
    "PKUPosterLayout": "https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout",
}

EXPECTED_PAPER_LINKS = {
    "GraphicDesignEvaluation": [
        "https://arxiv.org/abs/2410.08885",
        "https://doi.org/10.1145/3681758.3698010",
    ],
    "DesignBench": [
        "https://arxiv.org/abs/2506.06251",
    ],
}

EXPECTED_CITATION_TEXT = {
    "BannerRequest400": [
        "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
        "10.18653/v1/2025.emnlp-main.214",
    ],
    "Desigen": [
        "Weng, Haohan and Huang, Danqing",
        "10.1109/CVPR52733.2024.01209",
        "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    ],
    "DesignBench": [
        "DesignBench: A Comprehensive Benchmark for MLLM-based Front-end Code Generation",
        "Michael R. Lyu",
    ],
    "POSTAPosterArt": [
        "Chen, Haoyu and Xu, Xiaojie and Li, Wenbo",
        "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    ],
}

FORBIDDEN_PLACEHOLDERS = (
    "<!--",
    "-->",
    "TODO: YAML",
    "Dataset Paper Title",
    "TODO: Add BibTeX citation here",
    "Please input license information",
    "Please input description",
    "https://arxiv.org/abs/2005.00614",
    "https://www.aclweb.org/anthology/W17-1601.pdf",
    "https://www.aclweb.org/anthology/2020.acl-main.485/",
    "Dinan et al 2020",
    "Blodgett et al 2020",
    "Larson 2017",
    "author={Chen, Haoyu and others}",
    "author={Xiao, Shishi and Wang, Yufei",
)

OFFICIAL_TASK_CATEGORIES = {
    "audio-classification",
    "automatic-speech-recognition",
    "conversational",
    "depth-estimation",
    "document-question-answering",
    "feature-extraction",
    "fill-mask",
    "image-classification",
    "image-feature-extraction",
    "image-segmentation",
    "image-to-3d",
    "image-to-image",
    "image-to-text",
    "image-text-to-text",
    "mask-generation",
    "object-detection",
    "question-answering",
    "reinforcement-learning",
    "robotics",
    "sentence-similarity",
    "summarization",
    "table-question-answering",
    "tabular-classification",
    "tabular-regression",
    "text-classification",
    "text-generation",
    "text-to-3d",
    "text-to-audio",
    "text-to-image",
    "text-to-speech",
    "text2text-generation",
    "time-series-forecasting",
    "token-classification",
    "translation",
    "unconditional-image-generation",
    "video-classification",
    "visual-question-answering",
    "voice-activity-detection",
    "zero-shot-classification",
    "zero-shot-image-classification",
}


def dataset_card_path(dataset_name: str) -> Path:
    return ROOT / "datasets" / dataset_name / "README.md"


def tracked_dataset_card_paths() -> list[Path]:
    return sorted((ROOT / "datasets").glob("*/README.md"))


def _frontmatter(readme: str) -> list[str]:
    lines = readme.splitlines()
    if not lines or lines[0] != "---":
        return []
    try:
        end = lines.index("---", 1)
    except ValueError:
        return []
    return lines[1:end]


def _task_categories(frontmatter: list[str]) -> list[str]:
    categories = []
    in_task_categories = False
    for line in frontmatter:
        if line == "task_categories: []":
            return []
        if line == "task_categories:":
            in_task_categories = True
            continue
        if in_task_categories and line.startswith("  - "):
            categories.append(line.removeprefix("  - ").strip())
            continue
        if in_task_categories and line and not line.startswith(" "):
            break
    return categories


def test_tracked_dataset_cards_are_not_empty():
    for path in tracked_dataset_card_paths():
        if not (ROOT / ".git").exists() and not path.exists():
            continue
        assert path.read_text().strip(), f"{path.relative_to(ROOT)} is empty"


def test_dataset_card_task_categories_are_official():
    readme_paths = [
        *tracked_dataset_card_paths(),
        ROOT
        / ".agents"
        / "skills"
        / "create-dataset"
        / "templates"
        / "MyHFDataset"
        / "README.md",
    ]

    for path in readme_paths:
        categories = _task_categories(_frontmatter(path.read_text()))
        for category in categories:
            assert category in OFFICIAL_TASK_CATEGORIES, (
                f"{path.relative_to(ROOT)} uses unsupported task category {category!r}"
            )


def test_dataset_card_point_of_contact_values_are_render_safe():
    for path in tracked_dataset_card_paths():
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.startswith("- **Point of Contact:**"):
                continue

            value = line.removeprefix("- **Point of Contact:**").strip()
            assert value, f"{path.relative_to(ROOT)}:{line_number} has an empty contact"
            assert "TODO" not in value
            assert "[More Information Needed]" not in value
            assert not re.search(r"\]\(\s*\)", value), (
                f"{path.relative_to(ROOT)}:{line_number} has an empty contact link"
            )
            assert not re.search(r"\[[^\]]+\]\([^)]+\)", value), (
                f"{path.relative_to(ROOT)}:{line_number} uses a markdown contact link"
            )
            assert not re.fullmatch(r'["\'().\s]+', value), (
                f"{path.relative_to(ROOT)}:{line_number} is punctuation only"
            )


def test_root_readme_uses_public_hub_repo_ids():
    readme = (ROOT / "README.md").read_text()

    for dataset_name, repo_id in EXPECTED_HUB_REPOS.items():
        expected_url = f"https://huggingface.co/datasets/{repo_id}"
        assert expected_url in readme, f"{dataset_name} should link to {expected_url}"

    stale_urls = [
        "https://huggingface.co/datasets/creative-graphic-design/Camera",
        "https://huggingface.co/datasets/creative-graphic-design/CGLDataset",
        "https://huggingface.co/datasets/creative-graphic-design/CGLDatasetV2",
        "https://huggingface.co/datasets/creative-graphic-design/PKUPosterLayout",
    ]
    for stale_url in stale_urls:
        assert stale_url not in readme


def test_root_readme_original_badges_use_source_medium_labels():
    readme = (ROOT / "README.md").read_text()
    expected_label_by_logo = {
        "github": "GitHub",
        "githubpages": "Project%20Page",
        "homepage": "Project%20Page",
        "huggingface": "HF%20Hub",
    }
    badges = re.findall(
        r"Original-([^-]+(?:%20[^-]+)?)-0F766E\?logo=([^&]+)&logoColor=white",
        readme,
    )

    assert badges
    assert len(badges) == readme.count("Original-")
    for label, logo in badges:
        assert label == expected_label_by_logo[logo]


def test_root_readme_dataset_entries_include_input_output_notes():
    readme = (ROOT / "README.md").read_text()

    for dataset_name in EXPECTED_ROOT_README_IO_DATASETS:
        pattern = (
            rf"- \*\*\[{re.escape(dataset_name)}[^\]]*\]\([^)]*\)\*\*\n"
            rf"(?P<body>(?:(?:  -|    -) .+\n)+)"
        )
        match = re.search(pattern, readme)
        assert match, f"{dataset_name} entry not found"

        body = match.group("body")
        lines = body.splitlines()
        assert "img.shields.io" in lines[0], (
            f"{dataset_name} should place badges directly below the dataset name"
        )
        assert lines[2].startswith("    - ➡️ Input: "), (
            f"{dataset_name} should describe input after the description"
        )
        assert lines[3].startswith("    - ⬅️ Output: "), (
            f"{dataset_name} should describe output after input"
        )


def test_dataset_cards_include_known_public_hub_links():
    for dataset_name, expected_url in EXPECTED_CARD_LINKS.items():
        card = dataset_card_path(dataset_name).read_text()
        assert expected_url in card, f"{dataset_name} should link to {expected_url}"


def test_known_paper_links_are_current():
    root_readme = (ROOT / "README.md").read_text()

    for dataset_name, expected_links in EXPECTED_PAPER_LINKS.items():
        card = dataset_card_path(dataset_name).read_text()
        for expected_link in expected_links:
            assert expected_link in root_readme
            assert expected_link in card

    stale_graphic_design_eval_links = [
        "https://arxiv.org/abs/2410.10022",
        "https://doi.org/10.1145/3680528.3687588",
    ]
    for stale_link in stale_graphic_design_eval_links:
        assert stale_link not in root_readme
        assert (
            stale_link not in dataset_card_path("GraphicDesignEvaluation").read_text()
        )


def test_known_citation_text_is_specific():
    for dataset_name, expected_fragments in EXPECTED_CITATION_TEXT.items():
        card = dataset_card_path(dataset_name).read_text()
        loader = (ROOT / "datasets" / dataset_name / f"{dataset_name}.py").read_text()
        for expected_fragment in expected_fragments:
            assert expected_fragment in card
            assert expected_fragment in loader


def test_dataset_cards_do_not_contain_blocking_placeholders():
    for path in sorted((ROOT / "datasets").glob("*/README.md")):
        text = path.read_text()
        for placeholder in FORBIDDEN_PLACEHOLDERS:
            assert placeholder not in text, (
                f"{path.relative_to(ROOT)} contains {placeholder!r}"
            )


def test_loader_citations_do_not_contain_blocking_placeholders():
    for path in sorted((ROOT / "datasets").glob("*/*.py")):
        text = path.read_text()
        if "_CITATION" not in text:
            continue
        for placeholder in FORBIDDEN_PLACEHOLDERS:
            assert placeholder not in text, (
                f"{path.relative_to(ROOT)} contains {placeholder!r}"
            )


def test_dataset_card_paper_lines_are_not_concatenated():
    for path in sorted((ROOT / "datasets").glob("*/README.md")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "**Paper" not in line:
                continue
            assert line.count("**Paper") == 1, (
                f"{path.relative_to(ROOT)}:{line_number} has a malformed paper line"
            )


def test_sync_script_covers_all_expected_hub_repos():
    assert DATASET_CARD_REPOS == EXPECTED_HUB_REPOS


def test_merge_remote_frontmatter_preserves_hub_dataset_info():
    local = """---
license: unknown
pretty_name: Example
---

# Dataset Card for Example

Local body.
"""
    remote = """---
dataset_info:
  features:
  - name: image
    dtype: image
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

Remote body.
"""

    merged = merge_remote_frontmatter(local, remote)

    assert "# Dataset Card for Example" in merged
    assert "Local body." in merged
    assert "dataset_info:" in merged
    assert "configs:" in merged
    assert "Remote body." not in merged
