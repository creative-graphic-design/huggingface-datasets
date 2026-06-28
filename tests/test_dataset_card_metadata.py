from pathlib import Path


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


def test_dataset_card_task_categories_are_official():
    repo_root = Path(__file__).resolve().parents[1]
    failures = []
    readme_paths = [
        *sorted((repo_root / "datasets").glob("*/README.md")),
        repo_root
        / ".agents"
        / "skills"
        / "create-dataset"
        / "templates"
        / "MyHFDataset"
        / "README.md",
    ]

    for readme_path in readme_paths:
        categories = _task_categories(
            _frontmatter(readme_path.read_text(encoding="utf-8"))
        )
        for category in categories:
            if category not in OFFICIAL_TASK_CATEGORIES:
                failures.append(
                    f"{readme_path.relative_to(repo_root)}: unsupported task category {category!r}"
                )

    assert failures == []
