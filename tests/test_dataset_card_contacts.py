import re
from pathlib import Path


CONTACT_PREFIX = "- **Point of Contact:**"


def test_dataset_card_point_of_contact_values_are_render_safe():
    repo_root = Path(__file__).resolve().parents[1]
    failures = []

    for readme_path in sorted((repo_root / "datasets").glob("*/README.md")):
        for line_number, line in enumerate(
            readme_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.startswith(CONTACT_PREFIX):
                continue

            value = line.removeprefix(CONTACT_PREFIX).strip()
            if not value:
                failures.append(
                    f"{readme_path.relative_to(repo_root)}:{line_number}: empty"
                )
            if "TODO" in value or "[More Information Needed]" in value:
                failures.append(
                    f"{readme_path.relative_to(repo_root)}:{line_number}: placeholder"
                )
            if re.search(r"\]\(\s*\)", value):
                failures.append(
                    f"{readme_path.relative_to(repo_root)}:{line_number}: empty link"
                )
            if re.search(r"\[[^\]]+\]\([^)]+\)", value):
                failures.append(
                    f"{readme_path.relative_to(repo_root)}:{line_number}: markdown link"
                )
            if re.fullmatch(r'["\'().\s]+', value):
                failures.append(
                    f"{readme_path.relative_to(repo_root)}:{line_number}: punctuation only"
                )

    assert failures == []
