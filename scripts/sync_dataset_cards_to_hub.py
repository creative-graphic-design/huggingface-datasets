from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]

DATASET_CARD_REPOS = {
    "AesEvalBench": "creative-graphic-design/AesEvalBench",
    "BannerRequest400": "creative-graphic-design/BannerRequest400",
    "Camera": "creative-graphic-design/CAMERA",
    "CGLDataset": "creative-graphic-design/CGL-Dataset",
    "CGLDatasetV2": "creative-graphic-design/CGL-Dataset-v2",
    "CreativePSD": "creative-graphic-design/CreativePSD",
    "CTXFont": "creative-graphic-design/CTXFont",
    "DesignBench": "creative-graphic-design/DesignBench",
    "DEsignBenchPrompts": "creative-graphic-design/DEsignBench-Prompts",
    "Desigen": "creative-graphic-design/Desigen",
    "GenPoster100K": "creative-graphic-design/GenPoster100K",
    "GraphicDesignEvaluation": "creative-graphic-design/GraphicDesignEvaluation",
    "LICA": "creative-graphic-design/LICA",
    "Magazine": "creative-graphic-design/Magazine",
    "ObjectRemovalAlpha": "creative-graphic-design/ObjectRemovalAlpha",
    "PKUPosterLayout": "creative-graphic-design/PKU-PosterLayout",
    "PittImageVideoAdsDataset": "creative-graphic-design/PittImageVideoAdsDataset",
    "POSTAPosterArt": "creative-graphic-design/POSTAPosterArt",
    "PosterErase": "creative-graphic-design/PosterErase",
    "PosterIQ": "creative-graphic-design/PosterIQ",
    "PosterRewardBench": "creative-graphic-design/PosterRewardBench",
    "PubLayNet": "creative-graphic-design/PubLayNet",
    "Rico": "creative-graphic-design/Rico",
}

PRESERVED_REMOTE_FRONTMATTER_KEYS = ("dataset_info", "configs")


def split_frontmatter(text: str) -> tuple[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, flags=re.DOTALL)
    if not match:
        return "", text
    return match.group(1), match.group(2)


def frontmatter_has_key(frontmatter: str, key: str) -> bool:
    return re.search(rf"^{re.escape(key)}:", frontmatter, flags=re.MULTILINE) is not None


def extract_top_level_block(frontmatter: str, key: str) -> str:
    lines = frontmatter.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == f"{key}:"),
        None,
    )
    if start is None:
        return ""

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "-")) and re.match(r"^[^:]+:", line):
            end = index
            break

    return "\n".join(lines[start:end]).rstrip()


def merge_remote_frontmatter(local_text: str, remote_text: str) -> str:
    local_frontmatter, local_body = split_frontmatter(local_text)
    remote_frontmatter, _ = split_frontmatter(remote_text)

    additions = []
    for key in PRESERVED_REMOTE_FRONTMATTER_KEYS:
        if frontmatter_has_key(local_frontmatter, key):
            continue
        block = extract_top_level_block(remote_frontmatter, key)
        if block:
            additions.append(block)

    merged_frontmatter = local_frontmatter.strip()
    if additions:
        merged_frontmatter = "\n".join(
            part for part in [merged_frontmatter, *additions] if part
        )

    if not merged_frontmatter:
        return local_body
    return f"---\n{merged_frontmatter}\n---\n\n{local_body.lstrip()}"


def local_card_path(dataset_name: str) -> Path:
    return ROOT / "datasets" / dataset_name / "README.md"


def load_remote_card(repo_id: str) -> str:
    path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
    return Path(path).read_text()


def build_upload_payload(dataset_name: str) -> tuple[str, str]:
    repo_id = DATASET_CARD_REPOS[dataset_name]
    local_text = local_card_path(dataset_name).read_text()
    remote_text = load_remote_card(repo_id)
    return repo_id, merge_remote_frontmatter(local_text, remote_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local dataset cards to Hugging Face Hub README.md files."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_CARD_REPOS),
        help="Dataset name to sync. May be passed more than once.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync all tracked dataset cards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build payloads and print changed README sizes without uploading.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually upload README.md files. Required unless --dry-run is used.",
    )
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(DATASET_CARD_REPOS)
    if args.dataset:
        return args.dataset
    raise SystemExit("Pass --dataset at least once, or pass --all.")


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing to upload without --yes. Use --dry-run to preview.")
    if args.yes and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN must be set before uploading.")

    api = HfApi()
    for dataset_name in selected_datasets(args):
        repo_id, payload = build_upload_payload(dataset_name)
        local_size = len(local_card_path(dataset_name).read_text())
        payload_size = len(payload)
        if args.dry_run:
            print(f"DRY RUN {dataset_name}: {repo_id} {local_size} -> {payload_size}")
            continue

        api.upload_file(
            path_or_fileobj=payload.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update {dataset_name} dataset card",
        )
        print(f"Uploaded {dataset_name}: {repo_id}")


if __name__ == "__main__":
    main()
