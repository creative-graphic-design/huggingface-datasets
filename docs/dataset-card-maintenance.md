# Dataset Card Maintenance

Use this checklist when fixing or publishing Hugging Face dataset cards in this
repository.

## Common Failure Modes

- Hub README body is empty or near-empty because only Hub-generated
  `dataset_info` / `configs` frontmatter was uploaded.
- Root README links to a local directory name instead of the actual public Hub
  repo ID, for example `Camera` vs `CAMERA`.
- Dataset-card template comments contain unrelated paper links. Hugging Face can
  extract those comments into metadata tags, such as unrelated arXiv IDs.
- Markdown lines become malformed after deleting HTML comments, for example two
  `**Paper` labels on one line.
- BibTeX entries use placeholders, `and others`, preprint stubs after a venue
  version exists, or metadata copied from the wrong paper.

## Review Checklist

1. Compare root README Hub badges against the public Hub repo IDs.
2. Check each local `datasets/<name>/README.md` is non-empty and has useful body
   text, not only YAML frontmatter.
3. Remove dataset-card template comments before publishing to Hub.
4. Verify paper links and BibTeX against canonical sources:
   arXiv for preprints, ACL Anthology for ACL/EMNLP papers, CVF Open Access or
   DOI pages for CVPR papers, and DOI pages when a DOI is available.
5. Keep `README.md` and the loader `_CITATION` in sync when both exist.
6. Preserve Hub-generated `dataset_info` and `configs` when uploading local
   cards to Hub.
7. Do not push to Hub until the GitHub PR has been reviewed.

## Local Validation

Run the static dataset-card checks:

```shell
uv run pytest -q tests/test_dataset_cards.py
```

Run focused lint checks for edited maintenance scripts and loaders:

```shell
uv run ruff check tests/test_dataset_cards.py scripts/sync_dataset_cards_to_hub.py
```

Preview Hub upload payloads without publishing:

```shell
uv run python scripts/sync_dataset_cards_to_hub.py --dry-run --dataset Desigen
```

Upload only after review, with `HF_TOKEN` set:

```shell
uv run python scripts/sync_dataset_cards_to_hub.py --yes --dataset Desigen
```

## Adding New Regression Checks

When a new issue is found, encode it in `tests/test_dataset_cards.py` before
fixing the card. Prefer checks for durable patterns:

- known public Hub repo IDs
- known current paper URLs
- specific citation title/author fragments
- forbidden placeholders and unrelated template references
- malformed paper lines

This keeps the maintenance knowledge reusable without requiring future reviewers
to remember every previous failure manually.
