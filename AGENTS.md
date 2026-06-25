# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

This is a Python monorepo for Hugging Face datasets focused on graphic design and creative AI evaluation. The repository uses `uv` as the package manager and follows a workspace structure where each dataset is independently managed under `datasets/`.

## Documentation

- In the root README dataset list, describe each entry as dataset contents, not as a paper task. Prefer concise noun-phrase descriptions such as "Graphic design aesthetic evaluation data..." over wording like "Benchmark for..." or "Dataset for...".
- In the root README dataset badges, set the `Original` badge `logo` to the source medium using Simple Icons slugs from https://simpleicons.org/: use `github` for GitHub repositories, `githubpages` for GitHub Pages project sites, `huggingface` for upstream Hugging Face datasets, and `homepage` for independent project pages when no more specific Simple Icons brand applies.

## Development Commands

### Testing

```shell
# Run all tests for a specific dataset
uv run pytest -vsx datasets/<DatasetName>/tests/
```

### Dataset Creation

Use the `/create-dataset` skill (see `.agents/skills/create-dataset/SKILL.md`) to create new datasets from templates:

```shell
# The skill automates:
# 1. Creating dataset structure from templates
# 2. Generating DatasetName.py, tests, README, pyproject.toml
# 3. Guiding through data source configuration
```

Alternatively, manually initialize:

```shell
uv init --app -p 3.10 datasets/<DatasetName>
```

**Improving the create-dataset skill:**
If you have feedback or improvements for the `/create-dataset` skill, use the `/skill-creator` skill to update the skill files in `.agents/skills/create-dataset/`. The skill-creator provides guidance on skill structure, best practices, and helps maintain consistency across skills.

### Downloading Data Files

Some datasets use Google Drive for data hosting:

```shell
uv run --with gdown gdown <google_drive_file_id>
```
