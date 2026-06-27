---
name: publish-dataset
description: Publish Hugging Face datasets from this monorepo to the Hugging Face Hub. Use when the user asks to push, publish, upload, release, or verify whether a dataset under datasets/DatasetName is already published to the Hub.
---

# Publish Dataset

Publish generated dataset rows to the Hugging Face Hub through this repository's dataset tests. Do not publish dataset rows by uploading the local dataset source directory.

## Workflow

1. Identify the dataset directory: `datasets/<DatasetName>`.
2. Read `datasets/<DatasetName>/tests/<DatasetName>_test.py`.
3. Determine the Hub `repo_id` from the test fixtures or constants, usually `org_name` plus `dataset_name`.
4. Identify whether `test_load_dataset` needs a dataset-specific download env var, such as `<DATASET>_RUN_DOWNLOAD_TESTS=1`.
5. Check whether the Hub dataset already exists and contains generated data:

   ```bash
   uv run --frozen python - <<'PY'
   from huggingface_hub import HfApi

   repo_id = "ORG/DATASET"
   files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
   print(f"{repo_id}: {len(files)} files")
   print("README.md", "README.md" in files)
   print("parquet files", sum(path.endswith(".parquet") for path in files))
   PY
   ```

6. If `README.md` and expected Parquet files/config directories already exist, report that the dataset is already published and do not republish unless the user explicitly asks.
7. If missing or partially published, run the write tests.
8. Verify the Hub files again after publish.
9. Remove local `__pycache__` created under the dataset directory and confirm `git status --short`.

## Publish Command

Use the test suite as the publishing interface:

```bash
HF_WRITE_TESTS=1 uv run --frozen pytest -vsx \
  datasets/<DatasetName>/tests/<DatasetName>_test.py::test_load_dataset \
  datasets/<DatasetName>/tests/<DatasetName>_test.py::test_push_readme_to_hub
```

For gated full-download tests, include the dataset-specific env var found in the test file:

```bash
HF_WRITE_TESTS=1 <DATASET>_RUN_DOWNLOAD_TESTS=1 uv run --frozen pytest -vsx \
  datasets/<DatasetName>/tests/<DatasetName>_test.py::test_load_dataset \
  datasets/<DatasetName>/tests/<DatasetName>_test.py::test_push_readme_to_hub
```

Keep the same long-running pytest session alive until it exits. Do not start a second publish while one is still running.

## Verification

After publish, verify:

- pytest finished successfully
- `README.md` exists in the Hub dataset repo
- Parquet shards exist for each expected split/config
- for multi-config datasets, every expected config has at least one `.parquet` file
- local generated caches are removed
- local worktree has no unintended changes

Example multi-config check:

```bash
uv run --frozen python - <<'PY'
from huggingface_hub import HfApi

repo_id = "ORG/DATASET"
files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
configs = sorted({path.split("/")[0] for path in files if "/" in path and path.endswith(".parquet")})
print(len(configs))
for config in configs:
    count = sum(path.startswith(config + "/") and path.endswith(".parquet") for path in files)
    print(f"{config}: {count}")
PY
```

## Important Rules

- Use `DatasetDict.push_to_hub()` through `test_load_dataset`; this creates the generated Parquet dataset.
- Use `HfApi.upload_file()` only for metadata such as `README.md`.
- Do not use `HfApi.upload_folder()` or upload the local `datasets/<DatasetName>` source tree to publish dataset rows.
- If a publish fails partway through, inspect which config failed and prefer rerunning the smallest applicable pytest target instead of blindly rerunning the full publish.
