---
name: create-dataset
description: This skill should be used when the user asks to "create a dataset", "create a new huggingface dataset", "add a dataset", "implement a dataset", or discusses creating Hugging Face datasets in this monorepo. Provides a concrete MyHFDataset example to copy and customize.
version: 2.0.0
---

# Create Hugging Face Dataset Skill

Guides the creation of new Hugging Face datasets using a concrete example (`MyHFDataset`) stored in `.claude/skills/create-dataset/templates/MyHFDataset/`. Users copy and customize the example to create their own datasets.

## When to Use

Activate when user wants to create a new Hugging Face dataset in this monorepo.

## Workflow

### Step 1: Copy Example Structure

Ask the user for their dataset name (e.g., `CustomDataset`), then copy the MyHFDataset example:

1. **Copy the directory structure:**

   ```bash
   cp -r .claude/skills/create-dataset/templates/MyHFDataset datasets/{DatasetName}
   ```

2. **Rename files:**
   ```bash
   mv datasets/{DatasetName}/MyHFDataset.py datasets/{DatasetName}/{DatasetName}.py
   mv datasets/{DatasetName}/tests/MyHFDataset_test.py datasets/{DatasetName}/tests/{DatasetName}_test.py
   ```

### Step 2: Customize the Example

Guide the user to customize the copied files. Ask them for:

1. **dataset_name**: Their dataset name in CamelCase (e.g., CustomDataset)
2. **description**: Brief description of the dataset
3. **homepage**: Source project URL
4. **license**: License (e.g., MIT, Apache-2.0, Unknown)
5. **data_urls**: URLs to actual data files

**Files to customize:**

1. **`{DatasetName}.py`** - Main dataset implementation:

   - Replace class name: `MyHFDataset` → `{DatasetName}`
   - Update `_DESCRIPTION`, `_HOMEPAGE`, `_LICENSE` constants
   - Update `_URLS` with actual data source URLs
   - Implement `_info()` with correct features
   - Implement `_generate_examples()` with correct data parsing

2. **`tests/{DatasetName}_test.py`** - Test file:

   - Update `dataset_name()` fixture return value: `"MyHFDataset"` → `"{DatasetName}"`
   - Update `org_name()` fixture if needed: `"your-org"` → actual org name

3. **`README.md`** - Documentation:

   - Replace title: `MyHFDataset` → `{DatasetName}`
   - Update homepage, repository, paper URLs
   - Fill in dataset-specific details

4. **`pyproject.toml`** - Package configuration:
   - Update `name`: `"my-hf-dataset"` → `"{dataset-name}"` (kebab-case)
   - Update `description` with dataset-specific text

### Step 2.1: Configure Dependencies

After customizing the example, update `datasets/{DatasetName}/pyproject.toml` dependencies based on dataset requirements:

**Decision Tree:**

- **Does your dataset include images (ds.Image() features)?**
  - **YES** → Add `datasets[vision]>=2.0.0,<4.0.0`
  - **NO** → Leave dependencies empty (inherits from workspace root)

**For Image-Based Datasets:**

Update pyproject.toml:

```toml
[project]
name = "{dataset-name}"
version = "0.1.0"
description = "..."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "datasets[vision]>=2.0.0,<4.0.0",
]
```

**Important Notes:**

- Use `datasets[vision]`, **not** direct `pillow` dependency
- The `datasets[vision]` extra includes Pillow and related vision dependencies
- This is required for `ds.Image()` features to work properly

**For Non-Image Datasets:**

Keep dependencies empty:

```toml
dependencies = []
```

The workspace root provides base `datasets` package, so non-image datasets inherit all necessary dependencies.

### Step 3: Gather Dataset-Specific Requirements

After generating template files, ask user:

1. **Data Sources:**

   - URLs for data files
   - Format (JSON, JSONL, CSV, ZIP, images)?

2. **Configuration:**

   - Single config or multiple configs?
   - Config names (if multiple)?
   - **Note:** For multi-config datasets, see Step 4.4.1 for recommended patterns

3. **Data Structure:**

   - What fields exist in the data?
   - Field types (text, numbers, images, nested)?
   - Sample data (if available)?

4. **Metadata (optional):**
   - BibTeX citation?

### Step 4: Implement Dataset Logic

Update customized `datasets/MyHFDataset/MyHFDataset.py` (or your dataset name):

#### 4.1 Update \_URLS

Replace:

```python
_URLS = {
    "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
    "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}
```

With actual:

```python
_URLS = {
    "data": "https://actual-source.com/data.json",
}
```

#### 4.2 Update \_CITATION

If user provides:

```python
_CITATION = """\
@article{...}
"""
```

Otherwise keep placeholder.

#### 4.3 Update \_DESCRIPTION

If generic, enhance by fetching from homepage. Otherwise keep user-provided.

#### 4.4 Configure Builder

For multi-config datasets with type safety, see **Step 4.4.1** below for recommended patterns.

**Single config:**

```python
class MyHFDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
```

**Multiple configs:**

```python
BUILDER_CONFIGS = [
    ds.BuilderConfig(name="config1", version=VERSION, description="..."),
    ds.BuilderConfig(name="config2", version=VERSION, description="..."),
]
```

**Advanced (with StrEnum):**

```python
from enum import StrEnum, auto
from dataclasses import dataclass

class MyHFDatasetType(StrEnum):
    config1 = auto()
    config2 = auto()

@dataclass
class MyHFDatasetConfig(ds.BuilderConfig):
    name: MyHFDatasetType

BUILDER_CONFIG_CLASS = MyHFDatasetConfig
BUILDER_CONFIGS = [
    MyHFDatasetConfig(name=MyHFDatasetType.config1, version=VERSION),
]
```

#### 4.4.1 Best Practices for Multi-Config Datasets

When working with multiple configurations, these patterns improve type safety, enable exhaustiveness checking, and make your code more maintainable.

**Why These Patterns Matter:**

- **Type Safety**: Enums prevent typos and invalid config names at compile time
- **Exhaustiveness Checking**: Type checker ensures all config cases are handled
- **Maintainability**: Adding new configs triggers compile errors where implementation is needed

**Pattern 1: StrEnum with Direct `name` Field**

Define config types using StrEnum and use the enum directly in the `name` field:

```python
from enum import StrEnum, auto
from dataclasses import dataclass

class MyHFDatasetType(StrEnum):
    config1 = auto()
    config2 = auto()

@dataclass
class MyHFDatasetConfig(ds.BuilderConfig):
    name: MyHFDatasetType

BUILDER_CONFIG_CLASS = MyHFDatasetConfig
BUILDER_CONFIGS = [
    MyHFDatasetConfig(name=MyHFDatasetType.config1, version=VERSION, description="..."),
    MyHFDatasetConfig(name=MyHFDatasetType.config2, version=VERSION, description="..."),
]

# Optional: Add type hint for better IDE support
config: MyHFDatasetConfig
```

**Pattern 2: `match/case` with `assert_never`**

Use `match/case` statements with `assert_never` to ensure all config cases are handled:

```python
from typing import assert_never

def _info(self) -> ds.DatasetInfo:
    match self.config.name:
        case MyHFDatasetType.config1:
            features = ds.Features({
                "field1": ds.Value("string"),
            })
        case MyHFDatasetType.config2:
            features = ds.Features({
                "field2": ds.Value("int32"),
            })
        case _:
            assert_never(self.config.name)

    return ds.DatasetInfo(
        description=_DESCRIPTION,
        features=features,
        homepage=_HOMEPAGE,
        license=_LICENSE,
        citation=_CITATION,
    )
```

**Apply this pattern in all config-dependent methods:**

- `_info()` - for defining features per config
- `_split_generators()` - for config-specific data loading
- `_generate_examples()` - for config-specific example generation

**Complete Example:**

```python
from enum import StrEnum, auto
from dataclasses import dataclass
from typing import List, assert_never
import datasets as ds

class MyDatasetType(StrEnum):
    small = auto()
    large = auto()

@dataclass
class MyDatasetConfig(ds.BuilderConfig):
    name: MyDatasetType

class MyDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")

    config: MyDatasetConfig

    BUILDER_CONFIG_CLASS = MyDatasetConfig
    BUILDER_CONFIGS = [
        MyDatasetConfig(name=MyDatasetType.small, version=VERSION, description="Small subset"),
        MyDatasetConfig(name=MyDatasetType.large, version=VERSION, description="Full dataset"),
    ]

    DEFAULT_CONFIG_NAME = "small"

    def _info(self) -> ds.DatasetInfo:
        match self.config.name:
            case MyDatasetType.small:
                features = ds.Features({"text": ds.Value("string")})
            case MyDatasetType.large:
                features = ds.Features({
                    "text": ds.Value("string"),
                    "metadata": ds.Value("string"),
                })
            case _:
                assert_never(self.config.name)

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager) -> List[ds.SplitGenerator]:
        match self.config.name:
            case MyDatasetType.small:
                data_url = _URLS["small"]
            case MyDatasetType.large:
                data_url = _URLS["large"]
            case _:
                assert_never(self.config.name)

        filepath = dl_manager.download_and_extract(data_url)
        return [ds.SplitGenerator(name=ds.Split.TRAIN, gen_kwargs={"filepath": filepath})]

    def _generate_examples(self, filepath):
        with open(filepath) as f:
            for idx, line in enumerate(f):
                data = json.loads(line)

                match self.config.name:
                    case MyDatasetType.small:
                        yield idx, {"text": data["text"]}
                    case MyDatasetType.large:
                        yield idx, {
                            "text": data["text"],
                            "metadata": data.get("metadata", ""),
                        }
                    case _:
                        assert_never(self.config.name)
```

**When to Use:**

- Use StrEnum + direct `name` field for **any multi-config dataset**
- Use `match/case` with `assert_never` when **config affects behavior** in methods

#### 4.5 Define Features

```python
def _info(self):
    features = ds.Features({
        "text": ds.Value("string"),
        "number": ds.Value("int32"),
        "image": ds.Image(),
        "nested": {"field": ds.Value("string")},
    })

    return ds.DatasetInfo(
        description=_DESCRIPTION,
        features=features,
        homepage=_HOMEPAGE,
        license=_LICENSE,
        citation=_CITATION,
    )
```

**Types:**

- `ds.Value("string")`, `ds.Value("int32")`, `ds.Value("float")`, `ds.Value("bool")`
- `ds.Image()`, `ds.Audio()`
- `{...}` for nested, `[ds.Value(...)]` for lists

**Important:** If using `ds.Image()` features, ensure you've added `datasets[vision]>=2.0.0,<4.0.0` to your `pyproject.toml` dependencies (see Step 2.1).

#### 4.6 Implement \_split_generators

**Simple:**

```python
def _split_generators(self, dl_manager):
    files = dl_manager.download(_URLS)
    with open(files["data"]) as f:
        data = json.load(f)

    return [ds.SplitGenerator(name=ds.Split.TRAIN, gen_kwargs={"data": data})]
```

**Multiple splits:**

```python
def _split_generators(self, dl_manager):
    files = dl_manager.download_and_extract(_URLS[self.config.name])

    return [
        ds.SplitGenerator(name=ds.Split.TRAIN, gen_kwargs={"filepath": os.path.join(files, "train.jsonl")}),
        ds.SplitGenerator(name=ds.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(files, "dev.jsonl")}),
    ]
```

**With images:**

```python
def _split_generators(self, dl_manager):
    files = dl_manager.download_and_extract(_URLS)
    with open(files["metadata"]) as f:
        data = json.load(f)

    for item in data:
        item["image_path"] = dl_manager.download(f"{BASE_URL}/{item['image_name']}")

    return [ds.SplitGenerator(name=ds.Split.TRAIN, gen_kwargs={"data": data})]
```

#### 4.7 Implement \_generate_examples

**Simple:**

```python
def _generate_examples(self, data):
    for idx, item in enumerate(data):
        yield idx, {"text": item["text"], "label": item["label"]}
```

**From file:**

```python
def _generate_examples(self, filepath):
    with open(filepath) as f:
        for idx, line in enumerate(f):
            yield idx, json.loads(line)
```

**With images:**

```python
from PIL import Image

def _generate_examples(self, data):
    for idx, item in enumerate(data):
        yield idx, {
            "text": item["text"],
            "image": Image.open(item["image_path"]),
        }
```

### Step 5: Update Tests

Update `datasets/MyHFDataset/tests/MyHFDataset_test.py` (or your dataset name):

**Single config:**

```python
def test_load_dataset(dataset_path: str, trust_remote_code: bool = True):
    dataset = ds.load_dataset(path=dataset_path, trust_remote_code=trust_remote_code)
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == EXPECTED_COUNT
```

**Multiple configs:**

```python
@pytest.mark.parametrize(
    argnames=("config_name", "expected_num_train"),
    argvalues=(("config1", 100), ("config2", 200)),
)
def test_load_dataset(dataset_path: str, config_name: str, expected_num_train: int, trust_remote_code: bool = True):
    dataset = ds.load_dataset(path=dataset_path, name=config_name, trust_remote_code=trust_remote_code)
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train
```

#### Test Ordering

**IMPORTANT: Test function order matters!**

The test file must define tests in this specific order:

1. **First:** `test_load_dataset` - This test calls `dataset.push_to_hub()` which creates the repository on Hugging Face Hub
2. **Second:** `test_push_readme_to_hub` - This test uploads the README to the repository

**Why this order is critical:**

```python
def test_load_dataset(dataset_path: str, repo_id: str):
    dataset = ds.load_dataset(path=dataset_path, trust_remote_code=True)
    # This creates the repository on HF Hub:
    dataset.push_to_hub(repo_id=repo_id)

# This test must come AFTER test_load_dataset
def test_push_readme_to_hub(hf_api: HfApi, repo_id: str, script_dir: str):
    readme_path = os.path.join(script_dir, "README.md")
    # This will fail if the repository doesn't exist yet:
    hf_api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
```

**Key points:**

- `test_load_dataset` must be defined first because it creates the repository
- `test_push_readme_to_hub` must be defined second because it requires the repository to exist
- Pytest runs tests alphabetically by default, so "test_load_dataset" runs before "test_push_readme_to_hub" ✅
- Having them in the correct order in the file makes the dependency clear to anyone reading the code

### Step 6: Update README Files

Update both the dataset README and the repository root README.

#### 6.0 Add YAML Frontmatter to Dataset README

The README must include YAML frontmatter at the very top for Hugging Face Hub metadata.

**Option 1: Use the Tagging App (Recommended for beginners)**

1. Visit https://huggingface.co/spaces/huggingface/datasets-tagging
2. Fill in the form with your dataset information
3. Copy the generated YAML and paste at the top of your README

**Option 2: Write YAML Manually**

Add YAML frontmatter block at the top of README.md (following the official specification at https://github.com/huggingface/hub-docs/blob/main/datasetcard.md):

```yaml
---
# Core fields (recommended for all datasets)
language:
  - en # BCP-47 language code(s). Examples: en, ja, zh, multilingual
license: unknown # STRING (not list!). Use HF license identifiers or "unknown"
pretty_name: YourDatasetName # Human-readable display name
tags:
  - your-domain # Custom tags for discoverability
  - specific-task

# Recommended fields
annotations_creators:
  - machine-generated # Options: crowdsourced, found, expert-generated, machine-generated
language_creators:
  - found # Options: crowdsourced, found, expert-generated, machine-generated
size_categories:
  - n<1K # Format: n<1K, 1K<n<10K, 10K<n<100K, 100K<n<1M, 1M<n<10M, etc.
source_datasets:
  - original # Or list other datasets if derived
task_categories:
  - other # See HF docs for full list (image-to-text, text-to-image, etc.)
task_ids: [] # Usually empty

# Optional fields (add if applicable)
# paperswithcode_id: your-dataset-name  # If dataset is on Papers with Code
# language_details:  # More specific language codes
#   - en-US
#   - fr-FR
---
```

**Official Field Reference** (based on https://github.com/huggingface/hub-docs/blob/main/datasetcard.md):

**Core fields** (recommended for all datasets):

- `language` (list): BCP-47 language codes (en, ja, zh, etc.). Use "multilingual" for multiple languages
- `license` (string): **NOTE: STRING not list!** Use HF license identifiers, "other", or "unknown"
- `pretty_name` (string): Human-readable dataset name displayed on Hub
- `tags` (list): Custom keywords for discoverability

**Recommended fields**:

- `annotations_creators` (list): How annotations were created - crowdsourced, found, expert-generated, machine-generated
- `language_creators` (list): How language data was created - crowdsourced, found, expert-generated, machine-generated
- `size_categories` (list): Dataset size - n<1K, 1K<n<10K, 10K<n<100K, 100K<n<1M, 1M<n<10M, etc.
- `source_datasets` (list): "original" if new, or names of source datasets if derived
- `task_categories` (list): Standard HF task types (see https://huggingface.co/docs/hub/datasets-cards)
- `task_ids` (list): Usually empty unless using specific task identifiers

**Optional fields**:

- `paperswithcode_id` (string): Link to Papers with Code dataset
- `language_details` (list): More specific BCP-47 codes (e.g., "en-US", "fr-FR")
- `configs` (list): Define multiple dataset configurations
- `dataset_info`: Auto-generated by datasets-cli

**Common Mistakes to Avoid**:

- ❌ `license: [unknown]` - license must be a STRING, not a list
- ❌ `multilinguality` field - NOT in official spec, don't use it
- ✅ `license: unknown` - correct format

**Examples from existing datasets:**

- CTXFont: `tags: [design, typography, font-prediction, web-design, graphic-design, context-aware]`
- BannerRequest400: `task_categories: [image-to-text, text-to-image]`, `license: mit`
- GraphicDesignEvaluation: `annotations_creators: [crowdsourced]`, `license: apache-2.0`

**Reference Documentation**:

- Official spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md
- Task categories: https://huggingface.co/docs/hub/datasets-cards
- License identifiers: https://huggingface.co/docs/hub/repositories-licenses

#### 6.1 Update Dataset README

Review `datasets/MyHFDataset/README.md` (or your dataset name) and update any placeholder information:

- Replace `{{ arxiv_url }}`, `{{ publication_venue }}`, `{{ publication_url }}` with actual values
- If information is not available, leave TODO comments for future updates
- Verify license information matches the source repository
- Add complete citation information if available

**Update the Contributions Section**

The Contributions section (at the end of README) credits those who made the dataset available. There are two common patterns:

**Pattern A: Credit Original Dataset Authors**

Use this when adapting an existing dataset from a paper or public repository:

```markdown
### Contributions

Thanks to [@original-author-username](https://github.com/original-author-username) for creating this dataset.
```

Examples:

- CTXFont: `Thanks to [@nanxuanzhao](https://github.com/nanxuanzhao) for creating this dataset.`
- GraphicDesignEvaluation: `Thanks to [@DaichiHaraguchi](https://github.com/DaichiHaraguchi) for creating this dataset.`

**Pattern B: Credit the HF Implementation**

Use this when creating a new dataset or when crediting your organization:

```markdown
### Contributions

This Hugging Face dataset implementation was created by [your-name] to make [DatasetName] more accessible to the research community.
```

Example:

- BannerRequest400: Credits the organization without specific individuals

**Guidelines:**

- If adapting a published dataset: Credit the original authors using Pattern A
- If you're the original dataset creator: You can use either pattern
- Include both if appropriate: "Thanks to [@author] for creating the dataset and [@contributor] for the HF implementation"
- Use GitHub usernames when known and publicly available

#### 6.2 Update Repository Root README

Add an entry for your new dataset to the repository root README.md:

**Location:** `/README.md` (repository root)

**Format:**

```markdown
- **{DatasetName}**: [Brief 1-2 sentence description]. [[Data Source]]({homepage_url}) [[Conference/Journal]]({paper_url}) [[arXiv]]({arxiv_url})
```

**Guidelines:**

1. **Description**: Keep it brief (1-2 sentences) focusing on what the dataset is for and its key features
2. **Links**: Include data source, conference/journal paper (if published), and arXiv (if available)
3. **Format**: Bullet list format, one line per dataset

**Example:**

```markdown
- **BannerRequest400**: Multimodal benchmark for advertising banner generation with 100 logos and 5,200 specifications. [[Data Source]](https://github.com/sony/BannerAgency/tree/main/BannerRequest400) [[EMNLP'25]](https://aclanthology.org/2025.emnlp-main.214/) [[arXiv]](https://arxiv.org/abs/2503.11060)
```

### Step 7: Test and Debug

```bash
uv run pytest -vsx datasets/MyHFDataset/tests
```

If errors, fix and re-run.

Verify loading:

```python
import datasets as ds
ds = ds.load_dataset("datasets/MyHFDataset/MyHFDataset.py", trust_remote_code=True)
print(ds["train"][0])
```

## Example Session

**User:** "Create a dataset called MyData"

**Assistant:**

```
I'll help you create MyData dataset!

Please provide:
1. description: Brief description of the dataset
2. homepage: Source project URL (e.g., https://github.com/org/project)
3. license: License (e.g., MIT)
4. dataset_version (optional, default 1.0.0): Version number
```

**User provides info**

**Assistant:**

```
Generating dataset structure...

[Read templates, substitute values, create files]

✅ Created:
- datasets/MyData/MyData.py
- datasets/MyData/tests/MyData_test.py
- datasets/MyData/tests/__init__.py
- datasets/MyData/README.md
- datasets/MyData/pyproject.toml

Now for dataset-specific info:
1. Data source URLs?
2. Data format (JSON, JSONL, CSV, etc.)?
3. Single config or multiple?
4. What fields exist in the data?
```

**User provides data details**

**Assistant:**

```
Implementing dataset logic...

[Update _URLS, Features, _split_generators, _generate_examples, tests, README.md]

✅ Implementation complete! Running tests...
```

```bash
uv run pytest -vsx datasets/MyData/tests
```

**Result:** Tests pass → Dataset ready!

## Checklist

- [ ] User info collected
- [ ] Files generated from templates
- [ ] \_URLS updated
- [ ] Features defined
- [ ] \_split_generators implemented
- [ ] \_generate_examples implemented
- [ ] Tests updated
- [ ] README.md updated
- [ ] Tests passing
- [ ] Dataset loads correctly
