---
name: create-dataset
description: This skill should be used when the user asks to "create a dataset", "create a new huggingface dataset", "add a dataset", "implement a dataset", or discusses creating Hugging Face datasets in this monorepo.
version: 1.0.0
---

# Create Hugging Face Dataset Skill

Guides the creation of new Hugging Face datasets by generating code from templates stored in `.claude/skills/create-dataset/templates/`.

## When to Use

Activate when user wants to create a new Hugging Face dataset in this monorepo.

## Workflow

### Step 1: Gather Information

Ask user for:

1. **dataset_name** (required): Dataset name in CamelCase (e.g., MyDataset, BannerRequest400)
2. **description** (required): Brief description of the dataset
3. **homepage** (required): Source project URL (e.g., https://github.com/org/project)
4. **license** (required): License (e.g., MIT, Apache-2.0)
5. **dataset_version** (optional, default: 1.0.0): Version number

### Step 2: Generate Files from Templates

Read template files from `.claude/skills/create-dataset/templates/{{ dataset_name }}/` and replace all `{{ xxx }}` placeholders with actual values.

**Template files to process:**

1. `{{ dataset_name }}.py` → `datasets/{dataset_name}/{dataset_name}.py`
2. `tests/{{ dataset_name }}_test.py` → `datasets/{dataset_name}/tests/{dataset_name}_test.py`
3. `tests/__init__.py` → `datasets/{dataset_name}/tests/__init__.py`
4. `README.md` → `datasets/{dataset_name}/README.md`
5. `pyproject.toml` → `datasets/{dataset_name}/pyproject.toml`

**Replacement rules:**

- `{{ dataset_name }}` → user provided dataset_name
- `{{ description }}` → user provided description
- `{{ homepage }}` → user provided homepage
- `{{ license }}` → user provided license
- `{{ dataset_version }}` → user provided version (default: 1.0.0)

Note: Template files contain comments explaining additional transformations (e.g., module name lowercase, dependencies based on image usage).

### Step 3: Gather Dataset-Specific Requirements

After generating template files, ask user:

1. **Data Sources:**

   - URLs for data files
   - Format (JSON, JSONL, CSV, ZIP, images)?

2. **Configuration:**

   - Single config or multiple configs?
   - Config names (if multiple)?

3. **Data Structure:**

   - What fields exist in the data?
   - Field types (text, numbers, images, nested)?
   - Sample data (if available)?

4. **Metadata (optional):**
   - BibTeX citation?

### Step 4: Implement Dataset Logic

Update generated `datasets/{dataset_name}/{dataset_name}.py`:

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

**Single config:**

```python
class {DatasetName}Dataset(ds.GeneratorBasedBuilder):
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

class {DatasetName}Type(StrEnum):
    config1 = auto()
    config2 = auto()

@dataclass
class {DatasetName}Config(ds.BuilderConfig):
    name: {DatasetName}Type

BUILDER_CONFIG_CLASS = {DatasetName}Config
BUILDER_CONFIGS = [
    {DatasetName}Config(name={DatasetName}Type.config1, version=VERSION),
]
```

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

Update `datasets/{dataset_name}/tests/{dataset_name}_test.py`:

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

### Step 6: Update README.md

Add to repository root `README.md`:

```markdown
### {DatasetName}

[Brief description]

**Data Source:** [{homepage}]({homepage})

**Test:**
\`\`\`shell
uv run pytest -vsx datasets/{DatasetName}/tests
\`\`\`

**Load:**
\`\`\`python
import datasets as ds
dataset = ds.load_dataset("creative-graphic-design/{DatasetName}", trust_remote_code=True)
\`\`\`
```

### Step 7: Test and Debug

```bash
uv run pytest -vsx datasets/{dataset_name}/tests
```

If errors, fix and re-run.

Verify loading:

```python
import datasets as ds
ds = ds.load_dataset("datasets/{dataset_name}/{dataset_name}.py", trust_remote_code=True)
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
