# Quick Reference: RNACapPredictor Model Registry

## TL;DR - Start Here

```python
from rnacappredictor.model_registry import recommend_variant, build_model_paths
import pandas as pd

# 1. Get recommended model for your sample type
version, variant = recommend_variant(sample_type="blind", has_insdel=False)

# 2. Load pre-computed training library
paths = build_model_paths(version, variant, models_dir="../models")
df_train_mixes = pd.read_parquet(paths["mixes"])

# 3. Prepare your test data and predict
from rnacappredictor.predict_cap import predict_cap
results = predict_cap(df_train_mixes, df_test, include_insdel=False)

# 4. Save with reproducibility metadata
results.to_csv(f"results/{sample_name}_{variant}_results.csv")
```

---

## Key Concepts

### What's the Model Registry?

A **centralized system** for managing all RNACapPredictor model variants, providing:
- Single source of truth for all configurations
- Automatic model variant recommendation
- Easy model switching
- Full reproducibility metadata

### The 4 Model Variants

Choose based on **two questions:**

| Sample Type | Has INSDEL? | Use This Variant |
|------------|-------------|-----------------|
| Blind/Unknown | ❌ No | `exclude_zero_FALSE_no_insdel` ⭐ |
| Blind/Unknown | ✅ Yes | `exclude_zero_FALSE_with_insdel` |
| Controlled/Known | ❌ No | `exclude_zero_TRUE_no_insdel` |
| Controlled/Known | ✅ Yes | `exclude_zero_TRUE_with_insdel` |

**⭐ Most Common:** `exclude_zero_FALSE_no_insdel` for blind biological samples

---

## Core Functions

### `recommend_variant(sample_type, has_insdel)`
Returns `(version, variant)` tuple automatically.

```python
# For blind samples
version, variant = recommend_variant("blind", False)
# Returns: ("1.0", "exclude_zero_FALSE_no_insdel")

# For controlled samples  
version, variant = recommend_variant("controlled", False)
# Returns: ("1.0", "exclude_zero_TRUE_no_insdel")
```

### `get_model_config(version, variant)`
Get full configuration dictionary for a variant.

```python
config = get_model_config("1.0", "exclude_zero_FALSE_no_insdel")
# Returns dict with:
#   - mixes_file, features_file
#   - exclude_zero_caps, include_insdel
#   - use_case, description
#   - generation_stats, pros, cons
```

### `build_model_paths(version, variant, models_dir)`
Build absolute file paths for model files.

```python
paths = build_model_paths("1.0", "exclude_zero_FALSE_no_insdel")
# Returns:
# {
#   "mixes": "/path/to/df_train_mixes_exclude_zero_FALSE_step002.parquet",
#   "features": "/path/to/training_features_exclude_zero_FALSE_step002.npz"
# }
```

### `get_available_variants(version)`
List all variants for a version.

```python
variants = get_available_variants("1.0")
# Returns: ["exclude_zero_TRUE_no_insdel", "exclude_zero_TRUE_with_insdel", ...]
```

### `print_model_info(version, variant=None)`
Display human-readable model information.

```python
print_model_info("1.0", "exclude_zero_FALSE_no_insdel")
# Prints detailed info about the variant
```

---

## Key Files

| File | Purpose |
|------|---------|
| `rnacappredictor/model_registry.py` | Registry code + helper functions |
| `models/metadata.json` | Model metadata in JSON format |
| `models/MODEL_SELECTION_GUIDE.md` | When to use each variant |
| `notebooks/Final_Template_RNAcapPredictor_v2.ipynb` | Working example notebook |
| `MODELS_AND_REPRODUCIBILITY.md` | Full documentation |

---

## Decision Tree

```
What type of sample are you analyzing?
│
├─ Blind/Unknown (FM219 biological samples, etc.)
│  ├─ Have INS/DEL columns? 
│  │  ├─ YES → exclude_zero_FALSE_with_insdel
│  │  └─ NO  → exclude_zero_FALSE_no_insdel ⭐ RECOMMENDED
│  └─ Version: 1.0
│
└─ Controlled/Validation (FM179-FM181 training data, etc.)
   ├─ Have INS/DEL columns?
   │  ├─ YES → exclude_zero_TRUE_with_insdel
   │  └─ NO  → exclude_zero_TRUE_no_insdel
   └─ Version: 1.0
```

---

## Common Workflows

### Workflow 1: Single Blind Sample Prediction

```python
from rnacappredictor.model_registry import recommend_variant, build_model_paths
from rnacappredictor.predict_cap import predict_cap
import pandas as pd

# Get model
v, var = recommend_variant("blind", has_insdel=False)
paths = build_model_paths(v, var)

# Load training
df_train = pd.read_parquet(paths["mixes"])

# Load test
df_test = pd.read_csv("fingerprints.csv")
df_test["cap"] = "unknown"
df_test["experiment"] = "sample_1"

# Predict
results = predict_cap(df_train, df_test, include_insdel=False)
results.to_csv(f"results/sample_1_{var}_results.csv")
```

### Workflow 2: Batch Processing

```python
from rnacappredictor.model_registry import recommend_variant, build_model_paths
from rnacappredictor.predict_cap import predict_cap
import pandas as pd
from pathlib import Path

# Setup
v, var = recommend_variant("blind", has_insdel=False)
paths = build_model_paths(v, var)
df_train = pd.read_parquet(paths["mixes"])

# Process multiple samples
for fingerprint_file in Path("data").glob("*.csv"):
    df_test = pd.read_csv(fingerprint_file)
    df_test["cap"] = "unknown"
    df_test["experiment"] = fingerprint_file.stem
    
    results = predict_cap(df_train, df_test, include_insdel=False)
    results.to_csv(f"results/{fingerprint_file.stem}_{var}_results.csv")
```

### Workflow 3: Compare Multiple Variants

```python
from rnacappredictor.model_registry import get_available_variants, build_model_paths
from rnacappredictor.predict_cap import predict_cap
import pandas as pd

df_test = pd.read_csv("fingerprints.csv")
df_test["cap"] = "unknown"
df_test["experiment"] = "sample_1"

# Try all variants
for variant in get_available_variants("1.0"):
    paths = build_model_paths("1.0", variant)
    df_train = pd.read_parquet(paths["mixes"])
    
    results = predict_cap(df_train, df_test, include_insdel=(
        "INSDEL" in variant  # Match include_insdel to variant
    ))
    results.to_csv(f"results/sample_1_{variant}_comparison.csv")
```

---

## What Does Each Variant Mean?

### `exclude_zero_TRUE`
- **Meaning:** Only generates mixtures where ALL caps are present (no zeros)
- **Prevents:** "Catch-all bias" (last cap getting residual fractions)
- **Best For:** Controlled experiments where all caps are expected
- **Limitation:** Cannot predict single-cap samples

### `exclude_zero_FALSE` 
- **Meaning:** Generates all combinations including single-cap mixtures
- **Allows:** Any cap combination, including incomplete mixtures
- **Best For:** Blind samples, exploratory analysis
- **Caveat:** May have catch-all bias

### `no_insdel`
- **Uses:** Only A%, C%, G%, T% columns
- **Works:** With standard fingerprint data
- **Default:** Most common choice

### `with_insdel`
- **Uses:** A%, C%, G%, T%, INS%, DEL% columns
- **Requires:** Input fingerprints with insertion/deletion data
- **Benefit:** Better discrimination using INS/DEL signal

---

## Troubleshooting

**Q: "Variant not found" error**  
A: Check available variants with `get_available_variants("1.0")`

**Q: "Column not found" error**  
A: Ensure `include_insdel` parameter matches the variant
- `no_insdel` variant → `include_insdel=False`
- `with_insdel` variant → `include_insdel=True`

**Q: Results don't match old notebook**  
A: Verify you're using the same variant. Check saved metadata files.

**Q: Which model for my sample?**  
A: If unsure, start with `recommend_variant("blind", False)`

---

## Model Metadata

Every analysis saves metadata for reproducibility:

```
results/
├── sample_1_exclude_zero_FALSE_no_insdel_results.csv
└── sample_1_exclude_zero_FALSE_no_insdel_metadata.txt  ← Read this!
```

**Metadata file contains:**
- Sample name
- Model version & variant
- Model parameters (exclude_zero_caps, include_insdel)
- Use case
- Generation statistics

---

## Import Patterns

**Get registry functions:**
```python
from rnacappredictor.model_registry import (
    recommend_variant,
    get_model_config,
    build_model_paths,
    get_available_variants,
    print_model_info,
    get_version_metadata
)
```

**Get prediction function:**
```python
from rnacappredictor.predict_cap import predict_cap
```

---

## Pro Tips

✅ **Always** check `MODEL_SELECTION_GUIDE.md` before choosing a variant

✅ **Save metadata** with results for reproducibility

✅ **Use `recommend_variant()`** for automatic selection

✅ **Pin model version** if doing comparative studies

✅ **Document which variant** you used in your methods

❌ **Don't** mix results from different `exclude_zero_caps` settings

❌ **Don't** assume variant, always specify explicitly

❌ **Don't** forget `include_insdel` must match variant

---

## See Also

- **Full Docs:** `MODELS_AND_REPRODUCIBILITY.md`
- **Selection Guide:** `models/MODEL_SELECTION_GUIDE.md`
- **Example Notebook:** `notebooks/Final_Template_RNAcapPredictor_v2.ipynb`
- **Registry Code:** `rnacappredictor/model_registry.py`
