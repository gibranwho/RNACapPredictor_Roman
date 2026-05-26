# Model Management & Reproducibility Guide

This document explains the improvements made to the RNACapPredictor project for better model management, reproducibility, and ease of use.

## What's New

We've implemented a comprehensive **model registry system** that provides:

1. **Centralized Model Configuration** (`model_registry.py`)
2. **Model Metadata** (`models/metadata.json`)
3. **Model Selection Guide** (`models/MODEL_SELECTION_GUIDE.md`)
4. **Updated Template Notebook** (`notebooks/Final_Template_RNAcapPredictor_v2.ipynb`)

---

## Overview

### Problem Solved

Previously, the workflow required:
- ❌ Manually specifying model file paths
- ❌ Hardcoding model variant selection logic
- ❌ Unclear when to use which model variant
- ❌ No version tracking
- ❌ Difficult to reproduce results

Now, you have:
- ✅ Single source of truth for all model configs
- ✅ Automatic model variant recommendation
- ✅ Clear documentation for each variant
- ✅ Full reproducibility metadata
- ✅ Easy model switching

---

## Components

### 1. Model Registry (`rnacappredictor/model_registry.py`)

**Purpose:** Centralized Python module storing all model metadata and providing convenient functions.

**Key Features:**
- Single `MODEL_REGISTRY` dictionary with all variants
- Helper functions for model selection and configuration
- Automatic path building
- Recommendation engine based on sample type

**Usage:**

```python
from rnacappredictor.model_registry import (
    recommend_variant,
    get_model_config,
    build_model_paths,
    print_model_info
)

# Get recommendation
version, variant = recommend_variant(sample_type="blind", has_insdel=False)
# Returns: ("1.0", "exclude_zero_FALSE_no_insdel")

# Get full configuration
config = get_model_config("1.0", "exclude_zero_FALSE_no_insdel")
# Returns: dict with all metadata

# Build file paths
paths = build_model_paths("1.0", "exclude_zero_FALSE_no_insdel")
# Returns: {"mixes": "path/to/mixes.parquet", "features": "path/to/features.npz"}

# Display information
print_model_info("1.0", "exclude_zero_FALSE_no_insdel")
```

### 2. Model Metadata (`models/metadata.json`)

**Purpose:** JSON-based model metadata file for easy parsing and integration.

**Contents:**
- Training data source and date
- Supported caps and RTs
- All 4 model variants with their parameters
- Generation statistics
- Use cases for each variant

**Example:**
```json
{
  "version": "1.0",
  "training_data": "FM179-FM181_fingerprints.csv",
  "training_date": "2025-05-26",
  "variants": {
    "exclude_zero_FALSE_no_insdel": {
      "mixes_file": "df_train_mixes_exclude_zero_FALSE_step002.parquet",
      "features_file": "training_features_exclude_zero_FALSE_step002.npz",
      "use_case": "BLIND SAMPLES / FIRST-PASS SCREENING",
      ...
    }
  }
}
```

### 3. Model Selection Guide (`models/MODEL_SELECTION_GUIDE.md`)

**Purpose:** Comprehensive user guide for selecting the right model variant.

**Contains:**
- Quick decision tree
- Detailed descriptions of all 4 variants
- Pros/cons for each variant
- Usage examples
- FAQ

**Key Decision:**
```
Blind/Unknown Sample?
├─ YES, standard analysis → exclude_zero_FALSE_no_insdel ⭐ RECOMMENDED
├─ YES, with INSDEL → exclude_zero_FALSE_with_insdel
├─ NO (Controlled), standard → exclude_zero_TRUE_no_insdel
└─ NO (Controlled), with INSDEL → exclude_zero_TRUE_with_insdel
```

### 4. Updated Template Notebook (`notebooks/Final_Template_RNAcapPredictor_v2.ipynb`)

**Purpose:** Production-ready prediction notebook using the model registry.

**Key Improvements:**
- Uses `model_registry.py` for automatic model selection
- Clear configuration section at the top
- Built-in validation and diagnostics
- Reproducibility metadata saved with results
- Better error handling and progress reporting

**Workflow:**
1. Configure sample info and fingerprint paths
2. Let the system recommend a model (or choose manually)
3. Load pre-generated training mixtures (no generation!)
4. Prepare and validate data
5. Run predictions
6. Save results with reproducibility metadata

---

## Quick Start

### For Blind/Unknown Samples (Recommended Path)

```python
# In your notebook:
from rnacappredictor.model_registry import recommend_variant, build_model_paths
import pandas as pd

# Step 1: Get recommendation
version, variant = recommend_variant(sample_type="blind", has_insdel=False)
# → ("1.0", "exclude_zero_FALSE_no_insdel")

# Step 2: Get model paths
paths = build_model_paths(version, variant, models_dir="../models")

# Step 3: Load pre-computed training library
df_train_mixes = pd.read_parquet(paths["mixes"])

# Step 4: Prepare your data and predict
from rnacappredictor.predict_cap import predict_cap
results = predict_cap(df_train_mixes, df_test, include_insdel=False)
```

### For Controlled/Validation Samples

```python
from rnacappredictor.model_registry import recommend_variant, build_model_paths

# Get recommendation for controlled samples
version, variant = recommend_variant(sample_type="controlled", has_insdel=False)
# → ("1.0", "exclude_zero_TRUE_no_insdel")

# Rest is identical...
```

---

## Understanding Model Variants

### The 4 Available Models

| Variant | Blind? | With INSDEL? | Best For | Catch-All Bias? |
|---------|--------|-------------|----------|-----------------|
| `exclude_zero_TRUE_no_insdel` | ❌ No | ❌ No | Validation, Controlled | ✓ Prevented |
| `exclude_zero_TRUE_with_insdel` | ❌ No | ✅ Yes | Validation with INSDEL | ✓ Prevented |
| `exclude_zero_FALSE_no_insdel` | ✅ Yes | ❌ No | **Blind Samples (DEFAULT)** | ⚠️ Possible |
| `exclude_zero_FALSE_with_insdel` | ✅ Yes | ✅ Yes | Blind + INSDEL | ⚠️ Possible |

### What is "Catch-All Bias"?

When `exclude_zero_caps=False`, the last cap in mixture combinations can receive all residual fractions, making it appear artificially inflated.

**Solution:** Use `exclude_zero_TRUE` variants for controlled samples where you know all caps should be present.

---

## File Structure

```
RNACapPredictor_Roman/
├── models/
│   ├── metadata.json                              # Model metadata
│   ├── MODEL_SELECTION_GUIDE.md                   # Selection guide
│   ├── df_train_mixes_exclude_zero_TRUE_step002.parquet
│   ├── training_features_exclude_zero_TRUE_step002.npz
│   ├── df_train_mixes_exclude_zero_FALSE_step002.parquet
│   ├── training_features_exclude_zero_FALSE_step002.npz
│   ├── df_train_mixes_exclude_zero_TRUE_INSDEL_step002.parquet
│   ├── training_features_exclude_zero_TRUE_INSDEL_step002.npz
│   ├── df_train_mixes_exclude_zero_FALSE_INSDEL_step002.parquet
│   └── training_features_exclude_zero_FALSE_INSDEL_step002.npz
│
├── rnacappredictor/
│   ├── model_registry.py                         # ✨ NEW: Model registry
│   ├── predict_cap.py                            # Core prediction functions
│   └── ...
│
├── notebooks/
│   ├── Final_Template_RNAcapPredictor.ipynb      # Original template
│   ├── Final_Template_RNAcapPredictor_v2.ipynb   # ✨ NEW: Updated template
│   └── Model_files.ipynb                         # Model generation (reference)
│
└── ...
```

---

## Usage Examples

### Example 1: Batch Processing Multiple Samples

```python
from rnacappredictor.model_registry import build_model_paths, recommend_variant
import pandas as pd

# Get recommended model
version, variant = recommend_variant("blind", has_insdel=False)
paths = build_model_paths(version, variant)

# Load once
df_train_mixes = pd.read_parquet(paths["mixes"])

# Process multiple samples
for sample_id in ["FM219", "FM220", "FM221"]:
    df_test = pd.read_csv(f"data/{sample_id}_fingerprints.csv")
    results = predict_cap(df_train_mixes, df_test)
    results.to_csv(f"results/{sample_id}_predictions.csv")
```

### Example 2: Compare Model Variants

```python
from rnacappredictor.model_registry import get_available_variants, build_model_paths, get_model_config

version = "1.0"

# Try all variants
for variant in get_available_variants(version):
    config = get_model_config(version, variant)
    paths = build_model_paths(version, variant)
    
    print(f"\n{variant}")
    print(f"  Use Case: {config['use_case']}")
    print(f"  Exclude Zero: {config['exclude_zero_caps']}")
    print(f"  Include INSDEL: {config['include_insdel']}")
```

### Example 3: Display Model Information

```python
from rnacappredictor.model_registry import print_model_info

# Show all variants for a version
print_model_info("1.0")

# Show details for specific variant
print_model_info("1.0", "exclude_zero_FALSE_no_insdel")
```

---

## Reproducibility

Every prediction now includes metadata for reproducibility:

```
sample_name_variant_results.csv
sample_name_variant_metadata.txt
```

**Example metadata file:**
```
Sample Batch: FM219
Model Version: 1.0
Model Variant: exclude_zero_FALSE_no_insdel
Exclude Zero Caps: False
Include INSDEL: False
Use Case: BLIND SAMPLES / FIRST-PASS SCREENING

Generation Stats:
  total_combinations: 3329692
  generation_time_seconds: 11522
```

This ensures you can always:
- ✅ Identify which model was used
- ✅ Understand model parameters
- ✅ Reproduce the exact same analysis
- ✅ Track model versions across time

---

## Adding New Model Versions

To add a new model version (e.g., after retraining):

1. **Generate the models** using `Model_files.ipynb`
2. **Save files** with consistent naming in `models/`
3. **Update `model_registry.py`:**
   ```python
   MODEL_REGISTRY = {
       "1.0": { ... },  # existing
       "2.0": {         # new
           "metadata": { ... },
           "variants": { ... }
       }
   }
   ```
4. **Update `models/metadata.json`** with new version metadata

---

## Best Practices

### ✅ Do's

- ✅ Use `recommend_variant()` for automatic selection
- ✅ Save metadata alongside results for reproducibility
- ✅ Check `MODEL_SELECTION_GUIDE.md` before choosing variant
- ✅ Use consistent model versions for comparative studies
- ✅ Document which variant you used in your methods section

### ❌ Don'ts

- ❌ Don't hardcode model file paths
- ❌ Don't mix results from different `exclude_zero_caps` settings
- ❌ Don't forget to specify `include_insdel` correctly
- ❌ Don't upgrade model versions mid-analysis
- ❌ Don't assume default model without checking `MODEL_SELECTION_GUIDE.md`

---

## FAQ

**Q: Which model should I use?**  
A: Start with `recommend_variant("blind", False)` which returns `"exclude_zero_FALSE_no_insdel"` for unknown samples.

**Q: Do I need to regenerate models?**  
A: No! All models are pre-computed. Just load from parquet/npz files using the registry.

**Q: How do I handle new RT types?**  
A: You need to retrain (regenerate models) using `Model_files.ipynb`. Update `model_registry.py` with new version.

**Q: What if I want to use both INSDEL and exclude_zero_TRUE?**  
A: Use `exclude_zero_TRUE_with_insdel` variant.

**Q: Can I create custom variants?**  
A: Yes, by regenerating models with different parameters in `Model_files.ipynb` and adding them to the registry.

---

## Migration Guide

### From Old Notebook to New Notebook

**Old way:**
```python
MIXES_PATH = "../models/df_train_mixes_exclude_zero_FALSE_step002.parquet"
FEATURES_PATH = "../models/training_features_exclude_zero_FALSE_step002.npz"
df_train_mixes = pd.read_parquet(MIXES_PATH)
```

**New way:**
```python
from rnacappredictor.model_registry import build_model_paths
paths = build_model_paths("1.0", "exclude_zero_FALSE_no_insdel")
df_train_mixes = pd.read_parquet(paths["mixes"])
```

**Benefits:**
- Centralized configuration
- Automatic path management
- Easy variant switching
- Version tracking
- Better error handling

---

## Summary

The new model management system provides:

| Feature | Before | After |
|---------|--------|-------|
| Model selection | Manual hardcoding | Automatic recommendation |
| Path management | Error-prone strings | Centralized paths |
| Version tracking | None | Full metadata |
| Reproducibility | Limited | Complete |
| Model info | Documentation | Auto-generated |
| Variant switching | Time-consuming | One-line change |

**Next Steps:**
1. Review `models/MODEL_SELECTION_GUIDE.md`
2. Try the new notebook: `notebooks/Final_Template_RNAcapPredictor_v2.ipynb`
3. Update your workflows to use `model_registry.py`
4. Save metadata with results for reproducibility

---

## Questions or Issues?

Refer to:
- `models/MODEL_SELECTION_GUIDE.md` - How to pick the right model
- `rnacappredictor/model_registry.py` - Technical documentation
- `notebooks/Final_Template_RNAcapPredictor_v2.ipynb` - Working example
