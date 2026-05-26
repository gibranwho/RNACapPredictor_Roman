# Model Selection Guide for RNACapPredictor

This guide explains which model variant to use for different scenarios.

## Quick Decision Tree

```
Is your sample type known/controlled?
├─ YES → Use "exclude_zero_TRUE" variants
│  ├─ Standard analysis → exclude_zero_TRUE_no_insdel
│  └─ With INS/DEL signal → exclude_zero_TRUE_with_insdel
└─ NO (Blind/Unknown) → Use "exclude_zero_FALSE" variants
   ├─ Standard analysis → exclude_zero_FALSE_no_insdel ⭐ RECOMMENDED START
   └─ With INS/DEL signal → exclude_zero_FALSE_with_insdel
```

---

## Detailed Model Descriptions

### 1. `exclude_zero_TRUE_no_insdel` (Validation Model)

**Use When:**
- Working with known cap compositions
- Internal validation/testing
- You expect ALL caps to be present in the sample
- You want the fastest predictions

**Pros:**
- ✓ Prevents "catch-all" bias (last cap acting as catch-all)
- ✓ More accurate predictions when all caps are expected
- ✓ Smaller model size (1.9M combinations)
- ✓ Faster generation and predictions

**Cons:**
- ✗ Cannot predict single-cap samples
- ✗ May miss incomplete mixtures
- ✗ Not suitable for blind/exploratory analysis

**Model Size:** 1,906,884 combinations

**Example Use Case:**
```python
# For FM179-FM181 training data validation
model = "exclude_zero_TRUE_no_insdel"
```

---

### 2. `exclude_zero_TRUE_with_insdel` (Validation Model with INS/DEL)

**Use When:**
- Working with known cap compositions
- Insertion/deletion patterns are informative
- You have INS/DEL data in your input fingerprints
- You expect ALL caps to be present

**Pros:**
- ✓ Uses insertion/deletion signal for better discrimination
- ✓ Prevents catch-all bias
- ✓ Better for complex fingerprint analysis

**Cons:**
- ✗ Requires INS/DEL columns in input fingerprints
- ✗ Larger feature space
- ✗ Cannot predict single-cap samples

**Model Size:** 1,906,884 combinations (same as without INSDEL)

**Example Use Case:**
```python
# For controlled validation with INSDEL data
model = "exclude_zero_TRUE_with_insdel"
```

---

### 3. `exclude_zero_FALSE_no_insdel` (Blind Sample Model - RECOMMENDED)

**Use When:**
- Analyzing blind/unknown samples
- First-pass screening of biological samples
- You want to explore all possible cap combinations
- Fastest way to get results

**Pros:**
- ✓ Can predict single-cap samples
- ✓ Explores all cap combinations (including incomplete mixtures)
- ✓ Good for exploratory analysis
- ✓ **Standard recommended variant** for most use cases
- ✓ No special requirements for input

**Cons:**
- ✗ May have catch-all bias (last cap can receive residual)
- ✗ Larger model (3.3M combinations)
- ✗ Longer generation time (~3 hours)

**Model Size:** 3,329,692 combinations

**Example Use Case:**
```python
# For FM219 biological samples - RECOMMENDED FIRST STEP
model = "exclude_zero_FALSE_no_insdel"  # <- START HERE
```

---

### 4. `exclude_zero_FALSE_with_insdel` (Comprehensive Blind Model)

**Use When:**
- Analyzing blind samples with insertion/deletion data
- Want most comprehensive analysis
- Have INS/DEL columns in input
- Need to explore all possibilities including INS/DEL signal

**Pros:**
- ✓ Can predict any cap combination
- ✓ Uses insertion/deletion signal
- ✓ Most flexible and comprehensive variant
- ✓ Best for detailed analysis

**Cons:**
- ✗ Requires INS/DEL columns in input fingerprints
- ✗ Largest model and feature space
- ✗ Longest generation time (~3 hours)

**Model Size:** 3,329,692 combinations

**Example Use Case:**
```python
# For comprehensive analysis with full fingerprint signal
model = "exclude_zero_FALSE_with_insdel"
```

---

## Usage Example in Python

### Using the Model Registry

```python
from rnacappredictor.model_registry import (
    recommend_variant,
    get_model_config,
    build_model_paths,
    print_model_info
)

# Option 1: Get recommendation based on sample type
version, variant = recommend_variant(
    sample_type="blind",  # or "controlled", "validation"
    has_insdel=False      # Do you have INSDEL columns?
)
print(f"Recommended: {version}/{variant}")

# Option 2: Get full configuration
config = get_model_config(version="1.0", variant="exclude_zero_FALSE_no_insdel")
print(f"Use case: {config['use_case']}")
print(f"Description: {config['description']}")

# Option 3: Build file paths
paths = build_model_paths(
    version="1.0",
    variant="exclude_zero_FALSE_no_insdel",
    models_dir="../models"
)
print(f"Mixes file: {paths['mixes']}")
print(f"Features file: {paths['features']}")

# Option 4: Print human-readable info
print_model_info(version="1.0", variant="exclude_zero_FALSE_no_insdel")
```

### In Your Notebook

```python
import pandas as pd
from rnacappredictor.model_registry import build_model_paths
from rnacappredictor.predict_cap import predict_cap

# For blind FM219 samples - RECOMMENDED STARTING POINT
version = "1.0"
variant = "exclude_zero_FALSE_no_insdel"

paths = build_model_paths(version, variant, models_dir="../models")

# Load training mixtures
df_train_mixes = pd.read_parquet(paths["mixes"])

# Load your test data
df_test = pd.read_csv("fingerprints.csv")

# Make predictions
results = predict_cap(
    df_train_mixes,
    df_test,
    include_insdel=False,  # Must match variant
    print_top_k=15
)
```

---

## Decision Flowchart

```
START: Analyzing new sample
│
├─ Is the sample composition known? 
│  ├─ YES
│  │  ├─ Do you have INS/DEL data?
│  │  │  ├─ YES → exclude_zero_TRUE_with_insdel
│  │  │  └─ NO  → exclude_zero_TRUE_no_insdel ✓
│  │  └─ Run validation/testing
│  │
│  └─ NO (Blind/Unknown)
│     ├─ Do you have INS/DEL data?
│     │  ├─ YES → exclude_zero_FALSE_with_insdel
│     │  └─ NO  → exclude_zero_FALSE_no_insdel ⭐ RECOMMENDED
│     └─ Run exploratory analysis
│
└─ Evaluate results
   ├─ If catch-all bias suspected → Try exclude_zero variant
   ├─ If signal weak → Try with_insdel variant
   └─ If results good → Use for further analysis
```

---

## FAQ

**Q: Which model should I use for my first analysis?**  
A: Start with `exclude_zero_FALSE_no_insdel` for blind samples or `exclude_zero_TRUE_no_insdel` for known compositions.

**Q: What is "catch-all bias"?**  
A: When you exclude zero caps, the last cap in the mixture can receive all residual fractions, making it appear inflated. The `exclude_zero_TRUE` variants prevent this.

**Q: Do I need INS/DEL data?**  
A: No, the `no_insdel` variants work with just A%, C%, G%, T%. Use `with_insdel` only if you have insertion/deletion columns and want to leverage that signal.

**Q: How long does prediction take?**  
A: Predictions are fast (~seconds per sample). Model generation takes longer: ~2 hours for `exclude_zero_TRUE`, ~3 hours for `exclude_zero_FALSE`.

**Q: Can I mix models?**  
A: No, always use consistent training/test sets. Never mix `exclude_zero=True` results with `exclude_zero=False` results.

**Q: Which model is most accurate?**  
A: For blind samples, `exclude_zero_FALSE_with_insdel` is most comprehensive. For known samples, `exclude_zero_TRUE_no_insdel` is most focused.

---

## Reproducibility

All models were generated on **2025-05-26** from training data **FM179-FM181_fingerprints.csv** with:
- Step size: 0.02
- Replicates per cap: 5
- Caps: Ap₄A-U1, NAD-U1, TMG-U1, m⁷Gp₃A-U1, ppp-U1, y-meGTP-U1
- RTs: INDURO, ProtoScript, Marathon, GoScript, EpiScript

See `models/metadata.json` for full technical details.
