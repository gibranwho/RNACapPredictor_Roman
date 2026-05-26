"""
Model Registry for RNACapPredictor

This module provides a centralized registry of all trained model variants,
including metadata about their training parameters, capabilities, and use cases.

Usage:
    from rnacappredictor.model_registry import MODEL_REGISTRY, get_model_config
    
    # Get a specific model variant
    config = get_model_config(version="1.0", variant="exclude_zero_FALSE_no_insdel")
    
    # List all available variants for a version
    variants = get_available_variants(version="1.0")
"""

from typing import Dict, List, Optional
import os
import json

# ============================================================================
# MODEL REGISTRY DEFINITION
# ============================================================================
# This is the single source of truth for all model variants and their configs

MODEL_REGISTRY = {
    "1.0": {
        "metadata": {
            "training_data": "FM179-FM181_fingerprints.csv",
            "training_date": "2025-05-26",
            "caps": [
                "Ap₄A-U1",
                "NAD-U1",
                "TMG-U1",
                "m⁷Gp₃A-U1",
                "ppp-U1",
                "y-meGTP-U1"
            ],
            "rts": ["INDURO", "ProtoScript", "Marathon", "GoScript", "EpiScript"],
            "step_size": 0.02,
            "replicates_per_cap": 5,
            "description": "Initial model trained on FM179-FM181 in vitro samples"
        },
        "variants": {
            "exclude_zero_TRUE_no_insdel": {
                "mixes_file": "df_train_mixes_exclude_zero_TRUE_step002.parquet",
                "features_file": "training_features_exclude_zero_TRUE_step002.npz",
                "exclude_zero_caps": True,
                "include_insdel": False,
                "use_case": "VALIDATION / CONTROLLED EXPERIMENTS",
                "description": (
                    "For known/controlled samples where you expect all caps to be present. "
                    "Prevents 'catch-all' bias. Recommended for internal validation."
                ),
                "pros": [
                    "Prevents catch-all bias (last cap acting as catch-all)",
                    "Better accuracy when all caps are expected",
                    "Smaller model size"
                ],
                "cons": [
                    "Cannot predict single-cap samples",
                    "May miss incomplete mixtures",
                    "Not suitable for blind samples"
                ],
                "generation_stats": {
                    "total_combinations": 1906884,
                    "generation_time": "1:49:10"
                }
            },
            "exclude_zero_TRUE_with_insdel": {
                "mixes_file": "df_train_mixes_exclude_zero_TRUE_INSDEL_step002.parquet",
                "features_file": "training_features_exclude_zero_TRUE_INSDEL_step002.npz",
                "exclude_zero_caps": True,
                "include_insdel": True,
                "use_case": "VALIDATION / CONTROLLED EXPERIMENTS (with INS/DEL patterns)",
                "description": (
                    "For known/controlled samples where insertion/deletion patterns matter. "
                    "True mixtures only (no single-cap)."
                ),
                "pros": [
                    "Uses insertion/deletion signal for better discrimination",
                    "Prevents catch-all bias",
                    "Better for complex fingerprint analysis"
                ],
                "cons": [
                    "Requires INS/DEL columns in input fingerprints",
                    "Larger feature space",
                    "Cannot predict single-cap samples"
                ],
                "generation_stats": {
                    "total_combinations": 1906884,
                    "generation_time": "~1:49:10"
                }
            },
            "exclude_zero_FALSE_no_insdel": {
                "mixes_file": "df_train_mixes_exclude_zero_FALSE_step002.parquet",
                "features_file": "training_features_exclude_zero_FALSE_step002.npz",
                "exclude_zero_caps": False,
                "include_insdel": False,
                "use_case": "BLIND SAMPLES / FIRST-PASS SCREENING",
                "description": (
                    "For blind/unknown samples. Explores all cap combinations including single-cap. "
                    "Recommended starting point for biological samples."
                ),
                "pros": [
                    "Can predict single-cap samples",
                    "Explores all possible combinations",
                    "Good for exploratory analysis",
                    "Standard recommended variant"
                ],
                "cons": [
                    "May have catch-all bias (last cap gets residual)",
                    "Larger model",
                    "Longer generation time"
                ],
                "generation_stats": {
                    "total_combinations": 3329692,
                    "generation_time": "3:12:02"
                }
            },
            "exclude_zero_FALSE_with_insdel": {
                "mixes_file": "df_train_mixes_exclude_zero_FALSE_INSDEL_step002.parquet",
                "features_file": "training_features_exclude_zero_FALSE_INSDEL_step002.npz",
                "exclude_zero_caps": False,
                "include_insdel": True,
                "use_case": "BLIND SAMPLES / EXPLORATORY (with INS/DEL patterns)",
                "description": (
                    "For blind/unknown samples with insertion/deletion signal. "
                    "Most comprehensive model including single-cap and INS/DEL patterns."
                ),
                "pros": [
                    "Can predict any cap combination including single-cap",
                    "Uses insertion/deletion signal",
                    "Most flexible variant",
                    "Best for comprehensive analysis"
                ],
                "cons": [
                    "Requires INS/DEL columns in input",
                    "Largest model and feature space",
                    "Longest generation time"
                ],
                "generation_stats": {
                    "total_combinations": 3329692,
                    "generation_time": "~3:12:02"
                }
            }
        }
    }
}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_available_versions() -> List[str]:
    """Return list of available model versions."""
    return list(MODEL_REGISTRY.keys())


def get_available_variants(version: str) -> List[str]:
    """Return list of available variants for a specific version."""
    if version not in MODEL_REGISTRY:
        raise ValueError(f"Version {version} not found. Available: {get_available_versions()}")
    return list(MODEL_REGISTRY[version]["variants"].keys())


def get_model_config(version: str, variant: str) -> Dict:
    """
    Get the full configuration for a specific model variant.
    
    Parameters
    ----------
    version : str
        Model version (e.g., "1.0")
    variant : str
        Variant name (e.g., "exclude_zero_FALSE_no_insdel")
    
    Returns
    -------
    dict
        Configuration dictionary with model metadata, file paths, and descriptions
    
    Raises
    ------
    ValueError
        If version or variant not found
    """
    if version not in MODEL_REGISTRY:
        raise ValueError(
            f"Version '{version}' not found. "
            f"Available: {get_available_versions()}"
        )
    
    if variant not in MODEL_REGISTRY[version]["variants"]:
        available = get_available_variants(version)
        raise ValueError(
            f"Variant '{variant}' not found for version '{version}'. "
            f"Available: {available}"
        )
    
    return MODEL_REGISTRY[version]["variants"][variant]


def get_version_metadata(version: str) -> Dict:
    """
    Get metadata for a specific model version.
    
    Parameters
    ----------
    version : str
        Model version (e.g., "1.0")
    
    Returns
    -------
    dict
        Metadata dictionary with training info, caps, RTs, etc.
    """
    if version not in MODEL_REGISTRY:
        raise ValueError(
            f"Version '{version}' not found. "
            f"Available: {get_available_versions()}"
        )
    
    return MODEL_REGISTRY[version]["metadata"]


def recommend_variant(
    sample_type: str,
    has_insdel: bool = False
) -> tuple:
    """
    Recommend a model variant based on sample characteristics.
    
    Parameters
    ----------
    sample_type : str
        Type of sample: "blind", "controlled", or "validation"
    has_insdel : bool
        Whether input has insertion/deletion columns
    
    Returns
    -------
    tuple
        (version, variant) tuple recommended for the sample type
    """
    recommendations = {
        "blind": {
            False: ("1.0", "exclude_zero_FALSE_no_insdel"),
            True: ("1.0", "exclude_zero_FALSE_with_insdel")
        },
        "controlled": {
            False: ("1.0", "exclude_zero_TRUE_no_insdel"),
            True: ("1.0", "exclude_zero_TRUE_with_insdel")
        },
        "validation": {
            False: ("1.0", "exclude_zero_TRUE_no_insdel"),
            True: ("1.0", "exclude_zero_TRUE_with_insdel")
        }
    }
    
    if sample_type not in recommendations:
        raise ValueError(
            f"Unknown sample_type '{sample_type}'. "
            f"Must be one of: {list(recommendations.keys())}"
        )
    
    return recommendations[sample_type][has_insdel]


def build_model_paths(
    version: str,
    variant: str,
    models_dir: str = "../models"
) -> Dict[str, str]:
    """
    Build absolute file paths for model files.
    
    Parameters
    ----------
    version : str
        Model version
    variant : str
        Variant name
    models_dir : str
        Base directory containing model files
    
    Returns
    -------
    dict
        Dictionary with 'mixes' and 'features' paths
    """
    config = get_model_config(version, variant)
    
    return {
        "mixes": os.path.join(models_dir, config["mixes_file"]),
        "features": os.path.join(models_dir, config["features_file"])
    }


def print_model_info(version: str, variant: str = None):
    """
    Print human-readable information about a model.
    
    Parameters
    ----------
    version : str
        Model version
    variant : str, optional
        Specific variant to print. If None, prints all variants for the version.
    """
    if version not in MODEL_REGISTRY:
        print(f"❌ Version '{version}' not found")
        return
    
    metadata = get_version_metadata(version)
    
    print(f"\n{'='*80}")
    print(f"MODEL VERSION: {version}")
    print(f"{'='*80}")
    print(f"Training Data: {metadata['training_data']}")
    print(f"Training Date: {metadata['training_date']}")
    print(f"Description: {metadata['description']}")
    print(f"\nSupported Caps ({len(metadata['caps'])}):")
    for cap in metadata['caps']:
        print(f"  • {cap}")
    print(f"\nSupported RTs ({len(metadata['rts'])}):")
    for rt in metadata['rts']:
        print(f"  • {rt}")
    print(f"\nParameters: step_size={metadata['step_size']}, "
          f"replicates_per_cap={metadata['replicates_per_cap']}")
    
    if variant:
        if variant not in MODEL_REGISTRY[version]["variants"]:
            print(f"\n❌ Variant '{variant}' not found")
            return
        
        print(f"\n{'-'*80}")
        print(f"VARIANT: {variant}")
        print(f"{'-'*80}")
        
        var_config = get_model_config(version, variant)
        print(f"Use Case: {var_config['use_case']}")
        print(f"\nDescription:\n  {var_config['description']}")
        
        print(f"\nPros:")
        for pro in var_config['pros']:
            print(f"  ✓ {pro}")
        
        print(f"\nCons:")
        for con in var_config['cons']:
            print(f"  ✗ {con}")
        
        print(f"\nFiles:")
        print(f"  Mixes: {var_config['mixes_file']}")
        print(f"  Features: {var_config['features_file']}")
        
        print(f"\nGeneration Stats:")
        for key, value in var_config['generation_stats'].items():
            print(f"  • {key}: {value}")
    else:
        print(f"\n{'-'*80}")
        print(f"AVAILABLE VARIANTS ({len(MODEL_REGISTRY[version]['variants'])})")
        print(f"{'-'*80}")
        
        for var_name in get_available_variants(version):
            var_config = get_model_config(version, var_name)
            print(f"\n• {var_name}")
            print(f"  Use Case: {var_config['use_case']}")
            print(f"  Exclude Zero Caps: {var_config['exclude_zero_caps']}")
            print(f"  Include INSDEL: {var_config['include_insdel']}")


if __name__ == "__main__":
    # Example usage
    print("Available versions:", get_available_versions())
    print("Available variants for v1.0:", get_available_variants("1.0"))
    print_model_info("1.0")
