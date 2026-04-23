import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from tqdm import tqdm


# Function to create feature vector for each sample
def create_features(df, all_rt_names, include_insdel):
    features = []
    labels = []
    caps = []
    experiments = []
    if include_insdel:
        nucls = ['A%_INSDEL', 'C%_INSDEL', 'G%_INSDEL', 'T%_INSDEL', 'INS%_INSDEL', 'DEL%_INSDEL']
    else:
        nucls = ['A%', 'C%', 'G%', 'T%']
    
    for nucl in nucls:
        if nucl not in df.columns:
            raise ValueError(f"Column {nucl} not found in DataFrame")
    
    # Group by cap type and experiment
    for (cap_type, experiment), group in df.groupby(['cap', 'experiment']):
        # Initialize feature vector with zeros
        feature_vec = np.zeros(len(all_rt_names) * len(nucls))

        # For each RT in this group, add its ACGT percentages
        for _, row in group.iterrows():
            rt_idx = np.where(all_rt_names == row['RT'])[0][0]
            base_idx = rt_idx * len(nucls)
            feature_vec[base_idx:base_idx+len(nucls)] = row[nucls]
        
        features.append(feature_vec)
        labels.append(f"{cap_type} ({experiment})")
        caps.append(cap_type)
        experiments.append(experiment)

    return np.array(features), np.array(labels), np.array(caps), np.array(experiments)


# Predict using k-NN with cosine similarity and masked training set depending on the RTs present in the test sample
def predict(X_test_sample, X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=X_train.shape[0], metric='cosine')
    mask = X_test_sample != 0
    X_train_masked = X_train.copy()
    X_train_masked[:, ~mask] = 0  # Use only training RTs that are present in the test sample
    knn.fit(X_train_masked[:, mask], y_train)
    
    # Get distances and indices of 5 nearest neighbors
    distances, indices = knn.kneighbors(X_test_sample[mask].reshape(1, -1))
    
    # Convert distances to similarities (1 - distance)
    similarities = 1 - distances[0]
    
    # Get the corresponding labels
    neighbor_labels = y_train[indices[0]]
    
    return list(zip(neighbor_labels, similarities)), knn


def save_knn_model (model, filename='knn_model.pkl'):
    """
    Save the kNN model to a file using pickle.

    Args:
        model: The trained kNN model to save
        filename: The output filename (default: 'knn_model.pkl')
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print (f"kNN model saved to {filename}")


def load_knn_model(filename='knn_model.pkl'):
    """
    Load a kNN model from a pickle file.
    
    Args:
        filename: The input filename (default: 'knn_model.pkl')
    
    Returns:
        The loaded kNN model
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"kNN model loaded from {filename}")
    return model


def predict_cap(df_train, df_test, show_true_cap=False, include_insdel=False,
                print_top_k=50, save_mode=False, model_filename='knn_model.pkl'):
    """
    Make predictions using k-NN classifier with optional model persistence.
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        show_true_cap: Whether to show true cap (default: False)
        include_insdel: Whether to include insertions/deletions (default: False)
        print_top_k: Number of top predictions to print (default: 50)
        save_model: Whether to save the trained kNN model (default: False)
        model_filename: Filename for saving the model (default: 'knn_model.pkl')
    
    Returns:
        DataFrame with prediction results
    """
    # Get unique RT names from all datasets
    all_rt_names = df_train['RT'].unique()

    # Create features and labels for each dataset
    X_train, y_train, caps_train, experiments_train = create_features(df_train, all_rt_names, include_insdel)
    X_test, y_test, caps_test, experiments_test = create_features(df_test, all_rt_names, include_insdel)

    # Make predictions and collect kNN models
    test_predictions = []
    knn_models=[]
    for x in X_test:
        preds, knn = predict(x,X_train, y_train)
        test_predictions.append(preds)
        knn_models.append(knn)
    
    # Save the first model if requested
    if save_model and knn_models:
        save_knn_model(knn_models[0], model_filename)
    
    # Create a list to store results for DataFrame
    results = []
    
    # Print predictions with similarities and build DataFrame
    for i, (true, preds) in enumerate(zip(y_test, test_predictions)):
        used_rts = df_test[df_test['experiment'] == experiments_test[i]]['RT']
        mean_reads = df_test[df_test['experiment'] == experiments_test[i]]['num_reads_ACGT'].mean()
        
        print(f"Experiment: {experiments_test[i]}")
        if show_true_cap:
            print(f"True cap: {caps_test[i]}")
        print(f"{len(used_rts)} RTs considered for prediction({used_rts.tolist()}) with mean "
              f"number of reads {mean_reads}")
        
        # Store results for each prediction
        result_dict = {
            'experiment': experiments_test[i],
            'true_cap': caps_test[i] if show_true_cap else None,
            'num_rts': len(used_rts),
            'used_rts': used_rts.tolist(),
            'mean_reads': mean_reads
        }
        
        for k, (pred, sim) in enumerate(preds[:print_top_k]):
            print(f"Top-{k+1} prediction: {pred:8} with similarity {sim:.3f}")
            result_dict[f'prediction_{k+1}'] = pred
            result_dict[f'similarity_{k+1}'] = sim
            
        results.append(result_dict)
        print("\n")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    return results_df


def predict_cap_with_saved_model(df_test, X_train, y_train, all_rt_names, 
                                 model_filename='knn_model.pkl', show_true_cap=False, 
                                 include_insdel=False, print_top_k=50):
    """
    Make predictions using a pre-trained kNN model loaded from pickle file.
    
    Args:
        df_test: Test dataframe
        X_train: Training features
        y_train: Training labels
        all_rt_names: Array of RT names
        model_filename: Path to saved kNN model (default: 'knn_model.pkl')
        show_true_cap: Whether to show true cap
        include_insdel: Whether to include insertions/deletions
        print_top_k: Number of top predictions to print
    
    Returns:
        DataFrame with prediction results
    """
    # Load the saved model
    knn = load_knn_model(model_filename)
    
    # Create test features
    X_test, y_test, caps_test, experiments_test = create_features(df_test, all_rt_names, include_insdel)
    
    # Make predictions using loaded model
    results = []
    for i, x in enumerate(X_test):
        mask = x != 0
        distances, indices = knn.kneighbors(x[mask].reshape(1, -1))
        similarities = 1 - distances[0]
        neighbor_labels = y_train[indices[0]]
        preds = list(zip(neighbor_labels, similarities))
        
        used_rts = df_test[df_test['experiment'] == experiments_test[i]]['RT']
        mean_reads = df_test[df_test['experiment'] == experiments_test[i]]['num_reads_ACGT'].mean()
        
        print(f"Experiment: {experiments_test[i]}")
        if show_true_cap:
            print(f"True cap: {caps_test[i]}")
        print(f"{len(used_rts)} RTs considered for prediction({used_rts.tolist()}) with mean "
              f"number of reads {mean_reads}")
        
        result_dict = {
            'experiment': experiments_test[i],
            'true_cap': caps_test[i] if show_true_cap else None,
            'num_rts': len(used_rts),
            'used_rts': used_rts.tolist(),
            'mean_reads': mean_reads
        }
        
        for k, (pred, sim) in enumerate(preds[:print_top_k]):
            print(f"Top-{k+1} prediction: {pred:8} with similarity {sim:.3f}")
            result_dict[f'prediction_{k+1}'] = pred
            result_dict[f'similarity_{k+1}'] = sim
            
        results.append(result_dict)
        print("\n")
    
    results_df = pd.DataFrame(results)
    return results_df

                                     
def mix_fingerprints(
        df: pd.DataFrame, 
        cap_frac_dict: dict,
        include_insdel: bool = False,
    ):
    """
    Create a synthetic fingerprint mixture from pure cap fingerprints.
    
    Args:
        df: DataFrame with pure cap fingerprints
        cap_frac_dict: Dictionary mapping cap names to their fractions in mixture
        include_insdel: Whether to include insertion/deletion columns
    
    Returns:
        DataFrame with mixed fingerprint
    """
    if include_insdel:
        nuc_cols = ['num_A', 'num_C', 'num_G', 'num_T', 'num_INS', 'num_DEL']
        pct_cols = ['A%_INSDEL', 'C%_INSDEL', 'G%_INSDEL', 'T%_INSDEL', 'INS%_INSDEL', 'DEL%_INSDEL']
    else:
        nuc_cols = ['num_A', 'num_C', 'num_G', 'num_T']
        pct_cols = ['A%', 'C%', 'G%', 'T%']

    if len(df) != 5 * len(cap_frac_dict.keys()):
        raise ValueError("df must have 5 * len(cap_frac_dict.keys()) rows")
    if not set(cap_frac_dict.keys()).issubset(set(df['cap'].unique())):
        raise ValueError("frac_dict keys must be a subset of cap types in df")
    if not np.isclose(sum(cap_frac_dict.values()), 1):
        raise ValueError("fractions must sum to 1")
    
    res_df = []
    for rt in df['RT'].unique():
        rt_df = df[df['RT'] == rt]

        # Convert df to dict
        rt_dict = rt_df.set_index('cap')[nuc_cols].to_dict('index')
        
        # Compute weighted averages
        weighted_counts = {nuc: sum(rt_dict[cap][nuc] * cap_frac_dict[cap] 
                                  for cap in cap_frac_dict)
                         for nuc in nuc_cols}
        weighted_counts['RT'] = rt
        res_df.append(weighted_counts)

    df_result = pd.DataFrame(res_df)

    # Normalize counts to percentages
    total = df_result[nuc_cols].sum(axis=1)
    df_result[pct_cols] = df_result[nuc_cols].div(total, axis=0)
    df_result = df_result.drop(columns=nuc_cols)

    # Save cap_frac_dict info to the result
    df_result['cap'] = ' + '.join(f'{cap} ({frac:.1%})' for cap, frac in cap_frac_dict.items())
    df_result['experiment'] = ' + '.join(df['experiment'].unique())
    return df_result


def generate_fingerprint_mixes(
    df_train, 
    step_size=0.02, 
    include_insdel=False,
    cap_order=None,
    replicates_per_cap=5,
    exclude_zero_caps=False,
):
    """
    Generate synthetic fingerprint mixtures for any number of caps.
    
    This function creates training data by mixing pure cap fingerprints in different
    proportions. The exclude_zero_caps parameter prevents the "catch-all" bias where
    one cap (typically the last) acts as a catch-all for unmodeled fractions.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training fingerprints containing pure-cap rows.
    step_size : float
        Fraction grid step size, e.g. 0.02 or 0.05.
    include_insdel : bool
        Passed through to mix_fingerprints().
    cap_order : list[str] or None
        Ordered list of cap names to include in the mixture model.
        If None, uses sorted unique caps from df_train.
    replicates_per_cap : int
        Number of pure replicates expected per cap.
    exclude_zero_per_cap : bool
        If True, excludes combinations where any cap has fraction=0.
        This ṕrevents the last cap from acting as a "catch-all" for unmodeled
        fractions. Recommended:True for better model accuracy.
        (default: False for backward compatibility, but True is recommended)
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic fingerprint mixtures for training.
    
    Notes
    -----
    When exclude_zero_caps=True, the training set only includes true mixtures
    where all caps are represented in some proportion. This prevents systematic
    bias where one cap receives inflated predictions.
    """
    
    if cap_order is None:
        cap_order = sorted(df_train['cap'].unique())

    caps_in_df = set(df_train['cap'].unique())
    missing = set(cap_order) - caps_in_df
    if missing:
        raise ValueError(f"df_train is missing caps needed for mixing: {missing}")

    # Keep only selected caps
    df_train = df_train[df_train['cap'].isin(cap_order)].copy()

    # Check replicate count
    cap_counts = df_train['cap'].value_counts()
    bad_caps = [cap for cap in cap_order if cap_counts.get(cap, 0) != replicates_per_cap]
    if bad_caps:
        raise ValueError(
            f"Each cap must have exactly {replicates_per_cap} rows. "
            f"Problem caps: {[(cap, cap_counts.get(cap, 0)) for cap in bad_caps]}"
        )

    # Match the assumption used inside mix_fingerprints()
    expected_total_rows = replicates_per_cap * len(cap_order)
    if len(df_train) != expected_total_rows:
        raise ValueError(
            f"df_train must contain exactly {expected_total_rows} rows after filtering "
            f"({replicates_per_cap} per cap for {len(cap_order)} caps). "
            f"Found {len(df_train)} rows."
        )

    n_steps = int(round(1.0 / step_size))
    fractions = [i * step_size for i in range(n_steps + 1)]
    
    def generate_combinations_recursive(remaining_caps, remaining_total, current_combo=None):
        """
        Recursively assign fractions so all cap fractions sum to 1.
        The last cap gets the remaining total.
        """
        if current_combo is None:
            current_combo = {}

        if len(remaining_caps) == 1:
            last_value = remaining_total
            last_steps = last_value / step_size
            if abs(round(last_steps) - last_steps) < 1e-9:
                combo = {**current_combo, remaining_caps[0]: round(last_value, 10)}

                #NEW: Filter out combinations with zero caps if requested
                if exclude_zero_caps:
                    if any(v == 0.0 for v in combo.values()):
                        return # Skip this combination

                yield combo
            return

        first_cap = remaining_caps[0]
        rest_caps = remaining_caps[1:]

        for frac in fractions:
            if frac > remaining_total:
                break

            # Skip zero fractions if exclude_zero_caps is True
            if exclude_zero_caps and frac == 0.0 and len(remaining_caps) > 1:
                continue

            for combo in generate_combinations_recursive(
                rest_caps,
                remaining_total - frac,
                {**current_combo, first_cap: round(frac, 10)}
            ):
                yield combo

    total = sum(1 for _ in generate_combinations_recursive(cap_order, 1.0))

    print(f"\n{'='*70}")
    print(f"Generating fingerprint mixtures")
    print(f"{'='*70}")
    print(f"Caps in mixture: {cap_order}")
    print(f"Step size: {step_size}")
    print(f"Total combinations to generate: {total}")
    if exclude_zero_caps:
        print(f"⚠️  IMPORTANT: exclude_zero_caps=True")
        print(f"   Only generating TRUE MIXTURES (all caps present)")
        print(f"   This prevents catch-all bias (recommended!)")
    else:
        print(f"   Note: Include all combinations (including single-cap)")
        print(f"   Set exclude_zero_caps=True to prevent catch-all bias")
    print(f"{'='*70}\n")

    df_train_mixes = pd.concat(
        [
            mix_fingerprints(
                df_train,
                cap_frac_dict=cap_frac_dict,
                include_insdel=include_insdel
            )
            for cap_frac_dict in tqdm(
                generate_combinations_recursive(cap_order, 1.0),
                total=total,
                desc='Generating combinations'
            )
        ],
        ignore_index=True
    )

    return df_train_mixes


def main():
    raise NotImplementedError("Not implemented yet.")
    # df_train = pd.read_csv('data/train.csv')
    # df_test = pd.read_csv('data/test.csv')
    # predict_cap(df_train, df_test)


if __name__ == "__main__":
    main()
