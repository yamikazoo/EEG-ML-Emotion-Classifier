
import numpy as np
import pandas as pd

def create_asymmetry_features(df, symmetric_pairs):
    """Calculates Differential Asymmetry (DA) and Rational Asymmetry (RA) features."""
    asymmetry_features = {}
    all_features = [col for col in df.columns if col.split('_')[-1].isdigit()]
    base_features = [col.rsplit('_', 1)[0] for col in all_features if col.endswith('_1')]

    for left_idx, right_idx in symmetric_pairs:
        for feature_name in base_features:
            col_L = f"{feature_name}_{left_idx}"
            col_R = f"{feature_name}_{right_idx}"
            da_col_name = f"DA_{feature_name}_{left_idx}-{right_idx}"
            asymmetry_features[da_col_name] = df[col_L] - df[col_R]
            ra_col_name = f"RA_{feature_name}_{left_idx}-{right_idx}"
            sum_cols = df[col_L] + df[col_R]
            ra_values = (df[col_L] - df[col_R]) / sum_cols
            ra_values.replace([np.inf, -np.inf], 0, inplace=True) 
            asymmetry_features[ra_col_name] = ra_values

    asymmetry_df = pd.DataFrame(asymmetry_features, index=df.index)
    
    return pd.concat([df[all_features], asymmetry_df], axis=1)