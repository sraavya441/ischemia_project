import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def regenerate_dataset():
    print("Loading original ecg_dataset.csv...")
    df = pd.read_csv('ecg_dataset.csv', header=None)
    
    # The last column is the label
    label_col = df.columns[-1]
    
    print(f"Original shape: {df.shape}")
    print("Original distribution:")
    print(df[label_col].value_counts())
    
    # Check for NaN values in the target column and drop them (sometimes there's a trailing empty row)
    df = df.dropna(subset=[label_col])
    
    # Filter out invalid labels (e.g., if there's a 300.0 label from an artifact)
    df = df[df[label_col].isin([0.0, 1.0, 2.0])]
    
    # Split into features (X) and labels (y)
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    
    print("\nApplying SMOTE (Synthetic Minority Over-sampling Technique)...")
    print("This will mathematically generate synthetic heartbeats to match the majority class (24,054).")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"\nNew Balanced shape: {X_resampled.shape}")
    
    # Recombine into DataFrame
    df_balanced = pd.DataFrame(X_resampled)
    df_balanced[label_col] = y_resampled
    
    print("New distribution:")
    print(df_balanced[label_col].value_counts())
    
    # Shuffle the dataset
    print("\nShuffling dataset to ensure random distribution...")
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = 'ecg_dataset_72k_balanced.csv'
    print(f"Saving massive 72k dataset to {output_path} (This might take a minute)...")
    df_balanced.to_csv(output_path, index=False, header=False)
    
    print(f"Success! Your massive balanced dataset has been fully restored.")

if __name__ == "__main__":
    regenerate_dataset()
