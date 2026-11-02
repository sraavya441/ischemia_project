import pandas as pd
import numpy as np
import os

print("Loading original dataset...")
dataset_path = r"c:\Users\durga\Downloads\srs\project\project-1\ecg_dataset.csv"
output_path = r"c:\Users\durga\Downloads\srs\project\project-1\balanced_ecg_dataset.csv"

# Load data
df = pd.read_csv(dataset_path, header=None)

# The last column is the label
label_col = df.columns[-1]

print(f"Original shape: {df.shape}")
print("Original distribution:")
print(df[label_col].value_counts())

# Separate classes
# 0 = STEMI, 1 = NSTEMI, 2 = Normal
df_stemi = df[df[label_col] == 0]
df_nstemi = df[df[label_col] == 1]
df_normal = df[df[label_col] == 2]

print(f"STEMI signals found: {len(df_stemi)}")
print(f"NSTEMI signals found: {len(df_nstemi)}")
print(f"Normal signals found: {len(df_normal)}")

# Balance the Normal class by subsampling
# Setting a random_state for reproducibility
num_samples = len(df_stemi)  # Should be 3007
df_normal_sampled = df_normal.sample(n=num_samples, random_state=42)

print(f"Sampled Normal signals: {len(df_normal_sampled)}")

# Combine all into a balanced dataset
df_balanced = pd.concat([df_stemi, df_nstemi, df_normal_sampled])

# Shuffle the combined dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset shape: {df_balanced.shape}")
print("New distribution:")
print(df_balanced[label_col].value_counts())

print("Saving balanced dataset...")
df_balanced.to_csv(output_path, index=False, header=False)
print(f"Balanced dataset saved successfully to {output_path}")
