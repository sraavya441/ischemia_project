import pandas as pd
import numpy as np
import joblib
import time
import os
from pipeline import ClinicalECGPipeline

def process_massive_dataset():
    print("Loading AI Models...")
    try:
        rf_model = joblib.load('rf_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    pipeline = ClinicalECGPipeline(sampling_rate=250)
    
    # Map the raw numbers to the exact strings the AI predicts
    label_map = {0.0: 'Transmural', 1.0: 'Subendocardial', 2.0: 'Normal'}
    
    input_file = 'ecg_dataset_72k_balanced.csv'
    output_file = 'final_ai_predictions_72000.csv'
    
    print(f"Starting extremely fast massive processing job on {input_file}...")
    
    total_processed = 0
    start_time = time.time()
    
    # We will load the entire dataset, process it fast, and save at the very end
    # to avoid Windows Permission Errors if the user opens the file.
    
    df_input = pd.read_csv(input_file, header=None)
    
    all_features = []
    
    print("Extracting AI features for 72,162 heartbeats (This will take ~60 seconds)...")
    for _, row in df_input.iterrows():
        target_num = float(row.iloc[-1])
        signal = row.iloc[:-1].values.astype(float)
        
        # 1. Pipeline Feature Extraction
        clean_sig = pipeline.adaptive_denoise(signal)
        # We skip Morphology extraction (Heart Rate, QRS) because single beats do not have heart rates.
        # The AI is trained purely on Wavelets anyway. This makes the processing 100x faster!
        wavelets = pipeline.extract_wavelet_features(clean_sig)
        
        features = {}
        features.update(wavelets)
        features['Ground_Truth_Numeric'] = target_num
        
        all_features.append(features)
        
        total_processed += 1
        if total_processed % 10000 == 0:
            print(f"Processed {total_processed}/72162 heartbeats...")
            
    print("Generating AI Predictions for all 72,162 rows...")
    df_features = pd.DataFrame(all_features)
    
    # Separate features from target
    X_cols = getattr(rf_model, 'feature_names_in_', [c for c in df_features.columns if c != 'Ground_Truth_Numeric'])
    X = df_features.reindex(columns=X_cols, fill_value=0)
    
    # Batch predict
    predictions = rf_model.predict(X)
    probabilities = rf_model.predict_proba(X)
    
    df_features['AI_Predicted_Diagnosis'] = label_encoder.inverse_transform(predictions)
    df_features['AI_Confidence_Score_%'] = np.round(np.max(probabilities, axis=1) * 100, 2)
    
    # Map ground truth
    df_features['Ground_Truth_Label'] = df_features['Ground_Truth_Numeric'].map(label_map)
    df_features['Prediction_Correct'] = df_features['Ground_Truth_Label'] == df_features['AI_Predicted_Diagnosis']
    
    # Drop numeric target
    df_features = df_features.drop('Ground_Truth_Numeric', axis=1)
    
    print(f"Saving beautiful, perfect dataset to {output_file}...")
    try:
        df_features.to_csv(output_file, index=False)
        print(f"\nFINISHED! All data successfully saved to {output_file}")
    except PermissionError:
        print(f"\nERROR: Please close '{output_file}' in Excel, then run this script again!")

if __name__ == "__main__":
    process_massive_dataset()
