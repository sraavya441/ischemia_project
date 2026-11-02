import pandas as pd
import numpy as np
import joblib

def generate_risk_report():
    print("Loading models...")
    rf_model = joblib.load('rf_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    print("Loading extracted clinical features...")
    # Load the 1000 rows we already extracted features for
    df = pd.read_csv('ml_ready_features.csv')
    
    # Get feature columns expected by the model
    expected_cols = getattr(rf_model, 'feature_names_in_', df.columns)
    
    # Extract features for prediction
    X = df.reindex(columns=expected_cols, fill_value=0)
    
    print("Generating AI Risk Predictions...")
    predictions = rf_model.predict(X)
    probabilities = rf_model.predict_proba(X)
    
    # Decode labels
    decoded_predictions = label_encoder.inverse_transform(predictions)
    
    # Get max confidence score as Risk %
    confidence = np.max(probabilities, axis=1) * 100
    
    # Add to DataFrame
    df['AI_Predicted_Diagnosis'] = decoded_predictions
    df['AI_Confidence_Score_%'] = np.round(confidence, 2)
    
    # Check if prediction matches ground truth
    if 'label' in df.columns:
        df['Prediction_Correct'] = df['label'] == df['AI_Predicted_Diagnosis']
    
    print("Saving to risk_prediction.csv...")
    df.to_csv('risk_prediction.csv', index=False)
    
    print(f"Success! Generated risk_prediction.csv with {len(df)} patient records.")
    
if __name__ == "__main__":
    generate_risk_report()
