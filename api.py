from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from pipeline import ClinicalECGPipeline

app = FastAPI(title="CardioVision ML Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
try:
    rf_model = joblib.load('rf_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Machine learning models loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load models. Error: {e}")
    rf_model = None
    label_encoder = None

# Initialize pipeline
ecg_pipeline = ClinicalECGPipeline(sampling_rate=250)

@app.get("/")
def read_root():
    return {"message": "CardioVision API is running! Go to /docs to test it."}

class ECGPayload(BaseModel):
    data: list[float]

@app.post("/predict")
async def predict_ecg(payload: ECGPayload):
    if not rf_model or not label_encoder:
        raise HTTPException(status_code=500, detail="Models are not loaded on the server.")
        
    signal_data = payload.data
    
    # Save to file for Streamlit Live Sync
    import json
    with open('latest_signal.json', 'w') as f:
        json.dump(signal_data, f)
    
    if len(signal_data) == 0:
        raise HTTPException(status_code=400, detail="Empty data payload received.")
        
    # Convert to numpy array
    signal = np.array(signal_data, dtype=float)
    
    try:
        # 1. Adaptive Denoising
        clean_sig = ecg_pipeline.adaptive_denoise(signal)
        
        # 2. Extract Features
        morphology = ecg_pipeline.extract_morphology(clean_sig)
        wavelets = ecg_pipeline.extract_wavelet_features(clean_sig)
        
        # 3. Combine Features
        features = {}
        features.update(morphology)
        features.update(wavelets)
        
        # Replace NaNs with 0 (as trained)
        df_features = pd.DataFrame([features]).fillna(0)
        
        # 4. Predict
        # We must ensure the column order exactly matches what the RF was trained on.
        expected_cols = getattr(rf_model, 'feature_names_in_', df_features.columns)
        df_features = df_features.reindex(columns=expected_cols, fill_value=0)
        
        prediction = rf_model.predict(df_features)
        
        # 5. Decode label
        diagnosis = label_encoder.inverse_transform(prediction)[0]
        
        return {
            "status": "success",
            "diagnosis": diagnosis,
            "features_extracted": list(features.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
