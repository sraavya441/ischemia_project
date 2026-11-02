import pandas as pd
from xgboost import XGBClassifier
import joblib

print("Loading data for model saving...")
features_path = r"c:\Users\durga\Downloads\srs\project\project-1\extracted_features.csv"
df = pd.read_csv(features_path).dropna()

X = df.drop('label', axis=1)
y = df['label']

print("Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb.fit(X, y)

model_path = r"c:\Users\durga\Downloads\srs\project\project-1\xgb_model.json"
xgb.save_model(model_path)
print(f"Model successfully saved to {model_path}")
