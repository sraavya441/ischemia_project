import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

print("Loading final_ai_predictions_72000.csv...")
features_path = "final_ai_predictions_72000.csv"

df = pd.read_csv(features_path)

# Drop severity and previous AI prediction columns
cols_to_drop = ['severity', 'AI_Predicted_Diagnosis', 'AI_Confidence_Score_%', 'Prediction_Correct']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Rename Ground_Truth_Label to label for consistency
if 'Ground_Truth_Label' in df.columns:
    df = df.rename(columns={'Ground_Truth_Label': 'label'})

X = df.drop('label', axis=1)
y_strings = df['label']

# Encode string labels to numeric for XGBoost
le = LabelEncoder()
y = le.fit_transform(y_strings)
target_names = le.classes_

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples\n")

print("--- Training Random Forest Classifier ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report (RF):")
print(classification_report(y_test, y_pred_rf, labels=np.unique(y_test)))

print("--- Training XGBoost Classifier ---")
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report (XGB):")
print(classification_report(y_test, y_pred_xgb, labels=np.unique(y_test)))

print("\n--- Feature Importance (Top 10) ---")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

for i in range(min(10, len(feature_names))):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

top_features = pd.DataFrame({
    'Feature': [feature_names[indices[i]] for i in range(min(10, len(feature_names)))],
    'Importance': [importances[indices[i]] for i in range(min(10, len(feature_names)))]
})
top_features.to_csv("top_features.csv", index=False)

print("\nSaving models to disk via joblib...")
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Models saved successfully.")
