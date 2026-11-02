import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

print("Loading data and model...")
df = pd.read_csv('final_ai_predictions_72000.csv')
cols_to_drop = ['severity', 'AI_Predicted_Diagnosis', 'AI_Confidence_Score_%', 'Prediction_Correct']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
if 'Ground_Truth_Label' in df.columns:
    df = df.rename(columns={'Ground_Truth_Label': 'label'})

X = df.drop('label', axis=1).fillna(0)
y_strings = df['label']

le = joblib.load('label_encoder.pkl')
rf = joblib.load('rf_model.pkl')
y = le.transform(y_strings)

# Recreate the exact test set from training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Predicting on unseen test set...")
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"True Test Accuracy: {acc}")
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:\n", report)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
out_dir = r"C:\Users\S\.gemini\antigravity\brain\c906275e-8840-4431-b2b9-495c6a83bad0"
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# 2. Precision/Recall Bar Chart
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
classes = le.classes_
precisions = [report_dict[c]['precision'] for c in classes]
recalls = [report_dict[c]['recall'] for c in classes]
f1s = [report_dict[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precisions, width, label='Precision', color='#1f77b4')
rects2 = ax.bar(x, recalls, width, label='Recall', color='#ff7f0e')
rects3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#2ca02c')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim([0.0, 1.1])
ax.legend(loc='lower right')

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "metrics_bar_chart.png"), dpi=300)
plt.close()

print("Charts saved successfully.")
