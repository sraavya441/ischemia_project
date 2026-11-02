import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Data
print("Loading ecg_dataset_72k_balanced.csv (Full Dataset)...")
# Removed nrows=10000 to train on the full 72,000 dataset for maximum accuracy
df = pd.read_csv('ecg_dataset_72k_balanced.csv', header=None) 

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Convert labels to integer class indices if they aren't already
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, 'lstm_label_encoder.pkl')

print(f"Dataset shape: {X.shape}")

# LSTMs expect 3D input: (samples, time_steps, features)
# Here, features=1 since it's a univariate time series (ECG amplitude)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

# Create DataLoaders
train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Define Advanced LSTM Model
class ECGLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=3):
        super(ECGLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Added bidirectional=True which dramatically improves ECG accuracy
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        # Multiply by 2 because it's bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # Multiply num_layers by 2 for bidirectional
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

num_classes = len(np.unique(y))
model = ECGLSTM(num_classes=num_classes).to(device)

# 3. Train the Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Added weight decay

epochs = 10 # Reduced for faster CPU training
print("\nStarting Training...")
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f} - Time: {epoch_time:.2f}s")

# Save the model FIRST just in case evaluation crashes
torch.save(model.state_dict(), 'lstm_model.pth')
print("\nModel saved to lstm_model.pth!")

# 4. Evaluate and Metrics
print("\nEvaluating Model in batches to prevent memory crash...")
model.eval()
all_preds = []
test_data = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

with torch.no_grad():
    for inputs, _ in test_loader:
        test_outputs = model(inputs)
        _, predicted = torch.max(test_outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())

y_pred = np.array(all_preds)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}\n")

print("Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
