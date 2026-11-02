import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading extracted features...")
df = pd.read_csv(r"c:\Users\durga\Downloads\srs\project\project-1\extracted_features.csv").dropna()
X = df.drop('label', axis=1).values
y = df['label'].values

# Standardize features for GAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define Tabular GAN
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

latent_dim = 16
feature_dim = X.shape[1]
epochs = 500  # Keeping it low for fast CPU execution
batch_size = 128

print("Training GAN for data augmentation...")
augmented_X = []
augmented_y = []

# Train a separate GAN for each class to generate specific class data
for target_class in [0, 1, 2]:
    print(f"Training GAN for Class {target_class}...")
    X_class = X_scaled[y == target_class]
    tensor_X = torch.FloatTensor(X_class)
    
    G = Generator(latent_dim, feature_dim)
    D = Discriminator(feature_dim)
    
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
    
    # DataLoader
    dataset = torch.utils.data.TensorDataset(tensor_X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for real_data in dataloader:
            real_data = real_data[0]
            batch_size_real = real_data.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size_real, 1)
            fake_labels = torch.zeros(batch_size_real, 1)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = D(real_data)
            d_loss_real = criterion(outputs, real_labels)
            
            z = torch.randn(batch_size_real, latent_dim)
            fake_data = G(z)
            outputs = D(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            outputs = D(fake_data)
            g_loss = criterion(outputs, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
    # Generate 1000 synthetic samples for this class
    with torch.no_grad():
        z_synthetic = torch.randn(1000, latent_dim)
        synthetic_data = G(z_synthetic).numpy()
        # Inverse transform to original scale
        synthetic_data = scaler.inverse_transform(synthetic_data)
        
        augmented_X.extend(synthetic_data)
        augmented_y.extend([target_class] * 1000)

print(f"Generated {len(augmented_X)} synthetic samples.")

# Combine Real and Synthetic
X_combined = np.vstack((X, augmented_X))
y_combined = np.hstack((y, augmented_y))

print(f"Original Dataset Size: {X.shape[0]}")
print(f"Augmented Dataset Size: {X_combined.shape[0]}")

# Retrain ML Model on Augmented Data
print("\n--- Retraining XGBoost on Augmented Data ---")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print(f"New XGBoost Accuracy (with GAN Augmentation): {accuracy_score(y_test, y_pred):.4f}")

# Save the augmented dataset just in case
cols = df.columns[:-1]
aug_df = pd.DataFrame(X_combined, columns=cols)
aug_df['label'] = y_combined
aug_df.to_csv(r"c:\Users\durga\Downloads\srs\project\project-1\augmented_features.csv", index=False)
print("Saved augmented features to augmented_features.csv")
