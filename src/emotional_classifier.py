import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # NEW IMPORT
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# configuration
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5 
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_EMOTIONS = 27
NUM_CHANNELS = 14
CSV_FILE_PATH = "./EEGEmotions/training/eeg_features_extracted.csv" 
MODEL_SAVE_PATH = "best_cnn_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EEGDataset(Dataset):
  def __init__(self, features, labels, num_channels=14):
    num_samples = features.shape[0]
    num_total_features = features.shape[1]
    self.num_features_per_channel = num_total_features // num_channels
    if num_total_features % num_channels != 0:
        raise ValueError("Total feature count is not divisible by num_channels")

    features_grouped_by_channel = features.reshape(
        num_samples, num_channels, self.num_features_per_channel
    )
    features_transposed = features_grouped_by_channel.transpose(0, 2, 1)

    self.features = torch.tensor(features_transposed, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)
    print(f"Dataset shape: {self.features.shape}")

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

class EEG_CNN_Model(nn.Module):
  def __init__(self, num_features, num_channels, num_classes):
    super(EEG_CNN_Model, self).__init__()
    
    self.conv_block1 = nn.Sequential(
        nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(64)
    )
    
    self.conv_block2 = nn.Sequential(
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(128)
    )
    
    self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    self.flatten = nn.Flatten()
    
    self.fc = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes)
    )

  def forward(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.global_avg_pool(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x

if __name__ == "__main__":
    
  print(f"Loading pre-extracted features from {CSV_FILE_PATH}...")
  try:
    data = pd.read_csv(CSV_FILE_PATH)
  except FileNotFoundError:
    print(f"[ERROR] CSV file not found at: {CSV_FILE_PATH}")
    exit()

  data = data.dropna()
  labels = data['Emo_Label_Cowen(27)'].values - 1
  metadata_cols = ['Emo_Label_Ekman(6)', 'Emo_Label_Cowen(27)', 
                   'ParticipantID', 'Age', 'Gender', 'Nation', 
                   'eeg_component_number']
  feature_columns = [col for col in data.columns if col not in metadata_cols]
  features = data[feature_columns].apply(pd.to_numeric, errors='coerce').values
  num_total_features = features.shape[1]
  num_features_per_channel = num_total_features // NUM_CHANNELS
  
  print(f"Loaded {features.shape[0]} samples. Features per channel: {num_features_per_channel}")

  scaler = StandardScaler()
  features_scaled = scaler.fit_transform(features)
  X_train, X_val, y_train, y_val = train_test_split(
      features_scaled, labels, test_size=0.1, random_state=42, stratify=labels
  )

  train_dataset = EEGDataset(X_train, y_train, num_channels=NUM_CHANNELS)
  val_dataset = EEGDataset(X_val, y_val, num_channels=NUM_CHANNELS)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

  model = EEG_CNN_Model(
      num_features=num_features_per_channel, 
      num_channels=NUM_CHANNELS, 
      num_classes=NUM_EMOTIONS
  ).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
  best_val_loss = float('inf')
  
  print("\nStarting model training...")

  for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for features_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
      features_batch = features_batch.to(device)
      labels_batch = labels_batch.to(device)
      
      optimizer.zero_grad()
      outputs = model(features_batch)
      loss = criterion(outputs, labels_batch)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    # validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    with torch.no_grad():
      for features_batch, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        outputs = model(features_batch)
        loss = criterion(outputs, labels_batch)
        val_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

    train_loss_avg = running_loss / len(train_loader)
    val_loss_avg = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    scheduler.step(val_loss_avg) 

    if val_loss_avg < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss_avg:.4f}. Saving model...")
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss_avg:.4f} | "
          f"Val Loss: {val_loss_avg:.4f} | "
          f"Val Accuracy: {accuracy:.2f}% | "
          f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

  print("\nFinished Training!")
  print(f"Best model weights saved to {MODEL_SAVE_PATH}")