import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_EMOTIONS = 27
CSV_FILE_PATH = "./EEGEmotions/training/eeg_features_extracted.csv" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EEGDataset(Dataset):
  def __init__(self, features, labels):
    self.features = torch.tensor(features, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

class EEGNet(nn.Module):
  """A simple Neural Network for EEG classification."""
  def __init__(self, input_features, num_classes):
    super(EEGNet, self).__init__()
    self.layer1 = nn.Linear(input_features, 128)
    self.relu1 = nn.ReLU()
    self.layer2 = nn.Linear(128, 64)
    self.relu2 = nn.ReLU()
    self.output_layer = nn.Linear(64, num_classes)

  def forward(self, x):
    x = self.relu1(self.layer1(x))
    x = self.relu2(self.layer2(x))
    x = self.output_layer(x)
    return x

if __name__ == "__main__":
    
  print(f"Loading pre-extracted features from {CSV_FILE_PATH}...")
  try:
    data = pd.read_csv(CSV_FILE_PATH)
  except FileNotFoundError:
    print(f"[ERROR] CSV file not found at: {CSV_FILE_PATH}")
    print("Please make sure the file is in the same directory as the script.")
    exit()

  data = data.dropna()
  
  if 'Emo_Label_Cowen(27)' not in data.columns:
      print("[ERROR] 'Emo_Label_Cowen(27)' column not found in CSV.")
      exit()

  # create 0-based indexing for labels
  labels = data['Emo_Label_Cowen(27)'].values - 1

  metadata_cols = ['Emo_Label_Ekman(6)', 'Emo_Label_Cowen(27)', 
                   'ParticipantID', 'Age', 'Gender', 'Nation', 
                   'eeg_component_number']
  
  feature_columns = [col for col in data.columns if col not in metadata_cols]
  features = data[feature_columns].apply(pd.to_numeric, errors='coerce').values
  
  print(f"Successfully loaded {features.shape[0]} samples.")
  print(f"Each sample has {features.shape[1]} features.")

  print("Normalizing features (Standard Scaling)...")
  scaler = StandardScaler()
  features_scaled = scaler.fit_transform(features)

  print("Splitting data into training (90%) and validation (10%)...")
  X_train, X_val, y_train, y_val = train_test_split(
      features_scaled, 
      labels, 
      test_size=0.1,
      random_state=42,
      stratify=labels
  )

  print("Creating datasets and dataloaders...")
  train_dataset = EEGDataset(X_train, y_train)
  val_dataset = EEGDataset(X_val, y_val)

  # define num of workers to increase performance 
  train_loader = DataLoader(
      train_dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=True, 
      num_workers=os.cpu_count() // 2 
  )
  val_loader = DataLoader(
      val_dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=False, 
      num_workers=os.cpu_count() // 2
  )

  print("Initializing model, loss function and optimizer")
  input_feature_count = features.shape[1]
  model = EEGNet(input_features=input_feature_count, num_classes=NUM_EMOTIONS).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    # print details after training 
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss_avg:.4f} | "
          f"Val Loss: {val_loss_avg:.4f} | "
          f"Val Accuracy: {accuracy:.2f}%")

  print("\nFinished Training!")