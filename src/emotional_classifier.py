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
  raw_data, raw_labels = load_data(DATA_DIR)
  
  print("Extracting features from EEG data...")
  all_features = np.array([extract_features(reading) for reading in raw_data])
  all_labels = np.array(raw_labels)

  # --- 3. Create Dataset and Split into Train/Validation ---
  print("Creating dataset and splitting into training and validation...")
  full_dataset = EEGDataset(all_features, all_labels)
  train_size = int(0.9 * len(full_dataset))
  val_size = len(full_dataset) - train_size
  train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
  
  print("Creating dataloaders... ")
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

  print("Initializing model, loss function and optimizer")
  input_feature_count = all_features.shape[1]
  model = EEGNet(input_features=input_feature_count, num_classes=NUM_EMOTIONS).to(device)

  criterion = nn.CrossEntropyLoss() # Good for multi-class classification
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  print("\nStarting model training...")
  for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for features, labels in train_loader:
      features = features.to(device)
      labels = labels.to(device)
      
      optimizer.zero_grad()
      
      outputs = model(features)
      loss = criterion(outputs, labels)
      
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # No need to calculate gradients during validation
      for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
      f"Loss: {running_loss/len(train_loader):.4f}, "
      f"Validation Accuracy: {accuracy:.2f}%")

  print("\nFinished Training!")