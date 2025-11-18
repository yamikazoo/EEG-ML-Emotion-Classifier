import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5 
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_EMOTIONS = 27
NUM_CHANNELS = 14
CSV_FILE_PATH = "./EEGEmotions/training/eeg_features_extracted.csv" 
MODEL_SAVE_PATH = "best_cnn_model.pth"
RANDOM_STATE = 42
SYMMETRIC_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_asymmetry_features(df, symmetric_pairs):
    """Calculates Differential Asymmetry (DA) and Rational Asymmetry (RA) features."""
    asymmetry_features = {}
    all_features = [col for col in df.columns if col.split('_')[-1].isdigit()]
    base_features = [col.rsplit('_', 1)[0] for col in all_features if col.endswith('_1')]

    for left_idx, right_idx in symmetric_pairs:
        for feature_name in base_features:
            col_L = f"{feature_name}_{left_idx}"
            col_R = f"{feature_name}_{right_idx}"
            da_col_name = f"DA_{feature_name}_{left_idx}-{right_idx}"
            asymmetry_features[da_col_name] = df[col_L] - df[col_R]
            ra_col_name = f"RA_{feature_name}_{left_idx}-{right_idx}"
            sum_cols = df[col_L] + df[col_R]
            ra_values = (df[col_L] - df[col_R]) / sum_cols
            ra_values.replace([np.inf, -np.inf], 0, inplace=True) 
            asymmetry_features[ra_col_name] = ra_values

    asymmetry_df = pd.DataFrame(asymmetry_features, index=df.index)
    
    return pd.concat([df[all_features], asymmetry_df], axis=1)

class EEGDataset(Dataset):
  def __init__(self, features, labels, num_channels=NUM_CHANNELS):
    num_samples = features.shape[0]
    num_total_features = features.shape[1]
    
    self.num_features_per_channel = num_total_features // num_channels
    if num_total_features % num_channels != 0:
        raise ValueError(f"Total features ({num_total_features}) not divisible by channels ({num_channels}). Check feature ordering.")

    features_grouped_by_channel = features.reshape(
        num_samples, num_channels, self.num_features_per_channel
    )
    features_transposed = features_grouped_by_channel.transpose(0, 2, 1)

    self.features = torch.tensor(features_transposed, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)
    print(f"Dataset ready")

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
  print(f"Loading data from {CSV_FILE_PATH}...")
  data = pd.read_csv(CSV_FILE_PATH).dropna()

  data_engineered = create_asymmetry_features(data, SYMMETRIC_PAIRS)
  labels = data['Emo_Label_Cowen(27)'].values - 1 
  feature_columns = data_engineered.columns.tolist()
  features = data_engineered.values
  feature_names = data_engineered.columns.to_list()
  
  print(f"After Feature Engineering, total features: {features.shape[1]}")
  print("Step 1: Feature Selection using Random Forest...")
  
  scaler_temp = StandardScaler()
  features_scaled_temp = scaler_temp.fit_transform(features)
  
  X_train_rf, _, y_train_rf, _ = train_test_split(
      features_scaled_temp, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
  )

  rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
  rf_model.fit(X_train_rf, y_train_rf)
  
  importances = rf_model.feature_importances_
  feature_importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance': importances
  }).sort_values(by='Importance', ascending=False)
  
  CUTOFF_PERCENT = 0.75
  cutoff_index = int(len(feature_importance_df) * CUTOFF_PERCENT) 
  selected_feature_names = feature_importance_df['Feature'].head(cutoff_index).tolist()
  
  features_final = data_engineered.values
  
  print("Step 2: Final Scaling and Splitting for CNN training...")
  scaler_final = StandardScaler()
  features_scaled = scaler_final.fit_transform(features_final)
  
  X_train, X_val, y_train, y_val = train_test_split(
      features_scaled, labels, test_size=0.1, random_state=RANDOM_STATE, stratify=labels
  )
  
  num_features_retained = X_train.shape[1]
  num_features_per_channel = num_features_retained // NUM_CHANNELS 

  # init dataloaders + dataset 
  train_dataset = EEGDataset(X_train, y_train, num_channels=NUM_CHANNELS)
  val_dataset = EEGDataset(X_val, y_val, num_channels=NUM_CHANNELS)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

  # init model
  model = EEG_CNN_Model(
      num_features=num_features_per_channel, 
      num_channels=NUM_CHANNELS, 
      num_classes=NUM_EMOTIONS
  ).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
  best_val_loss = float('inf')
  
  print("\nStep 3: Starting Advanced CNN Training...")

  for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for features_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
      features_batch = features_batch.to(device)
      labels_batch = labels_batch.to(device)
      
      optimizer.zero_grad()
      outputs = model(features_batch)
      loss = criterion(outputs, labels_batch)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      _, predicted = torch.max(outputs.data, 1)
      train_total += labels_batch.size(0)
      train_correct += (predicted == labels_batch).sum().item()


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
    training_accuracy = 100 * train_correct / train_total 
    val_loss_avg = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    scheduler.step(val_loss_avg) 

    if val_loss_avg < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss_avg:.4f}. Saving model...")
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss_avg:.4f} | "
          f"Training Accuracy: {training_accuracy:.2f}% | "
          f"Val Loss: {val_loss_avg:.4f} | "
          f"Val Accuracy: {accuracy:.2f}% | "
          f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

  print("\nFinished Training!")
  print(f"Best model weights saved to {MODEL_SAVE_PATH}")