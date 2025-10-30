import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Dataloader, random_split
from tqdm import tqdm
import os
import numpy as np

DATA_DIR = "./eeg_raw"

SAMPLING_RATE = 128
NUM_CHANNELS = 14
NUM_EMOTIONS = 27

LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(data_dir):
  all_files = os.listdir(data_dir)
  eeg_data_list = []
  labels_list = []
  print = (f"Found {len(all_files)} files. Loading data...")

  for filename in tqdm(all_files):
    if filename.endswitch(".txt"):
      try:
        parts = filename.split('_')
        emotion_id = int(float(parts[1].replace('.txt','')))
        file_path = os.path.join(data_dir, filename)
        single_reading = np.loadtxt(file_path)
        eeg_data_list.append(single_reading)
        labels_list.append(emotion_id - 1)
      except Exception as e:
        print(f"Could not process file {filename}: {e}")

  print(f"Successfully loaded {len(eeg_data_list)} readings.")
  return eeg_data_list, labels_list

def extract_features(eeg_reading):
  mean_vals = np.mean(eeg_reading, axis=0)
  std_vals = np.std(eeg_reading, axis=0)

  feature_vector = np.concatenate((mean_vals, std_vals))

  return feature_vector

