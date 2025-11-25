import os
import numpy as np
import pandas as pd
import torch
import gradio as gr

from sklearn.preprocessing import StandardScaler

from emotional_classifier import (
    NUM_EMOTIONS,
    NUM_CHANNELS,
    SYMMETRIC_PAIRS,
    create_asymmetry_features,
    EEG_CNN_Model,
)

CSV_FILE_PATH = "EEGEmotions-27/training/eeg_features_extracted.csv"
MODEL_PATH = "best_cnn_model.pth"

EMOTIONS = [
    "admiration", "adoration", "aesthetic appreciation", "amusement", "anger",
    "anxiety", "awe", "awkwardness", "boredom", "calmness", "confusion",
    "craving", "disgust", "empathic pain", "entrancement", "excitement",
    "fear", "horror", "interest", "joy", "nostalgia", "relief", "romance",
    "sadness", "satisfaction", "sexual desire", "surprise",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

all_features_tensor = None   # (N, F_per_channel, C)
cowen_labels_zero_based = None  # (N,) 0~26
cowen_labels_raw = None         # (N,) 1~27
participant_ids = None          # (N,)
num_samples = None
num_features_per_channel = None
model = None


def prepare_features():
    global all_features_tensor, \
       cowen_labels_zero_based, \
       cowen_labels_raw, \
       participant_ids, \
       num_samples, \
       num_features_per_channel

    print(f"Loading CSV from {CSV_FILE_PATH} ...")
    data = pd.read_csv(CSV_FILE_PATH).dropna()

    cowen_labels_raw = data["Emo_Label_Cowen(27)"].values.astype(int)  # 1~27
    participant_ids = data["ParticipantID"].values.astype(int)

    cowen_labels_zero_based = cowen_labels_raw - 1

    data_engineered = create_asymmetry_features(data, SYMMETRIC_PAIRS)
    features = data_engineered.values  # (N, F_total)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    num_samples_local, num_total_features = features_scaled.shape

    if num_total_features % NUM_CHANNELS != 0:
        raise ValueError(
            f"ValueError: Invalid feature layout — total features ({num_total_features}) are not divisible by the number of channels ({NUM_CHANNELS}). The dataset structure may be corrupted or mismatched."
        )

    n_feat_per_ch = num_total_features // NUM_CHANNELS

    # (N, F_total) -> (N, C, F_per_channel) -> (N, F_per_channel, C)
    features_grouped = features_scaled.reshape(
        num_samples_local, NUM_CHANNELS, n_feat_per_ch
    )
    features_transposed = np.transpose(features_grouped, (0, 2, 1))

    all_features_tensor = torch.tensor(
        features_transposed, dtype=torch.float32
    )  # (N, F_per_channel, C)

    num_samples = num_samples_local
    num_features_per_channel = n_feat_per_ch

    print(f"Prepared features: {all_features_tensor.shape}, samples: {num_samples}")


def load_model():
    global model, num_features_per_channel

    if all_features_tensor is None:
        prepare_features()

    print(f"Loading model from {MODEL_PATH} ...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"FileNotFoundError: The model file could not be found at: {MODEL_PATH}")

    m = EEG_CNN_Model(
        num_features=num_features_per_channel,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_EMOTIONS,
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()

    model = m
    print("Model loaded and ready.")


def predict(sample_index: int):
    if model is None:
        load_model()

    idx = int(sample_index)
    if idx < 0 or idx >= num_samples:
        return {"error": f"The index must be between 0 and {num_samples-1} (inclusive)."}, ""

    x = all_features_tensor[idx].unsqueeze(0).to(device)  # (1, F, C)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))            # 0~26
    true_idx = int(cowen_labels_zero_based[idx])  # 0~26

    prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(NUM_EMOTIONS)}

    pid = int(participant_ids[idx])
    cowen_raw = int(cowen_labels_raw[idx])      # 1~27

    info_text = (
        f"Sample #{idx}\n"
        f"- ParticipantID: {pid}\n"
        f"- Cowen Label: {cowen_raw} ({EMOTIONS[true_idx]})\n"
        f"- Predicted (Model): {EMOTIONS[pred_idx]}"
    )

    return prob_dict, info_text


def main():
    try:
        prepare_features()
        load_model()
    except Exception as e:
        print(f"Warning during initialization: {e}")
        print("Gradio will still launch. You can test with an index later to check for errors.")


    index_input = gr.Slider(
        minimum=0,
        maximum=(num_samples - 1) if num_samples is not None else 100,
        step=1,
        value=0,
        label="Sample Index (0-based)",
    )

    output_label = gr.Label(
        num_top_classes=5,
        label="Predicted Emotion Probabilities",
    )

    output_text = gr.Textbox(
        label="Sample Info (Participant / Labels / Prediction)",
        lines=6,
    )

    demo = gr.Interface(
        fn=predict,
        inputs=index_input,
        outputs=[output_label, output_text],
        title="EEGEmotions-27 CNN Emotion Classifier (Sample Viewer)",
        description=(
            "Select the N-th sample from the `eeg_features_extracted.csv` file to see how the trained CNN model predicts its emotion.\n\n"
            "The interface also displays the Participant ID and Cowen (27-class), along with the model’s prediction."
        ),
    )

    demo.launch()


if __name__ == "__main__":
    main()
