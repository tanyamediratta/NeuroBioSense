from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


st.set_page_config(
    page_title="NeuroBioSense Emotion Dissonance",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR

VIDEO_MODEL_PATH = OUT_DIR / "video_3dcnn_emotion.pt"
BIO_MODEL_PATH = OUT_DIR / "bio_1dcnn_sliding_window.pt"
FUSION_MODEL_PATH = OUT_DIR / "fusion_3dcnn_1dcnn_difference.pt"
BIO_MEAN_PATH = OUT_DIR / "bio_sliding_mean.npy"
BIO_STD_PATH = OUT_DIR / "bio_sliding_std.npy"
METADATA_PATH = OUT_DIR / "final_project_metadata.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_MAP = {"N": 0, "A": 1, "D": 2, "F": 3, "J": 4, "SA": 5, "SU": 6}
ID_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}
EMOTION_NAMES = {
    "N": "Neutral",
    "A": "Anger",
    "D": "Disgust",
    "F": "Fear",
    "J": "Joy",
    "SA": "Sadness",
    "SU": "Surprise",
}

N_FRAMES = 16
FRAME_SIZE = 112
BIO_SEQ_LEN = 64
BIO_COLUMNS = ["BVP", "EDA", "TEMP", "X", "Y", "Z"]

KIN_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KIN_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


class Video3DCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        for name, p in self.backbone.named_parameters():
            if name.startswith("stem") or name.startswith("layer1"):
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, num_classes),
        )

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.features(x))


class Bio1DCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.classifier(x)


def emotion_label(class_id: int) -> str:
    code = ID_TO_EMOTION[int(class_id)]
    return f"{EMOTION_NAMES[code]} ({code})"


def status_chip(path: Path) -> str:
    return "Available" if path.exists() else "Missing"


def emotion_group(class_id: int) -> str:
    code = ID_TO_EMOTION[int(class_id)]

    if code == "N":
        return "neutral"

    if code == "J":
        return "positive"

    return "negative"


def fusion_insight(video_pred: int, bio_pred: int) -> tuple[str, str]:
    video_code = ID_TO_EMOTION[int(video_pred)]
    bio_code = ID_TO_EMOTION[int(bio_pred)]

    negative = {"A", "D", "F", "SA", "SU"}

    if video_code == bio_code:
        return (
            "Emotionally Aligned",
            "The emotion shown in the advertisement matches the viewer's biosignal response.",
        )

    if video_code == "J" and bio_code in negative:
        return (
            "Fake Happiness",
            "The advertisement appears joyful, but the biosignal response suggests stress, shock, sadness, fear, anger, or discomfort.",
        )

    if video_code == "N" and bio_code in negative:
        return (
            "Hidden Stress Response",
            "The video appears neutral, but the biosignal response suggests internal stress or discomfort.",
        )

    if video_code in negative and bio_code == "N":
        return (
            "Detached Response",
            "The video shows a strong or high-arousal emotion, but the viewer's biosignal response appears neutral.",
        )

    if video_code in negative and bio_code == "J":
        return (
            "Unexpected Positive Response",
            "The video shows a negative or high-arousal emotion, but the viewer's biosignal response appears positive.",
        )

    if video_code == "J" and bio_code == "N":
        return (
            "Low Emotional Engagement",
            "The advertisement appears happy, but the viewer's biosignal response appears neutral.",
        )

    if video_code == "N" and bio_code == "J":
        return (
            "Positive Internal Response",
            "The video appears neutral, but the viewer's biosignal response suggests positive engagement.",
        )

    return (
        "Emotional Dissonance",
        "The emotion shown by the video differs from the viewer's biosignal emotion.",
    )


def purchase_intent_estimate(video_pred: int, bio_pred: int) -> tuple[str, str]:
    video_group = emotion_group(video_pred)
    bio_group = emotion_group(bio_pred)
    match = video_group == bio_group

    if match and bio_group == "positive":
        return (
            "Likely to Buy",
            "The viewer's biosignal response is positive and aligned with the advertisement emotion group.",
        )

    if video_group == "positive" and bio_group == "negative":
        return (
            "Unlikely to Buy",
            "The advertisement appears positive, but the viewer's biosignal response is negative or high-arousal, indicating emotional dissonance.",
        )

    if bio_group == "negative":
        return (
            "Unlikely to Buy",
            "The viewer's biosignal response suggests discomfort, stress, shock, fear, anger, sadness, or rejection.",
        )

    if bio_group == "neutral" and video_group == "positive":
        return (
            "Maybe / Low Intent",
            "The advertisement appears positive, but the viewer's biosignal response is neutral, suggesting limited engagement.",
        )

    if bio_group == "positive" and video_group == "neutral":
        return (
            "Maybe / Moderate Intent",
            "The viewer shows a positive internal response even though the advertisement appears neutral.",
        )

    if match and bio_group == "neutral":
        return (
            "Maybe / Neutral Intent",
            "Both the advertisement and biosignal response appear neutral, so purchase intent is uncertain.",
        )

    return (
        "Maybe / Uncertain",
        "The emotional pattern does not strongly indicate purchase or rejection.",
    )


@st.cache_resource
def load_models():
    if not VIDEO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {VIDEO_MODEL_PATH}")

    if not BIO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {BIO_MODEL_PATH}")

    video_model = Video3DCNN(num_classes=7).to(DEVICE)
    video_ckpt = torch.load(VIDEO_MODEL_PATH, map_location=DEVICE)
    video_model.load_state_dict(video_ckpt["model_state_dict"])
    video_model.eval()

    bio_model = Bio1DCNN(in_channels=len(BIO_COLUMNS), num_classes=7).to(DEVICE)
    bio_ckpt = torch.load(BIO_MODEL_PATH, map_location=DEVICE)
    bio_model.load_state_dict(bio_ckpt["model_state_dict"])
    bio_model.eval()

    return video_model, bio_model


def read_video_clip(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError("Could not read frames from uploaded video.")

    if total < N_FRAMES:
        frame_indices = np.linspace(0, max(0, total - 1), N_FRAMES).astype(int)
    else:
        start = max(0, (total - N_FRAMES) // 2)
        frame_indices = np.arange(start, start + N_FRAMES)

    wanted = set(frame_indices.tolist())
    frames = []
    current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(frame)

        current += 1

        if len(frames) == N_FRAMES:
            break

    cap.release()

    if not frames:
        raise ValueError("Uploaded video did not produce a valid clip.")

    while len(frames) < N_FRAMES:
        frames.append(frames[-1])

    arr = np.stack(frames, axis=0)
    x = torch.from_numpy(arr).float().permute(3, 0, 1, 2) / 255.0
    x = (x - KIN_MEAN) / KIN_STD

    return x.unsqueeze(0)


def resample_biosignal(seq: np.ndarray, seq_len: int = BIO_SEQ_LEN) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros((seq_len, len(BIO_COLUMNS)), dtype=np.float32)

    if len(seq) == 1:
        return np.repeat(seq, seq_len, axis=0).astype(np.float32)

    old_x = np.linspace(0, 1, len(seq))
    new_x = np.linspace(0, 1, seq_len)

    out = []
    for c in range(seq.shape[1]):
        out.append(np.interp(new_x, old_x, seq[:, c]))

    return np.stack(out, axis=1).astype(np.float32)


def read_biosignal_csv(uploaded_file) -> torch.Tensor:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in BIO_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Biosignal CSV is missing columns: {missing}")

    values = df[BIO_COLUMNS].values.astype(np.float32)

    if BIO_MEAN_PATH.exists() and BIO_STD_PATH.exists():
        mean = np.load(BIO_MEAN_PATH)
        std = np.load(BIO_STD_PATH)
        values = (values - mean) / (std + 1e-6)

    values = resample_biosignal(values, BIO_SEQ_LEN)

    return torch.tensor(values.T, dtype=torch.float32).unsqueeze(0)


def predict_emotions(video_file, bio_file):
    video_model, bio_model = load_models()

    suffix = Path(video_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    try:
        video_tensor = read_video_clip(video_path).to(DEVICE)
    finally:
        Path(video_path).unlink(missing_ok=True)

    bio_tensor = read_biosignal_csv(bio_file).to(DEVICE)

    with torch.no_grad():
        video_logits = video_model(video_tensor)
        bio_logits = bio_model(bio_tensor)

        video_probs = torch.softmax(video_logits, dim=1).cpu().numpy()[0]
        bio_probs = torch.softmax(bio_logits, dim=1).cpu().numpy()[0]

    video_pred = int(video_probs.argmax())
    bio_pred = int(bio_probs.argmax())

    return video_pred, bio_pred, video_probs, bio_probs


def probability_table(probs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Emotion": [emotion_label(i) for i in range(7)],
            "Confidence": probs,
        }
    ).sort_values("Confidence", ascending=False)


st.title("NeuroBioSense Emotion Dissonance Detection")
st.caption(
    "3D CNN for advertisement video emotion, 1D CNN for biosignal emotion, "
    "rule-based fusion insight, and estimated purchase intent"
)

with st.sidebar:
    st.header("Model Files")
    st.write(f"Device: `{DEVICE}`")
    st.write(f"Video model: **{status_chip(VIDEO_MODEL_PATH)}**")
    st.write(f"Biosignal model: **{status_chip(BIO_MODEL_PATH)}**")
    st.write(f"Fusion artifact: **{status_chip(FUSION_MODEL_PATH)}**")
    st.write(f"Normalization: **{status_chip(BIO_MEAN_PATH)} / {status_chip(BIO_STD_PATH)}**")

    st.divider()
    st.header("Project Metrics")
    st.metric("Video 3D CNN", "49.8%")
    st.metric("Biosignal 1D CNN", "40.7%")
    st.metric("Rule-Based Fusion", "83.78%")
    st.caption(
        "Final fusion uses affective grouping: Neutral = N, Positive = J, "
        "Negative/High-Arousal = A, D, F, SA, SU."
    )

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Upload Inputs")
    video_file = st.file_uploader(
        "Upload advertisement video",
        type=["mp4", "mov", "avi", "mkv", "mpeg4"],
    )
    bio_file = st.file_uploader("Upload biosignal slice CSV", type=["csv"])
    run_btn = st.button(
        "Run Prediction",
        type="primary",
        disabled=video_file is None or bio_file is None,
    )

with col_right:
    st.subheader("What This Demo Shows")
    st.write(
        "The video model predicts the emotion shown in the advertisement. "
        "The biosignal model predicts the viewer's internal emotion from physiological signals. "
        "A rule-based fusion layer groups emotions into Neutral, Positive, and Negative/High-Arousal "
        "to display match/mismatch, fusion insight, and estimated purchase intent."
    )

st.divider()

if run_btn:
    try:
        with st.spinner("Running video and biosignal models..."):
            video_pred, bio_pred, video_probs, bio_probs = predict_emotions(video_file, bio_file)

        video_group = emotion_group(video_pred)
        bio_group = emotion_group(bio_pred)
        match = video_group == bio_group

        difference_result = "Match" if match else "Mismatch"
        insight_title, insight_text = fusion_insight(video_pred, bio_pred)
        purchase_title, purchase_text = purchase_intent_estimate(video_pred, bio_pred)

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "Video Emotion",
            emotion_label(video_pred),
            f"{video_probs[video_pred] * 100:.1f}% confidence",
        )

        m2.metric(
            "Biosignal Emotion",
            emotion_label(bio_pred),
            f"{bio_probs[bio_pred] * 100:.1f}% confidence",
        )

        m3.metric("Difference", difference_result)
        m4.metric("Fusion Insight", insight_title)
        m5.metric("Purchase Intent", purchase_title)

        st.subheader("Affective Grouping")
        st.write(
            f"Video group: **{video_group.title()}** | "
            f"Biosignal group: **{bio_group.title()}**"
        )

        st.subheader("Fusion Insight")
        if match:
            st.success(insight_title)
        else:
            st.warning(insight_title)
        st.write(insight_text)

        st.subheader("Estimated Purchase Intent")
        if "Likely" in purchase_title:
            st.success(purchase_title)
        elif "Unlikely" in purchase_title:
            st.error(purchase_title)
        else:
            st.info(purchase_title)
        st.write(purchase_text)

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.subheader("Video Emotion Confidence")
            video_df = probability_table(video_probs)
            st.bar_chart(video_df.set_index("Emotion")["Confidence"])
            st.dataframe(video_df, use_container_width=True, hide_index=True)

        with chart_right:
            st.subheader("Biosignal Emotion Confidence")
            bio_df = probability_table(bio_probs)
            st.bar_chart(bio_df.set_index("Emotion")["Confidence"])
            st.dataframe(bio_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(str(exc))

st.divider()

st.subheader("Submission Notes")
st.write(
    "Saved PyTorch models are reused from `.pt` files, so retraining is not required during the demo. "
    "The biosignal model shown here is the improved sliding-window 1D CNN trained directly on the full "
    "4-Hertz biosignal dataset. The final fusion output uses a rule-based affective grouping layer, "
    "which achieved 83.78% accuracy against the original seven-class match/mismatch labels. "
    "Estimated purchase intent is rule-based because the dataset does not contain actual purchase/no-purchase labels."
)

if METADATA_PATH.exists():
    with st.expander("View Saved Metadata"):
        st.json(json.loads(METADATA_PATH.read_text()))
