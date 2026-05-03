
# NeuroBioSense: Emotion Dissonance Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## Overview

NeuroBioSense is a multimodal deep learning system that analyzes advertisement videos and viewer biosignal responses to detect:

- Video emotion shown in the advertisement
- Viewer emotion predicted from biosignals
- Emotional match or mismatch
- Fusion-based emotional insight
- Estimated purchase intent

The system integrates:

- A 3D CNN for video emotion recognition
- A 1D CNN for biosignal emotion recognition
- A final rule-based fusion layer for emotional dissonance analysis
- A Streamlit frontend for project demonstration

---

## Problem Statement

Modern advertisements are designed to create specific emotional reactions, but the emotion shown in an advertisement may not always match the viewer's real physiological response.

This project aims to answer:

1. What emotion does the advertisement convey?
2. What emotion does the viewer actually feel based on biosignals?
3. Is there emotional alignment or emotional dissonance?
4. Based on the emotional response, is the viewer likely to buy the product?

---

## System Architecture

### Video Emotion Model

Advertisement Video → 3D CNN → Video Emotion

- Model: Pretrained R3D-18 + Addition of final layers for emotion detection
- Input: Advertisement video frames
- Output classes: Neutral, Anger, Disgust, Fear, Joy, Sadness, Surprise
- Captures spatial and temporal features from video clips

---

### Biosignal Emotion Model

BVP, EDA, TEMP, X, Y, Z → 1D CNN → Biosignal Emotion

- Uses physiological biosignal data
- Trained using sliding-window samples from the full 4-Hertz biosignal dataset
- Improved the number of biosignal samples from linked video slices to 37,089 windows
- Captures emotional patterns from physiological signals

---

### Final Rule-Based Fusion Layer

Video Emotion + Biosignal Emotion → Affective Grouping → Match / Mismatch → Insight

Instead of comparing all seven emotions directly, the final fusion layer groups emotions into broader affective categories:

| Emotion | Fusion Group |
|---|---|
| Neutral | Neutral |
| Joy | Positive |
| Anger | Negative / High-Arousal |
| Disgust | Negative / High-Arousal |
| Fear | Negative / High-Arousal |
| Sadness | Negative / High-Arousal |
| Surprise | Negative / High-Arousal |

This improved the fusion accuracy because biosignals are better at detecting broad emotional states than exact emotion categories.

Example:

```text
Video Emotion: Joy
Biosignal Emotion: Fear
Difference: Mismatch
Fusion Insight: Fake Happiness
```

---

## Purchase Intent Estimation

The dataset does not contain actual purchase/no-purchase labels. Therefore, purchase intent is estimated using a rule-based layer based on emotional dissonance.

Rules used:

- Positive video emotion + positive biosignal emotion → Likely to Buy
- Positive video emotion + negative biosignal emotion → Unlikely to Buy
- Negative biosignal response → Low purchase intent
- Neutral response → Uncertain or moderate purchase intent

This is not a supervised purchase prediction model. It is an interpretable estimate based on the viewer's emotional response.

---

## Model Performance

| Component | Accuracy |
|---|---:|
| Video 3D CNN | 49.8% |
| Biosignal 1D CNN | 40.7% |
| Original Learned Fusion Model | 74.5% |
| Final Rule-Based Fusion Layer | 83.78% |

The original learned fusion model achieved 74.5% accuracy but was biased toward the majority mismatch class. To improve reliability and explainability, the final system uses a rule-based affective grouping layer, which achieved 83.78% accuracy.

---
## Frontend Features

The Streamlit frontend displays:

- Uploaded advertisement video
- Uploaded biosignal slice CSV
- Video emotion prediction
- Biosignal emotion prediction
- Match / mismatch result
- Fusion insight
- Estimated purchase intent
- Confidence charts
- Saved model status

---

## Installation

Install required dependencies:

```bash
pip install streamlit torch torchvision opencv-python numpy pandas
```

---

## How to Run

Place the frontend file and saved model files in the same folder.

Required files:

```text
neurobiosense_frontend.py
video_3dcnn_emotion.pt
bio_1dcnn_sliding_window.pt
bio_sliding_mean.npy
bio_sliding_std.npy
fusion_3dcnn_1dcnn_difference.pt
final_project_metadata.json
```

Run the app:

```bash
python -m streamlit run neurobiosense_frontend.py
```

Then open:

```text
http://localhost:8501
```

---

## Outputs

The system produces:

- Video Emotion
- Biosignal Emotion
- Match / Mismatch
- Fusion Insight
- Estimated Purchase Intent
- Emotion Confidence Charts

---

## Data Linking Method

The original video-to-biosignal linking used a deterministic hash-based method:

```text
video_id → hash → biosignal slice
```

This ensures reproducibility, but it is not the same as real timestamp-based alignment. This is noted as a limitation of the project.

---

## Limitations

- The dataset does not contain purchase intent labels.
- Purchase intent is estimated using rules, not supervised learning.
- Biosignal and video alignment is hash-based rather than timestamp-based.
- Some emotion classes are underrepresented, especially Fear and Sadness.
- The final fusion layer is rule-based, but this makes it more explainable for project demonstration.

---

## Conclusion

NeuroBioSense detects emotional dissonance between advertisement content and viewer biosignal response. The final system combines a 3D CNN video model, a 1D CNN biosignal model, and an explainable rule-based fusion layer. The fusion layer achieved 83.78% accuracy and provides useful insights such as emotional match/mismatch, fake happiness, and estimated purchase intent.
```
