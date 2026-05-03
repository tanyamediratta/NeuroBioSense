# NeuroBioSense: Emotion Dissonance Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## Overview

NeuroBioSense is a multimodal deep learning system that analyzes advertisement videos and viewer biosignal responses to detect:

- Emotional alignment
- Emotional dissonance
- Estimated purchase intent

The system integrates:
- A 3D CNN for video emotion recognition  
- A 1D CNN for biosignal emotion recognition  
- A rule-based fusion layer for behavioral insights  

---

## Problem Statement

Modern advertisements are designed to evoke emotions, but there is often a mismatch between **intended emotion** and **actual human response**.

This project aims to answer:

1. What emotion does the advertisement convey?
2. What emotion does the viewer actually feel?
3. Is there emotional alignment or dissonance?

---

## System Architecture

### Video Emotion Model
Advertisement Video → 3D CNN → Video Emotion

- Model: Pretrained R3D-18  
- Captures spatial and temporal features  

---

### Biosignal Emotion Model
BVP, EDA, TEMP, X, Y, Z → 1D CNN → Biosignal Emotion

- Uses sliding window training on biosignal data  
- Captures physiological emotional patterns  

---

### Fusion Insight Layer
Video Emotion + Biosignal Emotion → Match / Mismatch → Insight

Example:
Video Emotion: Joy
Biosignal Emotion: Fear
Difference: Mismatch
Insight: Fake Happiness

---

## Purchase Intent Estimation

Since no labeled dataset exists, purchase intent is estimated using rules:

- Positive + Positive → Likely to Buy  
- Positive + Negative → Unlikely to Buy  
- Neutral → Low / Uncertain Intent  

Note: This is not a supervised prediction.

---

## Model Performance

| Model                     | Accuracy |
|--------------------------|----------|
| Video 3D CNN             | 49.8%    |
| Biosignal 1D CNN         | 40.7%    |
| Fusion Difference Model  | 74.5%    |

Note: Fusion model showed class imbalance bias; final system uses direct comparison.


---

## Installation

Install required dependencies:

```bash
pip install streamlit torch torchvision opencv-python numpy pandas
```
---
## Outputs

Video Emotion  
Biosignal Emotion  
Match / Mismatch  
Fusion Insight  
Estimated Purchase Intent  
Confidence Charts  
---
### Data Linking Method

video_id → hash → biosignal slice
Ensures reproducibility
Not equivalent to real timestamp alignment
---
### Conclusion

NeuroBioSense highlights the gap between intended advertisement emotion and actual human response, enabling deeper insights into consumer behavior.
