#  Semantic Segmentation for Drone Landing Zone Detection


<div align="center">
  <img src="./1.png" alt="Project Teaser" width="800"/>
  <br>
  <em>Real-time semantic segmentation for autonomous drone landing</em>
</div>
---

##  Project Overview

This project implements a **real-time semantic segmentation system** for autonomous drone landing zone detection using **Knowledge Distillation** and **Quantization** techniques. The system identifies 23 different classes including trees, grass, water, paved areas, buildings, vehicles, and obstacles from aerial drone imagery.


### Key Features

-  **Real-time segmentation** at 30-50 FPS
-  **Lightweight model** (3 MB) for edge deployment
-  **23-class** semantic segmentation
-  **Flask web interface** with camera support
-  **Knowledge Distillation** for model compression
-  **Dynamic Quantization** for fast inference

---

##  Why This Project

### The Problem

| Challenge | Impact |
|-----------|--------|
| Drones need safe landing zones | Manual landing is unsafe in unknown terrain |
| Drones have limited compute power | Large models won't run on edge devices |
| Real-time decisions needed | Slow inference = crashed drones |
| Varied terrain types | Need to identify many different surfaces |

### Our Solution

| Solution | Benefit |
|----------|---------|
| **Semantic Segmentation** | Identifies safe vs unsafe zones |
| **Knowledge Distillation** | 45 MB → 8 MB (5.6x smaller) |
| **Dynamic Quantization** | 8 MB → 3 MB (2.7x smaller) |
| **2 ms Inference** | 30-50 FPS real-time |
| **23 Classes** | Covers all terrain types |

---

##  Dataset

### Aerial Semantic Segmentation Drone Dataset

- **Total Images:** 400
- **Resolution:** Various (resized to 256×256 for training)
- **Classes:** 23 different classes
- **Source:** Kaggle - Semantic Drone Dataset

### The 23 Classes

| # | Class | Color | # | Class | Color | # | Class | Color |
|---|-------|-------|---|-------|-------|---|-------|-------|
| 1 |  Tree |  | 9 |  Pool |  | 17 | Fence-pole |  |
| 2 |  Grass |  | 10 |  Person |  | 18 | Window |  |
| 3 |  Vegetation |  | 11 |  Dog |  | 19 | Door |  |
| 4 |  Dirt |  | 12 |  Car |  | 20 |  Obstacle |  |
| 5 | Gravel |  | 13 |  Bicycle |  | 21-23 | Additional |  |
| 6 | Rocks |  | 14 |  Roof |  | | |
| 7 |  Water |  | 15 |  Wall |  | | |
| 8 |  Paved Area |  | 16 | Fence | | | |


---





#  COMPLETE TRAINING PIPELINE

```text
                        COMPLETE TRAINING PIPELINE

        ┌──────────────────────────────────────────────┐
        │              PHASE 1                         │
        │          DATA PREPARATION                    │
        └──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────┐
        │           Aerial Drone Dataset               │
        │                                              │
        │      • 400 RGB Images                        │
        │      • 23 Semantic Classes                   │
        │      • Pixel-wise Annotations                │
        └──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────┐
        │            Data Augmentation                 │
        │                                              │
        │  • Horizontal Flip                           │
        │  • Vertical Flip                             │
        │  • Random Rotation                           │
        │  • Brightness Adjustment                     │
        │  • Contrast Adjustment                       │
        └──────────────────────────────────────────────┘
                        │
                        ▼

        ┌──────────────────────────────────────────────┐
        │              PHASE 2                         │
        │          TEACHER TRAINING                    │
        └──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────┐
        │         Teacher Model (VGG-UNet)             │
        │                                              │
        │ Input : 256 × 256 × 3                        │
        │                                              │
        │ Encoder (VGG16)                              │
        │   64 → 128 → 256 → 512 → 512                │
        │                                              │
        │ Decoder (UNet)                               │
        │   512 → 256 → 128 → 64 → 23                 │
        │                                              │
        │ Output : 256 × 256 × 23 Softmax              │
        │                                              │
        │ Model Size : 45 MB                           │
        │ Speed      : 15 ms/image                     │
        │ Accuracy   : Baseline (~100%)                │
        │ Epochs     : 30                              │
        └──────────────────────────────────────────────┘
                        │
                        ▼

        ┌──────────────────────────────────────────────┐
        │              PHASE 3                         │
        │       KNOWLEDGE DISTILLATION                 │
        └──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────┐
        │          Distillation Process                │
        │                                              │
        │ Teacher predicts Soft Labels                │
        │                  │                           │
        │                  ▼                           │
        │ Student learns from:                        │
        │                                              │
        │ 1. Ground Truth Labels                      │
        │ 2. Teacher Soft Predictions                 │
        │                                              │
        │ Total Loss =                                │
        │ α × Soft Loss + (1-α) × Hard Loss           │
        │                                              │
        │ Soft Loss = KL Divergence                   │
        │ Hard Loss = Cross Entropy                   │
        │ α = 0.7                                     │
        └──────────────────────────────────────────────┘
                        │
                        ▼

        ┌──────────────────────────────────────────────┐
        │              PHASE 4                         │
        │         STUDENT TRAINING                     │
        └──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────┐
        │      Student Model (Lightweight UNet)        │
        │                                              │
        │ Input : 256 × 256 × 3                        │
        │                                              │
        │ Encoder                                      │
        │   32 → 64 → 128 → 256                       │
        │                                              │
        │ Decoder                                      │
        │   256 → 128 → 64 → 32 → 23                  │
        │                                              │
        │ Output : 256 × 256 × 23 Softmax              │
        │                                              │
        │ Model Size : 8 MB                            │
        │ Speed      : 5 ms/image                      │
        │ Accuracy   : ~95% of Teacher                 │
        │ Epochs     : 25                              │
        └──────────────────────────────────────────────┘
                        │
                        ▼

        ┌──────────────────────────────────────────────┐
        │              PHASE 5                         │
        │             QUANTIZATION                     │
        └──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────┐
        │         Dynamic Quantization                 │
        │                                              │
        │ Student Model (Float32)                      │
        │               │                              │
        │               ▼                              │
        │ Weights : Float32 → INT8                     │
        │ Activations : Float32                        │
        │                                              │
        │ Quantized Model                              │
        │                                              │
        │ Size  : 8 MB → 3 MB                          │
        │ Speed : 5 ms → 2 ms                          │
        │ Accuracy : ~95% of Teacher                   │
        └──────────────────────────────────────────────┘
                        │
                        ▼

        ┌──────────────────────────────────────────────┐
        │              PHASE 6                         │
        │              DEPLOYMENT                      │
        └──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────┐
        │        student_dynamic_quant.tflite          │
        │                                              │
        │ Size      : 3 MB                             │
        │ Speed     : 2 ms/image                       │
        │ FPS       : 30–50                            │
        │ Accuracy  : ~95% of Teacher                  │
        │ Platform  : Flask Web Application            │
        │                                              │
        │ Features                                     │
        │ • Image Upload                               │
        │ • Camera Support                             │
        │ • Real-time Segmentation                     │
        └──────────────────────────────────────────────┘
```

---

#  Pipeline Summary

| Phase | Description |
|--------|-------------|
| **Phase 1** | Dataset preparation and augmentation |
| **Phase 2** | Train high-capacity VGG-UNet teacher model |
| **Phase 3** | Transfer teacher knowledge to student network |
| **Phase 4** | Train lightweight UNet student model |
| **Phase 5** | Quantize model to INT8 for faster inference |
| **Phase 6** | Deploy optimized TensorFlow Lite model |

---

#  Final Model

| Property | Value |
|----------|-------|
| Model | `student_dynamic_quant.tflite` |
| Size | **3 MB** |
| Speed | **2 ms/image** |
| FPS | **30–50 FPS** |
| Accuracy | **~95% of Teacher Model** |
| Framework | TensorFlow Lite |
| Deployment | Flask Web Application |

---

#  Key Advantages

- Lightweight deployment model (3 MB)
- Real-time semantic segmentation
- Faster inference using INT8 quantization
- Knowledge distillation preserves high accuracy
- Suitable for edge devices and embedded systems
- Supports image upload and live camera inference
##  Model Performance

| Model | Size | Speed | IoU | FPS |
|-------|------|-------|-----|-----|
| Teacher (VGG-UNet) | 45 MB | 15 ms | 0.72 | 5-10 |
| Student (Distilled) | 8 MB | 5 ms | 0.67 | 15-20 |
| **Dynamic Quantized** | **3 MB** | **2 ms** | **0.65** | **30-50** |



---

## **Project Explanation **

# **Project Overview: Semantic Segmentation for Drone Landing Zone Detection**

This project develops a **real-time semantic segmentation system** for autonomous drone landing using **Knowledge Distillation** and **Quantization**. The system processes aerial images to identify 23 different classes including trees, grass, water, buildings, and obstacles, enabling safe landing zone detection.

The **training pipeline** uses a VGG-UNet as a **Teacher model** (45 MB) trained on 400 drone images. Through **Knowledge Distillation**, a lightweight **Student model** (8 MB) learns to mimic the Teacher's behavior while being 5x smaller. Finally, **Dynamic Quantization** compresses the Student to a **3 MB TFLite model** that runs at 2-3 ms inference speed (30-50 FPS) on standard CPU.

The **deployment system** includes a Flask web application with camera support, allowing real-time segmentation from webcam or uploaded images. The model identifies safe landing zones by segmenting the scene into 23 classes, providing instant visual feedback with colored overlays.

**Key Achievements:**
-  **15x smaller** than original (45 MB → 3 MB)
-  **5-7x faster** inference (15 ms → 2 ms)
-  **90-95%** accuracy retention
-  **Real-time** 30-50 FPS processing
-  **Edge-ready** deployment with Flask

The system successfully demonstrates how large, accurate models can be compressed for real-time drone deployment without significant accuracy loss, making autonomous landing safer and more reliable. 

























#  PHASAL AI - Plant Disease Detection with Reinforcement Learning

An intelligent plant disease detection system that combines Convolutional Neural Networks (CNN) with Soft Actor-Critic (SAC) reinforcement learning to provide optimal spray recommendations for crop diseases.

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Reinforcement Learning](#reinforcement-learning)
- [Reward System](#reward-system)
- [Retraining Process](#retraining-process)
- [Database Schema](#database-schema)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

##  Overview

PHASAL AI is a complete end-to-end solution for plant disease management that:
- **Detects** diseases from leaf images using a CNN model (39 disease classes)
- **Recommends** optimal spray amounts using SAC reinforcement learning
- **Learns** from user feedback through a reward system
- **Improves** continuously with real-world data

### Key Metrics
- **39** Disease classes detected
- **30,000+** Training scenarios
- **500ml** Maximum spray recommendation
- **7 days** Reward evaluation period
- **10+** Experiences trigger retraining

##  Features

-  **Image Upload & Disease Detection**
-  **SAC Reinforcement Learning Agent**
-  **Before/After Reward System**
-  **Continuous Retraining**
-  **Weather Data Integration**
-  **Responsive Web Interface**
-  **SQLite Database Storage**
-  **Real-time Recommendations**

##  System Architecture


┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASAL AI ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   User Uploads   │
                              │   Leaf Image     │
                              └────────┬─────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │     CNN Disease Detection    │
                        │  (TensorFlow/Keras Model)    │
                        └──────────────┬───────────────┘
                                       │
                              ┌────────┴────────┐
                              │                 │
                              ▼                 ▼
                ┌──────────────────┐  ┌──────────────────┐
                │  Single Detection│  │  Reward System   │
                │  Mode            │  │  (Before/After)  │
                └────────┬─────────┘  └────────┬─────────┘
                         │                     │
                         ▼                     ▼
                ┌──────────────────┐  ┌──────────────────┐
                │   SAC Agent      │  │  Store Experience│
                │   (Spray Amount) │  │  (State,Action,  │
                └────────┬─────────┘  │   Reward,Next)   │
                         │            └────────┬─────────┘
                         │                     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Retraining Engine  │
                         │   (Update Model)     │
                         └──────────────────────┘


---

# Complete Pipeline

## Phase 1 — Data Collection

```text
                      User Uploads Image
                              │
                              ▼

        ┌──────────────────────┐      ┌────────────────────────────┐
        │ Single Detection     │      │ Reward Mode               │
        ├──────────────────────┤      ├────────────────────────────┤
        │ Upload Image         │      │ Upload Before Image        │
        │ CNN Detect Disease   │      │ Spray Recommendation       │
        │ SAC Recommend Spray  │      │ Apply Spray               │
        │ Display Result       │      │ Upload After Image         │
        └──────────────────────┘      │ Calculate Reward          │
                                      └────────────────────────────┘
```

### Single Detection Workflow

1. User uploads a leaf image.
2. CNN model detects the disease.
3. Environmental parameters are collected.
4. SAC agent predicts spray quantity.
5. Recommendation is displayed.

### Reward Workflow

1. Upload infected leaf.
2. Receive spray recommendation.
3. Apply spray.
4. Wait approximately 7 days.
5. Upload healed leaf.
6. System calculates reward.
7. Experience is stored for reinforcement learning.

---

#  Stage 2 — Reinforcement Learning Processing

## State Vector

The SAC agent receives a **6-dimensional state vector**.

```text
State = [
    disease_id,
    severity,
    temperature,
    humidity,
    soil_moisture,
    time_since_last_spray
]
```

Example

```text
[4, 0.73, 28, 65, 45, 0]
```

---

## SAC Agent

```text
             STATE
               │
               ▼
      ┌─────────────────────┐
      │ Soft Actor Critic   │
      ├─────────────────────┤
      │ Actor Network       │
      │ Critic Network      │
      └─────────┬───────────┘
                ▼
        Spray Recommendation
```

### Actor Network

Input

- 6-dimensional state vector

Output

- Normalized spray amount (0–1)

---

### Critic Network

Input

- State + Action

Output

- Q-value (expected reward)

---

## Spray Amount Calculation

### Step 1 — Disease Detection

```text
Disease      : Tomato Late Blight
Severity     : 0.73
Confidence   : 0.94
```

### Step 2 — Sensor Data

```text
Temperature     : 28.5°C
Humidity        : 65.2%
Soil Moisture   : 45.8%
```

### Step 3 — Build State

```text
[4, 0.73, 28.5, 65.2, 45.8, 0]
```

### Step 4 — Prediction Sources

| Method | Weight |
|---------|-------:|
| SAC Agent | 60% |
| Rule-based Expert System | 20% |
| Vector Database Similar Cases | 20% |

Example

| Source | Spray |
|---------|------:|
| SAC Agent | 180 ml |
| Rule-Based | 237 ml |
| Vector DB | 192 ml |

Final spray

```text
180×0.6 + 237×0.2 + 192×0.2

= 194 ml
```

Displayed to user

```text
Recommended Spray : 194 ml

Urgency : HIGH
Spray within 24 hours.
```

---

#  Stage 3 — Reward Calculation

## Reward Formula

```text
Reward

= Disease Reduction
- Waste Penalty
- Environmental Penalty
- Time Penalty
```

where

```text
Disease Reduction

= Severity × Spray Efficiency × 2
```

```text
Waste Penalty

= |Applied − Optimal| / 500 × 0.5
```

```text
Environmental Penalty

= max(0, Applied − Optimal) / 500 × 0.3
```

```text
Time Penalty

= 0.2
if spray is applied too early
```

---

## Reward Levels

| Improvement | Reward | Meaning |
|-------------|--------|---------|
| >60% | 2.0 | Excellent |
| 40–60% | 1.5 | Good |
| 20–40% | 1.0 | Fair |
| 0–20% | 0.5 | Poor |
| 0% | 0.0 | Neutral |
| <0% | -1.0 | Condition Worsened |

---

#  Stage 4 — Learning & Retraining

## Stored Experience

```json
{
  "state_before": [4,0.73,28,65,45,0],
  "action": 0.36,
  "reward": 1.5,
  "state_after": [4,0.25,27,68,55,0],
  "done": true
}
```

---

## Retraining Pipeline

```text
Experience Buffer
        │
        ▼
Collect ≥10 Experiences
        │
        ▼
Load Existing SAC Model
        │
        ▼
Fine Tune with Real Data
        │
        ▼
Save Improved Model
        │
        ▼
Deploy Updated Model
```

---

#  RL Training

## `train_sac_agent.py`

### Purpose

Trains the SAC agent entirely from simulation before deployment.

### Responsibilities

- Creates virtual crop environment
- Generates 30,000 disease scenarios
- Trains SAC agent
- Saves

```text
crop_sac_model.zip
```

### Simulated Parameters

- 39 disease classes
- Disease severity (30–95%)
- Temperature (15–35°C)
- Humidity (40–90%)
- Soil moisture (35–75%)
- Time since spray (0–168 hours)

---

#  Online Learning

## `retrain_rl_agent.py`

### Purpose

Improves the deployed RL model using real user feedback.

### Workflow

```text
Load Experiences
        │
        ▼
Load Existing SAC Model
        │
        ▼
Fine Tune
        │
        ▼
Save Improved Model
        │
        ▼
Deploy Updated Model
```

Model is retrained automatically after sufficient real-world experiences are collected.

---

#  Data Flow Architecture

```text
User Upload
      │
      ▼
CNN Disease Detection
      │
      ▼
State Builder
      │
      ▼
SAC Agent
      │
      ▼
Spray Recommendation
      │
      ▼
SQLite Database
      │
      ▼
Reward Calculation
      │
      ▼
Experience Buffer
      │
      ▼
Retraining Engine
      │
      ▼
Updated SAC Model
```

---

#  Databases

The system maintains three primary databases.

| Database | Purpose |
|----------|---------|
| `spray_logs` | Single detection history |
| `plant_sessions` | Before/After reward sessions |
| `rl_experiences` | Reinforcement learning experiences |

---

#  Technologies

- TensorFlow / Keras
- Soft Actor-Critic (Stable-Baselines3)
- SQLite
- Flask
- OpenCV
- NumPy
- Python
- Vector Database (Similarity Search)


#  What I Learned

Working on **PHASAL AI** and the **Semantic Segmentation Project** strengthened my understanding of computer vision, deep learning, and intelligent decision-making systems. Through these projects, I learned:

##  PHASAL AI

- How to integrate CNN-based disease detection with Reinforcement Learning for precision agriculture.
- How the Soft Actor-Critic (SAC) algorithm learns optimal spray recommendations through rewards and penalties.
- How to design state vectors using disease severity and environmental parameters.
- How reward engineering influences reinforcement learning performance.
- How experience replay enables continuous learning from real user feedback.
- How combining Reinforcement Learning, rule-based reasoning, and historical case retrieval improves recommendation quality.
- How to build an end-to-end AI pipeline that continuously improves through retraining.

---

##  Semantic Segmentation Project

- How semantic segmentation performs **pixel-level classification**, assigning a class label to every pixel in an image.
- How encoder-decoder architectures such as **VGG-UNet** capture both high-level features and fine spatial details.
- How skip connections help preserve object boundaries and improve segmentation accuracy.
- How Knowledge Distillation transfers knowledge from a large teacher model to a lightweight student model while maintaining high accuracy.
- How Dynamic Quantization reduces model size and inference time for deployment on resource-constrained devices.
- How data augmentation improves model generalization and robustness.
- How semantic segmentation models are trained, validated, and optimized for real-time applications.
- The trade-off between model accuracy, speed, and deployment efficiency.
- How TensorFlow Lite enables lightweight deployment on edge devices.

---

## 💡 Key Takeaways

- Pixel-level understanding provides richer information than image classification for real-world applications.
- Reinforcement Learning enables AI systems to improve through interaction and feedback.
- Knowledge Distillation and Quantization make deep learning models suitable for edge deployment.
- Combining multiple AI techniques results in more reliable and practical intelligent systems.
- Building end-to-end AI solutions requires integrating data processing, model training, optimization, deployment, and continuous learning into a single workflow.