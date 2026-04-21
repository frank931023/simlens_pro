# Methodology

## System Architecture Overview

We propose the SimLens system with a three-layer architecture designed to achieve traceable and interpretable short video recommendation simulation. The core concept of the entire system is to separate "objective video features" and "user behavior weights," establishing a clear mathematical relationship between them. Unlike traditional black-box neural network approaches, SimLens guarantees 100% traceability, allowing every prediction result to be decomposed into specific feature contributions. The system architecture is shown in Figure 1.

```
┌─────────────────────────────────────────────────────────────┐
│                    SimLens System Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 1: Objective Feature Extraction                       │
│  ┌───────────────────────────────────────────────────┐      │
│  │ Input: Video Segment                              │      │
│  │ Process: Automated Feature Calculation            │      │
│  │ Output: Standardized Feature Scores (1-10)        │      │
│  │   - ASD: Average Shot Duration                    │      │
│  │   - SCR: Shot Change Rate                         │      │
│  │   - OFM: Optical Flow Magnitude                   │      │
│  │   - SD: Speech Density                            │      │
│  │   - CC: Content Complexity                        │      │
│  └───────────────────────────────────────────────────┘      │
│                          ↓                                    │
│  Layer 2: User Weight Learning                               │
│  ┌───────────────────────────────────────────────────┐      │
│  │ Input: MicroLens-100K User Watch History          │      │
│  │ Process: Linear Regression for User-Specific      │      │
│  │          Weight Learning                          │      │
│  │ Output: Feature Weight Vector                     │      │
│  │   - ASD Weight: w₁                                │      │
│  │   - SCR Weight: w₂                                │      │
│  │   - OFM Weight: w₃                                │      │
│  │   - SD Weight: w₄                                 │      │
│  │   - CC Weight: w₅                                 │      │
│  └───────────────────────────────────────────────────┘      │
│                          ↓                                    │
│  Layer 3: Retention Score Calculation                        │
│  ┌───────────────────────────────────────────────────┐      │
│  │ Retention_Score = Σ(Feature_i × Weight_i)         │      │
│  │ Output: Retention Rate Prediction Score           │      │
│  └───────────────────────────────────────────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

Figure 1: SimLens Three-Layer Architecture Diagram

## Layer 1: Objective Video Feature Extraction

Layer 1 is responsible for automatically extracting five objective, computable features from videos. The selection of these features is based on research literature in the video analysis field, and each feature can be fully automated through open-source tools without requiring manual annotation or subjective evaluation.

### 1.1 Average Shot Duration (ASD)

**Definition**: The average duration (in seconds) of all shots in a video.

**Calculation Method**: Automatic scene detection using PySceneDetect. PySceneDetect identifies shot boundaries by analyzing content differences between adjacent frames.

**Mathematical Formula**:
```
ASD = (T_total) / (N_shots)
```
Where:
- T_total: Total video duration (seconds)
- N_shots: Total number of detected shots

**Normalization**: Map raw ASD values to a 1-10 scale:
```
ASD_score = 10 - min(9, (ASD - 1) / 1.0)
```

**Score Interpretation**:
- 1-3 points: Long shots (ASD > 8 seconds), slow pace
- 4-6 points: Medium shot length (ASD 4-8 seconds), moderate pace
- 7-10 points: Short shots (ASD < 4 seconds), fast pace

**Research Basis**: Film studies show that average shot duration has decreased from 12 seconds in the 1930s to 2-4 seconds in modern times, reflecting changes in audience preferences for visual pacing.

### 1.2 Shot Change Rate (SCR)

**Definition**: The number of shot transitions per minute.

**Calculation Method**: Directly calculated based on ASD.

**Mathematical Formula**:
```
SCR = 60 / ASD
```

**Normalization**: Map SCR to a 1-10 scale:
```
SCR_score = min(10, (SCR - 2) / 2.8)
```

**Score Interpretation**:
- 1-3 points: Low change rate (< 6 cuts/min), minimal visual variation
- 4-6 points: Medium change rate (6-15 cuts/min), moderate visual variation
- 7-10 points: High change rate (> 15 cuts/min), frequent visual variation

**Research Basis**: High change rates are associated with maintaining audience attention, and fast-paced editing can enhance the appeal of short videos.

### 1.3 Optical Flow Magnitude (OFM)

**Definition**: The intensity of motion between video frames, reflecting the degree of object or camera movement in the scene.

**Calculation Method**: Calculate optical flow vectors between adjacent frames using OpenCV's Farneback optical flow algorithm.

**Mathematical Formula**:
```
OFM = (1/N) Σ sqrt(u²_i + v²_i)
```
Where:
- N: Total number of pixels
- u_i, v_i: Horizontal and vertical optical flow components of pixel i

**Normalization**: Map OFM to a 1-10 scale:
```
OFM_score = min(10, OFM / 10)
```

**Score Interpretation**:
- 1-3 points: Static scene (OFM < 30), almost no motion
- 4-6 points: Medium motion (OFM 30-70), moderate object or camera movement
- 7-10 points: High-intensity motion (OFM > 70), rapid movement or dramatic camera changes

**Research Basis**: Motion intensity is an important indicator of visual stimulation, affecting audience attention and engagement.

### 1.4 Speech Density (SD)

**Definition**: The proportion of time in a video that contains speech.

**Calculation Method**: Detect speech segments using the Whisper speech recognition model.

**Mathematical Formula**:
```
SD = (T_speech / T_total) × 100%
```
Where:
- T_speech: Total duration containing speech (seconds)
- T_total: Total video duration (seconds)

**Normalization**: Map SD to a 1-10 scale:
```
SD_score = SD × 10
```

**Score Interpretation**:
- 1-3 points: Low speech density (< 30%), mainly background music or ambient sound
- 4-6 points: Medium speech density (30-60%), speech mixed with other audio
- 7-10 points: High speech density (> 60%), continuous speech content

**Research Basis**: Speech density reflects the intensity of information delivery; high speech density typically indicates high information content.

### 1.5 Content Complexity (CC)

**Definition**: The diversity and richness of visual content in a video.

**Calculation Method**: Extract visual feature vectors from video frames using the CLIP model, and calculate the standard deviation of feature vectors as a complexity indicator.

**Mathematical Formula**:
```
CC = std(CLIP_features)
```
Where:
- CLIP_features: Set of CLIP feature vectors extracted from video frames
- std: Standard deviation function

**Normalization**: Map CC to a 1-10 scale:
```
CC_score = min(10, CC × 20)
```

**Score Interpretation**:
- 1-3 points: Low complexity (CC < 0.2), uniform visual content
- 4-6 points: Medium complexity (CC 0.2-0.4), some variation in visual content
- 7-10 points: High complexity (CC > 0.4), rich and diverse visual content

**Research Basis**: Visual diversity reflects content richness, affecting audience visual interest.

### 1.6 Tool and Implementation Specifications

All feature extraction tools are open-source and publicly available, ensuring research reproducibility and transparency.

**PySceneDetect**:
- Version: 0.6.x or higher
- Detection Method: ContentDetector
- Threshold: 27.0 (default value)
- Open-Source License: BSD 3-Clause

**OpenCV**:
- Version: 4.x
- Optical Flow Algorithm: cv2.calcOpticalFlowFarneback()
- Parameters: pyr_scale=0.5, levels=3, winsize=15
- Open-Source License: Apache 2.0

**Whisper**:
- Model: base or small
- Language: Auto-detect
- Task: transcribe
- Open-Source License: MIT

**CLIP**:
- Model: ViT-B/32
- Sampling Rate: 1 frame per second
- Feature Dimension: 512
- Open-Source License: MIT

**Computational Requirements**:
- CPU: 8 cores or more
- GPU: NVIDIA GPU recommended (CUDA support)
- Memory: 16GB or more
- Processing Speed: Approximately 1-2 minutes per video (depending on video length and hardware configuration)

## Layer 2: User Weight Learning

Layer 2 is responsible for learning user-specific feature weights from real user behavior data. Unlike traditional rule-based or expert knowledge approaches, we directly learn preference patterns from users' viewing history.

### 2.1 Data Source: MicroLens-100K

**Dataset Statistics**:
- Number of Users: 100,000
- Number of Videos: 19,738
- Interaction Records: 719,405
- Key Information: Each interaction record contains user ID, video ID, and watch time

**Data Preprocessing**:
1. Filter out records with watch time < 1 second (likely accidental touches)
2. Calculate retention rate for each record: retention = watch_time / video_duration
3. Extract five features for each video: [ASD, SCR, OFM, SD, CC]

### 2.2 Weight Learning Method

**Problem Definition**: For user u, learn weight vector w_u = [w₁, w₂, w₃, w₄, w₅] such that predicted retention rate is closest to actual retention rate.

**Mathematical Model**: Linear Regression
```
retention_predicted = w₁×ASD + w₂×SCR + w₃×OFM + w₄×SD + w₅×CC + b
```

**Training Process**:
1. Split user's viewing history into training set (80%) and validation set (20%)
2. Solve for weights using least squares method:
```
w_u = argmin Σ(retention_actual - retention_predicted)²
```
3. Evaluate performance on validation set

**Regularization**: To prevent overfitting, use L2 regularization:
```
w_u = argmin [Σ(retention_actual - retention_predicted)² + λ||w||²]
```
Where λ = 0.01

**Alternative Methods**: Besides linear regression, the following can also be used:
- Ridge Regression: Stronger L2 regularization
- Lasso Regression: L1 regularization, can produce sparse weights
- Gradient Boosting: Capture non-linear relationships

### 2.3 Weight Interpretation

The learned weight vector reflects user content preferences:

**Positive Weights (w_i > 0)**: The higher this feature value, the more likely the user will watch for longer
- Example: w_SCR = 0.8 indicates user prefers fast-paced videos with high change rates

**Negative Weights (w_i < 0)**: The higher this feature value, the more likely the user will leave early
- Example: w_ASD = -0.5 indicates user dislikes slow-paced videos with long shots

**Zero Weights (w_i ≈ 0)**: This feature has little impact on user's viewing decisions
- Example: w_SD = 0.1 indicates user is insensitive to speech density

### 2.4 Cold Start Handling

For new users (no viewing history), use the following strategies:

**Method 1: Global Average Weights**
```
w_new = (1/N) Σ w_u
```
Use average weights of all users as initial values

**Method 2: Cluster Weights**
1. Perform K-means clustering on existing users (based on weight vectors)
2. New users use weights from the nearest cluster center

**Method 3: Demographic Mapping**
If user age, gender, and other information are available, use average weights from similar demographic groups

## Layer 3: Retention Score Calculation

Layer 3 combines video features and user weights to calculate the final retention rate prediction score.

### 3.1 Retention Score Calculation Formula

For user u watching video v, the retention rate score is calculated as follows:

```
Retention_Score = Σ(Feature_i × Weight_i) + b
                = w₁×ASD + w₂×SCR + w₃×OFM + w₄×SD + w₅×CC + b
```

Where:
- Feature_i: Normalized score (1-10) of video v on the i-th feature
- Weight_i: Weight of user u on the i-th feature
- b: Bias term

### 3.2 Score Normalization

To make scores more interpretable, we normalize raw scores to the [0, 1] interval:

```
Retention_Score_normalized = sigmoid(Retention_Score_raw)
                           = 1 / (1 + e^(-Retention_Score_raw))
```

**Score Interpretation**:
- 0.0-0.3: Low retention, user may skip quickly
- 0.3-0.7: Medium retention, user will watch but may not finish
- 0.7-1.0: High retention, user is very likely to watch completely

### 3.3 Temporal Analysis

For long videos, we can segment them into multiple clips (e.g., one clip every 10 seconds), calculate retention rate for each clip separately, and generate a retention rate curve:

```
Retention_Curve = [Score_t1, Score_t2, ..., Score_tn]
```

This enables us to:
1. Identify peaks and valleys in the video
2. Predict when users are likely to leave
3. Provide optimization suggestions for content creators

### 3.4 Traceability Advantage

The core advantage of SimLens is 100% traceability. Every retention rate prediction can be fully decomposed into contributions from individual features:

**Example Decomposition**:
```
Video A for User B Retention Rate Prediction: 0.75 (High Retention)

Detailed Calculation:
- ASD Contribution: 8 × 0.3 = 2.4
- SCR Contribution: 9 × 0.8 = 7.2  ← Main positive factor
- OFM Contribution: 7 × 0.5 = 3.5
- SD Contribution: 4 × (-0.2) = -0.8
- CC Contribution: 6 × 0.4 = 2.4
- Bias Term: 1.0
Total: 15.7 → sigmoid(15.7) ≈ 0.75

Explanation: User B highly values shot change rate (weight 0.8), and Video A has 
a very high change rate (9 points), thus predicting high retention. The negative 
contribution from speech density is offset by positive contributions from other features.
```

**Comparison with Black-Box Methods**:

| Aspect | SimLens | Deep Neural Networks |
|--------|---------|---------------------|
| Interpretability | Fully traceable | Black box |
| Feature Contribution | Clearly quantified | Cannot decompose |
| Debugging Capability | Can identify problem features | Difficult to debug |
| User Trust | High (explainable) | Low (opaque) |
| Computational Cost | Low | High |

### 3.5 Practical Application Scenarios

**Scenario 1: Content Creator Optimization**
After a creator uploads a video, the system analyzes:
- "Your video has low retention (0.3) at 15-25 seconds, mainly due to excessively low shot change rate (3 points)"
- "Suggestion: Increase scene transitions or add dynamic elements in this segment"

**Scenario 2: Personalized Recommendation**
When the recommendation system recommends videos to users:
- Calculate user's predicted retention rate for candidate videos
- Prioritize recommending high retention rate (> 0.7) videos
- Avoid recommending low retention rate (< 0.3) videos

**Scenario 3: User Behavior Analysis**
Platform analyzes user groups:
- "Young users (18-25 years) have average SCR weight of 0.9, preferring fast-paced content"
- "Middle-aged users (35-50 years) have average SD weight of 0.7, preferring information-dense content"

## Overall System Workflow

### 4.1 Offline Phase

**Step 1: Video Feature Extraction**
- Extract five features for all videos in the video library
- Store feature vectors in database
- Time Complexity: O(N_videos)

**Step 2: User Weight Learning**
- Learn weights from viewing history for each user
- Store weight vectors in database
- Time Complexity: O(N_users × N_history)

### 4.2 Online Phase

**Step 1: Retrieve Candidate Videos**
- Obtain candidate set using collaborative filtering or content filtering
- Candidate Set Size: 100-1000 videos

**Step 2: Retention Rate Prediction**
- Calculate retention rate score for each candidate video
- Time Complexity: O(N_candidates) - very fast

**Step 3: Ranking and Recommendation**
- Sort by retention rate score in descending order
- Return Top-K videos (K = 10-50)

### 4.3 Performance Optimization

**Feature Caching**: Pre-compute and cache feature vectors for all videos

**Weight Updates**: Periodically (daily or weekly) update user weights, rather than real-time updates

**Batch Computation**: Use matrix operations to batch calculate retention rates for multiple videos:
```
Scores = Features_matrix × Weights_vector
```

**Distributed Computing**: Feature extraction can be parallelized to support large-scale video libraries

## Summary

SimLens' three-layer architecture achieves the following goals:

1. **Objectivity**: Layer 1 uses fully automated feature extraction without subjective evaluation
2. **Data-Driven**: Layer 2 learns weights from real user behavior data rather than relying on assumptions
3. **Traceability**: Layer 3 ensures every prediction can be fully decomposed and explained
4. **Scalability**: The entire system can efficiently handle large-scale video and user data
5. **Practicality**: Provides actionable insights for content creators and recommendation systems

This design makes SimLens a scientifically rigorous and practically feasible video recommendation simulation system.
