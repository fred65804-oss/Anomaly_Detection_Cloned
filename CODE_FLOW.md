# Hybrid IDS - Complete Code Flow Explanation

## 🚀 Overview

The system has **TWO main flows**:
1. **TRAINING FLOW** - Build and train the model
2. **INFERENCE FLOW** - Use the trained model to make predictions

---

## 📊 PHASE 1: TRAINING FLOW

```
training/train.py (Entry Point)
    ↓
[STEP 1] Load Data
    ↓
[STEP 2] Create Binary Targets
    ↓
[STEP 3] Prepare Features
    ↓
[STEP 4] Train/Val Split
    ↓
[STEP 5] Train Pipeline
    ↓
artifacts/latest/ (Save Models)
```

---

### **STEP 1: Load Data**
**File:** `training/train.py` lines 85-110

```python
# Two modes:
# Mode A: Single file
df = pd.read_csv("training/UNSW-NB15.csv")
df_train, df_test = train_test_split(df, test_size=0.2)

# Mode B: Two separate files
df_train = pd.read_csv("training/UNSW-NB15_Train.csv")
df_test = pd.read_csv("training/UNSW-NB15_Test.csv")

# Result: 
# df_train.shape ≈ (2M rows, 49 columns)
# df_test.shape ≈ (500K rows, 49 columns)
```

**What happens:**
- Reads CSV file(s) from `training/` folder
- Normalizes column names (strips whitespace)
- If single file: splits 80/20 into train/test
- If two files: loads them separately

---

### **STEP 2: Create Binary Targets**
**File:** `training/train.py` lines 120-160

```python
# Auto-detect label column
label_col = "label"  # or "attack_class", "attack", etc.

# Create binary target (0 = normal, 1 = attack)
is_normal = (df_train[label_col] == "normal")
y_train = (~is_normal).astype(int)

# Result:
# y_train has only values: 0 (normal) or 1 (attack)
# Class distribution: ~80% normal, ~20% attacks
```

**What happens:**
- Finds the label column by searching for known names
- Detects what represents "normal" class
- Creates binary column: `is_intrusion = (label != "normal")`
- This is used for supervised learning

---

### **STEP 3: Prepare Features**
**File:** `training/train.py` lines 165-180

```python
X_train = df_train.drop(columns=[label_col])  # Features only
X_test = df_test.drop(columns=[label_col])

# Split train into train/val (80/20 of the 80%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,  # Keep class balance
    random_state=42
)

# Result:
# X_train: 1,625,629 rows × 49 columns (64% of original)
# X_val:     406,407 rows × 49 columns (16% of original)
# X_test:    508,010 rows × 49 columns (20% of original)
```

**What happens:**
- Separates features (X) from labels (y)
- Further splits train into 80% train + 20% validation
- Validation set is used for hyperparameter tuning
- Test set is held out for final evaluation

---

### **STEP 4: Train/Val Split is Already Done Above**

Now we have:
```
X_train (1.6M rows) → For training components
X_val (406K rows)   → For tuning weights/thresholds
X_test (508K rows)  → For final evaluation
```

---

### **STEP 5: Train Pipeline (THE BIG ONE)**
**File:** `pipeline/pipeline.py` lines 62-320

This is where all the magic happens. The pipeline has **10 inner steps**:

#### **[1/10] PREPROCESSING**
```python
X_train_processed = preprocessor.fit_transform(X_train)

# What it does:
# - Identifies categorical columns: ['proto', 'state', 'service']
# - Drops constant columns (only 1 unique value)
# - Drops all-unique columns (every row different)
# - One-hot encodes categoricals
# - Result: ~160 columns (was 49, expanded due to one-hot)
```

**Flow:**
```
Raw Data (49 cols)
    ↓
Detect Categoricals (3 cols)
    ↓
One-Hot Encode (+ 150 new binary cols)
    ↓
Drop Constant/Unique (- 3 cols)
    ↓
Result: ~160 cols
```

---

#### **[2/10] FEATURE ENGINEERING**
```python
X_train_eng = add_statistical_features(X_train_processed)

# Creates new features like:
bytes_ratio = src_bytes / (dst_bytes + 1)
total_bytes = src_bytes + dst_bytes
srv_ratio = srv_count / (count + 1)
packet_rate = count / (duration + 1)

# Result: ~175 columns (was ~160, added ~15 engineered features)
```

---

#### **[3/10] DROP CORRELATED FEATURES**
```python
correlated = drop_correlated_features(X_train_eng, threshold=0.95)
X_train_reduced = X_train_eng.drop(columns=correlated)

# What it does:
# - Calculates correlation matrix
# - Finds feature pairs with correlation > 0.95
# - Drops one from each pair
# - Result: ~100 columns (was ~175, dropped ~75 redundant features)
```

---

#### **[4/10] SCALING**
```python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)

# What it does:
# - Scales features using median and IQR
# - Resistant to outliers (better than StandardScaler)
# - Result: same shape (~100 cols), but values normalized
```

---

#### **[5/10] AUTOENCODER (Trained on NORMAL Data Only)**
```python
# Extract only normal traffic for autoencoder
normal_idx = (y_train == 0)  # Where label is "normal"
X_normal = X_train_scaled[normal_idx]  # ~1.3M rows (normal only)

# Cap at 200K for memory
X_normal_capped = X_normal.sample(min(200_000, len(X_normal)))

# Train autoencoder
autoencoder = AutoencoderIDS(input_dim=100, encoding_dim=32)
autoencoder.fit(X_normal_capped, epochs=10, batch_size=256)

# Architecture:
# Input(100) → Dense(128, relu) → Dense(64, relu)
#           → Dense(32, relu)  [BOTTLENECK - Encoder output]
#           → Dense(64, relu) → Dense(128, relu) → Output(100)
# 
# Loss: MSE (learns to reconstruct normal traffic)
# 
# Purpose:
# - Encoder: Compresses 100 dims → 32 dims (feature extraction)
# - Decoder: Decompresses 32 dims → 100 dims (reconstruction)
# - Reconstruction Error: Used as anomaly score
```

**What's Happening:**
```
Normal Traffic (Attack=0)
    ↓ (Only uses these)
Autoencoder Training
    ↓
Learn what "Normal" looks like
    ↓
If Attack Traffic appears → High reconstruction error
    ↓ (This error is an anomaly score)
```

---

#### **[6/10] PCA (Fallback if Autoencoder Disabled)**
```python
if use_pca:
    pca = PCATransformer(n_components=0.95)  # Retain 95% variance
    X_pca = pca.fit_transform(X_normal_capped)
    # Result: Reduced from ~100 dims to ~50 dims (typical)
```

---

#### **[7/10] ANOMALY DETECTORS (Trained on NORMAL Data Only)**
```python
# Isolation Forest
iso_forest = IsolationForestDetector(
    n_estimators=100,
    max_samples=256,
    contamination=0.15,  # Expect 15% anomalies
    random_state=42
)
iso_forest.fit(X_normal_capped)

# LOF (Local Outlier Factor)
lof = LocalOutlierFactorDetector(
    n_neighbors=20,
    contamination=0.15,
    novelty=True
)
lof.fit(X_normal_capped[:50_000])  # Cap at 50K

# What they do:
# IF: Isolates outliers using random partitioning
#     Returns: isolation score (higher = more anomalous)
# LOF: Detects local density anomalies
#      Returns: local outlier factor (higher = more anomalous)
```

**Why Train Only on Normal Data:**
```
Normal Traffic (labeled 0)
    ↓ (ONLY this goes into detectors)
These detectors learn: "This is what normal looks like"
    ↓
At inference:
  - Normal traffic → Low anomaly scores
  - Attack traffic → High anomaly scores
  - Novel attacks → Also high scores (they're anomalous!)
```

---

#### **[8/10] RANDOM FOREST (Supervised - Trained on MIXED Data)**
```python
# Encode features using autoencoder
X_encoded = autoencoder.encode(X_train_scaled)  # ~100 → 32 dims

# Cap RF training at 500K rows (stratified to keep class balance)
rf = SupervisedRF(
    n_estimators=50,
    max_depth=15,
    class_weight='balanced'  # Handle imbalance
)
rf.fit(X_encoded, y_train)

# What it does:
# - Takes 32-dim encoded features from autoencoder
# - Trains on BOTH normal (y=0) and attack (y=1) samples
# - Learns decision boundaries between the two classes
# - Returns probability of being an attack (0 to 1)
```

**Key Difference:**
```
Unsupervised Detectors:
  - Trained only on normal data
  - Learn: "What is anomalous?"
  - Good for novel attacks

Supervised Model (RF):
  - Trained on both normal and attack data
  - Learn: "What distinguishes attacks from normal?"
  - Good for known attack patterns
```

---

#### **[9/10] NORMALIZER + WEIGHTS/THRESHOLD**
```python
# Score validation set through all detectors
sup_probs_val = rf.predict_proba(X_encoded_val)  # RF probabilities
iso_scores_val = iso_forest.score(X_scaled_val)  # IF anomaly scores
lof_scores_val = lof.score(X_scaled_val)        # LOF anomaly scores

# Normalize anomaly scores to [0, 1]
normalizer = ScoreNormalizer()
normalizer.fit({"iso_forest": iso_scores_val, "lof": lof_scores_val})

iso_norm = normalizer.transform({"iso_forest": iso_scores_val, ...})

# Grid search for best weight and threshold
best_weight, best_threshold = optimize_supervised_weight(
    sup_probs_val,
    {"iso_forest": iso_norm, "lof": lof_norm},
    y_val,
    weight_min=0.1,
    weight_max=0.7,
    threshold_min=0.3,
    threshold_max=0.7,
    min_recall=0.90  # Must catch 90% of attacks
)

# Result: best_weight ≈ 0.65, best_threshold ≈ 0.50
```

**What's Happening:**
```
For each (weight, threshold) combination:
  1. Combine: hybrid = weight × RF_prob + (1-weight) × mean(IF, LOF)
  2. Predict: is_attack = (hybrid ≥ threshold)
  3. Compute: F1 score
  4. Check: Recall ≥ 0.90?
  5. Track: Best F1 score with this (weight, threshold)

Result: Most balanced configuration
```

---

#### **[10/10] CREATE ENSEMBLE**
```python
ensemble = HybridEnsemble(
    supervised_weight=0.65,
    method='weighted_avg'
)

# Formula at inference:
# avg_anomaly = (IF_normalized + LOF_normalized) / 2
# hybrid = 0.65 × RF_prob + 0.35 × avg_anomaly
```

---

### **SAVE MODELS**
**File:** `utils/model_manager.py`

After training, all components are saved:
```
artifacts/latest/
├── config.json                    ← Configuration
├── preprocessor.joblib            ← Fitted preprocessor
├── scaler.joblib                  ← Fitted RobustScaler
├── pca.joblib                     ← Fitted PCA (if used)
├── autoencoder.weights.h5         ← Keras autoencoder weights
├── anomaly_detectors.joblib       ← Fitted IF + LOF
├── supervised_model.joblib        ← Fitted Random Forest
├── normalizer.joblib              ← Fitted score normalizer
├── ensemble.joblib                ← Hybrid ensemble config
└── threshold_optimizer.joblib     ← Threshold config
```

**Total:** ~150-200 MB of models

---

---

## 🌐 PHASE 2: INFERENCE FLOW (API)

```
app/main.py (FastAPI Server)
    ↓
@app.on_event("startup")
    ↓
app/dependencies.py → ModelManager.load_pipeline("latest")
    ↓
Load all 10 saved model files
    ↓
Create HybridIDSPipeline from saved artifacts
    ↓
Ready for predictions!

User sends POST /predict
    ↓
app/main.py:152 → predict_single()
    ↓
pipeline.predict_proba(df)
    ↓
[Results returned with alert_level]
```

---

### **STEP 1: API STARTUP**
**File:** `app/main.py` lines 73-90

```python
@app.on_event("startup")
async def startup_event():
    pipeline = get_pipeline()
    # Triggers lazy loading from artifacts/latest/
    # Models loaded into memory (~200MB)
```

---

### **STEP 2: LOAD PIPELINE**
**File:** `app/dependencies.py` lines 20-45

```python
@lru_cache(maxsize=1)  # Cache in memory (load only once)
def get_pipeline():
    manager = ModelManager("artifacts")
    pipeline = manager.load_pipeline("latest")
    return pipeline
    
# What load_pipeline does:
# 1. Read config.json
# 2. Load preprocessor.joblib
# 3. Load scaler.joblib
# 4. Load pca.joblib
# 5. Load autoencoder.weights.h5 + rebuild Keras model
# 6. Load anomaly_detectors.joblib (IF + LOF)
# 7. Load supervised_model.joblib (RF)
# 8. Load normalizer.joblib
# 9. Load ensemble.joblib
# 10. Load threshold_optimizer.joblib
# Result: Fully initialized HybridIDSPipeline object
```

---

### **STEP 3: SINGLE PREDICTION**
**File:** `app/main.py` lines 152-175

```python
@app.post("/predict")
async def predict_single(sample: NetworkTrafficInput, pipeline=Depends(get_pipeline)):
    # Input example:
    # {
    #   "src_bytes": 500,
    #   "dst_bytes": 1000,
    #   "proto": "TCP",
    #   "state": "FIN",
    #   ... (more fields)
    # }
    
    # Convert to DataFrame
    sample_dict = sample.to_dict()  # Only non-None fields
    df = pd.DataFrame([sample_dict])
    
    # Get prediction
    prob = pipeline.predict_proba(df)[0]      # Probability (0-1)
    is_intrusion = pipeline.predict(df)[0]    # Binary (0 or 1)
    
    # Determine alert level
    alert_level, alert_message = determine_alert_level(float(prob), bool(is_intrusion))
    
    # Return response
    return PredictionOutput(
        is_intrusion=bool(is_intrusion),
        confidence=float(prob),
        intrusion_probability=float(prob),
        alert_level=alert_level,
        alert_message=alert_message
    )
```

---

### **STEP 4: PREDICTION INTERNALS**
**File:** `pipeline/pipeline.py` lines 340-420

When `pipeline.predict_proba(X)` is called:

```
Input: df with 1 row

└─→ _transform_features(X)
    │
    ├─→ preprocessor.transform(X)
    │   └─ One-hot encode categoricals
    │   └─ Align to training columns
    │
    ├─→ add_statistical_features()
    │   └─ Create derived features
    │
    ├─→ drop_correlated_features()
    │   └─ Remove redundant features
    │
    ├─→ scaler.transform()
    │   └─ Normalize using fitted RobustScaler
    │
    └─→ X_scaled (ready for detectors)

└─→ _get_supervised_probs(X_scaled)
    │
    ├─→ autoencoder.encode(X_scaled)
    │   └─ 100 dims → 32 dims
    │
    └─→ rf.predict_proba(X_encoded)
        └─ Returns prob of being intrusion (0-1)

└─→ _get_anomaly_scores(X_scaled)
    │
    ├─→ iso_forest.score(X_scaled)
    │   └─ Isolation Forest anomaly score
    │
    ├─→ lof.score(X_scaled)
    │   └─ LOF anomaly score
    │
    └─→ anomaly_scores_dict = {"iso": [...], "lof": [...]}

└─→ normalizer.transform(anomaly_scores_dict)
    │
    └─→ Clip each score to [0, 1] using percentiles

└─→ ensemble.predict_proba(sup_probs, normalized_scores)
    │
    ├─→ avg_anomaly = mean(iso_norm, lof_norm)
    │
    └─→ hybrid = 0.65 × sup_prob + 0.35 × avg_anomaly

└─→ threshold_optimizer.predict(hybrid)
    │
    └─→ is_intrusion = (hybrid ≥ 0.50)

└─→ Return: hybrid_prob (0-1) and is_intrusion (0 or 1)
```

---

### **STEP 5: ALERT LEVEL DETERMINATION**
**File:** `app/main.py` lines 30-60

```python
def determine_alert_level(probability, is_intrusion):
    if probability >= 0.9:
        return ("CRITICAL", "Immediate action required")
    elif probability >= 0.7:
        return ("HIGH", "Proceed with caution")
    elif probability >= 0.5:
        return ("MEDIUM", "Suspicious activity")
    elif probability >= 0.4:
        return ("LOW", "Monitoring required")
    else:
        if is_intrusion:
            return ("LOW", "Low confidence anomaly")
        else:
            return ("NORMAL", "No intrusion detected")
```

---

### **EXAMPLE PREDICTION WALKTHROUGH**

```
User sends:
{
  "src_bytes": 100,
  "dst_bytes": 5000,
  "proto": "TCP",
  "state": "FIN",
  "count": 20,
  "duration": 5.0
}

Pipeline processes:
1. Preprocessor: One-hot encodes "TCP" and "FIN"
2. Feature Engineering: 
   - bytes_ratio = 100 / 5001 ≈ 0.02
   - total_bytes = 5100
   - packet_rate = 20 / 5.0 = 4.0
3. Drop Correlations: Removes redundant features
4. Scaling: Normalizes all values
5. Autoencoder Encoding: 100 dims → 32 dims
6. RF Probability: 0.15 (15% confidence it's attack)
7. IF Anomaly: 0.8 (80% anomalous)
8. LOF Anomaly: 0.7 (70% anomalous)
9. Normalize: Both → [0, 1] ≈ 0.75 each
10. Ensemble: 0.65 × 0.15 + 0.35 × 0.725 = 0.343
11. Threshold: 0.343 < 0.50 → is_intrusion = 0 (Normal)

Return:
{
  "is_intrusion": false,
  "confidence": 0.343,
  "alert_level": "NORMAL",
  "alert_message": "No intrusion detected - Normal Traffic"
}
```

---

---

## 🔄 COMPLETE FLOW DIAGRAM

```
TRAINING PHASE
==============

train.py
├─ Load Data (UNSW-NB15.csv) → 2.5M rows
├─ Create Binary Targets → y ∈ {0, 1}
├─ Train/Val/Test Split → 64%/16%/20%
│
└─ pipeline.fit(X_train, y_train, X_val, y_val)
    │
    ├─ [1] Preprocessor.fit(X_train)
    │       └─ One-hot encode, drop constant columns
    │
    ├─ [2] Feature Engineering
    │       └─ Create bytes_ratio, packet_rate, etc.
    │
    ├─ [3] Drop Correlated Features
    │       └─ threshold=0.95
    │
    ├─ [4] Scaler.fit(X_train)
    │       └─ RobustScaler
    │
    ├─ [5] Autoencoder.fit(X_normal_only)
    │       ├─ Encoder: 100 → 32 dims
    │       └─ Learns to reconstruct normal traffic
    │
    ├─ [6] IF.fit(X_normal_only)
    │       └─ Isolation Forest
    │
    ├─ [7] LOF.fit(X_normal_only)
    │       └─ Local Outlier Factor
    │
    ├─ [8] RF.fit(X_encoded, y_mixed)
    │       └─ Trained on both normal and attack
    │
    ├─ [9] Normalizer.fit(X_val)
    │       └─ Learn p5, p95 for each detector
    │
    ├─ [10] Optimize weights/thresholds on X_val
    │        └─ Grid search: F1 score optimization
    │
    └─ Save 10 model files to artifacts/latest/


INFERENCE PHASE
===============

API Startup
│
├─ app.on_event("startup")
│   └─ get_pipeline() → Load from artifacts/
│       ├─ Load config.json
│       ├─ Load preprocessor.joblib
│       ├─ Load scaler.joblib
│       ├─ Load pca.joblib
│       ├─ Load autoencoder.weights.h5
│       ├─ Load anomaly_detectors.joblib
│       ├─ Load supervised_model.joblib
│       ├─ Load normalizer.joblib
│       ├─ Load ensemble.joblib
│       └─ Load threshold_optimizer.joblib
│
└─ Ready! Cached in memory via @lru_cache(maxsize=1)

User Request: POST /predict
│
├─ Validate input (Pydantic)
├─ Convert to DataFrame
│
├─ pipeline.predict_proba(X)
│   │
│   ├─ _transform_features(X)
│   │   ├─ Preprocessor.transform()
│   │   ├─ Feature Engineering
│   │   ├─ Drop Correlations
│   │   └─ Scaler.transform()
│   │
│   ├─ _get_supervised_probs(X_scaled)
│   │   ├─ Autoencoder.encode() → 32 dims
│   │   └─ RF.predict_proba() → [0, 1]
│   │
│   ├─ _get_anomaly_scores(X_scaled)
│   │   ├─ IF.score() → raw scores
│   │   └─ LOF.score() → raw scores
│   │
│   ├─ Normalizer.transform() → [0, 1]
│   │
│   ├─ Ensemble.predict_proba()
│   │   └─ 0.65 × RF + 0.35 × mean(IF, LOF)
│   │
│   └─ Return: hybrid_prob ∈ [0, 1]
│
├─ pipeline.predict(X)
│   └─ is_intrusion = (hybrid_prob ≥ 0.50)
│
├─ determine_alert_level(prob, is_intrusion)
│   └─ Return: alert_level + message
│
└─ Return PredictionOutput JSON
```

---

## 📝 Key Takeaways

1. **Preprocessing** transforms raw data into usable features
2. **Autoencoder** learns normal patterns and extracts compressed features
3. **Unsupervised Detectors** (IF, LOF) catch anomalies and novel attacks
4. **Supervised Model** (RF) recognizes known attack patterns
5. **Ensemble** combines both approaches with learned weights
6. **At Inference:** All 10 components work together to make a single prediction

The system is designed to catch both **known attacks** (via RF) and **novel/zero-day attacks** (via unsupervised detectors).
