# Hybrid IDS — Complete API Flow

> This document traces every HTTP request from the moment it arrives at the
> FastAPI server until a response is returned, showing exactly what happens
> inside the pipeline at each step.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Server Startup & Model Loading](#2-server-startup--model-loading)
3. [Endpoint Map](#3-endpoint-map)
4. [Single Prediction Flow (`POST /predict`)](#4-single-prediction-flow-post-predict)
5. [Batch Prediction Flow (`POST /predict/batch`)](#5-batch-prediction-flow-post-predictbatch)
6. [Feature Transformation Pipeline](#6-feature-transformation-pipeline)
7. [Score Computation Pipeline](#7-score-computation-pipeline)
8. [Score Normalisation](#8-score-normalisation)
9. [Ensemble Combination](#9-ensemble-combination)
10. [Threshold & Classification](#10-threshold--classification)
11. [Alert Level Determination](#11-alert-level-determination)
12. [Health & Model Info Endpoints](#12-health--model-info-endpoints)
13. [Hot Reload Endpoint](#13-hot-reload-endpoint)
14. [Request / Response Schemas](#14-request--response-schemas)
15. [Error Handling](#15-error-handling)
16. [End-to-End Worked Example](#16-end-to-end-worked-example)

---

## 1. High-Level Overview

```
                          ┌─────────────────────────────────────────────────────────────────────────┐
                          │                        FastAPI Server (port 8000)                        │
                          │                                                                         │
  HTTP Request            │  ┌──────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────┐ │  HTTP Response
 ─────────────────────►   │  │  Schema   │──►│   Transform      │──►│   Score &     │──►│  Alert   │ │ ──────────────►
  NetworkTrafficInput     │  │ Validate  │   │   Features       │   │   Ensemble    │   │  Level   │ │  PredictionOutput
                          │  └──────────┘   └──────────────────┘   └──────────────┘   └──────────┘ │
                          │                                                                         │
                          │         Backed by: HybridIDSPipeline (lazy-loaded singleton)            │
                          └─────────────────────────────────────────────────────────────────────────┘
```

**One-sentence summary:** The client sends raw network traffic features as JSON →
the API transforms them through the same preprocessing/feature-engineering/scaling
pipeline used during training → feeds the scaled features through an autoencoder,
random forest, isolation forest, and LOF → normalises and combines the scores
with max logic → applies a threshold → returns a prediction with an
alert level.

---

## 2. Server Startup & Model Loading

When `python app/main.py` (or `uvicorn app.main:app`) is executed:

```
┌──────────────────────────────────────────────────────────────────────┐
│  main.py                                                            │
│                                                                      │
│  uvicorn.run(app, host="0.0.0.0", port=8000)                        │
│      │                                                               │
│      ▼                                                               │
│  @app.on_event("startup")                                            │
│      │                                                               │
│      ├── get_pipeline()          ← dependencies.py                   │
│      │       │                                                       │
│      │       ├── _pipeline is None?  → YES                           │
│      │       │       │                                               │
│      │       │       ▼                                               │
│      │       │   ModelManager("artifacts")                           │
│      │       │       │                                               │
│      │       │       ▼                                               │
│      │       │   load_pipeline("latest")                             │
│      │       │       │                                               │
│      │       │       ├── Read config.json → IDSConfig                │            
│      │       │       ├── Load params.joblib                          │            
│      │       │       │     └── feature_names, correlated_features    │
│      │       │       ├── Load preprocessor.joblib                    │
│      │       │       ├── Load scaler.joblib (RobustScaler)           │
│      │       │       ├── Load pca.joblib (if file exists)            │
│      │       │       ├── Load autoencoder.weights.h5                 │
│      │       │       │     ├── Rebuild Keras architecture            │
│      │       │       │     ├── Load trained weights                  │
│      │       │       │     └── Build encoder sub-model               │
│      │       │       ├── Load anomaly_detectors.joblib (IF + LOF)    │
│      │       │       ├── Load supervised_model.joblib (RF)           │
│      │       │       ├── Load normalizer.joblib (p5/p95)             │
│      │       │       ├── Load ensemble.joblib (weight + method)      │
│      │       │       └── Load threshold_optimizer.joblib             │
│      │       │                                                       │
│      │       └── Store in global _pipeline                           │
│      │                                                               │
│      └── Print: threshold, supervised_weight                         │
│                                                                      │
│  Server is now ready to accept requests                              │
└──────────────────────────────────────────────────────────────────────┘
```

### Artifact Files Loaded

| File                        | Python Object Restored              | Purpose                              |
|-----------------------------|--------------------------------------|--------------------------------------|
| `config.json`               | `IDSConfig`                         | All hyperparameters                  |
| `params.joblib`             | `dict`                              | `feature_names`, `correlated_features` |
| `preprocessor.joblib`       | `Preprocessor`                      | Categorical cols, drop cols, training cols |
| `scaler.joblib`             | `ScalerWrapper` (RobustScaler)      | Fitted center/scale values           |
| `pca.joblib`                | `PCATransformer`                    | Fitted PCA (fallback encoder)        |
| `autoencoder.weights.h5`    | `AutoencoderIDS` weights            | Trained neural network weights       |
| `anomaly_detectors.joblib`  | `{IF, LOF}` dict                    | Fitted IF + LOF sklearn models       |
| `supervised_model.joblib`   | `SupervisedRF`                      | Fitted RandomForest                  |
| `normalizer.joblib`         | `ScoreNormalizer`                   | Fitted p5/p95 per detector           |
| `ensemble.joblib`           | `HybridEnsemble`                    | Weight (0.65) + method (max ensemble) |
| `threshold_optimizer.joblib`| `ThresholdOptimizer`                | Best threshold (e.g. 0.50)           |

### Autoencoder Loading Strategy

```
Attempt 1:  autoencoder.weights.h5 exists?
    ├── YES → Rebuild architecture from config → load_weights() → ✓
    └── NO  ─┐
             ▼
Attempt 2:  autoencoder.keras exists? (legacy format)
    ├── YES → tf.keras.models.load_model() → validate input_dim matches → ✓
    └── NO  → WARN: Using random weights, disable AE anomaly scoring
```

### Lazy-Loading Singleton Pattern

```python
# dependencies.py
_pipeline = None                    # Module-level global

def get_pipeline():
    global _pipeline
    if _pipeline is None:           # First call → load from disk
        _pipeline = ModelManager("artifacts").load_pipeline("latest") # Load the pipeline with all artifacts imported first
    return _pipeline                # All subsequent calls → return cached
```

This means the model is loaded **once** at first use and shared across all
requests. No per-request loading overhead.

---

## 3. Endpoint Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                     http://localhost:8000                             │
│                                                                      │
│  GET  /              → Root info (name, version, docs link)          │
│  GET  /health        → Health check + model loaded status            │
│  GET  /model/info    → Config, threshold, weights, features          │
│  POST /predict       → Single sample → prediction + alert            │
│  POST /predict/batch → Multiple samples → list of predictions        │
│  POST /model/reload  → Hot-reload model from disk                    │
│  GET  /docs          → Swagger UI (auto-generated)                   │
│  GET  /redoc         → ReDoc documentation (auto-generated)          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Single Prediction Flow (`POST /predict`)

This is the core endpoint. Here is every step from request to response:

```
 Client                          FastAPI                         Pipeline
   │                                │                                │
   │  POST /predict                 │                                │
   │  { "dur": 0.12,               │                                │
   │    "spkts": 6, ... }          │                                │
   │ ──────────────────────────────►│                                │
   │                                │                                │
   │                     ┌──────────┤                                │
   │                     │ 1. Pydantic Validation                    │
   │                     │    NetworkTrafficInput(**body)             │
   │                     │    - All fields Optional                  │
   │                     │    - Supports KDD + UNSW features         │
   │                     └──────────┤                                │
   │                                │                                │
   │                     ┌──────────┤                                │
   │                     │ 2. to_dict()                              │
   │                     │    Keep only non-None fields              │
   │                     │    {"dur": 0.12, "spkts": 6, ...}        │
   │                     └──────────┤                                │
   │                                │                                │
   │                     ┌──────────┤                                │
   │                     │ 3. pd.DataFrame([sample_dict])            │
   │                     │    1 row × N columns                      │
   │                     └──────────┤                                │
   │                                │                                │
   │                                │  pipeline.predict_proba(df)    │
   │                                │ ──────────────────────────────►│
   │                                │                                │
   │                                │         (see §6-§9 below)      │
   │                                │                                │
   │                                │  prob = 0.87                   │
   │                                │◄──────────────────────────────│
   │                                │                                │
   │                                │  pipeline.predict(df)          │
   │                                │ ──────────────────────────────►│
   │                                │  is_intrusion = (prob ≥ 0.50)  │
   │                                │  = True                        │
   │                                │◄──────────────────────────────│
   │                                │                                │
   │                     ┌──────────┤                                │
   │                     │ 4. determine_alert_level(0.87, True)      │
   │                     │    → ("HIGH", "Intrusion detected...")     │
   │                     └──────────┤                                │
   │                                │                                │
   │  Response 200 OK               │                                │
   │  {                             │                                │
   │    "is_intrusion": true,       │                                │
   │    "confidence": 0.87,         │                                │
   │    "intrusion_probability":    │                                │
   │       0.87,                    │                                │
   │    "alert_level": "HIGH",      │                                │
   │    "alert_message": "..."      │                                │
   │  }                             │                                │
   │◄──────────────────────────────│                                │
```

---

## 5. Batch Prediction Flow (`POST /predict/batch`)

```
 Client                          FastAPI                         Pipeline
   │                                │                                │
   │  POST /predict/batch           │                                │
   │  { "samples": [               │                                │
   │      { "dur": 0.12, ... },    │                                │
   │      { "dur": 1.50, ... },    │                                │
   │      { "dur": 0.00, ... }     │                                │
   │  ] }                          │                                │
   │ ──────────────────────────────►│                                │
   │                                │                                │
   │                     ┌──────────┤                                │
   │                     │ 1. Validate each sample                   │
   │                     │ 2. [s.to_dict() for s in samples]         │
   │                     │ 3. pd.DataFrame(samples_dicts)            │
   │                     │    → 3 rows × N columns                   │
   │                     └──────────┤                                │
   │                                │                                │
   │                                │  pipeline.predict_proba(df)    │
   │                                │ ──────────────────────────────►│
   │                                │  probs = [0.87, 0.92, 0.15]   │
   │                                │◄──────────────────────────────│
   │                                │                                │
   │                                │  pipeline.predict(df)          │
   │                                │ ──────────────────────────────►│
   │                                │  preds = [1, 1, 0]             │
   │                                │◄──────────────────────────────│
   │                                │                                │
   │                     ┌──────────┤                                │
   │                     │ 4. For each (pred, prob):                  │
   │                     │    determine_alert_level()                 │
   │                     │    → PredictionOutput per sample           │
   │                     │                                            │
   │                     │ 5. intrusions_detected = sum(preds) = 2   │
   │                     └──────────┤                                │
   │                                │                                │
   │  Response 200 OK               │                                │
   │  {                             │                                │
   │    "predictions": [...],       │                                │
   │    "count": 3,                 │                                │
   │    "intrusions_detected": 2    │                                │
   │  }                             │                                │
   │◄──────────────────────────────│                                │
```

**Key difference from single predict:** All samples are batched into one
DataFrame and processed in a single pass through the pipeline — the autoencoder,
RF, IF, and LOF each process the full batch at once. This is significantly
faster than calling `/predict` N times.

---

## 6. Feature Transformation Pipeline

This is what happens inside `pipeline._transform_features(X)` — called by
both `predict_proba()` and `predict()`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        _transform_features(X)                               │
│                                                                             │
│  Raw DataFrame (1 row × N user-supplied columns)                           │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  STEP 1: preprocessor.transform(X)                          │            │
│  │                                                              │            │
│  │  a) Strip whitespace from column names                       │            │
│  │  b) Drop label columns: attack_class, label, attack, etc.   │            │
│  │  c) Drop constant / all-unique columns (from training)       │            │
│  │  d) One-hot encode categorical columns:                      │            │
│  │       KDD:  protocol_type, service, flag                     │            │
│  │       UNSW: proto, state, service                            │            │
│  │     → bool columns cast to int8                              │            │
│  │  e) Align to training_columns:                               │            │
│  │       df.reindex(columns=training_columns, fill_value=0)     │            │
│  │     → Missing columns filled with 0                          │            │
│  │     → Extra columns dropped                                  │            │
│  │     → Column ORDER matches training exactly                  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│      │                                                                      │
│      ▼  ~160 columns (after one-hot expansion)                             │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  STEP 2: add_statistical_features(X)                        │            │
│  │                                                              │            │
│  │  UNSW features (if columns exist):                           │
│  │    bytes_ratio    = sbytes / (dbytes + 1)                    │            │
│  │    pkt_ratio      = spkts / (dpkts + 1)                     │            │
│  │    total_pkts     = spkts + dpkts                            │            │
│  │    bytes_per_pkt  = (sbytes+dbytes) / (spkts+dpkts+1)       │            │
│  │    load_asymmetry = |sload-dload| / (sload+dload+1)         │            │
│  │    jit_asymmetry  = |sjit-djit| / (sjit+djit+1)            │            │
│  │                                                              │            │
│  │  KDD features (if columns exist):                            │            │
│  │    bytes_ratio, total_bytes, srv_ratio, packet_rate          │            │
│  └─────────────────────────────────────────────────────────────┘            │
│      │                                                                      │
│      ▼  ~170 columns                                                       │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  STEP 3: add_context_aware_features(X)                      │            │
│  │                                                              │            │
│  │  UNSW features (if columns exist):                           │            │
│  │    handshake_efficiency = (synack+ackdat) / (tcprtt+1)      │            │
│  │    connection_density   = ct_dst_src_ltm × ct_srv_dst       │            │
│  │    src_byte_rate        = sbytes / (dur+1)                  │            │
│  │    response_efficiency  = response_body_len / (dbytes+1)    │            │
│  │    ttl_asymmetry        = |sttl - dttl|                     │            │
│  │                                                              │            │
│  │  KDD features (if columns exist):                            │
│  │    bytes_per_connection, connection_regularity,              │            │
│  │    service_focus, host_diversity, byte_asymmetry, etc.       │            │
│  └─────────────────────────────────────────────────────────────┘            │
│      │                                                                      │
│      ▼  ~175 columns                                                       │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  STEP 4: Drop correlated features                            │            │
│  │                                                              │            │
│  │  Uses self.correlated_features (saved list from training)    │            │
│  │  df.drop(columns=correlated_features, errors='ignore')       │            │
│  └─────────────────────────────────────────────────────────────┘            │
│      │                                                                      │
│      ▼  ~100 columns                                                       │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  STEP 5: Cast + Scale                                        │            │
│  │                                                              │            │
│  │  X.astype('float32')                                        │            │
│  │  scaler.transform(X)  ← fitted RobustScaler                │            │
│  │                                                              │            │
│  │  RobustScaler formula per feature:                           │            │
│  │    x_scaled = (x - median) / IQR                            │            │
│  │  (Robust to outliers — uses median & interquartile range)    │            │
│  └─────────────────────────────────────────────────────────────┘            │
│      │                                                                      │
│      ▼  numpy float32 array, shape: (n_samples, ~100)                      │
│                                                                             │
│  This X_scaled is returned for use by the scoring pipeline                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Column Alignment — How Inference Handles Missing/Extra Features

```
Training saw these columns (saved in preprocessor.training_columns):
  [dur, spkts, dpkts, sbytes, ..., proto_tcp, proto_udp, state_FIN, ...]
       160 columns after one-hot encoding

User sends only 8 fields:
  {dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl}

After reindex(columns=training_columns, fill_value=0):
  [dur=0.12, spkts=6, dpkts=4, sbytes=258, ..., proto_tcp=0, proto_udp=0, ...]
       160 columns — missing ones filled with 0
```

---

## 7. Score Computation Pipeline

After features are transformed, `predict_proba()` computes scores from
**three independent paths** and combines them:

```
                           X_scaled
                     (n_samples × ~100)
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
     ┌─────────────┐  ┌───────────┐    ┌───────────┐
     │ Autoencoder  │  │ Isolation │    │    LOF     │
     │   Encoder    │  │  Forest   │    │  (Novelty  │
     │              │  │           │    │   mode)    │
     │  encode(X)   │  │  score(X) │    │  score(X)  │
     │  → 32-dim    │  │           │    │            │
     └──────┬───────┘  └─────┬─────┘    └─────┬─────┘
            │                │                │
            ▼                │                │
     ┌─────────────┐        │                │
     │ Random      │        │                │
     │ Forest      │        │                │
     │             │        │                │
     │ predict_    │        │                │
     │  proba()    │        │                │
     └──────┬──────┘        │                │
            │                │                │
            ▼                ▼                ▼
      RF probability    IF raw score     LOF raw score
      (0 to 1)          (≈ 0.4–0.6)     (≈ 1–100+)
            │                │                │
            │                └────────┬───────┘
            │                         │
            │                    ┌────┴─────┐
            │                    │Normalizer│
            │                    │ (p5/p95) │
            │                    └────┬─────┘
            │                         │
            │                    ┌────┴─────┐
            │                    │IF: 0–1   │
            │                    │LOF: 0–1  │
            │                    └────┬─────┘
            │                         │
            └───────────┬─────────────┘
                        │
                   ┌────┴──────┐
                   │ Ensemble  │
                   │(weighted  │
                   │  average) │
                   └────┬──────┘
                        │
                        ▼
                 Hybrid Probability
                    (0 to 1)
```

### Path A — Supervised (Random Forest)

```
X_scaled (n × ~100)
    │
    ▼
autoencoder.encode(X_scaled)
    │   Keras encoder sub-model
    │   Input(~100) → Dense(128,relu) → Dropout → Dense(64,relu) → Dropout → Dense(32,relu)
    │
    ▼
X_encoded (n × 32)       ← compressed representation
    │
    ▼
supervised_model.predict_proba(X_encoded)
    │   RandomForest with 50 trees, class_weight='balanced'
    │   Returns predict_proba(X)[:, 1]  (probability of class=intrusion)
    │
    ▼
sup_probs (n,)            ← values in [0, 1]
```

### Path B — Unsupervised (Anomaly Detectors)

```
X_scaled (n × ~100)
    │
    ├──► IsolationForest.score(X)
    │       = -model.score_samples(X)
    │       Higher = more anomalous
    │       Typical range: 0.35 – 0.65
    │
    └──► LOF.score(X)
            = -model.score_samples(X)
            Higher = more anomalous
            Typical range: 1 – 100+ (very skewed)
```

### Path C — Autoencoder Reconstruction Error (conditional)

```
Only included if autoencoder.weights_loaded == True

X_scaled (n × ~100)
    │
    ▼
autoencoder.reconstruction_error(X)
    │   reconstructed = autoencoder.predict(X)
    │   error = mean( (X - reconstructed)² , axis=1 )
    │
    ▼
recon_error (n,)          ← higher for attacks (AE only learned normal patterns)
```

---

## 8. Score Normalisation

Raw scores from different detectors live on completely different scales.
The `ScoreNormalizer` maps them all to [0, 1]:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ScoreNormalizer                                                     │
│                                                                      │
│  Fitted during training on validation scores:                        │
│    For each detector:                                                │
│      p5_val  = np.percentile(scores, 5)                             │
│      p95_val = np.percentile(scores, 95)                            │
│    Stored in: self.percentiles[detector_name] = (p5_val, p95_val)   │
│                                                                      │
│  At inference:                                                       │
│    For each detector:                                                │
│      normalized = (score - p5_val) / (p95_val - p5_val + ε)        │
│      clipped    = clip(normalized, 0, 1)                            │
│                                                                      │
│  Example:                                                            │
│    IF score = 0.52,  p5 = 0.40, p95 = 0.58                         │
│    normalized = (0.52 - 0.40) / (0.58 - 0.40) = 0.667              │
│                                                                      │
│    LOF score = 45.0, p5 = 1.2,  p95 = 80.0                         │
│    normalized = (45.0 - 1.2) / (80.0 - 1.2) = 0.556                │
│                                                                      │
│  Unseen detector fallback:                                           │
│    normalized = (score - min) / (max - min + ε)                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. Ensemble Combination

The `HybridEnsemble` combines the supervised RF probability with the
normalised unsupervised scores:

```
┌──────────────────────────────────────────────────────────────────────┐
│  HybridEnsemble (method = 'weighted_avg')                            │
│                                                                      │
│  Inputs:                                                             │
│    sup_probs:          RF probability per sample     [0, 1]         │
│    anomaly_scores_dict: {IF: [0,1], LOF: [0,1]}     normalised     │
│                                                                      │
│  Computation:                                                        │
│    avg_anomaly = mean(IF_score, LOF_score)                          │
│                                                                      │
│    supervised_weight = 0.65                                          │
│    unsup_weight      = 1 - 0.65 = 0.35                              │
│                                                                      │
│    hybrid_prob = 0.65 × RF_prob + 0.35 × avg_anomaly               │
│                                                                      │
│  Example:                                                            │
│    RF_prob     = 0.92  (RF says "very likely attack")               │
│    IF_norm     = 0.67  (IF says "moderately anomalous")             │
│    LOF_norm    = 0.56  (LOF says "somewhat anomalous")              │
│                                                                      │
│    avg_anomaly = (0.67 + 0.56) / 2 = 0.615                         │
│    hybrid_prob = 0.65 × 0.92 + 0.35 × 0.615                        │
│               = 0.598 + 0.215                                        │
│               = 0.813                                                │
│                                                                      │
│  Output:  hybrid_prob = 0.813                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### Why 0.65 / 0.35?

```
RF (supervised, 0.65):   Trained on labelled normal+attack data.
                         Directly learns attack patterns → high precision.

IF+LOF (unsupervised, 0.35):  Trained on normal-only data.
                               Detects NOVEL attacks RF hasn't seen,
                               but also noisier → lower weight.
```

### Alternative: `max` method (not currently active)

```
max_anomaly = max(IF_norm, LOF_norm)
hybrid_prob = max(0.65 × RF_prob, 0.35 × max_anomaly)

This lets a single highly-anomalous detector override RF,
which causes more false positives → switched to weighted_avg.
```

---

## 10. Threshold & Classification

```
┌──────────────────────────────────────────────────────────────────────┐
│  ThresholdOptimizer                                                  │
│                                                                      │
│  threshold = 0.50 (optimised on validation set during training)      │
│                                                                      │
│  predict(hybrid_prob):                                               │
│    is_intrusion = (hybrid_prob >= 0.50)                              │
│                                                                      │
│  Examples:                                                           │
│    hybrid_prob = 0.813 → is_intrusion = True  (attack)              │
│    hybrid_prob = 0.490 → is_intrusion = False (normal)              │
│    hybrid_prob = 0.500 → is_intrusion = True  (borderline attack)   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 11. Alert Level Determination

After obtaining the probability and binary prediction, `determine_alert_level()`
maps them to a human-readable severity:

```
┌──────────────────────────────────────────────────────────────────────┐
│  determine_alert_level(probability, is_intrusion)                    │
│                                                                      │
│  ┌──────────────────┬──────────┬───────────────────────────────────┐ │
│  │ Condition         │  Level   │ Message                           │ │
│  ├──────────────────┼──────────┼───────────────────────────────────┤ │
│  │ prob ≥ 0.90       │ CRITICAL │ "Critical threat detected with   │ │
│  │                   │          │  very high confidence -           │ │
│  │                   │          │  Immediate action required"       │ │
│  ├──────────────────┼──────────┼───────────────────────────────────┤ │
│  │ prob ≥ 0.70       │ HIGH     │ "Intrusion detected with high    │ │
│  │                   │          │  confidence rate: Proceed with    │ │
│  │                   │          │  caution"                         │ │
│  ├──────────────────┼──────────┼───────────────────────────────────┤ │
│  │ prob ≥ 0.50       │ MEDIUM   │ "Suspicious activity detected -  │ │
│  │                   │          │  Action recommended"              │ │
│  ├──────────────────┼──────────┼───────────────────────────────────┤ │
│  │ prob ≥ 0.40       │ LOW      │ "Borderline Anomaly detected:    │ │
│  │                   │          │  Close monitoring required"       │ │
│  ├──────────────────┼──────────┼───────────────────────────────────┤ │
│  │ prob < 0.40       │ LOW      │ "Low confidence anomaly - May be │ │
│  │ AND is_intrusion  │          │  it could be a False Anomaly"     │ │
│  ├──────────────────┼──────────┼───────────────────────────────────┤ │
│  │ prob < 0.40       │ NORMAL   │ "No intrusion detected -          │ │
│  │ AND NOT intrusion │          │  Normal Traffic"                  │ │
│  └──────────────────┴──────────┴───────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

Visual scale:

```
 0.0          0.40         0.50         0.70         0.90         1.0
  ├────────────┼────────────┼────────────┼────────────┼────────────┤
  │   NORMAL   │    LOW     │   MEDIUM   │    HIGH    │  CRITICAL  │
  │ (if !intr) │            │            │            │            │
  │   LOW      │            │            │            │            │
  │ (if intr)  │            │            │            │            │
  └────────────┴────────────┴────────────┴────────────┴────────────┘
```

---

## 12. Health & Model Info Endpoints

### `GET /health`

```
┌───────────────────────────────────────────────┐
│  Try:                                          │
│    pipeline = get_pipeline()                   │
│    model_loaded = pipeline.fitted  → True      │
│    version = "latest"                          │
│  Except:                                       │
│    model_loaded = False                        │
│    version = "unknown"                         │
│                                                │
│  Response:                                     │
│  {                                             │
│    "status": "healthy" or "degraded",          │
│    "model_loaded": true/false,                 │
│    "version": "latest"                         │
│  }                                             │
└───────────────────────────────────────────────┘
```

### `GET /model/info`

```
┌───────────────────────────────────────────────┐
│  Reads directly from the loaded pipeline:      │
│                                                │
│  Response:                                     │
│  {                                             │
│    "version": "latest",                        │
│    "threshold": 0.50,                          │
│    "supervised_weight": 0.65,                  │
│    "ensemble_method": "weighted_avg",          │
│    "use_autoencoder": true,                    │
│    "use_pca": true,                            │
│    "anomaly_detectors": [                      │
│      "isolation_forest", "lof"                 │
│    ],                                          │
│    "num_features": 100                         │
│  }                                             │
└───────────────────────────────────────────────┘
```

---

## 13. Hot Reload Endpoint

### `POST /model/reload?version=latest`

```
┌───────────────────────────────────────────────────────────────┐
│  1. reload_pipeline("latest")                                  │
│       │                                                        │
│       ├── global _pipeline = None                              │
│       ├── ModelManager.load_pipeline("latest")                 │
│       │     └── (full artifact loading — same as startup)      │
│       └── global _pipeline = new_pipeline                      │
│                                                                │
│  2. Response:                                                  │
│  {                                                             │
│    "message": "Model version 'latest' loaded successfully",    │
│    "threshold": 0.50,                                          │
│    "supervised_weight": 0.65                                   │
│  }                                                             │
│                                                                │
│  Use case: Retrain the model → POST /model/reload              │
│  No server restart needed.                                     │
└───────────────────────────────────────────────────────────────┘
```

---

## 14. Request / Response Schemas

### `NetworkTrafficInput` (Request Body)

All fields are **Optional**. Send only what your data has.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Supports TWO datasets simultaneously:                               │
│                                                                      │
│  UNSW-NB15 Features (~40 fields):                                    │
│    dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl,            │
│    sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit,         │
│    swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat,               │
│    smean, dmean, trans_depth, response_body_len,                    │
│    ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm,        │
│    ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd,    │
│    ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports,     │
│    proto, state                                                      │
│                                                                      │
│    + Capitalised variants: Spkts, Dpkts, Sload, Dload,             │
│      Sintpkt, Dintpkt, Sjit, Djit, smeansz, dmeansz, res_bdy_len  │
│                                                                      │
│  KDD Cup 1999 Features (~40 fields):                                 │
│    duration, protocol_type, service, flag, src_bytes, dst_bytes,    │
│    land, wrong_fragment, urgent, hot, num_failed_logins,            │
│    logged_in, num_compromised, root_shell, su_attempted,            │
│    num_root, num_file_creations, num_shells, num_access_files,      │
│    num_outbound_cmds, is_host_login, is_guest_login, count,        │
│    srv_count, serror_rate, srv_serror_rate, rerror_rate, ...        │
│                                                                      │
│  to_dict(): Returns {key: value} for non-None fields only           │
└─────────────────────────────────────────────────────────────────────┘
```

### `PredictionOutput` (Response Body)

```json
{
  "is_intrusion": true,
  "confidence": 0.87,
  "intrusion_probability": 0.87,
  "alert_level": "HIGH",
  "alert_message": "Intrusion detected with high confidence rate: Proceed with caution"
}
```

| Field                  | Type   | Description                                       |
|------------------------|--------|---------------------------------------------------|
| `is_intrusion`         | bool   | `True` if hybrid_prob ≥ threshold                |
| `confidence`           | float  | Same as intrusion_probability (0–1)              |
| `intrusion_probability`| float  | Raw hybrid ensemble probability (0–1)            |
| `alert_level`          | string | CRITICAL / HIGH / MEDIUM / LOW / NORMAL           |
| `alert_message`        | string | Human-readable explanation                        |

### `BatchPredictionResponse`

```json
{
  "predictions": [ /* list of PredictionOutput */ ],
  "count": 10,
  "intrusions_detected": 3
}
```

---

## 15. Error Handling

```
┌──────────────────────────────────────────────────────────────────────┐
│  Error Scenarios & HTTP Responses                                    │
│                                                                      │
│  Model not loaded:                                                   │
│    GET /health → 200 { "status": "degraded", "model_loaded": false } │
│    POST /predict → 500 "Prediction error: Pipeline must be fitted"   │
│                                                                      │
│  Invalid input (missing required types, wrong types):                │
│    POST /predict → 422 Unprocessable Entity (Pydantic validation)    │
│                                                                      │
│  Internal pipeline error (shape mismatch, NaN, etc.):                │
│    POST /predict → 500 "Prediction error: <details>"                │
│    POST /predict/batch → 500 "Batch prediction error: <details>"    │
│                                                                      │
│  Model reload failure:                                               │
│    POST /model/reload → 500 "Error reloading model: <details>"      │
│                                                                      │
│  Model info error:                                                   │
│    GET /model/info → 500 "Error getting model info: <details>"       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 16. End-to-End Worked Example

Let's trace a single DNS query sample from HTTP request to final response.

### Step 1 — Client sends request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dur": 0.001,
    "proto": "udp",
    "sbytes": 62,
    "dbytes": 141,
    "spkts": 1,
    "dpkts": 1,
    "sttl": 64,
    "dttl": 128,
    "rate": 1000.0,
    "sload": 496000.0,
    "dload": 1128000.0,
    "ct_dst_ltm": 2,
    "ct_src_ltm": 5
  }'
```

### Step 2 — Pydantic validates → `to_dict()` strips None fields

```python
sample_dict = {
    "dur": 0.001, "proto": "udp", "sbytes": 62, "dbytes": 141,
    "spkts": 1, "dpkts": 1, "sttl": 64, "dttl": 128,
    "rate": 1000.0, "sload": 496000.0, "dload": 1128000.0,
    "ct_dst_ltm": 2, "ct_src_ltm": 5
}
# ~70 other Optional fields were None → excluded
```

### Step 3 — Create DataFrame

```
   dur  proto  sbytes  dbytes  spkts  dpkts  sttl  dttl  rate    sload     dload  ct_dst_ltm  ct_src_ltm
0  0.001  udp    62     141      1      1     64    128  1000.0  496000.0  1128000.0    2           5
```

### Step 4 — `preprocessor.transform()`

```
Drop labels (none present) → Drop constant cols (none) →
One-hot encode "proto" → proto_udp=1, proto_tcp=0, proto_arp=0, ... →
Reindex to 160 training columns (missing ones = 0) →

Result: 1 row × 160 columns
```

### Step 5 — Feature engineering

```
bytes_ratio    = 62 / (141 + 1) = 0.437
pkt_ratio      = 1 / (1 + 1) = 0.500
total_pkts     = 1 + 1 = 2
bytes_per_pkt  = (62 + 141) / (1 + 1 + 1) = 67.67
load_asymmetry = |496000 - 1128000| / (496000 + 1128000 + 1) = 0.389
ttl_asymmetry  = |64 - 128| = 64

Result: 1 row × ~175 columns
```

### Step 6 — Drop correlated features → Scale

```
Drop ~75 correlated columns → 1 row × ~100 columns
RobustScaler: (x - median) / IQR per column
Result: numpy float32 array, shape (1, 100)
```

### Step 7 — Score computation

```
Path A — Supervised:
  encoder.predict(X_scaled)     → shape (1, 32)
  RF.predict_proba(X_encoded)   → [0.08]  (RF says 8% chance of attack)

Path B — Unsupervised:
  IF.score(X_scaled)  → [0.42]  (raw IF score)
  LOF.score(X_scaled) → [5.3]   (raw LOF score)
```

### Step 8 — Normalise

```
IF:  (0.42 - 0.40) / (0.58 - 0.40) = 0.111
LOF: (5.3 - 1.2) / (80.0 - 1.2)    = 0.052
```

### Step 9 — Ensemble

```
avg_anomaly = (0.111 + 0.052) / 2 = 0.082
hybrid_prob = 0.65 × 0.08 + 0.35 × 0.082
            = 0.052 + 0.029
            = 0.081
```

### Step 10 — Threshold + Alert

```
is_intrusion = (0.081 >= 0.50) → False
determine_alert_level(0.081, False) → ("NORMAL", "No intrusion detected - Normal Traffic")
```

### Step 11 — Response

```json
{
  "is_intrusion": false,
  "confidence": 0.081,
  "intrusion_probability": 0.081,
  "alert_level": "NORMAL",
  "alert_message": "No intrusion detected - Normal Traffic"
}
```

✅ Correctly classified as normal traffic.

---

## Summary Diagram — Complete Request Lifecycle

```
                    ┌────────────────┐
                    │   HTTP Client   │
                    └───────┬────────┘
                            │ POST /predict
                            │ JSON body
                            ▼
                    ┌────────────────┐
                    │  Pydantic      │
                    │  Validation    │
                    │  + to_dict()   │
                    └───────┬────────┘
                            │ dict of non-None fields
                            ▼
                    ┌────────────────┐
                    │ pd.DataFrame   │
                    │ (1 row × N)    │
                    └───────┬────────┘
                            │
              ══════════════╪══════════════════════════
              ║  _transform_features(X)               ║
              ║             │                          ║
              ║    ┌────────┴────────┐                 ║
              ║    │  Preprocessor   │                 ║
              ║    │  transform()    │                 ║
              ║    └────────┬────────┘                 ║
              ║             │                          ║
              ║    ┌────────┴────────┐                 ║
              ║    │  Feature Eng.   │                 ║
              ║    │  (stats+ctx)    │                 ║
              ║    └────────┬────────┘                 ║
              ║             │                          ║
              ║    ┌────────┴────────┐                 ║
              ║    │  Drop Corr.     │                 ║
              ║    └────────┬────────┘                 ║
              ║             │                          ║
              ║    ┌────────┴────────┐                 ║
              ║    │  RobustScaler   │                 ║
              ║    └────────┬────────┘                 ║
              ║             │  X_scaled                ║
              ══════════════╪══════════════════════════
                            │
           ┌────────────────┼───────────────┐
           │                │               │
           ▼                ▼               ▼
    ┌─────────────┐  ┌───────────┐  ┌───────────┐
    │  AE Encode  │  │    IF     │  │   LOF     │
    │  → 32-dim   │  │  score()  │  │  score()  │
    └──────┬──────┘  └─────┬─────┘  └─────┬─────┘
           │               │               │
           ▼               └───────┬───────┘
    ┌─────────────┐                │
    │     RF      │         ┌──────┴──────┐
    │ predict_    │         │ Normalizer  │
    │  proba()    │         │  (p5/p95)   │
    └──────┬──────┘         └──────┬──────┘
           │                       │
           │ RF_prob (0–1)         │ IF_norm, LOF_norm (0–1)
           │                       │
           └───────────┬───────────┘
                       │
                ┌──────┴──────┐
                │  Ensemble   │
                │  0.65×RF +  │
                │ 0.35×avg(   │
                │  IF, LOF)   │
                └──────┬──────┘
                       │
                       │ hybrid_prob
                       ▼
                ┌──────────────┐
                │  Threshold   │
                │  ≥ 0.50 ?    │
                └──────┬───────┘
                       │
                       │ is_intrusion (bool)
                       ▼
                ┌──────────────┐
                │  Alert Level │
                │  Mapping     │
                └──────┬───────┘
                       │
                       ▼
                ┌────────────────┐
                │ JSON Response  │
                │ {is_intrusion, │
                │  confidence,   │
                │  alert_level,  │
                │  alert_message}│
                └────────────────┘
```
