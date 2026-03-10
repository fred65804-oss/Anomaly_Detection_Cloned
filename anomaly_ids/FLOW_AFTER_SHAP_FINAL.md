# Hybrid IDS — Complete Code Flow

> **Last Updated**: February 2025  
> **Dataset**: UNSW-NB15 (also supports KDD Cup 99 and any CSV with a label column)  
> **Stack**: Python · FastAPI · TensorFlow/Keras · scikit-learn · SHAP · LIME

---

## Table of Contents

1. [Project Structure at a Glance](#1-project-structure-at-a-glance)
2. [Component Map](#2-component-map)
3. [Training Flow](#3-training-flow)
4. [Inference Flow](#4-inference-flow)
5. [Explanation Flow (SHAP / LIME)](#5-explanation-flow-shap--lime)
6. [Model Persistence Flow](#6-model-persistence-flow)
7. [API Layer](#7-api-layer)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [Key Design Decisions](#9-key-design-decisions)

---

## 1. Project Structure at a Glance

```
anomaly_ids/
├── training/
│   └── train.py                ← Entry point for training
├── pipeline/
│   ├── config.py               ← IDSConfig (all hyperparameters)
│   ├── pipeline.py             ← HybridIDSPipeline (master orchestrator)
│   ├── preprocessing.py        ← Preprocessor + drop_correlated_features
│   ├── feature_engineering.py  ← add_statistical_features, add_context_aware_features, PCATransformer
│   ├── autoencoder.py          ← AutoencoderIDS (Dense AE, encoding_dim=32)
│   ├── anomaly_models.py       ← IsolationForestDetector, LocalOutlierFactorDetector
│   ├── supervised.py           ← SupervisedRF (Random Forest)
│   ├── normalization.py        ← ScoreNormalizer (percentile-based)
│   ├── ensemble.py             ← HybridEnsemble + optimize_supervised_weight
│   └── threshold.py            ← ThresholdOptimizer
├── utils/
│   ├── model_manager.py        ← ModelManager (save / load all artifacts)
│   ├── explainer.py            ← IDSExplainer (SHAP + LIME wrapper)
│   ├── metrics.py              ← Evaluation helpers
│   └── logger.py               ← setup_logger
├── app/
│   ├── main.py                 ← FastAPI app + all endpoints
│   ├── dependencies.py         ← Lazy pipeline loading (get_pipeline)
│   └── schemas.py              ← Pydantic request / response models
└── artifacts/
    └── latest/                 ← Saved model artifacts (joblib + .h5)
```

---

## 2. Component Map

| Component | Class / Function | File | Role |
|---|---|---|---|
| Config | `IDSConfig` | `pipeline/config.py` | All hyperparameters (AE, RF, LOF, IF, ensemble, thresholds) |
| Preprocessor | `Preprocessor` | `pipeline/preprocessing.py` | Drop labels, one-hot encode categoricals, drop constant columns, align columns |
| Corr Drop | `drop_correlated_features()` | `pipeline/preprocessing.py` | Remove features with Pearson |r| > 0.95 |
| Feature Eng. | `add_statistical_features()` | `pipeline/feature_engineering.py` | Ratio features, log transforms (e.g. bytes ratio, packet size) |
| Feature Eng. | `add_context_aware_features()` | `pipeline/feature_engineering.py` | Coerce all numeric cols to float, `fillna(0)` |
| PCA | `PCATransformer` | `pipeline/feature_engineering.py` | Optional PCA dimensionality reduction |
| Scaler | `RobustScaler` | `pipeline/pipeline.py` (init) | Robust scaling (percentile-based, outlier-resistant) |
| Autoencoder | `AutoencoderIDS` | `pipeline/autoencoder.py` | Dense AE trained on normal-only data → produces 32-dim encoding + reconstruction error |
| IF Detector | `IsolationForestDetector` | `pipeline/anomaly_models.py` | Unsupervised anomaly scoring (trained on normal-only) |
| LOF Detector | `LocalOutlierFactorDetector` | `pipeline/anomaly_models.py` | Unsupervised anomaly scoring, novelty=True (trained on normal-only) |
| Supervised RF | `SupervisedRF` | `pipeline/supervised.py` | Random Forest on AE-encoded features → P(intrusion) |
| Normalizer | `ScoreNormalizer` | `pipeline/normalization.py` | Clips unsupervised scores to [0,1] using p5/p95 percentiles |
| Ensemble | `HybridEnsemble` | `pipeline/ensemble.py` | Combines supervised + unsupervised via MAX or weighted average |
| Weight Optim. | `optimize_supervised_weight()` | `pipeline/ensemble.py` | Grid search over (weight, threshold) on validation set, maximises F1 with recall ≥ 0.90 |
| Threshold | `ThresholdOptimizer` | `pipeline/threshold.py` | Picks classification threshold that maximises F1 on validation |
| Explainer | `IDSExplainer` | `utils/explainer.py` | SHAP KernelExplainer + LIME LimeTabularExplainer on post-transform features |
| Pipeline | `HybridIDSPipeline` | `pipeline/pipeline.py` | Ties all components together for fit / predict / explain |
| Model Mgr | `ModelManager` | `utils/model_manager.py` | joblib + .h5 save/load; auto-reloads SHAP background on load |

---

## 3. Training Flow

**Entry point**: `python training/train.py`

```
train.py
│
├─ [1/5] Load CSV(s)
│       Single file  →  train_test_split (80/20, stratified, random_state=42)
│       Two files    →  KDDTrain.csv + KDDTest.csv loaded separately
│       Strip whitespace from all column names
│
├─ [2/5] Create Binary Target
│       Auto-detect label column: "attack_class" | "label" | "class" | "target" | "attack" | "intrusion"
│       Auto-detect normal class: "normal" string  OR  integer 0
│       df["is_intrusion"] = (label != normal_label).astype(int)
│       Log novel attacks (in test but not in train)
│
├─ [3/5] Prepare Features
│       X_train_full = df_train.drop("is_intrusion")
│       y_train_full = df_train["is_intrusion"]
│
├─ [4/5] Train / Val Split
│       train_test_split(X_train_full, test_size=0.2, stratify=y_train_full)
│       → X_train, X_val, y_train, y_val
│
├─ [5/5] pipeline.fit(X_train, y_train, X_val, y_val)
│        ↓  (see Pipeline Fit below)
│
├─ init_explainer()
│       Sample 200 normal rows from df_train
│       Drop label + "is_intrusion" columns
│       pipeline.init_explainer(X_normal_background)
│       Save shap_background.joblib
│
└─ model_manager.save_pipeline(pipeline, version="latest")
```

### Pipeline Fit — 10 Steps (`pipeline.fit`)

```
Step 1 — Preprocessing
    preprocessor.fit_transform(X_train)
    → drops label cols, one-hot encodes categoricals (KDD: protocol/service/flag; others: auto-detected)
    → drops constant / all-unique columns
    → stores training_columns for test-time alignment
    → result: X_train_processed (DataFrame)

Step 2 — Feature Engineering
    add_statistical_features(X_train_processed)
    → adds ratio/log/packet-size derived features

    add_context_aware_features(X_train_processed)
    → pd.to_numeric(errors='coerce') on all numeric candidates
    → fillna(0) on the same set (prevents NaN from reaching sklearn)
    → result: enriched DataFrame

Step 3 — Drop Correlated Features
    drop_correlated_features(X_train_processed, threshold=0.95)
    → samples up to 50K rows for correlation matrix computation
    → drops one column from each highly-correlated pair
    → stores self.correlated_features (list) for inference-time drop
    → updates self.feature_names

Step 4 — Scaling
    X_train_processed.astype('float32')
    scaler.fit(X_train_processed).transform(X_train_processed)
    → RobustScaler (percentile-based, outlier-resistant)
    → result: X_train_scaled (numpy array)

    Extract normal-only subset:
    X_normal_train = X_train_scaled[y_train == 0]
    Cap at unsupervised_max_samples (e.g. 100K) for LOF cost control

Step 5 — Autoencoder (if config.use_autoencoder=True)
    AutoencoderIDS(input_dim, encoding_dim=32, dropout=0.2, epochs=20, batch_size=256)
    Architecture:
        Encoder: input → Dense(128,relu) → Dropout → Dense(64,relu) → Dropout → Dense(32,relu)
        Decoder: Dense(64,relu) → Dropout → Dense(128,relu) → Dense(input_dim,linear)
    autoencoder.fit(X_normal_train_us, X_normal_val)
    → trained on normal-only data
    → learns what "normal" looks like; high reconstruction error = anomaly

Step 6 — PCA (if config.use_pca_features=True)
    PCATransformer(n_components=...).fit(X_normal_train_us)
    → optional dimensionality reduction on normal data

Step 7 — Anomaly Detectors
    IsolationForestDetector.fit(X_normal_train_us)
    → score() returns  -decision_function  (higher = more anomalous)

    LocalOutlierFactorDetector.fit(X_normal_train_us)
    → novelty=True; score() returns -decision_function

Step 8 — Supervised Model (Random Forest)
    If rf_max_samples defined → stratified sub-sample keeping attack ratio
    If use_autoencoder → X_rf_encoded = autoencoder.encode(X_rf)
    SupervisedRF.fit(X_rf_encoded, y_rf)
    → RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced')
    → returns P(intrusion) via predict_proba()[:, 1]

Step 9 — Weight + Threshold Optimization (if X_val provided)
    X_val_processed = _transform_features(X_val)
    sup_probs_val      = _get_supervised_probs(X_val_processed)
    anomaly_scores_val = _get_anomaly_scores(X_val_processed)
    normalizer.fit(anomaly_scores_val)
    anomaly_scores_val_norm = normalizer.transform(anomaly_scores_val)

    Grid search over (supervised_weight, threshold):
        weight ∈ [weight_min, weight_max]  step=weight_step
        threshold ∈ [threshold_min, threshold_max]  step=threshold_step
        Objective: maximise F1 subject to recall ≥ 0.90
    Stores: config.supervised_weight, threshold_optimizer.best_threshold

Step 10 — Ensemble
    HybridEnsemble(supervised_weight=best_weight, method='max')
    self.fitted = True
```

---

## 4. Inference Flow

**Triggered by**: `POST /predict` or direct `pipeline.predict_proba(X)`

```
Raw Input (DataFrame, 1 or N rows)
│
├─ _transform_features(X)
│   ├─ preprocessor.transform(X)
│   │   → drop label cols, one-hot encode, align to training_columns (fill missing with 0)
│   ├─ add_statistical_features()   → ratio/log features
│   ├─ add_context_aware_features() → numeric coercion + fillna(0)
│   ├─ drop correlated features     (self.correlated_features)
│   ├─ fillna(0)                    (safety net for edge-case NaNs)
│   ├─ astype('float32')
│   └─ scaler.transform()
│   → result: X_scaled (numpy array)
│
├─ _get_supervised_probs(X_scaled)
│   ├─ autoencoder.encode(X_scaled)  → X_encoded (shape: n × 32)
│   └─ supervised_model.predict_proba(X_encoded)[:, 1]
│   → sup_probs (shape: n,)
│
├─ _get_anomaly_scores(X_scaled)
│   ├─ autoencoder.reconstruction_error(X_scaled)   [only if weights_loaded]
│   ├─ isolation_forest.score(X_scaled)
│   └─ lof.score(X_scaled)
│   → scores dict: {'autoencoder': [...], 'isolation_forest': [...], 'lof': [...]}
│
├─ normalizer.transform(anomaly_scores)
│   → clips each score to [0,1] using fitted p5/p95 percentiles
│   → anomaly_scores_norm dict
│
├─ ensemble.predict_proba(sup_probs, anomaly_scores_norm)
│   MAX method:
│       max_anom = max over all detectors per row
│       hybrid = max(supervised_weight × sup_probs,
│                    (1 - supervised_weight) × max_anom)
│   → hybrid_probs (shape: n,)  — single intrusion probability per row
│
└─ threshold_optimizer.predict(hybrid_probs)
    → (hybrid_probs >= best_threshold).astype(int)
    → binary labels  0 = Normal,  1 = Intrusion
```

### Alert Level Logic (`determine_alert_level` in `main.py`)

| Probability | is_intrusion | Alert Level | Message |
|---|---|---|---|
| ≥ 0.90 | True | CRITICAL | Immediate action required |
| ≥ 0.70 | True | HIGH | High priority investigation |
| ≥ 0.50 | True | MEDIUM | Medium priority review |
| ≥ 0.40 | True | LOW | Low priority, monitor |
| < 0.40 | False | NORMAL | Normal traffic |

---

## 5. Explanation Flow (SHAP / LIME)

**Triggered by**: `POST /predict/explain?method=shap&top_k=10`

### Initialization (done once at training time or on API load)

```
pipeline.init_explainer(X_background)
│
├─ _transform_features(X_background)
│   → X_bg_transformed (scaled numpy array, 100–200 normal rows)
│
└─ IDSExplainer(pipeline, feature_names, X_bg_transformed)
    ├─ shap.KernelExplainer(model=_predict_fn,
    │                        data=shap.sample(X_bg_transformed, 100))
    └─ lime.LimeTabularExplainer(training_data=X_bg_transformed,
                                  feature_names=feature_names,
                                  class_names=['Normal','Intrusion'],
                                  mode='classification')
```

### Prediction + Explanation (`explain_prediction`)

```
Raw Input X (1-row DataFrame)
│
├─ Standard prediction (predict_proba + predict)  →  prob, is_intrusion
│
├─ _transform_features(X)
│   → X_scaled (scaled numpy array, already in SHAP's background space)
│
├─ method='shap'
│   IDSExplainer.explain_shap(X_scaled, top_k)
│   │
│   ├─ shap_explainer.shap_values(X_scaled, nsamples=100)
│   │   SHAP perturbs X_scaled in scaled feature space
│   │   For each perturbation → _predict_fn(X_perturbed)
│   │       → _predict_proba_transformed(X_perturbed)  [NO re-transform]
│   │           → _get_supervised_probs() + _get_anomaly_scores()
│   │           → normalizer → ensemble
│   │           → returns both[:, 1]  (intrusion probability only, 1-D)
│   │   → shap_values: 2-D array (n_samples × n_features)
│   │
│   ├─ row = shap_values[0]          (first/only sample)
│   ├─ top_indices = argsort(|row|)[::-1][:top_k]
│   └─ returns list of {feature, shap_value, direction}
│       direction: "toward_intrusion" if shap_value > 0
│                  "toward_normal"    if shap_value < 0
│
├─ method='lime'
│   IDSExplainer.explain_lime(X_scaled, top_k)
│   │
│   ├─ lime_explainer.explain_instance(X_scaled[0], _predict_fn_lime, num_features=top_k)
│   │   _predict_fn_lime returns shape (n, 2): [P(normal), P(intrusion)]
│   └─ returns list of {feature, weight, direction}
│
└─ method='both'
    → { 'shap': [...], 'lime': [...] }
```

> **Why two predict functions?**  
> SHAP `KernelExplainer` works best with a scalar output (1-D), so `_predict_fn` returns only `both[:, 1]`.  
> LIME always needs `[P(class0), P(class1)]` (2-D), so `_predict_fn_lime` returns both columns.  
> Both bypass `_transform_features` — they call `_predict_proba_transformed` directly, which runs only the scoring steps on the already-scaled numpy array. This prevents double-transformation of the input.

---

## 6. Model Persistence Flow

### Saving (`ModelManager.save_pipeline`)

```
artifacts/latest/
├── config.json                ← IDSConfig.to_dict()
├── params.joblib              ← feature_names, correlated_features, threshold, fitted
├── preprocessor.joblib        ← Preprocessor (training_columns, categorical_cols, drop_cols)
├── scaler.joblib              ← RobustScaler
├── pca.joblib                 ← PCATransformer  (if use_pca_features=True)
├── autoencoder.weights.h5     ← Keras weights only (portable across TF versions)
├── anomaly_detectors.joblib   ← {'isolation_forest': ..., 'lof': ...}
├── supervised_model.joblib    ← SupervisedRF (RandomForest)
├── normalizer.joblib          ← ScoreNormalizer (percentile boundaries)
├── ensemble.joblib            ← HybridEnsemble (weight, method)
├── threshold_optimizer.joblib ← ThresholdOptimizer (best_threshold)
└── shap_background.joblib     ← 200 normal rows (post-label-drop, pre-transform)
                                   used to reinitialize IDSExplainer on API load
```

> **Note**: `IDSExplainer` is NOT saved (SHAP/LIME internals cannot be joblib-serialized reliably). It is always recreated from `shap_background.joblib` at load time.

### Loading (`ModelManager.load_pipeline`)

```
load_pipeline("latest")
│
├─ Read config.json            → IDSConfig
├─ Create HybridIDSPipeline(config)
├─ Load params.joblib          → feature_names, correlated_features, threshold
├─ Load preprocessor.joblib
├─ Load scaler.joblib
├─ Load pca.joblib             (if exists)
├─ Load autoencoder.weights.h5
│   → Rebuild AutoencoderIDS(input_dim, encoding_dim, …)
│   → autoencoder.load_weights(weights_path)
│   → Fallback: try legacy .keras format
│   → If neither loads: weights_loaded=False (autoencoder scoring disabled)
├─ Load anomaly_detectors.joblib
├─ Load supervised_model.joblib
├─ Load normalizer.joblib
├─ Load ensemble.joblib
├─ Load threshold_optimizer.joblib
├─ pipeline.fitted = True
└─ If shap_background.joblib exists:
       X_background = joblib.load(...)
       pipeline.init_explainer(X_background)   ← recreates IDSExplainer
   Else:
       pipeline.explainer = None
       (WARN logged — re-run training to regenerate)
```

---

## 7. API Layer

**Server**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`  
**Docs**: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Schema | Description |
|---|---|---|---|
| `GET` | `/` | — | Root, returns API name + docs link |
| `GET` | `/health` | `HealthResponse` | API health + model loaded status |
| `GET` | `/model/info` | `ModelInfo` | Threshold, weights, config, feature count |
| `POST` | `/predict` | `NetworkTrafficInput` → `PredictionOutput` | Single-sample prediction |
| `POST` | `/predict/explain?method=shap&top_k=10` | `NetworkTrafficInput` → `ExplainedPredictionOutput` | Prediction + SHAP/LIME feature importance |
| `POST` | `/predict/batch` | `BatchPredictionRequest` → `BatchPredictionResponse` | Multi-sample batch prediction |
| `POST` | `/model/reload?version=latest` | — | Hot-reload model from disk (no restart needed) |

### Key Schemas

```
NetworkTrafficInput   — All UNSW-NB15 + KDD columns (all optional via to_dict())
PredictionOutput      — is_intrusion, confidence, intrusion_probability, alert_level, alert_message
ExplainedPredictionOutput (extends PredictionOutput)
                      — top_features_shap: List[FeatureContribution]
                      — top_features_lime: List[FeatureContribution]
                      — explanation_method: str
FeatureContribution   — feature: str, shap_value/weight: float, direction: str
BatchPredictionResponse — predictions: List[PredictionOutput], count, intrusions_detected
ModelInfo             — version, threshold, supervised_weight, ensemble_method, use_autoencoder, ...
HealthResponse        — status, model_loaded, version
```

### Dependency Injection (`dependencies.py`)

```
get_pipeline()
  → Lazily loads pipeline on first request
  → Returns cached global _pipeline instance
  → Thread-safe (FastAPI's async model)

reload_pipeline(version)
  → Calls ModelManager.load_pipeline(version)
  → Replaces global _pipeline
  → auto-reinitializes explainer from shap_background.joblib
```

---

## 8. Data Flow Diagram

```
                           ┌──────────────────────────────────────────────┐
                           │               TRAINING TIME                  │
                           └──────────────────────────────────────────────┘

  CSV File(s)
      │
      ▼
  train.py ──► load & split ──► create binary label ──► X_train, y_train, X_val, y_val
                                                              │
                                                              ▼
                                              ┌─────────────────────────────────┐
                                              │    HybridIDSPipeline.fit()      │
                                              │                                 │
                                              │  Preprocessor.fit_transform()   │
                                              │        ↓                        │
                                              │  add_statistical_features()     │
                                              │  add_context_aware_features()   │
                                              │        ↓                        │
                                              │  drop_correlated_features()     │
                                              │        ↓                        │
                                              │  RobustScaler.fit_transform()   │
                                              │        ↓                        │
                                              │  AutoencoderIDS.fit()  ┐        │
                                              │  (normal data only)    │        │
                                              │        ↓               │        │
                                              │  PCATransformer.fit()  │        │
                                              │        ↓               │        │
                                              │  IsolationForest.fit() │ unsup  │
                                              │  LOFDetector.fit()     ┘        │
                                              │        ↓                        │
                                              │  AE.encode() → RF.fit()        │
                                              │        ↓                        │
                                              │  optimize weights & threshold   │
                                              │        ↓                        │
                                              │  HybridEnsemble created         │
                                              └─────────────────────────────────┘
                                                              │
                                              init_explainer(normal_background)
                                                              │
                                              ModelManager.save_pipeline()
                                              + save shap_background.joblib


                           ┌──────────────────────────────────────────────┐
                           │               INFERENCE TIME                 │
                           └──────────────────────────────────────────────┘

  HTTP POST /predict
  { "sbytes": 100, "proto": "tcp", ... }
      │
      ▼
  FastAPI main.py
  dependencies.py → get_pipeline()   (lazy-load on first call)
      │
      ▼
  NetworkTrafficInput.to_dict() → pd.DataFrame
      │
      ▼
  pipeline.predict_proba(df)
      │
      ├─ _transform_features()
      │   Preprocessor.transform → stat features → context features
      │   → drop corr cols → fillna(0) → float32 → RobustScaler.transform
      │
      ├─ AE.encode()  →  RF.predict_proba()  →  sup_probs
      │
      ├─ AE.reconstruction_error() + IF.score() + LOF.score()  →  anomaly_scores
      │
      ├─ ScoreNormalizer.transform()  →  anomaly_scores_norm
      │
      └─ HybridEnsemble.predict_proba()  →  hybrid_prob
             ↓
         ThresholdOptimizer.predict()  →  0 or 1
             ↓
         determine_alert_level()  →  NORMAL/LOW/MEDIUM/HIGH/CRITICAL
             ↓
         PredictionOutput JSON response


                           ┌──────────────────────────────────────────────┐
                           │           EXPLANATION TIME (SHAP)            │
                           └──────────────────────────────────────────────┘

  HTTP POST /predict/explain?method=shap&top_k=10
      │
      ▼
  pipeline.explain_prediction(df, method='shap', top_k=10)
      │
      ├─ _transform_features(df) → X_scaled   [transform once]
      │
      └─ IDSExplainer.explain_shap(X_scaled, top_k)
              │
              ├─ KernelExplainer.shap_values(X_scaled, nsamples=100)
              │    For each SHAP perturbation of X_scaled:
              │        _predict_fn(X_perturbed)
              │            → _predict_proba_transformed(X_perturbed)   [no re-transform]
              │                → RF → IF → LOF → normalize → ensemble
              │                → returns both[:, 1]   (1-D intrusion prob)
              │
              ├─ row = shap_values[0]       (2D array, take row 0)
              ├─ sort by |shap_value| descending → top_k
              └─ [{feature, shap_value, direction}, ...]
                        ↓
              ExplainedPredictionOutput JSON response
```

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Train AE & unsupervised on normal-only** | AE learns the "normal" manifold; high reconstruction error = anomaly |
| **RF trained on AE-encoded features** | 32-dim encoding is more compact and anomaly-aware than raw scaled features |
| **MAX ensemble strategy** | Either the supervised model OR any anomaly detector can independently raise an alert — maximises recall |
| **RobustScaler** | Percentile-based scaling is resistant to extreme outlier values common in network traffic |
| **Percentile score normalization (p5/p95)** | Brings IF and LOF scores to [0,1] without assuming Gaussian distribution |
| **`unsupervised_max_samples` cap** | LOF is O(n²); capping at 100K prevents multi-hour training on large datasets |
| **`rf_max_samples` cap + stratified sampling** | RF converges well before full-data pass; stratify preserves attack ratio |
| **Two SHAP predict functions** | `_predict_fn` → 1-D (required for KernelExplainer); `_predict_fn_lime` → 2-D (required for LIME) |
| **SHAP operates on post-transform space** | Raw input has string columns (e.g., `proto="tcp"`); SHAP must perturb in the numeric scaled space |
| **`shap_background.joblib` saved separately** | `IDSExplainer` (SHAP/LIME internals) cannot be joblib-serialized; background data is saved instead and used to recreate the explainer on load |
| **`top_k: int` annotation in FastAPI** | FastAPI query parameters default to `str`; explicit `int` annotation prevents `TypeError: slice indices must be integers` |
| **`fillna(0)` after `pd.to_numeric(errors='coerce')`** | UNSW-NB15 can have object-typed numeric columns; coercion produces NaN which breaks sklearn; fill with 0 before any model sees the data |
| **Column alignment via `reindex(fill_value=0)`** | One-hot encoding at inference may produce fewer dummy columns than training if a categorical value is absent; `reindex` ensures shape consistency |
