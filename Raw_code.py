"""
HYBRID IDS - OPTIMIZED VERSION

Key improvements:
1. Better anomaly detector parameters
2. Adaptive ensemble weighting
3. Per-detector threshold optimization
4. Focus on novel attack detection
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, log_loss
)
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ======================================================
# CONFIGURATION - OPTIMIZED FOR NOVEL ATTACKS
# ======================================================
CONFIG = {
    'use_autoencoder': True,
    'use_pca_features': True,
    'anomaly_detectors': ['isolation_forest', 'lof'],  # Only the best ones
    'supervised_weight': 0.3,  # Lower weight = more unsupervised influence
    'ensemble_method': 'max',  # Use max instead of average
    'optimize_weights': True,  # Optimize weights on validation
}

print("="*70)
print("HYBRID IDS - OPTIMIZED VERSION")
print("="*70)

# ======================================================
# 1. LOAD DATA
# ======================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

df_train = pd.read_csv(r"C:\Anomaly_Detection\anomaly_ids\training\KDDTrain.csv")
df_test = pd.read_csv(r"C:\Anomaly_Detection\anomaly_ids\training\KDDTest.csv")

print(f"Train: {len(df_train)}, Test: {len(df_test)}")

# ======================================================
# 2. PREPROCESSING
# ======================================================
print("\n" + "="*70)
print("PREPROCESSING")
print("="*70)

# Remove constant columns
drop_cols = [
    col for col in df_train.columns
    if df_train[col].nunique() == 1 or df_train[col].nunique() == len(df_train)
]
df_train.drop(columns=drop_cols, inplace=True, errors='ignore')
df_test.drop(columns=drop_cols, inplace=True, errors='ignore')

# Binary target
df_train["is_intrusion"] = (df_train["attack_class"].str.lower() != "normal").astype(int)
df_test["is_intrusion"] = (df_test["attack_class"].str.lower() != "normal").astype(int)

# Store attack info
test_attack_labels = df_test["attack_class"].copy()
train_attacks = set(df_train["attack_class"].unique())
test_attacks = set(df_test["attack_class"].unique())
novel_attacks = test_attacks - train_attacks
test_novel_mask = test_attack_labels.isin(novel_attacks)

print(f"Novel attacks: {len(novel_attacks)}")
print(f"Novel test samples: {test_novel_mask.sum()} ({test_novel_mask.mean()*100:.1f}%)")

# Drop attack columns
df_train.drop(columns=["attack_class", "attack_class_category"], errors="ignore", inplace=True)
df_test.drop(columns=["attack_class", "attack_class_category"], errors="ignore", inplace=True)

# Features
X = df_train.drop("is_intrusion", axis=1)
y = df_train["is_intrusion"]
X_test = df_test.drop("is_intrusion", axis=1)
y_test = df_test["is_intrusion"]

# One-hot encoding
categorical_cols = [col for col in X.columns if col in ["protocol_type", "service", "flag"]]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# ======================================================
# 3. TRAIN/VAL SPLIT
# ======================================================
print("\n" + "="*70)
print("CREATING SPLITS")
print("="*70)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42 # 'stratify = y' will keep ratio normal/attack same in both splits
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ======================================================
# 4. FEATURE ENGINEERING
# ======================================================
def add_statistical_features(df):
    df = df.copy()
    if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
        df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
        df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
    if 'count' in df.columns and 'srv_count' in df.columns:
        df['srv_ratio'] = df['srv_count'] / (df['count'] + 1)
    if 'duration' in df.columns and 'count' in df.columns:
        df['packet_rate'] = df['count'] / (df['duration'] + 1)
    return df

X_train = add_statistical_features(X_train)
X_val = add_statistical_features(X_val)
X_test = add_statistical_features(X_test)

# Remove correlated features (train only)
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

if to_drop:
    print(f"Dropping {len(to_drop)} correlated features")
    X_train.drop(columns=to_drop, inplace=True)
    X_val.drop(columns=to_drop, inplace=True)
    X_test.drop(columns=to_drop, inplace=True)

print(f"Final features: {X_train.shape[1]}")
feature_names = list(X_train.columns)

# ======================================================
# 5. SCALING
# ======================================================
print("\n" + "="*70)
print("SCALING")
print("="*70)

scaler = RobustScaler()
# Before scaling, we will ensure we have all columns as float type
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train_arr = scaler.fit_transform(X_train).astype('float32')
X_val_arr = scaler.transform(X_val).astype('float32')
X_test_arr = scaler.transform(X_test).astype('float32')

y_train_arr = y_train.values
y_val_arr = y_val.values
y_test_arr = y_test.values

# Extract normal traffic
X_normal_train = X_train_arr[y_train_arr == 0]
X_normal_val = X_val_arr[y_val_arr == 0]

print(f"Normal train: {len(X_normal_train)}")

# ======================================================
# 6. AUTOENCODER
# ======================================================
if CONFIG['use_autoencoder']:
    print("\n" + "="*70)
    print("TRAINING AUTOENCODER")
    print("="*70)
    
    input_dim = X_train_arr.shape[1]
    encoding_dim = 32
    
    encoder_input = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(encoder_input)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    history = autoencoder.fit(
        X_normal_train, X_normal_train,
        epochs=20,
        batch_size=256,
        validation_data=(X_normal_val, X_normal_val),
        verbose=0
    )
    
    encoder = keras.Model(encoder_input, encoded)
    
    X_train_encoded = encoder.predict(X_train_arr, verbose=0)
    X_val_encoded = encoder.predict(X_val_arr, verbose=0)
    X_test_encoded = encoder.predict(X_test_arr, verbose=0)
    
    # Reconstruction errors
    ae_scores_train = np.mean(np.square(X_train_arr - autoencoder.predict(X_train_arr, verbose=0)), axis=1)
    ae_scores_val = np.mean(np.square(X_val_arr - autoencoder.predict(X_val_arr, verbose=0)), axis=1)
    ae_scores_test = np.mean(np.square(X_test_arr - autoencoder.predict(X_test_arr, verbose=0)), axis=1)
    
    print(f"Train loss: {history.history['loss'][-1]:.6f}")

# ======================================================
# 7. PCA
# ======================================================
if CONFIG['use_pca_features']:
    print("\n" + "="*70)
    print("PCA")
    print("="*70)
    
    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(X_normal_train)
    
    X_train_pca = pca.transform(X_train_arr)
    X_val_pca = pca.transform(X_val_arr)
    X_test_pca = pca.transform(X_test_arr)
    
    print(f"PCA components: {pca.n_components_}")

# ======================================================
# 8. ANOMALY DETECTORS - OPTIMIZED PARAMETERS
# ======================================================
print("\n" + "="*70)
print("TRAINING ANOMALY DETECTORS")
print("="*70)

anomaly_models = {}
anomaly_scores = {'train': {}, 'val': {}, 'test': {}}

# Isolation Forest
if 'isolation_forest' in CONFIG['anomaly_detectors']:
    print("\n[1/2] Isolation Forest (aggressive)...")
    iso_forest = IsolationForest(
        n_estimators=300,  # More trees
        max_samples=256,  # Smaller samples = more sensitive
        contamination=0.30,  # Higher = more sensitive to anomalies
        random_state=42,
        n_jobs=-1,
        max_features=0.7  # Use fewer features per tree
    )
    iso_forest.fit(X_normal_train)
    anomaly_models['isolation_forest'] = iso_forest
    
    anomaly_scores['train']['isolation_forest'] = -iso_forest.score_samples(X_train_arr)
    anomaly_scores['val']['isolation_forest'] = -iso_forest.score_samples(X_val_arr)
    anomaly_scores['test']['isolation_forest'] = -iso_forest.score_samples(X_test_arr)
    print("Trained")

# LOF
if 'lof' in CONFIG['anomaly_detectors']:
    print("\n[2/2] LOF (aggressive)...")
    lof = LocalOutlierFactor(
        n_neighbors=10,  # Fewer neighbors = more sensitive
        contamination=0.30,  # Higher sensitivity
        novelty=True,
        n_jobs=-1
    )
    lof.fit(X_normal_train)
    anomaly_models['lof'] = lof
    
    anomaly_scores['train']['lof'] = -lof.score_samples(X_train_arr)
    anomaly_scores['val']['lof'] = -lof.score_samples(X_val_arr)
    anomaly_scores['test']['lof'] = -lof.score_samples(X_test_arr)
    print("Trained the Local Outlier Factor")

# Add autoencoder scores
if CONFIG['use_autoencoder']:
    anomaly_scores['train']['autoencoder'] = ae_scores_train
    anomaly_scores['val']['autoencoder'] = ae_scores_val
    anomaly_scores['test']['autoencoder'] = ae_scores_test

print(f"\n {len(anomaly_models) + (1 if CONFIG['use_autoencoder'] else 0)} detectors")

# ======================================================
# 9. SUPERVISED MODEL
# ======================================================
print("\n" + "="*70)
print("TRAINING SUPERVISED MODEL")
print("="*70)

supervised_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

if CONFIG['use_autoencoder']:
    supervised_model.fit(X_train_encoded, y_train_arr)
    sup_probs_train = supervised_model.predict_proba(X_train_encoded)[:, 1]
    sup_probs_val = supervised_model.predict_proba(X_val_encoded)[:, 1]
    sup_probs_test = supervised_model.predict_proba(X_test_encoded)[:, 1]
else:
    supervised_model.fit(X_train_arr, y_train_arr)
    sup_probs_train = supervised_model.predict_proba(X_train_arr)[:, 1]
    sup_probs_val = supervised_model.predict_proba(X_val_arr)[:, 1]
    sup_probs_test = supervised_model.predict_proba(X_test_arr)[:, 1]

print("Supervised trained")

# ======================================================
# 10. NORMALIZE SCORES
# ======================================================

# Some observations => Z-score was also applied over here. Z-score expects your data to have a symmetric distribution,
# and no tails. By default, network(or anomaly data to be precise) is not symmetrically distributed, and has long tails.
# So, implementing the Z-score was not ideal in this scenario 
print("\n" + "="*70)
print("NORMALIZING SCORES using percentile method")
print("="*70)

def normalize_scores(scores_dict):
    normalized = {}
    for name, scores in scores_dict.items():
        # Use percentile normalization
        p5 = np.percentile(scores, 5)
        p95 = np.percentile(scores, 95)
        normalized[name] = np.clip((scores - p5) / (p95 - p5 + 1e-10), 0, 1)
    return normalized

anomaly_scores_norm = {
    'train': normalize_scores(anomaly_scores['train']),
    'val': normalize_scores(anomaly_scores['val']),
    'test': normalize_scores(anomaly_scores['test'])
}

print("Normalized using percentile method")

# ======================================================
# 11. OPTIMIZE ENSEMBLE WEIGHTS ON VALIDATION
# ======================================================
print("\n" + "="*70)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("="*70)

if CONFIG['optimize_weights']:
    best_recall = 0
    best_sup_weight = 0.3
    
    print("\nTesting different supervised weights...")
    for sup_weight in np.arange(0.1, 0.7, 0.05):
        # Test with max ensemble
        anom_array = np.column_stack(list(anomaly_scores_norm['val'].values()))
        max_anom = np.max(anom_array, axis=1)
        hybrid_probs = np.maximum(sup_weight * sup_probs_val, (1 - sup_weight) * max_anom)
        
        # Find best threshold for this weight
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.3, 0.7, 0.05):
            preds = (hybrid_probs >= t).astype(int)
            recall = recall_score(y_val_arr, preds)
            if recall > best_recall:
                best_recall = recall
                best_sup_weight = sup_weight
                best_t = t
    
    CONFIG['supervised_weight'] = best_sup_weight
    print(f"\n Optimal supervised weight: {best_sup_weight:.2f}")
    print(f"  Best validation recall: {best_recall:.4f}")

# ======================================================
# 12. BUILD FINAL ENSEMBLE
# ======================================================
print("\n" + "="*70)
print("BUILDING ENSEMBLE")
print("="*70)

def hybrid_ensemble_max(sup_prob, anom_scores, sup_weight):
    """Use MAX ensemble - if ANY detector fires, flag it"""
    anom_array = np.column_stack(list(anom_scores.values()))
    max_anom = np.max(anom_array, axis=1)
    
    # Take maximum of weighted components
    return np.maximum(sup_weight * sup_prob, (1 - sup_weight) * max_anom)

hybrid_probs_train = hybrid_ensemble_max(sup_probs_train, anomaly_scores_norm['train'], CONFIG['supervised_weight'])
hybrid_probs_val = hybrid_ensemble_max(sup_probs_val, anomaly_scores_norm['val'], CONFIG['supervised_weight'])
hybrid_probs_test = hybrid_ensemble_max(sup_probs_test, anomaly_scores_norm['test'], CONFIG['supervised_weight'])

print(f"Method: MAX ensemble")
print(f"Weights: Supervised={CONFIG['supervised_weight']:.2f}, Unsupervised={1-CONFIG['supervised_weight']:.2f}")

# ======================================================
# 13. THRESHOLD OPTIMIZATION
# ======================================================
print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION")
print("="*70)

best_threshold = 0.5
best_recall = 0

for t in np.arange(0.2, 0.8, 0.01):
    preds = (hybrid_probs_val >= t).astype(int)
    recall = recall_score(y_val_arr, preds)
    if recall > best_recall:
        best_recall = recall
        best_threshold = t

print(f"Best threshold: {best_threshold:.3f} (Recall={best_recall:.4f})")

# ======================================================
# 14. EVALUATION
# ======================================================
print("\n" + "="*70)
print("VALIDATION PERFORMANCE")
print("="*70)

val_preds_hybrid = (hybrid_probs_val >= best_threshold).astype(int)
val_preds_sup = (sup_probs_val >= 0.5).astype(int)

val_logloss_hybrid = log_loss(y_val_arr, hybrid_probs_val)
val_logloss_sup = log_loss(y_val_arr, sup_probs_val)

print("\n[HYBRID]")
print(f"  Recall:    {recall_score(y_val_arr, val_preds_hybrid):.4f}")
print(f"  Precision: {precision_score(y_val_arr, val_preds_hybrid):.4f}")
print(f"  F1:        {f1_score(y_val_arr, val_preds_hybrid):.4f}")
print(f"  Log Loss:  {val_logloss_hybrid:.4f}")

print("\n[SUPERVISED]")
print(f"  Recall:    {recall_score(y_val_arr, val_preds_sup):.4f}")
print(f"  Precision: {precision_score(y_val_arr, val_preds_sup):.4f}")
print(f"  F1:        {f1_score(y_val_arr, val_preds_sup):.4f}")
print(f"  Log Loss:  {val_logloss_sup:.4f}")

print("\n" + "="*70)
print("TEST PERFORMANCE")
print("="*70)

test_preds_hybrid = (hybrid_probs_test >= best_threshold).astype(int)
test_preds_sup = (sup_probs_test >= 0.5).astype(int)

cm_hybrid = confusion_matrix(y_test_arr, test_preds_hybrid)
cm_sup = confusion_matrix(y_test_arr, test_preds_sup)

test_logloss_hybrid = log_loss(y_test_arr, hybrid_probs_test)
test_logloss_sup = log_loss(y_test_arr, sup_probs_test)

print("\n[HYBRID]")
print(f"  Recall:    {recall_score(y_test_arr, test_preds_hybrid):.4f}")
print(f"  Precision: {precision_score(y_test_arr, test_preds_hybrid):.4f}")
print(f"  F1:        {f1_score(y_test_arr, test_preds_hybrid):.4f}")
print(f"  Log Loss:  {test_logloss_hybrid:.4f}")
print(f"  FN:        {cm_hybrid[1,0]}")

print("\n[SUPERVISED]")
print(f"  Recall:    {recall_score(y_test_arr, test_preds_sup):.4f}")
print(f"  Precision: {precision_score(y_test_arr, test_preds_sup):.4f}")
print(f"  F1:        {f1_score(y_test_arr, test_preds_sup):.4f}")
print(f"  Log Loss:  {test_logloss_sup:.4f}")
print(f"  FN:        {cm_sup[1,0]}")

# ======================================================
# 15. NOVEL ATTACK ANALYSIS
# ======================================================

# This analysis is done, as to understand how the model behaves when we expose it to attacks, specifically
# those attacks which the model has never seen before
print("\n" + "="*70)
print("NOVEL ATTACK ANALYSIS")
print("="*70)

novel_idx = test_novel_mask.values # attacks which the model has never seen before
known_idx = (~test_novel_mask).values & (y_test_arr == 1) # Points which are not unknown attacks (attacks which the model has already seen) 

if novel_idx.sum() > 0:
    print(f"\nNovel Attacks ({novel_idx.sum()} samples):")
    hybrid_novel_recall = recall_score(y_test_arr[novel_idx], test_preds_hybrid[novel_idx]) # using 'novel_idx' will give indexes of all those rows which have novel attacks(which the model never sees during training)
    sup_novel_recall = recall_score(y_test_arr[novel_idx], test_preds_sup[novel_idx])
    novel_logloss_hybrid = log_loss(y_test_arr[novel_idx], hybrid_probs_test[novel_idx], labels=[0, 1])
    novel_logloss_sup = log_loss(y_test_arr[novel_idx], sup_probs_test[novel_idx], labels=[0, 1])
    print(f"  [HYBRID]     Recall: {hybrid_novel_recall:.4f}, Log Loss: {novel_logloss_hybrid:.4f}")
    print(f"  [SUPERVISED] Recall: {sup_novel_recall:.4f}, Log Loss: {novel_logloss_sup:.4f}")
    print(f"  Improvement: {(hybrid_novel_recall - sup_novel_recall)*100:+.1f}%")

if known_idx.sum() > 0:
    print(f"\nKnown Attacks ({known_idx.sum()} samples):")
    known_logloss_hybrid = log_loss(y_test_arr[known_idx], hybrid_probs_test[known_idx], labels=[0, 1])
    known_logloss_sup = log_loss(y_test_arr[known_idx], sup_probs_test[known_idx], labels=[0, 1])
    print(f"  [HYBRID]     Recall: {recall_score(y_test_arr[known_idx], test_preds_hybrid[known_idx]):.4f}, Log Loss: {known_logloss_hybrid:.4f}")
    print(f"  [SUPERVISED] Recall: {recall_score(y_test_arr[known_idx], test_preds_sup[known_idx]):.4f}, Log Loss: {known_logloss_sup:.4f}")

# ======================================================
# 16. IMPROVEMENT SUMMARY
# ======================================================
print("\n" + "="*70)
print("IMPROVEMENT SUMMARY")
print("="*70)

recall_imp = recall_score(y_test_arr, test_preds_hybrid) - recall_score(y_test_arr, test_preds_sup)
fn_reduction = cm_sup[1, 0] - cm_hybrid[1, 0]
logloss_imp = test_logloss_sup - test_logloss_hybrid  # Lower log loss is better

print(f"\nOverall (Test Set):")
print(f"  Recall improvement: {recall_imp:+.4f} ({recall_imp*100:+.1f}%)")
print(f"  Log Loss improvement: {logloss_imp:+.4f} (lower is better)")
print(f"  FN reduction:       {fn_reduction:+d} attacks")

print(f"\nTraining Data Log Loss:")
train_logloss_hybrid = log_loss(y_train_arr, hybrid_probs_train)
train_logloss_sup = log_loss(y_train_arr, sup_probs_train)
print(f"  [HYBRID]     {train_logloss_hybrid:.4f}")
print(f"  [SUPERVISED] {train_logloss_sup:.4f}")

print(f"\nValidation Data Log Loss:")
print(f"  [HYBRID]     {val_logloss_hybrid:.4f}")
print(f"  [SUPERVISED] {val_logloss_sup:.4f}")

print(f"\nTest Data Log Loss:")
print(f"  [HYBRID]     {test_logloss_hybrid:.4f}")
print(f"  [SUPERVISED] {test_logloss_sup:.4f}")

# ======================================================
# 17. SAVE MODELS
# ======================================================
joblib.dump(supervised_model, "hybrid_supervised.joblib")
joblib.dump(anomaly_models, "hybrid_anomaly_detectors.joblib")
joblib.dump(scaler, "hybrid_scaler.joblib")
joblib.dump(feature_names, "hybrid_features.joblib")
joblib.dump(best_threshold, "hybrid_threshold.joblib")
joblib.dump(CONFIG, "hybrid_config.joblib")

if CONFIG['use_autoencoder']:
    autoencoder.save("hybrid_autoencoder.keras")
    encoder.save("hybrid_encoder.keras")

if CONFIG['use_pca_features']:
    joblib.dump(pca, "hybrid_pca.joblib")

print("\n Models saved")
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
