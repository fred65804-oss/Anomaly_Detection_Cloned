"""
Evaluation module for Hybrid IDS
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss, classification_report
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_model(pipeline, X, y, dataset_name="Test", verbose=True):
    """
    Evaluate pipeline on a dataset
    
    Args:
        pipeline: Trained HybridIDSPipeline
        X: Features
        y: True labels
        dataset_name: Name of dataset for printing
        verbose: Whether to print results
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    y_probs = pipeline.predict_proba(X)
    y_pred = pipeline.predict(X)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'log_loss': log_loss(y, y_probs),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"{dataset_name.upper()} PERFORMANCE - HYBRID MODEL")
        print(f"{'='*70}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Log Loss:  {metrics['log_loss']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  TN: {metrics['confusion_matrix'][0,0]:6d}  FP: {metrics['confusion_matrix'][0,1]:6d}")
        print(f"  FN: {metrics['confusion_matrix'][1,0]:6d}  TP: {metrics['confusion_matrix'][1,1]:6d}")
    
    return metrics


def evaluate_supervised_only(pipeline, X, y, dataset_name="Test", verbose=True):
    """
    Evaluate supervised model only (for comparison)
    
    Args:
        pipeline: Trained HybridIDSPipeline
        X: Features
        y: True labels
        dataset_name: Name of dataset for printing
        verbose: Whether to print results
        
    Returns:
        Dictionary of metrics
    """
    # Transform features
    X_scaled = pipeline._transform_features(X)
    
    # Get supervised probabilities
    sup_probs = pipeline._get_supervised_probs(X_scaled)
    
    # Predict with threshold 0.5
    sup_pred = (sup_probs >= 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y, sup_pred),
        'precision': precision_score(y, sup_pred, zero_division=0),
        'recall': recall_score(y, sup_pred, zero_division=0),
        'f1': f1_score(y, sup_pred, zero_division=0),
        'log_loss': log_loss(y, sup_probs),
        'confusion_matrix': confusion_matrix(y, sup_pred)
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"{dataset_name.upper()} PERFORMANCE - SUPERVISED ONLY")
        print(f"{'='*70}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Log Loss:  {metrics['log_loss']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  TN: {metrics['confusion_matrix'][0,0]:6d}  FP: {metrics['confusion_matrix'][0,1]:6d}")
        print(f"  FN: {metrics['confusion_matrix'][1,0]:6d}  TP: {metrics['confusion_matrix'][1,1]:6d}")
    
    return metrics


def print_improvement_summary(hybrid_metrics, supervised_metrics, dataset_name="Test"):
    """
    Print comparison between hybrid and supervised models
    
    Args:
        hybrid_metrics: Metrics from hybrid model
        supervised_metrics: Metrics from supervised-only model
        dataset_name: Name of dataset
    """
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT SUMMARY - {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Recall improvement
    recall_imp = hybrid_metrics['recall'] - supervised_metrics['recall']
    print(f"\n  Recall improvement: {recall_imp:+.4f} ({recall_imp*100:+.1f}%)")
    
    # Precision change
    precision_imp = hybrid_metrics['precision'] - supervised_metrics['precision']
    print(f"  Precision change:   {precision_imp:+.4f} ({precision_imp*100:+.1f}%)")
    
    # F1 improvement
    f1_imp = hybrid_metrics['f1'] - supervised_metrics['f1']
    print(f"  F1 improvement:     {f1_imp:+.4f} ({f1_imp*100:+.1f}%)")
    
    # Log loss improvement (lower is better)
    logloss_imp = supervised_metrics['log_loss'] - hybrid_metrics['log_loss']
    print(f"  Log Loss improvement: {logloss_imp:+.4f} (lower is better)")
    
    # False negative reduction
    fn_reduction = supervised_metrics['confusion_matrix'][1,0] - hybrid_metrics['confusion_matrix'][1,0]
    print(f"  False negatives reduced: {fn_reduction:+d} attacks")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    from utils import ModelManager
    import pandas as pd
    
    # Load pipeline
    print("Loading pipeline...")
    model_manager = ModelManager("artifacts")
    pipeline = model_manager.load_pipeline("latest")
    
    # Load test data
    print("Loading test data...")
    df_test = pd.read_csv("training/KDDTest.csv")
    df_test["is_intrusion"] = (df_test["attack_class"].str.lower() != "normal").astype(int)
    
    X_test = df_test.drop(columns=["is_intrusion"], errors='ignore')
    y_test = df_test["is_intrusion"]
    
    # Evaluate
    print("\nEvaluating...")
    hybrid_metrics = evaluate_model(pipeline, X_test, y_test, "Test")
    supervised_metrics = evaluate_supervised_only(pipeline, X_test, y_test, "Test")
    
    # Print comparison
    print_improvement_summary(hybrid_metrics, supervised_metrics, "Test")
