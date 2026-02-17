"""
Novel Attack Analysis for Hybrid IDS
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_novel_attacks(pipeline, X_test, y_test, test_attack_labels, novel_attacks, verbose=True):
    """
    Analyze model performance on novel vs known attacks
    
    Args:
        pipeline: Trained HybridIDSPipeline
        X_test: Test features
        y_test: Test labels
        test_attack_labels: Attack type labels for test set
        novel_attacks: Set of novel attack types
        verbose: Whether to print results
        
    Returns:
        Dictionary with analysis results
    """
    # Get predictions
    y_probs_hybrid = pipeline.predict_proba(X_test)
    y_pred_hybrid = pipeline.predict(X_test)
    
    # Get supervised-only predictions
    X_scaled = pipeline._transform_features(X_test)
    sup_probs = pipeline._get_supervised_probs(X_scaled)
    sup_pred = (sup_probs >= 0.5).astype(int)
    
    # Create masks
    novel_mask = test_attack_labels.isin(novel_attacks)
    known_attack_mask = (~novel_mask) & (y_test == 1)
    normal_mask = (y_test == 0)
    
    results = {
        'novel': {},
        'known': {},
        'normal': {}
    }
    
    # Analyze Novel Attacks
    if novel_mask.sum() > 0:
        novel_idx = novel_mask.values
        
        results['novel']['count'] = novel_idx.sum()
        results['novel']['hybrid_recall'] = recall_score(y_test[novel_idx], y_pred_hybrid[novel_idx])
        results['novel']['supervised_recall'] = recall_score(y_test[novel_idx], sup_pred[novel_idx])
        results['novel']['hybrid_precision'] = precision_score(y_test[novel_idx], y_pred_hybrid[novel_idx], zero_division=0)
        results['novel']['supervised_precision'] = precision_score(y_test[novel_idx], sup_pred[novel_idx], zero_division=0)
        results['novel']['hybrid_logloss'] = log_loss(y_test[novel_idx], y_probs_hybrid[novel_idx], labels=[0, 1])
        results['novel']['supervised_logloss'] = log_loss(y_test[novel_idx], sup_probs[novel_idx], labels=[0, 1])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"NOVEL ATTACK ANALYSIS")
            print(f"{'='*70}")
            print(f"\nNovel Attacks ({results['novel']['count']} samples):")
            print(f"  Attack types: {len(novel_attacks)}")
            if len(novel_attacks) <= 10:
                print(f"  Types: {sorted(novel_attacks)}")
            
            print(f"\n  [HYBRID MODEL]")
            print(f"    Recall:    {results['novel']['hybrid_recall']:.4f}")
            print(f"    Precision: {results['novel']['hybrid_precision']:.4f}")
            print(f"    Log Loss:  {results['novel']['hybrid_logloss']:.4f}")
            
            print(f"\n  [SUPERVISED ONLY]")
            print(f"    Recall:    {results['novel']['supervised_recall']:.4f}")
            print(f"    Precision: {results['novel']['supervised_precision']:.4f}")
            print(f"    Log Loss:  {results['novel']['supervised_logloss']:.4f}")
            
            improvement = (results['novel']['hybrid_recall'] - results['novel']['supervised_recall']) * 100
            print(f"\n  Recall Improvement: {improvement:+.1f}%")
    
    # Analyze Known Attacks
    if known_attack_mask.sum() > 0:
        known_idx = known_attack_mask.values
        
        results['known']['count'] = known_idx.sum()
        results['known']['hybrid_recall'] = recall_score(y_test[known_idx], y_pred_hybrid[known_idx])
        results['known']['supervised_recall'] = recall_score(y_test[known_idx], sup_pred[known_idx])
        results['known']['hybrid_logloss'] = log_loss(y_test[known_idx], y_probs_hybrid[known_idx], labels=[0, 1])
        results['known']['supervised_logloss'] = log_loss(y_test[known_idx], sup_probs[known_idx], labels=[0, 1])
        
        if verbose:
            print(f"\nKnown Attacks ({results['known']['count']} samples):")
            print(f"  [HYBRID]     Recall: {results['known']['hybrid_recall']:.4f}, Log Loss: {results['known']['hybrid_logloss']:.4f}")
            print(f"  [SUPERVISED] Recall: {results['known']['supervised_recall']:.4f}, Log Loss: {results['known']['supervised_logloss']:.4f}")
    
    # Analyze Normal Traffic
    if normal_mask.sum() > 0:
        normal_idx = normal_mask.values
        
        results['normal']['count'] = normal_idx.sum()
        # For normal traffic, we care about precision (not flagging normal as attack)
        results['normal']['hybrid_precision'] = precision_score(1 - y_test[normal_idx], 1 - y_pred_hybrid[normal_idx], zero_division=0)
        results['normal']['supervised_precision'] = precision_score(1 - y_test[normal_idx], 1 - sup_pred[normal_idx], zero_division=0)
        
        if verbose:
            print(f"\nNormal Traffic ({results['normal']['count']} samples):")
            print(f"  [HYBRID]     Specificity: {results['normal']['hybrid_precision']:.4f}")
            print(f"  [SUPERVISED] Specificity: {results['normal']['supervised_precision']:.4f}")
    
    if verbose:
        print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    from utils import ModelManager
    
    # Load pipeline
    print("Loading pipeline...")
    model_manager = ModelManager("artifacts")
    pipeline = model_manager.load_pipeline("latest")
    
    # Load test data
    print("Loading test data...")
    df_train = pd.read_csv("training/KDDTrain.csv")
    df_test = pd.read_csv("training/KDDTest.csv")
    
    # Create targets
    df_test["is_intrusion"] = (df_test["attack_class"].str.lower() != "normal").astype(int)
    
    # Identify novel attacks
    train_attacks = set(df_train["attack_class"].unique())
    test_attacks = set(df_test["attack_class"].unique())
    novel_attacks = test_attacks - train_attacks
    
    print(f"Novel attacks: {len(novel_attacks)}")
    
    # Prepare data
    X_test = df_test.drop(columns=["is_intrusion"], errors='ignore')
    y_test = df_test["is_intrusion"]
    test_attack_labels = df_test["attack_class"]
    
    # Analyze
    results = analyze_novel_attacks(
        pipeline, X_test, y_test, 
        test_attack_labels, novel_attacks,
        verbose=True
    )
