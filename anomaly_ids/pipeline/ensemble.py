"""
Hybrid ensemble combining supervised and unsupervised detectors
"""

import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score
from scipy.special import logsumexp

class HybridEnsemble:
    """
    Hybrid ensemble using MAX strategy
    Combines supervised probabilities with unsupervised anomaly scores
    """
    
    def __init__(self, supervised_weight=0.3, method='max'):
        """
        Initialize hybrid ensemble
        
        Args:
            supervised_weight: Weight for supervised component (0 to 1)
            method: Ensemble method ('max' or 'weighted_avg')
        """
        self.supervised_weight = supervised_weight
        self.method = method
    
    def predict_proba(self, sup_probs, anomaly_scores_dict):
        """
        Combine supervised and unsupervised scores
        
        Args:
            sup_probs: Supervised model probabilities
            anomaly_scores_dict: Dictionary of normalized anomaly scores
            
        Returns:
            Hybrid anomaly probabilities
        """
        # Stack all anomaly scores in the form of a 2D array
        anom_array = np.column_stack(list(anomaly_scores_dict.values()))
        
        if self.method == 'max':
            # Since we have to maximize, we will take maximum anomaly score across all detectors
            max_anom = np.max(anom_array, axis=1)
            
            # Take maximum of weighted supervised and unsupervised
            unsup_weight = 1 - self.supervised_weight # Weights also work like probabilities
            hybrid_probs = np.maximum(
                self.supervised_weight * sup_probs,
                unsup_weight * max_anom
            )
        elif self.method == 'weighted_avg':
            # Now our motive changes to taking an average, so we will take average of all anomaly scores
            avg_anom = np.mean(anom_array, axis=1)
            
            # Weighted average of supervised and unsupervised
            unsup_weight = 1 - self.supervised_weight
            hybrid_probs = (
                self.supervised_weight * sup_probs +
                unsup_weight * avg_anom
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return hybrid_probs


def optimize_supervised_weight(sup_probs_val, anomaly_scores_val, y_val,
                               weight_min=0.1, weight_max=0.7, weight_step=0.05,
                               threshold_min=0.3, threshold_max=0.7, threshold_step=0.05,
                               method='max', optimize_for='f1',
                               min_recall=0.90):
    """
    Optimize supervised weight using validation set
    
    Args:
        sup_probs_val: Supervised probabilities on validation set
        anomaly_scores_val: Dictionary of normalized anomaly scores on validation set
        y_val: True labels for validation set
        weight_min: Minimum weight to test
        weight_max: Maximum weight to test
        weight_step: Step size for weight grid search
        threshold_min: Minimum threshold to test
        threshold_max: Maximum threshold to test
        threshold_step: Step size for threshold grid search
        method: Ensemble method ('max' or 'weighted_avg')
        optimize_for: Metric to optimize ('recall', 'f1', 'precision')
        min_recall: Minimum recall required (acts as a floor constraint)
        
    Returns:
        best_weight: Optimal supervised weight
        best_threshold: Optimal classification threshold
        best_score: Best validation score achieved
    """
    from sklearn.metrics import f1_score, precision_score
    
    best_score = 0
    best_weight = 0.3
    best_threshold = 0.5 # Defaulting to 0.5 as the threshold
    
    for sup_weight in np.arange(weight_min, weight_max + weight_step, weight_step):
        # Create ensemble with this weight
        ensemble = HybridEnsemble(supervised_weight=sup_weight, method=method)
        hybrid_probs = ensemble.predict_proba(sup_probs_val, anomaly_scores_val)
        
        # Find best threshold for this weight
        for t in np.arange(threshold_min, threshold_max, threshold_step):
            preds = (hybrid_probs >= t).astype(int)
            
            # Enforce minimum recall constraint
            rec = recall_score(y_val, preds)
            if rec < min_recall:
                continue  # Skip configs that sacrifice too much recall
            
            if optimize_for == 'f1':
                score = f1_score(y_val, preds)
            elif optimize_for == 'recall':
                score = rec
            elif optimize_for == 'precision':
                score = precision_score(y_val, preds)
            else:
                score = f1_score(y_val, preds)
            
            if score > best_score:
                best_score = score
                best_weight = sup_weight
                best_threshold = t
    
    # If no configuration met the min_recall constraint, fall back to
    # optimizing recall without the floor (safety net)
    if best_score == 0:
        print("  ⚠ No config met min_recall constraint. Relaxing to recall-only optimization.")
        for sup_weight in np.arange(weight_min, weight_max + weight_step, weight_step):
            ensemble = HybridEnsemble(supervised_weight=sup_weight, method=method)
            hybrid_probs = ensemble.predict_proba(sup_probs_val, anomaly_scores_val)
            for t in np.arange(threshold_min, threshold_max, threshold_step):
                preds = (hybrid_probs >= t).astype(int)
                score = f1_score(y_val, preds)
                if score > best_score:
                    best_score = score
                    best_weight = sup_weight
                    best_threshold = t
    
    return best_weight, best_threshold, best_score
