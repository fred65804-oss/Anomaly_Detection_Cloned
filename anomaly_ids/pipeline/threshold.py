"""
    Threshold optimization for binary classification
    Optimization is done on the basis of the number 
"""

import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


class ThresholdOptimizer:
    """
    Optimizes classification threshold on validation set
    """
    
    def __init__(self, threshold_min=0.2, threshold_max=0.8, threshold_step=0.01,
                 optimize_for='recall', default_threshold = 0.55):
        """
        Initialize threshold optimizer
        
        Args:
            threshold_min: Minimum threshold to test
            threshold_max: Maximum threshold to test
            threshold_step: Step size for grid search
            optimize_for: Metric to optimize ('recall', 'precision', 'f1')
            default_threshold: Default threshold to be used
        """
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
        self.optimize_for = optimize_for
        self.default_threshold = default_threshold
        self.best_threshold = default_threshold  # Start with the default, not 0.5
        self.best_score = 0
    
    def optimize(self, y_true, y_probs):
        """
        Find optimal threshold on validation set using F1 score
        
        Args:
            y_true: True labels
            y_probs: Predicted probabilities
            
        Returns:
            best_threshold: Optimal threshold
            best_score: Best score achieved
        """
        best_score = 0
        best_threshold = self.default_threshold
        
        for t in np.arange(self.threshold_min, self.threshold_max, self.threshold_step):
            preds = (y_probs >= t).astype(int)
            
            # Always use F1 for threshold optimization (balances precision & recall)
            score = f1_score(y_true, preds)
            
            if score > best_score:
                best_score = score
                best_threshold = t
        
        self.best_threshold = best_threshold
        self.best_score = best_score
        
        return best_threshold, best_score
    
    def predict(self, y_probs):
        """
        Make predictions using optimized threshold
        
        Args:
            y_probs: Predicted probabilities
            
        Returns:
            Binary predictions
        """
        return (y_probs >= self.best_threshold).astype(int)
