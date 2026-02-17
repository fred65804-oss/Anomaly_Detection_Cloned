"""
Score normalization using percentile method
Better for skewed distributions than Z-score normalization
"""

import numpy as np


class ScoreNormalizer:
    """
    Normalizes anomaly scores using percentile method
    Fits on training scores, transforms any set
    """
    
    def __init__(self, p_low=5, p_high=95):
        """
        Initialize normalizer
        
        Args:
            p_low: Lower percentile for normalization
            p_high: Upper percentile for normalization
        """
        self.p_low = p_low
        self.p_high = p_high
        self.percentiles = {}  # Stores p5, p95 for each detector
        self.fitted = False
    
    def fit(self, scores_dict):
        """
        Fit normalizer on training scores
        
        Args:
            scores_dict: Dictionary of {detector_name: scores_array}
        """
        self.percentiles = {}
        for name, scores in scores_dict.items():
            p_low_val = np.percentile(scores, self.p_low) # 5th percentile
            p_high_val = np.percentile(scores, self.p_high) # 95th percentile
            self.percentiles[name] = (p_low_val, p_high_val) # Storing the scores (Format: [model_name] : (5th percentile value, 95th percentile value))
        
        self.fitted = True
        return self
    
    def transform(self, scores_dict):
        """
        Normalize scores using fitted percentiles
        
        Args:
            scores_dict: Dictionary of {detector_name: scores_array}
            
        Returns:
            Dictionary of normalized scores (clipped to [0, 1])
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        normalized = {}
        for name, scores in scores_dict.items():
            if name not in self.percentiles:
                # Gracefully handle unseen detectors with fallback normalization
                s_min = np.min(scores)
                s_max = np.max(scores)
                normalized[name] = np.clip(
                    (scores - s_min) / (s_max - s_min + 1e-10),
                    0, 1
                )
                continue
            
            p_low_val, p_high_val = self.percentiles[name]
            # Normalize and clip to [0, 1]
            normalized[name] = np.clip(
                (scores - p_low_val) / (p_high_val - p_low_val + 1e-10),
                0, 1
            )
        
        return normalized
    
    def fit_transform(self, scores_dict):
        """
        Fit and transform in one step
        
        Args:
            scores_dict: Dictionary of {detector_name: scores_array}
            
        Returns:
            Dictionary of normalized scores
        """
        return self.fit(scores_dict).transform(scores_dict)
