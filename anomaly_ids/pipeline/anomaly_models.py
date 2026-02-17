"""
Anomaly detection models: Isolation Forest and Local Outlier Factor
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector wrapper
    Trained on normal traffic only
    """
    
    def __init__(self, n_estimators=300, max_samples=256, contamination=0.30, 
                 max_features=0.7, random_state=42, n_jobs=-1):
        """
        Initialize Isolation Forest detector
        
        Args:
            n_estimators: Number of trees
            max_samples: Number of samples per tree (smaller = more sensitive)
            contamination: Expected proportion of anomalies
            max_features: Features to use per tree
            random_state: Random seed
            n_jobs: Parallel jobs
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
            max_features=max_features
        )
        self.fitted = False
    
    def fit(self, X):
        """
        Fit on normal traffic data
        
        Args:
            X: Normal traffic samples (numpy array or DataFrame)
        """
        self.model.fit(X)
        self.fitted = True
        return self
    
    def score(self, X):
        """
        Get anomaly scores (higher = more anomalous)
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (negated decision function)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")
        return -self.model.score_samples(X)


class LocalOutlierFactorDetector:
    """
    Local Outlier Factor anomaly detector wrapper
    Trained on normal traffic only
    """
    
    def __init__(self, n_neighbors=10, contamination=0.30, n_jobs=-1):
        """
        Initialize LOF detector
        
        Args:
            n_neighbors: Number of neighbors (fewer = more sensitive)
            contamination: Expected proportion of anomalies
            n_jobs: Parallel jobs
        """
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,  # Enable novelty detection mode
            n_jobs=n_jobs
        )
        self.fitted = False
    
    def fit(self, X):
        """
        Fit on normal traffic data
        
        Args:
            X: Normal traffic samples (numpy array or DataFrame)
        """
        self.model.fit(X)
        self.fitted = True
        return self
    
    def score(self, X):
        """
        Get anomaly scores (higher = more anomalous)
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (negated decision function)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")
        return -self.model.score_samples(X)
