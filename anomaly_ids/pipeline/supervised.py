"""
Supervised Random Forest classifier for intrusion detection
"""

from sklearn.ensemble import RandomForestClassifier


class SupervisedRF:
    """
    Random Forest classifier wrapper for supervised intrusion detection
    """
    
    def __init__(self, n_estimators=100, max_depth=20, min_samples_split=20,
                 min_samples_leaf=10, max_features='sqrt', class_weight='balanced',
                 random_state=42, n_jobs=-1):
        """
        Initialize Random Forest classifier
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            max_features: Features to consider for splits
            class_weight: Class balancing strategy
            random_state: Random seed
            n_jobs: Parallel jobs
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state
        )
        self.fitted = False
    
    def fit(self, X, y):
        """
        Train the classifier
        
        Args:
            X: Training features (numpy array or DataFrame)
            y: Training labels (0=normal, 1=intrusion)
        """
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict_proba(self, X):
        """
        Get probability predictions for intrusion class
        
        Args:
            X: Data to predict
            
        Returns:
            Probabilities for class 1 (intrusion)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """
        Get binary predictions
        
        Args:
            X: Data to predict
            
        Returns:
            Binary predictions (0=normal, 1=intrusion)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
