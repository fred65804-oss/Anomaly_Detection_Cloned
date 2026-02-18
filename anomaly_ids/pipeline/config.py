"""
Configuration management for Hybrid IDS Pipeline
This file contains the entire configuration for RandomForest, AutoEncoder, LOF, PCA, Correlation, etc
"""
class IDSConfig:
    """Configuration class for Hybrid Intrusion Detection System"""
    
    def __init__(self, config_dict=None):
        """
        Initialize configuration with default or custom settings
        
        Args:
            config_dict: Optional dictionary with custom configuration
        """
        # Default configuration matching Raw_code.py
        self.use_autoencoder = True
        self.use_pca_features = True
        self.anomaly_detectors = ['isolation_forest', 'lof']
        self.supervised_weight = 0.35  # Increased from 0.10: RF sees full normal distribution, overrides IF/LOF false positives
        self.ensemble_method = 'max' # Using max ensemble
        self.optimize_weights = False  # Use manual optimal values instead of auto-optimization
        
        # Autoencoder parameters
        self.ae_encoding_dim = 32
        self.ae_epochs = 10             # Reduced from 20 (converges well before 20 on large data)
        self.ae_batch_size = 256
        self.ae_dropout = 0.2
        
        # Anomaly detector parameters
        self.iso_n_estimators = 100        # Reduced from 300 (300 is overkill with max_samples=256)
        self.iso_max_samples = 256
        self.iso_contamination = 0.15  # Match actual attack rate (~12.7%). 0.30 was too aggressive → high false positives
        self.iso_max_features = 0.7
        
        self.lof_n_neighbors = 20          # Increased from 10: more neighbors = smoother, less noise-sensitive boundary
        self.lof_contamination = 0.15  # Match actual attack rate (~12.7%). 0.30 was too aggressive → high false positives

        # Max rows fed to unsupervised detectors (IF, LOF, AE) during fit.
        # LOF is O(n^2) – training on millions of rows takes hours.
        # 200_000 gives better coverage of the normal distribution (was 100K = only 5.6% of normals).
        # Set to None to disable the cap (use all data).
        self.unsupervised_max_samples = 200_000
        
        # Supervised model parameters
        self.rf_n_estimators = 50          # Reduced from 100 (sufficient for large datasets)
        self.rf_max_depth = 15             # Reduced from 20 (prevents overfitting + faster)
        self.rf_min_samples_split = 20
        self.rf_min_samples_leaf = 10
        self.rf_max_features = 'sqrt'
        self.rf_class_weight = 'balanced'
        
        # PCA parameters
        self.pca_n_components = 0.95
        
        # Correlation threshold
        self.correlation_threshold = 0.95
        
        # Threshold optimization
        self.threshold_min = 0.2
        self.threshold_max = 0.8
        self.threshold_step = 0.01
        self.default_threshold = 0.50  # Optimal threshold found through testing

        # Weight optimization
        self.weight_min = 0.1
        self.weight_max = 0.7
        self.weight_step = 0.05
        
        # Override with custom config if provided
        if config_dict:
            self.update(config_dict)
    
    def update(self, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        return f"IDSConfig({self.to_dict()})"


# Default configuration instance
DEFAULT_CONFIG = IDSConfig()
