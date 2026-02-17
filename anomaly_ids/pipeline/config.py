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
        self.supervised_weight = 0.10  # Optimal weight found through testing
        self.ensemble_method = 'max' # Using max ensemble
        self.optimize_weights = False  # Use manual optimal values instead of auto-optimization
        
        # Autoencoder parameters
        self.ae_encoding_dim = 32
        self.ae_epochs = 20
        self.ae_batch_size = 256
        self.ae_dropout = 0.2
        
        # Anomaly detector parameters
        self.iso_n_estimators = 300
        self.iso_max_samples = 256
        self.iso_contamination = 0.30
        self.iso_max_features = 0.7
        
        self.lof_n_neighbors = 10
        self.lof_contamination = 0.30
        
        # Supervised model parameters
        self.rf_n_estimators = 100
        self.rf_max_depth = 20
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
