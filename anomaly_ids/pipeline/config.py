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
        self.supervised_weight = 0.65  # RF is dominant: at weighted_avg, unsup (LOF/IF) can't outvote RF when it says normal
        self.ensemble_method = 'weighted_avg'
        # ENABLING BELOW OPTION(SET TO 'TRUE') WILL RETRAIN THE ENTIRE MODEL(RECOMMENDED FOR NEWER DATASETS)
        self.optimize_weights = False  # Use manual optimal values instead of auto-optimization
        
        # Autoencoder parameters
        self.ae_encoding_dim = 32 # Final output dimensions
        self.ae_epochs = 10             # Reduced from 20
        self.ae_batch_size = 256
        self.ae_dropout = 0.2 # 20% of neurons will be dropped out from the previous layer
        
        # Anomaly detector parameters
        self.iso_n_estimators = 100        # Reduced from 300 (300 is overkill with max_samples=256)
        self.iso_max_samples = 256
        self.iso_contamination = 0.15  # Match actual attack rate (~12.7%). 0.30 was too aggressive → high false positives
        self.iso_max_features = 0.7
        
        self.lof_n_neighbors = 20          # More neighbors => smoother, less noise-sensitive boundary
        self.lof_contamination = 0.15  # Match actual attack rate (was approximately 12.7%)
        self.lof_max_samples = 50_000    # LOF is O(n*k*log n) — its own cap separate from AE/IF

        # Max rows used to train RF (In most cases, we will not consider the entire dataset, as some datasets can cross more than 1 million rows also)
        # For now, we will put a cap on rows fed to 500k
        # Can be set to 'None' to use the entire data
        self.rf_max_samples = 500_000

        # Max rows fed to unsupervised detectors (IF, LOF, AE) during fit.
        # LOF is O(n^2) – training on millions of rows takes hours.
        # 200_000 gives better coverage of the normal distribution
        # Set to None to disable the cap (use all data)
        self.unsupervised_max_samples = 200_000
        
        # Supervised model parameters(Random Forest)
        self.rf_n_estimators = 50          # Sufficient for large datasets
        self.rf_max_depth = 15             # Prevents overfitting + faster
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
        self.default_threshold = 0.50  # Optimal threshold found through testing. Can train the entire pipeline again, looking for a better threshold. Generally done for new datasets.

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
