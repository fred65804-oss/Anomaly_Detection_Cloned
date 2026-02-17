"""
    Main Hybrid IDS Pipeline - Orchestrates all components
"""

import numpy as np
import pandas as pd
from .config import IDSConfig
from .preprocessing import Preprocessor, ScalerWrapper, drop_correlated_features
from .feature_engineering import add_statistical_features, add_context_aware_features, PCATransformer
from .autoencoder import AutoencoderIDS
from .anomaly_models import IsolationForestDetector, LocalOutlierFactorDetector
from .supervised import SupervisedRF
from .normalization import ScoreNormalizer
from .ensemble import HybridEnsemble, optimize_supervised_weight
from .threshold import ThresholdOptimizer


class HybridIDSPipeline:
    """
    Complete Hybrid Intrusion Detection System Pipeline
    
    Combines preprocessing, feature engineering, autoencoder, 
    anomaly detectors, supervised classifier, and ensemble
    """
    
    def __init__(self, config=None):
        """
        Initialize pipeline with configuration
        
        Args:
            config: IDSConfig instance or dict
        """
        if config is None:
            self.config = IDSConfig()
        elif isinstance(config, dict):
            self.config = IDSConfig(config)
        else:
            self.config = config
        
        # Initialize components
        self.preprocessor = Preprocessor()
        self.scaler = ScalerWrapper()
        self.pca = None
        self.autoencoder = None
        self.encoder = None
        self.anomaly_detectors = {}
        self.supervised_model = None
        self.normalizer = ScoreNormalizer()
        self.ensemble = None
        self.threshold_optimizer = ThresholdOptimizer(
            threshold_min=self.config.threshold_min,
            threshold_max=self.config.threshold_max,
            threshold_step=self.config.threshold_step,
            default_threshold = getattr(self.config, 'default_threshold', 0.55)
        )
        
        self.feature_names = None
        self.correlated_features = []
        self.fitted = False
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """
        Fit the entire pipeline
        
        Args:
            X_train: Training features (DataFrame)
            y_train: Training labels (0=normal, 1=intrusion)
            X_val: Validation features (optional, for optimization)
            y_val: Validation labels (optional, for optimization)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            
        Returns:
            self
        """
        if verbose >= 1:
            print("="*70)
            print("FITTING HYBRID IDS PIPELINE")
            print("="*70)
        
        # 1. Preprocessing
        if verbose >= 1:
            print("\n[1/10] Preprocessing...")
        
        X_train_processed = self.preprocessor.fit_transform(X_train) # This function basically returns "self.fit(df).transform(df)"
        
        # 2. Feature Engineering
        if verbose >= 1:
            print("[2/10] Feature Engineering...")
        
        X_train_processed = add_statistical_features(X_train_processed) # Adds statistical features to the specified array
        X_train_processed = add_context_aware_features(X_train_processed)

        # 3. Drop Correlated Features
        if verbose >= 1:
            print("[3/10] Dropping correlated features...")
        
        X_train_processed, self.correlated_features = drop_correlated_features(
            X_train_processed, 
            threshold=self.config.correlation_threshold
        )
        # Setting the 'feature_names' to take the required arguments
        self.feature_names = X_train_processed.columns.tolist()

        # 4. Scaling
        if verbose >= 1:
            print("[4/10] Scaling...")
        
        X_train_processed = X_train_processed.astype('float32')
        X_train_scaled = self.scaler.fit(X_train_processed).transform(X_train_processed) # Scaling the processed array
        
        # Extract normal traffic for unsupervised training
        X_normal_train = X_train_scaled[y_train == 0] 
        if verbose >= 1:
            print(f"   Normal samples: {len(X_normal_train)}")
        
        # 5. Autoencoder (if enabled)
        if self.config.use_autoencoder:
            if verbose >= 1:
                print("[5/10] Training Autoencoder...")
            
            input_dim = X_train_scaled.shape[1] # All columns
            self.autoencoder = AutoencoderIDS(
                input_dim=input_dim,
                encoding_dim=self.config.ae_encoding_dim,
                dropout=self.config.ae_dropout,
                epochs=self.config.ae_epochs,
                batch_size=self.config.ae_batch_size
            )
            
            # Validation data if available
            if X_val is not None and y_val is not None:
                X_val_processed = self._transform_features(X_val) # Transforming raw validation data when the pipeline runs
                X_normal_val = X_val_processed[y_val == 0]
                self.autoencoder.fit(X_normal_train, X_normal_val, verbose=0) 
            else:
                self.autoencoder.fit(X_normal_train, verbose=0) # Only use raw normal training data
            
            if verbose >= 1:
                loss = self.autoencoder.get_last_train_loss() # The last recorded loss will be returned(the current run's loss)
                print(f"   Training loss: {loss:.6f}")
        else:
            if verbose >= 1:
                print("[5/10] Skipping Autoencoder (disabled)")
        
        # 6. PCA (if enabled)
        if self.config.use_pca_features:
            if verbose >= 1:
                print("[6/10] Fitting PCA...")
            
            self.pca = PCATransformer(
                n_components=self.config.pca_n_components,
                random_state=42
            )
            self.pca.fit(X_normal_train)
            
            if verbose >= 1:
                print(f"   PCA components: {self.pca.pca.n_components_}")
        else:
            if verbose >= 1:
                print("[6/10] Skipping PCA (disabled)")
        
        # 7. Train Anomaly Detectors
        if verbose >= 1:
            print("[7/10] Training Anomaly Detectors...")
        
        detector_count = 0
        if 'isolation_forest' in self.config.anomaly_detectors:
            if verbose >= 2:
                print("   - Isolation Forest...")
            iso_forest = IsolationForestDetector(
                n_estimators=self.config.iso_n_estimators,
                max_samples=self.config.iso_max_samples,
                contamination=self.config.iso_contamination,
                max_features=self.config.iso_max_features
            )
            iso_forest.fit(X_normal_train)
            self.anomaly_detectors['isolation_forest'] = iso_forest
            detector_count += 1
        
        if 'lof' in self.config.anomaly_detectors:
            if verbose >= 2:
                print("   - Local Outlier Factor...")
            lof = LocalOutlierFactorDetector(
                n_neighbors=self.config.lof_n_neighbors,
                contamination=self.config.lof_contamination
            )
            lof.fit(X_normal_train)
            self.anomaly_detectors['lof'] = lof
            detector_count += 1
        
        if verbose >= 1:
            print(f"   Trained {detector_count} detectors")
        
        # 8. Train Supervised Model
        if verbose >= 1:
            print("[8/10] Training Supervised Model...")
        
        self.supervised_model = SupervisedRF(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            max_features=self.config.rf_max_features,
            class_weight=self.config.rf_class_weight
        )
        
        # Train on encoded features if autoencoder enabled
        if self.config.use_autoencoder:
            X_train_encoded = self.autoencoder.encode(X_train_scaled)
            self.supervised_model.fit(X_train_encoded, y_train)
        else:
            self.supervised_model.fit(X_train_scaled, y_train)
        
        # 9. Optimize Weights and Threshold (if validation data provided)
        if X_val is not None and y_val is not None and self.config.optimize_weights:
            if verbose >= 1:
                print("[9/10] Optimizing weights and threshold...")
            
            # Transform validation data
            X_val_processed = self._transform_features(X_val) # Transforming the validation data within the pipeline only
            
            # Get scores on validation
            sup_probs_val = self._get_supervised_probs(X_val_processed)
            anomaly_scores_val = self._get_anomaly_scores(X_val_processed)
            
            # Normalize scores
            # Another function has been implemented in the same class(ScoreNormalizer class), that combines the approach of fit and transform functions
            self.normalizer.fit(anomaly_scores_val)
            anomaly_scores_val_norm = self.normalizer.transform(anomaly_scores_val)
            
            # Optimize using F1 score with a recall floor
            best_weight, best_threshold, best_score = optimize_supervised_weight(
                sup_probs_val, anomaly_scores_val_norm, y_val,
                weight_min=self.config.weight_min,
                weight_max=self.config.weight_max,
                weight_step=self.config.weight_step,
                threshold_min=self.config.threshold_min,
                threshold_max=self.config.threshold_max,
                threshold_step=self.config.threshold_step,
                method=self.config.ensemble_method,
                optimize_for='f1',
                min_recall=0.90
            )
            
            self.config.supervised_weight = best_weight
            self.threshold_optimizer.best_threshold = best_threshold
            
            if verbose >= 1:
                print(f"   Optimal weight: {best_weight:.3f}")
                print(f"   Optimal threshold: {best_threshold:.3f}")
                print(f"   Validation F1: {best_score:.4f}")
        else:
            if verbose >= 1:
                print("[9/10] Using default weights and threshold")
            
            # Still need to fit normalizer even if not optimizing weights
            if X_val is not None and y_val is not None:
                X_val_processed = self._transform_features(X_val)
                anomaly_scores_val = self._get_anomaly_scores(X_val_processed)
                self.normalizer.fit(anomaly_scores_val)
            else:
                # No validation data - fit on training data
                anomaly_scores_train = self._get_anomaly_scores(X_train_scaled)
                self.normalizer.fit(anomaly_scores_train)
        
        # 10. Create Ensemble
        if verbose >= 1:
            print("[10/10] Creating Hybrid Ensemble...")
        
        # Build the final ensemble with the final optimized weights
        self.ensemble = HybridEnsemble(
            supervised_weight=self.config.supervised_weight,
            method=self.config.ensemble_method
        )
        
        if verbose >= 1:
            print(f"   Method: {self.config.ensemble_method.upper()}")
            print(f"   Supervised weight: {self.config.supervised_weight:.3f}")
        
        self.fitted = True
        
        if verbose >= 1:
            print("\n" + "="*70)
            print("PIPELINE FITTING COMPLETE")
            print("="*70)
        
        return self
    
    def _transform_features(self, X):
        """Internal method to transform features through pipeline"""
        X_processed = self.preprocessor.transform(X)
        X_processed = add_statistical_features(X_processed)
        X_processed = add_context_aware_features(X_processed)
        X_processed = X_processed.drop(columns=self.correlated_features, errors='ignore')
        X_processed = X_processed.astype('float32')
        X_scaled = self.scaler.transform(X_processed)
        return X_scaled
    
    def _get_supervised_probs(self, X_scaled):
        """Get supervised model probabilities"""
        # Always use autoencoder if it exists (maintains expected dimensionality)
        if self.autoencoder is not None:
            X_encoded = self.autoencoder.encode(X_scaled) # Raw predictions
            return self.supervised_model.predict_proba(X_encoded) # Raw probabilities
        
        # Fall back to PCA if autoencoder doesn't exist but PCA is available
        elif self.pca is not None:
            X_encoded = self.pca.transform(X_scaled) # Raw predictions
            return self.supervised_model.predict_proba(X_encoded) # Predicted probabilities
        else: # If both autoencoder and pca are not available, we will move with raw predictions only
            return self.supervised_model.predict_proba(X_scaled)
    
    def _get_anomaly_scores(self, X_scaled):
        """Get anomaly scores from all detectors"""
        scores = {}
        
        # Autoencoder reconstruction error (only if trained weights were loaded)
        if self.autoencoder is not None and self.autoencoder.weights_loaded:
            scores['autoencoder'] = self.autoencoder.reconstruction_error(X_scaled)
        
        # Other anomaly detectors
        for name, detector in self.anomaly_detectors.items():
            scores[name] = detector.score(X_scaled)
        
        return scores
    
    def predict_proba(self, X):
        """
        Predict intrusion probabilities
        
        Args:
            X: Features (DataFrame)
            
        Returns:
            Probabilities for intrusion class
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Transform features
        X_scaled = self._transform_features(X)
        
        # Get supervised probabilities
        sup_probs = self._get_supervised_probs(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = self._get_anomaly_scores(X_scaled)
        
        # Normalize scores
        anomaly_scores_norm = self.normalizer.transform(anomaly_scores)
        
        # Ensemble
        hybrid_probs = self.ensemble.predict_proba(sup_probs, anomaly_scores_norm)
        
        return hybrid_probs
    
    def predict(self, X):
        """
        Predict binary intrusion labels
        
        Args:
            X: Features (DataFrame)
            
        Returns:
            Binary predictions (0=normal, 1=intrusion)
        """
        probs = self.predict_proba(X)
        return self.threshold_optimizer.predict(probs)
    
    def get_params(self):
        """Get pipeline parameters"""
        return {
            'config': self.config.to_dict(),
            'feature_names': self.feature_names,
            'correlated_features': self.correlated_features,
            'threshold': self.threshold_optimizer.best_threshold,
            'fitted': self.fitted
        }
