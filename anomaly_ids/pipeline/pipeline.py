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
        # Updating self.feature_names after feature engineering
        self.feature_names = X_train_processed.columns.tolist()

        # 4. Scaling
        if verbose >= 1:
            print("[4/10] Scaling...")
        
        X_train_processed = X_train_processed.astype('float32')
        X_train_scaled = self.scaler.fit(X_train_processed).transform(X_train_processed) # Scaling the processed array
        
        # Extract normal traffic for unsupervised training
        X_normal_train = X_train_scaled[y_train == 0] 
        if verbose >= 1:
            print(f"   Normal samples (full): {len(X_normal_train)}")

        # Cap normal samples for unsupervised models.
        # LOF is O(n^2) and training on millions of rows takes many hours.
        # 100K normal samples captures the distribution well without the cost.
        # Before randomization, we have already extracted only normal traffic(where target variable is 0)
        max_us = getattr(self.config, 'unsupervised_max_samples', None)
        if max_us is not None and len(X_normal_train) > max_us:
            rng = np.random.default_rng(42) # A new random number generator (will replace np.random). This introduces reproducibility
            idx = rng.choice(len(X_normal_train), size=max_us, replace=False)
            idx.sort()
            X_normal_train_us = X_normal_train[idx]
            if verbose >= 1:
                print(f"   Normal samples (capped for unsupervised): {len(X_normal_train_us)}")
        else:
            X_normal_train_us = X_normal_train
        
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
                X_normal_val = X_val_processed[y_val == 0] # Normal validation data
                self.autoencoder.fit(X_normal_train_us, X_normal_val, verbose=0) 
            else:
                self.autoencoder.fit(X_normal_train_us, verbose=0) # Only use raw normal training data
            
            if verbose >= 1:
                loss = self.autoencoder.get_last_train_loss() # The last recorded loss will be returned(It will be the current run's loss only)
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
            self.pca.fit(X_normal_train_us)
            
            if verbose >= 1:
                print(f"   PCA components: {self.pca.pca.n_components_}")
        else:
            if verbose >= 1:
                print("[6/10] Skipping PCA (disabled)")
        
        # 7. Train Anomaly Detectors
        if verbose >= 1:
            print("[7/10] Training Anomaly Detectors...")
        
        detector_count = 0 # How many anomaly models I have used
        if 'isolation_forest' in self.config.anomaly_detectors:
            if verbose >= 2:
                print("   - Isolation Forest...")
            iso_forest = IsolationForestDetector(
                n_estimators=self.config.iso_n_estimators,
                max_samples=self.config.iso_max_samples,
                contamination=self.config.iso_contamination,
                max_features=self.config.iso_max_features
            )
            iso_forest.fit(X_normal_train_us) # Fit only on normal data
            self.anomaly_detectors['isolation_forest'] = iso_forest
            detector_count += 1
        
        if 'lof' in self.config.anomaly_detectors:
            if verbose >= 2:
                print("   - Local Outlier Factor...")
            lof = LocalOutlierFactorDetector(
                n_neighbors=self.config.lof_n_neighbors,
                contamination=self.config.lof_contamination
            )
            lof.fit(X_normal_train_us)
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
        
        # Train on encoded features if autoencoder enabled.
        # Cap rows fed to RF — encoding huge amounts of rows through AE is slow and RF
        # converges well before that. Sample stratified so attack ratio is kept.
        rf_max = getattr(self.config, 'rf_max_samples', None)
        if rf_max is not None and len(X_train_scaled) > rf_max:
            rng_rf = np.random.default_rng(7) # Make a default generator
            # Stratified sample: keep attack ratio
            normal_idx = np.where(y_train.values == 0)[0]
            attack_idx = np.where(y_train.values == 1)[0]
            attack_ratio = len(attack_idx) / len(X_train_scaled) # Let's say if the original dataset had 0.45 as the ratio. So, the sample will also contain target variable(y_train in this case) in the ratio of 0.45
            # In the below line ->
            # "rf_max * attack_ratio" tells us how much percent of the total_rows I want in the sample(eg. 200_000 * 0.30 gives 60000 as the answer. This means that the sample should contain 60000 points as attack)
            # len(attack_idx) is the total number of attack points(data)
            # 'min' will ensure that either we take all anomaly points or take a fraction of it(explained above).
            # 'min' will also ensure that we don't try to ask for more attacks than what we usually have
            n_attack = min(len(attack_idx), int(rf_max * attack_ratio)) 
            n_normal = rf_max - n_attack # Total number of data points subtracted by the total attack points(total - anomaly = normal data points)
            # In the below code ->
            # selecting normal and attack data points(from normal_idx and attack_idx) based on a range
            # range selection => either select entire index range(normal_idx) or select the portion of attack(n_normal)
            sel_normal = rng_rf.choice(normal_idx, size=min(n_normal, len(normal_idx)), replace=False)
            sel_attack = rng_rf.choice(attack_idx, size=n_attack, replace=False)
            # The ratio of samples have been determined, now we will just concatenate the results and generate the entire sample
            sel_idx = np.sort(np.concatenate([sel_normal, sel_attack]))
            # Pulling out samples based on limits defined above
            X_rf = X_train_scaled[sel_idx] 
            y_rf = y_train.values[sel_idx] 
            if verbose >= 1:
                print(f"   RF training samples (capped): {len(X_rf)}")
        else: # If the current number of rows are not more than rf_max, or if we have not defined rf_max, we will enter here 
            X_rf = X_train_scaled
            y_rf = y_train.values if hasattr(y_train, 'values') else y_train

        # Supplying the samples to the autoencoder
        if self.config.use_autoencoder:
            X_train_encoded = self.autoencoder.encode(X_rf)
            self.supervised_model.fit(X_train_encoded, y_rf)
        else:
            self.supervised_model.fit(X_rf, y_rf)
        
        # 9. Optimize Weights and Threshold (if validation data provided)
        if X_val is not None and y_val is not None and self.config.optimize_weights:
            if verbose >= 1:
                print("[9/10] Optimizing weights and threshold...")

            # ── Fix for Issue 2: Split validation in half ─────────────────────
            # The normalizer learns p5/p95 percentiles from data.
            # If we fit it AND grid-search the threshold on the SAME data, the
            # threshold is tuned on already-perfectly-normalised scores → optimistic
            # (leaked) metrics. Solution: fit normalizer on val_norm_half, then
            # optimize threshold on the separate val_opt_half.
            y_val_arr = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
            n_val = len(y_val_arr)
            rng_val = np.random.default_rng(55)
            # Stratified split: keep class ratio in both halves
            normal_val_idx  = np.where(y_val_arr == 0)[0]
            attack_val_idx  = np.where(y_val_arr == 1)[0]
            rng_val.shuffle(normal_val_idx)
            rng_val.shuffle(attack_val_idx)
            # First half of each class → normalizer fitting
            norm_half_idx = np.sort(np.concatenate([
                normal_val_idx[:len(normal_val_idx) // 2],
                attack_val_idx[:len(attack_val_idx) // 2]
            ]))
            # Second half of each class → threshold optimisation
            opt_half_idx = np.sort(np.concatenate([
                normal_val_idx[len(normal_val_idx) // 2:],
                attack_val_idx[len(attack_val_idx) // 2:]
            ]))

            # Convert pandas index → positional if needed
            if hasattr(X_val, 'iloc'):
                X_val_norm_half = X_val.iloc[norm_half_idx]
                X_val_opt_half  = X_val.iloc[opt_half_idx]
            else:
                X_val_norm_half = X_val[norm_half_idx]
                X_val_opt_half  = X_val[opt_half_idx]
            y_val_opt_half = y_val_arr[opt_half_idx]

            if verbose >= 1:
                print(f"   Val split → normalizer: {len(norm_half_idx)} rows | threshold search: {len(opt_half_idx)} rows")

            # Step A: Fit normalizer on the FIRST half only
            X_val_norm_half_proc = self._transform_features(X_val_norm_half)
            anomaly_scores_norm_half = self._get_anomaly_scores(X_val_norm_half_proc)
            # Another function has been implemented in the same class(ScoreNormalizer class), that combines the approach of fit and transform functions
            self.normalizer.fit(anomaly_scores_norm_half)

            # Step B: Score the SECOND half and normalize using the fitted normalizer
            X_val_opt_half_proc = self._transform_features(X_val_opt_half)
            sup_probs_val       = self._get_supervised_probs(X_val_opt_half_proc)
            anomaly_scores_val  = self._get_anomaly_scores(X_val_opt_half_proc)
            anomaly_scores_val_norm = self.normalizer.transform(anomaly_scores_val)

            # Optimize using F1 score with a recall floor on the SECOND half
            best_weight, best_threshold, best_score = optimize_supervised_weight(
                sup_probs_val, anomaly_scores_val_norm, y_val_opt_half,
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
            
            # Still need to fit normalizer even if not optimizing weights.
            # Cap scoring to 50K rows — LOF decision_function is O(n*k) and
            # running it on 500K val rows takes many minutes.
            if X_val is not None and y_val is not None:
                X_val_processed = self._transform_features(X_val)
                norm_cap = 50_000
                if X_val_processed.shape[0] > norm_cap:
                    rng_nv = np.random.default_rng(77) # Range for novel attack detection
                    nv_idx = rng_nv.choice(X_val_processed.shape[0], size=norm_cap, replace=False) # Extract 50k validation data
                    nv_idx.sort()
                    X_val_norm = X_val_processed[nv_idx]
                else:
                    X_val_norm = X_val_processed
                anomaly_scores_val = self._get_anomaly_scores(X_val_norm)
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
        # Safety: fill any remaining NaNs (e.g. from engineered features on
        # edge-case inputs) with 0 before passing to sklearn estimators.
        X_processed = X_processed.fillna(0)
        X_processed = X_processed.astype('float32')
        X_scaled = self.scaler.transform(X_processed) # Only validation data is passed
        return X_scaled
    
    def _get_supervised_probs(self, X_scaled):
        """Get supervised model probabilities"""
        # Always use autoencoder if it exists (maintains expected dimensionality)
        if self.autoencoder is not None:
            X_encoded = self.autoencoder.encode(X_scaled) # Encoded features(To be supplied to RandomForest ahead)
            return self.supervised_model.predict_proba(X_encoded) # Raw probabilities
        
        # Fall back to PCA if autoencoder doesn't exist but PCA is available
        elif self.pca is not None:
            X_encoded = self.pca.transform(X_scaled)
            return self.supervised_model.predict_proba(X_encoded) # Predicted probabilities
        else: # If both autoencoder and pca are not available, we will move with raw predictions only
            return self.supervised_model.predict_proba(X_scaled)
    
    def _get_anomaly_scores(self, X_scaled):
        """
            Get anomaly scores from all detectors(Storing the results as dictionary)
        """
        scores = {}
        
        # Autoencoder reconstruction error (only if trained weights were loaded)
        if self.autoencoder is not None and self.autoencoder.weights_loaded:
            scores['autoencoder'] = self.autoencoder.reconstruction_error(X_scaled)
        
        # Other anomaly detectors
        for name, detector in self.anomaly_detectors.items():
            scores[name] = detector.score(X_scaled)
        
        return scores
    
    # Function to initialize IDSExplainer
    def init_explainer(self, X_background):
        """
            Initialize the explainer with background data
            This will be called after training, using a sample of normal training data

            Args:
                X_background: 100 to 200 rows of normal traffic (this will be a dataframe)
        """
        from utils.explainer import IDSExplainer
        # Making the data pass through transformation
        x_bg_transformed = self._transform_features(X_background)
        # Initializing Explainer object
        self.explainer = IDSExplainer(
            pipeline = self,
            feature_names = self.feature_names,
            X_train_background = x_bg_transformed
        )

    # SHAP/LIME Explanations
    def explain_prediction(self, X, method = 'shap', top_k = 10):
        """
            Explain why a prediction was made by black-box model(s)

            Args:
                X: Single(1 row) sample dataframe (raw, pre-transform)
                method: 'shap', 'lime', or 'both'
                top_k: Number of top features to return

            Returns:
                A Dictionary with feature importances   
        """
        if not hasattr(self, 'explainer') or self.explainer is None:
            raise ValueError("Explainer not initialized. Call init_explainer() first")

        # Transform X into the same scaled numeric space that SHAP's background
        # data lives in — SHAP must perturb in that space, not the raw string space.
        X_scaled = self._transform_features(X)

        if method == 'shap':
            return self.explainer.explain_shap(X_scaled, top_k)

        elif method == 'lime':
            return self.explainer.explain_lime(X_scaled, top_k)

        elif method == 'both':
            return {
                'shap': self.explainer.explain_shap(X_scaled, top_k),
                'lime': self.explainer.explain_lime(X_scaled, top_k)
            }

    def _predict_proba_transformed(self, X_scaled):
        """
        Run the scoring pipeline on an already-transformed (scaled) numpy array.
        Used by IDSExplainer._predict_fn so SHAP/LIME perturbations never pass
        through _transform_features a second time.

        Returns:
            np.ndarray of shape (n, 2): [P(normal), P(intrusion)] per row
        """
        sup_probs = self._get_supervised_probs(X_scaled)
        anomaly_scores = self._get_anomaly_scores(X_scaled)
        anomaly_scores_norm = self.normalizer.transform(anomaly_scores)
        hybrid_probs = self.ensemble.predict_proba(sup_probs, anomaly_scores_norm)
        return np.column_stack([1 - hybrid_probs, hybrid_probs])

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
