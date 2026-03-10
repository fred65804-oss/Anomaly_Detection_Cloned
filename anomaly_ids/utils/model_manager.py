"""
Model management utilities for saving and loading pipeline
"""

import os
import joblib
import json
import numpy as np
from pathlib import Path


class ModelManager:
    """Manages saving and loading of the complete pipeline"""
    
    def __init__(self, artifacts_dir="artifacts"):
        """
        Initialize model manager
        
        Args:
            artifacts_dir: Directory to save/load models
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_pipeline(self, pipeline, version="latest"):
        """
        Save complete pipeline and all its components
        
        Args:
            pipeline: HybridIDSPipeline instance
            version: Version identifier (default: "latest")
        """
        if not pipeline.fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        version_dir = self.artifacts_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving pipeline to {version_dir}...")
        
        # Save configuration
        config_path = version_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(pipeline.config.to_dict(), f, indent=2)
        print(f"  [OK] Config saved")
        
        # Save pipeline parameters
        params_path = version_dir / "params.joblib"
        joblib.dump(pipeline.get_params(), params_path)
        print(f"  [OK] Parameters saved")
        
        # Save preprocessor
        preprocessor_path = version_dir / "preprocessor.joblib"
        joblib.dump(pipeline.preprocessor, preprocessor_path)
        print(f"  [OK] Preprocessor saved")
        
        # Save scaler
        scaler_path = version_dir / "scaler.joblib"
        joblib.dump(pipeline.scaler, scaler_path)
        print(f"  [OK] Scaler saved")
        
        # Save PCA if used
        if pipeline.pca is not None:
            pca_path = version_dir / "pca.joblib"
            joblib.dump(pipeline.pca, pca_path)
            print(f"  [OK] PCA saved")
        
        # Save autoencoder if used
        if pipeline.autoencoder is not None:
            # Save weights only (more portable across TF/Keras versions)
            weights_path = version_dir / "autoencoder.weights.h5"
            pipeline.autoencoder.autoencoder.save_weights(weights_path)
            print(f"  [OK] Autoencoder weights saved")
        
        # Save anomaly detectors
        if pipeline.anomaly_detectors:
            detectors_path = version_dir / "anomaly_detectors.joblib"
            joblib.dump(pipeline.anomaly_detectors, detectors_path)
            print(f"  [OK] Anomaly detectors saved")
        
        # Save supervised model
        supervised_path = version_dir / "supervised_model.joblib"
        joblib.dump(pipeline.supervised_model, supervised_path)
        print(f"  [OK] Supervised model saved")
        
        # Save normalizer
        normalizer_path = version_dir / "normalizer.joblib"
        joblib.dump(pipeline.normalizer, normalizer_path)
        print(f"  [OK] Normalizer saved")
        
        # Save ensemble
        ensemble_path = version_dir / "ensemble.joblib"
        joblib.dump(pipeline.ensemble, ensemble_path)
        print(f"  [OK] Ensemble saved")
        
        # Save threshold optimizer
        threshold_path = version_dir / "threshold_optimizer.joblib"
        joblib.dump(pipeline.threshold_optimizer, threshold_path)
        print(f"  [OK] Threshold optimizer saved")
        
        print(f"\n[OK] Pipeline saved successfully to {version_dir}")
        
        return version_dir
    
    def load_pipeline(self, version="latest"):
        """
        Load complete pipeline from saved artifacts
        
        Args:
            version: Version identifier to load
            
        Returns:
            Loaded HybridIDSPipeline instance
        """
        from pipeline.pipeline import HybridIDSPipeline
        from pipeline.config import IDSConfig
        import tensorflow as tf
        
        version_dir = self.artifacts_dir / version
        
        if not version_dir.exists():
            raise ValueError(f"Version '{version}' not found in {self.artifacts_dir}")
        
        print(f"Loading pipeline from {version_dir}...")
        
        # Load configuration
        config_path = version_dir / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = IDSConfig(config_dict)
        
        # Create pipeline instance
        pipeline = HybridIDSPipeline(config)
        
        # Load parameters
        params_path = version_dir / "params.joblib"
        params = joblib.load(params_path)
        pipeline.feature_names = params['feature_names']
        pipeline.correlated_features = params['correlated_features']
        
        # Load preprocessor
        preprocessor_path = version_dir / "preprocessor.joblib"
        pipeline.preprocessor = joblib.load(preprocessor_path)
        
        # Load scaler
        scaler_path = version_dir / "scaler.joblib"
        pipeline.scaler = joblib.load(scaler_path)
        
        # Load PCA if exists
        pca_path = version_dir / "pca.joblib"
        if pca_path.exists():
            pipeline.pca = joblib.load(pca_path)
        
        # Load autoencoder if exists
        weights_path = version_dir / "autoencoder.weights.h5"
        legacy_path = version_dir / "autoencoder.keras"
        
        if weights_path.exists() or legacy_path.exists():
            from pipeline.autoencoder import AutoencoderIDS
            
            # Create instance with proper architecture
            input_dim = len(pipeline.feature_names)
            pipeline.autoencoder = AutoencoderIDS(
                input_dim=input_dim,
                encoding_dim=config.ae_encoding_dim,
                dropout=config.ae_dropout,
                epochs=config.ae_epochs,
                batch_size=config.ae_batch_size
            )
            
            weights_loaded = False
            
            # Strategy 1: Load from .weights.h5 (new format)
            if weights_path.exists():
                try:
                    print(f"  Loading autoencoder weights from {weights_path.name}...")
                    pipeline.autoencoder.autoencoder.load_weights(weights_path)
                    weights_loaded = True
                    pipeline.autoencoder.weights_loaded = True
                    print("  [OK] Autoencoder weights loaded successfully")
                except Exception as e:
                    print(f"  Failed to load .weights.h5: {str(e)[:100]}")
            
            # Strategy 2: Try legacy .keras format
            if not weights_loaded and legacy_path.exists():
                try:
                    print(f"  Attempting legacy load from {legacy_path.name}...")
                    loaded_model = tf.keras.models.load_model(legacy_path, compile=False)
                    # Validate input shape matches current feature count before using it
                    legacy_input_dim = loaded_model.input_shape[-1]
                    if legacy_input_dim != input_dim:
                        print(f"  [WARN] Legacy model input_dim={legacy_input_dim} != expected {input_dim}. Skipping.")
                    else:
                        pipeline.autoencoder.autoencoder = loaded_model
                        # Rebuild encoder from loaded model
                        for i, layer in enumerate(loaded_model.layers):
                            if hasattr(layer, 'units') and layer.units == config.ae_encoding_dim:
                                pipeline.autoencoder.encoder = tf.keras.Model(
                                    loaded_model.input, layer.output
                                )
                                break
                        weights_loaded = True
                        pipeline.autoencoder.weights_loaded = True
                        print("  [OK] Autoencoder loaded from legacy format")
                except Exception as e:
                    print(f"  Failed legacy load: {str(e)[:100]}")
            
            if not weights_loaded:
                print("  [WARN] Could not load autoencoder weights. Using random weights.")
                print("  Note: Disabling autoencoder anomaly scoring.")
                pipeline.autoencoder.weights_loaded = False
            
            # Ensure encoder is set
            if pipeline.autoencoder.encoder is None:
                for i, layer in enumerate(pipeline.autoencoder.autoencoder.layers):
                    if hasattr(layer, 'units') and layer.units == config.ae_encoding_dim:
                        pipeline.autoencoder.encoder = tf.keras.Model(
                            pipeline.autoencoder.autoencoder.input,
                            layer.output
                        )
                        break
        
        # Load anomaly detectors
        detectors_path = version_dir / "anomaly_detectors.joblib"
        if detectors_path.exists():
            pipeline.anomaly_detectors = joblib.load(detectors_path)
        
        # Load supervised model
        supervised_path = version_dir / "supervised_model.joblib"
        pipeline.supervised_model = joblib.load(supervised_path)
        
        # Load normalizer
        normalizer_path = version_dir / "normalizer.joblib"
        pipeline.normalizer = joblib.load(normalizer_path)
        
        # Load ensemble
        ensemble_path = version_dir / "ensemble.joblib"
        pipeline.ensemble = joblib.load(ensemble_path)
        
        # Load threshold optimizer
        threshold_path = version_dir / "threshold_optimizer.joblib"
        pipeline.threshold_optimizer = joblib.load(threshold_path)
        
        pipeline.fitted = True

        # Reinitialize explainer from saved background data (SHAP/LIME can't be joblib-serialized)
        background_path = version_dir / "shap_background.joblib"
        if background_path.exists():
            try:
                X_background = joblib.load(background_path)
                pipeline.init_explainer(X_background)
                print(f"  [OK] Explainer initialized from background data ({len(X_background)} samples)")
            except Exception as e:
                print(f"  [WARN] Could not initialize explainer: {e}")
                pipeline.explainer = None
        else:
            print(f"  [WARN] No shap_background.joblib found — explainer not available. Re-run training to generate it.")
            pipeline.explainer = None
        
        print(f"[OK] Pipeline loaded successfully from {version_dir}")
        
        return pipeline
    
    def list_versions(self):
        """List all saved versions"""
        versions = [d.name for d in self.artifacts_dir.iterdir() if d.is_dir()]
        return sorted(versions)
