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
        print(f"  ✓ Config saved")
        
        # Save pipeline parameters
        params_path = version_dir / "params.joblib"
        joblib.dump(pipeline.get_params(), params_path)
        print(f"  ✓ Parameters saved")
        
        # Save preprocessor
        preprocessor_path = version_dir / "preprocessor.joblib"
        joblib.dump(pipeline.preprocessor, preprocessor_path)
        print(f"  ✓ Preprocessor saved")
        
        # Save scaler
        scaler_path = version_dir / "scaler.joblib"
        joblib.dump(pipeline.scaler, scaler_path)
        print(f"  ✓ Scaler saved")
        
        # Save PCA if used
        if pipeline.pca is not None:
            pca_path = version_dir / "pca.joblib"
            joblib.dump(pipeline.pca, pca_path)
            print(f"  ✓ PCA saved")
        
        # Save autoencoder if used
        if pipeline.autoencoder is not None:
            # Save weights only (more portable across TF/Keras versions)
            weights_path = version_dir / "autoencoder.weights.h5"
            pipeline.autoencoder.autoencoder.save_weights(weights_path)
            print(f"  ✓ Autoencoder weights saved")
        
        # Save anomaly detectors
        if pipeline.anomaly_detectors:
            detectors_path = version_dir / "anomaly_detectors.joblib"
            joblib.dump(pipeline.anomaly_detectors, detectors_path)
            print(f"  ✓ Anomaly detectors saved")
        
        # Save supervised model
        supervised_path = version_dir / "supervised_model.joblib"
        joblib.dump(pipeline.supervised_model, supervised_path)
        print(f"  ✓ Supervised model saved")
        
        # Save normalizer
        normalizer_path = version_dir / "normalizer.joblib"
        joblib.dump(pipeline.normalizer, normalizer_path)
        print(f"  ✓ Normalizer saved")
        
        # Save ensemble
        ensemble_path = version_dir / "ensemble.joblib"
        joblib.dump(pipeline.ensemble, ensemble_path)
        print(f"  ✓ Ensemble saved")
        
        # Save threshold optimizer
        threshold_path = version_dir / "threshold_optimizer.joblib"
        joblib.dump(pipeline.threshold_optimizer, threshold_path)
        print(f"  ✓ Threshold optimizer saved")
        
        print(f"\n✓ Pipeline saved successfully to {version_dir}")
        
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
                    print("  ✓ Autoencoder weights loaded successfully")
                except Exception as e:
                    print(f"  Failed to load .weights.h5: {str(e)[:100]}")
            
            # Strategy 2: Try legacy .keras format
            if not weights_loaded and legacy_path.exists():
                try:
                    print(f"  Attempting legacy load from {legacy_path.name}...")
                    loaded_model = tf.keras.models.load_model(legacy_path, compile=False)
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
                    print("  ✓ Autoencoder loaded from legacy format")
                except Exception as e:
                    print(f"  Failed legacy load: {str(e)[:100]}")
            
            if not weights_loaded:
                print("  ⚠ Could not load autoencoder weights. Using random weights.")
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
        
        print(f"✓ Pipeline loaded successfully from {version_dir}")
        
        return pipeline
    
    def list_versions(self):
        """List all saved versions"""
        versions = [d.name for d in self.artifacts_dir.iterdir() if d.is_dir()]
        return sorted(versions)
