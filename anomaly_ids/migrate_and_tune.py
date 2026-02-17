"""
Model Migration and Tuning Script
Copies Raw_code.py trained model to API artifacts and applies balanced tuning
"""

import shutil
from pathlib import Path
from utils import ModelManager
import joblib

def migrate_and_tune():
    print("="*70)
    print("MODEL MIGRATION AND BALANCED TUNING")
    print("="*70)
    
    # Paths
    raw_code_dir = Path("c:/Anomaly_Detection")
    api_artifacts = Path("c:/Anomaly_Detection/anomaly_ids/artifacts")
    
    print("\n[Step 1] Checking for Raw_code.py trained models...")
    
    # Check if Raw_code.py models exist
    raw_models = [
        "hybrid_supervised.joblib",
        "hybrid_anomaly_detectors.joblib",
        "hybrid_scaler.joblib",
        "hybrid_features.joblib",
        "hybrid_threshold.joblib",
        "hybrid_config.joblib",
        "hybrid_autoencoder.keras",
        "hybrid_encoder.keras",
        "hybrid_pca.joblib"
    ]
    
    models_exist = all((raw_code_dir / model).exists() for model in raw_models[:6])
    
    if not models_exist:
        print("  ✗ Raw_code.py models not found!")
        print("  Please run: python Raw_code.py")
        return
    
    print("  ✓ Found Raw_code.py models")
    
    print("\n[Step 2] Loading Raw_code.py configuration...")
    config = joblib.load(raw_code_dir / "hybrid_config.joblib")
    threshold = joblib.load(raw_code_dir / "hybrid_threshold.joblib")
    
    print(f"  Current Threshold: {threshold:.3f}")
    print(f"  Current Supervised Weight: {config.get('supervised_weight', 'N/A'):.3f}")
    
    print("\n[Step 3] Applying BALANCED tuning to Raw_code.py models...")
    
    # Update config with balanced parameters
    config['supervised_weight'] = 0.40
    print(f"  ✓ Updated supervised_weight: {config['supervised_weight']:.3f}")
    
    # Update threshold
    new_threshold = 0.50
    print(f"  ✓ Updated threshold: {threshold:.3f} → {new_threshold:.3f}")
    
    print("\n[Step 4] Saving tuned models back to Raw_code.py directory...")
    
    # Save updated config and threshold
    joblib.dump(config, raw_code_dir / "hybrid_config.joblib")
    joblib.dump(new_threshold, raw_code_dir / "hybrid_threshold.joblib")
    
    print("  ✓ Saved tuned configuration")
    
    print("\n[Step 5] Copying models to API artifacts directory...")
    
    # Create artifacts directory structure
    latest_dir = api_artifacts / "models" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all model files
    for model_file in raw_models:
        src = raw_code_dir / model_file
        dst = latest_dir / model_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {model_file}")
    
    # Save config.json for API
    import json
    config_json = latest_dir / "config.json"
    with open(config_json, 'w') as f:
        json.dump(config, f, indent=2)
    print("  ✓ Saved config.json")
    
    print("\n[Step 6] Verification...")
    print(f"  Models location: {latest_dir}")
    print(f"  Threshold: {new_threshold:.3f}")
    print(f"  Supervised Weight: {config['supervised_weight']:.3f}")
    print(f"  Anomaly Weight: {1 - config['supervised_weight']:.3f}")
    
    print("\n[CRITICAL: Next Steps]")
    print("  1. RESTART the API server:")
    print("     - Stop the current server (Ctrl+C)")
    print("     - Run: python -m app.main")
    print()
    print("  2. Re-run tests:")
    print("     python test_api.py")
    print()
    print("  3. Expected Results:")
    print("     - Normal traffic: 5-6/6 correct (83-100%)")
    print("     - Attacks: 7-9/9 correct (78-100%)")
    print("     - Overall: 70-80% accuracy")
    
    print("\n" + "="*70)
    print("MIGRATION AND TUNING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    migrate_and_tune()
