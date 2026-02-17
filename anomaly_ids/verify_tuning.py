"""
Script to verify that the model in artifacts directory is actually tuned
"""
import joblib
from pathlib import Path
import json

def verify():
    print("="*70)
    print("VERIFYING MODEL CONFIGURATION")
    print("="*70)
    
    api_dir = Path("c:/Anomaly_Detection/anomaly_ids/artifacts/models/latest")
    config_path = api_dir / "hybrid_config.joblib"
    thresh_path = api_dir / "hybrid_threshold.joblib"
    
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        return
        
    config = joblib.load(config_path)
    threshold = joblib.load(thresh_path)
    
    sup_weight = config.get('supervised_weight', 'N/A')
    
    print(f"Path: {api_dir}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Supervised Weight: {sup_weight}")
    
    print("-" * 30)
    
    if abs(threshold - 0.50) < 0.01 and abs(sup_weight - 0.40) < 0.01:
        print("✅ SUCCESS: Model on disk IS tuned correctly.")
        print("If API gives wrong results, it is NOT loading this file.")
        print("Try: python -m app.main (from c:\\Anomaly_Detection\\anomaly_ids)")
    else:
        print("❌ FAILURE: Model on disk is NOT tuned.")
        print("Please run: python migrate_and_tune.py")

if __name__ == "__main__":
    verify()
