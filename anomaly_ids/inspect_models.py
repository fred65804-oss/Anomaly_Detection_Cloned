"""
Debug script to inspect saved model configuration
"""
import joblib
from pathlib import Path
import json

def inspect_models():
    print("="*70)
    print("MODEL INSPECTION")
    print("="*70)
    
    # 1. Inspect Raw_code.py output
    raw_dir = Path("c:/Anomaly_Detection")
    print(f"\n[1] Checking Raw_code.py output in: {raw_dir}")
    try:
        raw_config = joblib.load(raw_dir / "hybrid_config.joblib")
        raw_thresh = joblib.load(raw_dir / "hybrid_threshold.joblib")
        print(f"  Threshold: {raw_thresh:.4f}")
        print(f"  Supervised Weight: {raw_config.get('supervised_weight', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. Inspect API Artifacts (latest)
    api_dir = Path("c:/Anomaly_Detection/anomaly_ids/artifacts/models/latest")
    print(f"\n[2] Checking API Artifacts in: {api_dir}")
    try:
        api_config = joblib.load(api_dir / "hybrid_config.joblib")
        api_thresh = joblib.load(api_dir / "hybrid_threshold.joblib")
        print(f"  Threshold: {api_thresh:.4f}")
        print(f"  Supervised Weight: {api_config.get('supervised_weight', 'N/A')}")
        
        # Check config.json too
        if (api_dir / "config.json").exists():
            with open(api_dir / "config.json", 'r') as f:
                json_conf = json.load(f)
            print(f"  config.json Supervised Weight: {json_conf.get('supervised_weight', 'N/A')}")
            
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    inspect_models()
