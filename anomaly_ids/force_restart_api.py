"""
Force restart API with fresh model load
This script kills the API process and restarts it cleanly
"""
import subprocess
import sys
import time
import os
import signal

def force_restart_api():
    print("="*70)
    print("FORCE RESTART API SERVER")
    print("="*70)
    
    print("\n[Step 1] Stopping all Python processes running uvicorn...")
    
    # Kill all uvicorn/FastAPI processes on Windows
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq *uvicorn*'], 
                      capture_output=True)
        print("  ✓ Killed uvicorn processes")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Give it a moment to clean up
    time.sleep(2)
    
    print("\n[Step 2] Verifying model files...")
    from pathlib import Path
    import joblib
    
    api_dir = Path("c:/Anomaly_Detection/anomaly_ids/artifacts/models/latest")
    config = joblib.load(api_dir / "hybrid_config.joblib")
    threshold = joblib.load(api_dir / "hybrid_threshold.joblib")
    
    print(f"  Model on disk:")
    print(f"    - Threshold: {threshold:.4f}")
    print(f"    - Supervised Weight: {config.get('supervised_weight', 'N/A')}")
    
    if abs(threshold - 0.50) > 0.01 or abs(config.get('supervised_weight', 0) - 0.40) > 0.01:
        print("\n  ⚠️  WARNING: Model files don't have expected tuning!")
        print("  Run: python migrate_and_tune.py")
        return
    
    print("\n[Step 3] Starting API server...")
    print("  Command: python -m app.main")
    print("  Working Directory: c:\\Anomaly_Detection\\anomaly_ids")
    print("\n  Press Ctrl+C to stop the server when done testing\n")
    
    # Change to correct directory and start
    os.chdir("c:/Anomaly_Detection/anomaly_ids")
    
    # Start the API server
    try:
        subprocess.run([sys.executable, "-m", "app.main"])
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")

if __name__ == "__main__":
    force_restart_api()
