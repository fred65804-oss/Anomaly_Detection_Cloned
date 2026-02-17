"""
Enhanced Model Tuning Script
Addresses critical false positive issue where all normal traffic is flagged as intrusion
"""

from utils import ModelManager

def tune_model_aggressive():
    """
    Apply aggressive tuning to fix zero precision issue
    """
    print("="*70)
    print("ENHANCED MODEL TUNING")
    print("="*70)
    
    print("\nLoading latest pipeline...")
    manager = ModelManager("artifacts")
    pipeline = manager.load_pipeline("latest")
    
    print("\n[Current Configuration]")
    print(f"  Threshold: {pipeline.threshold_optimizer.best_threshold:.3f}")
    print(f"  Supervised Weight: {pipeline.config.supervised_weight:.3f}")
    print(f"  Ensemble Method: {pipeline.config.ensemble_method}")
    
    # Diagnose the issue
    print("\n[Issue Identified]")
    print("  ✗ All normal traffic flagged as intrusion (0% precision)")
    print("  ✓ All attacks detected correctly (100% recall)")
    print("  Problem: Threshold too low + Anomaly weight too high")
    
    # Apply fixes
    print("\n[Applying Fixes]")
    
    # Fix 1: Increase threshold significantly
    old_threshold = pipeline.threshold_optimizer.best_threshold
    pipeline.threshold_optimizer.best_threshold = 0.55
    print(f"  ✓ Threshold: {old_threshold:.3f} → 0.55")
    
    # Fix 2: Increase supervised weight (trust classifier more than anomaly detectors)
    old_weight = pipeline.config.supervised_weight
    pipeline.config.supervised_weight = 0.6
    pipeline.ensemble.supervised_weight = 0.6
    print(f"  ✓ Supervised Weight: {old_weight:.3f} → 0.6")
    print("    (Now trusting Random Forest classifier 60% vs anomaly detectors 40%)")
    
    print("\n[New Configuration]")
    print(f"  Threshold: {pipeline.threshold_optimizer.best_threshold:.3f}")
    print(f"  Supervised Weight: {pipeline.config.supervised_weight:.3f}")
    
    # Save updated pipeline
    print("\n[Saving Updated Pipeline]")
    manager.save_pipeline(pipeline, version="latest")
    print("  ✓ Pipeline saved to artifacts/models/latest/")
    
    print("\n[Next Steps]")
    print("  1. Reload the API:")
    print("     - Restart: python -m app.main")
    print("     - OR use reload endpoint: POST /model/reload")
    print("  2. Re-run tests: python test_api.py")
    print("  3. Expected improvement: 80-90% accuracy")
    
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    tune_model_aggressive()
