"""
Interactive debug script to test pipeline components step-by-step
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add anomaly_ids to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import HybridIDSPipeline, IDSConfig
from utils import ModelManager

def debug_pipeline():
    print("="*70)
    print("HYBRID IDS PIPELINE DEBUG")
    print("="*70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading dataset...")
    csv_files = list(Path("training").glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in training/")
        return
    
    csv_file = csv_files[0]
    print(f"✓ Found: {csv_file.name}")
    
    df = pd.read_csv(csv_file)
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)[:10]}...")  # First 10 columns
    
    # Step 2: Initialize pipeline
    print("\n[Step 2] Initializing pipeline...")
    config = IDSConfig()
    pipeline = HybridIDSPipeline(config=config)
    print(f"✓ Pipeline created")
    
    # Step 3: Prepare data
    print("\n[Step 3] Preparing data...")
    from sklearn.model_selection import train_test_split
    
    # Find label column
    label_cols = ["attack_class", "label", "attack", "class"]
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break
    
    if not label_col:
        print(f"❌ Label column not found. Available columns: {list(df.columns)}")
        return
    
    print(f"✓ Label column: {label_col}")
    print(f"✓ Unique labels: {df[label_col].unique()[:10]}")
    
    X = df.drop(columns=[label_col])
    y = (df[label_col] != "normal").astype(int)  # Binary: 0=normal, 1=attack
    
    print(f"✓ Class distribution: {y.value_counts().to_dict()}")
    
    # Small sample for quick testing
    X_small = X.iloc[:5000]
    y_small = y.iloc[:5000]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_small, y_small, test_size=0.2, stratify=y_small, random_state=42
    )
    
    print(f"✓ Train set: {X_train.shape}, Val set: {X_val.shape}")
    
    # Step 4: Fit pipeline
    print("\n[Step 4] Fitting pipeline (this may take a few minutes)...")
    try:
        pipeline.fit(X_train, y_train, X_val, y_val, verbose=2)
        print("✓ Pipeline fitted successfully!")
    except Exception as e:
        print(f"❌ Error during fitting: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Make predictions
    print("\n[Step 5] Making predictions...")
    try:
        probs = pipeline.predict_proba(X_val.iloc[:10])
        preds = pipeline.predict(X_val.iloc[:10])
        
        print(f"✓ Predictions shape: {preds.shape}")
        print(f"✓ Sample predictions: {preds}")
        print(f"✓ Sample probabilities: {probs[:5]}")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Evaluate
    print("\n[Step 6] Evaluating on validation set...")
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        val_preds = pipeline.predict(X_val)
        
        acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        
        print(f"✓ Accuracy:  {acc:.4f}")
        print(f"✓ Precision: {prec:.4f}")
        print(f"✓ Recall:    {rec:.4f}")
        print(f"✓ F1-Score:  {f1:.4f}")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 7: Save model
    print("\n[Step 7] Saving pipeline...")
    try:
        manager = ModelManager("artifacts")
        manager.save_pipeline(pipeline, version="debug")
        print(f"✓ Model saved to artifacts/debug/")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)

if __name__ == "__main__":
    debug_pipeline()
