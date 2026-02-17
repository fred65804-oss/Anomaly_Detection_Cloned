"""
Main training script for Hybrid IDS
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import HybridIDSPipeline, IDSConfig
from utils import ModelManager, setup_logger


def train_hybrid_ids(train_csv="KDDTrain.csv", 
                     test_csv="KDDTest.csv",
                     artifacts_dir="artifacts",
                     config_dict=None,
                     verbose=1):
    """
    Train Hybrid IDS pipeline
    
    Args:
        train_csv: Path to training CSV
        test_csv: Path to test CSV
        artifacts_dir: Directory to save models
        config_dict: Optional configuration dictionary
        verbose: Verbosity level
        
    Returns:
        pipeline: Trained pipeline
        data: Dictionary with train/val/test data
    """
    logger = setup_logger("training", log_file=f"{artifacts_dir}/training.log")
    
    logger.info("="*70)
    logger.info("HYBRID IDS TRAINING")
    logger.info("="*70)
    
    # 1. Load Data
    logger.info("\n[1/5] Loading data...")
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    logger.info(f"  Train samples: {len(df_train)}")
    logger.info(f"  Test samples: {len(df_test)}")
    
    # 2. Create Binary Target
    logger.info("\n[2/5] Creating binary targets...")
    
    # Auto-detect label column (look for common names)
    label_col_candidates = ["attack_class", "label", "class", "target", "attack", "intrusion"]
    label_col = None
    normal_label = "normal"  # Default normal class identifier
    
    for col in label_col_candidates:
        if col in df_train.columns:
            label_col = col
            # Auto-detect what represents "normal" (case-insensitive)
            unique_vals = df_train[col].astype(str).str.lower().unique()
            if "normal" in unique_vals:
                normal_label = "normal"
            elif "0" in unique_vals:
                normal_label = "0"
            break
    
    if label_col is None:
        raise ValueError(f"No label column found. Expected one of: {label_col_candidates}")
    
    logger.info(f"  Using label column: '{label_col}' (normal class: '{normal_label}')")
    
    # Create binary intrusion target
    df_train["is_intrusion"] = (df_train[label_col].astype(str).str.lower() != normal_label).astype(int)
    df_test["is_intrusion"] = (df_test[label_col].astype(str).str.lower() != normal_label).astype(int)
    
    logger.info(f"  Train intrusions: {df_train['is_intrusion'].sum()} ({df_train['is_intrusion'].mean()*100:.1f}%)")
    logger.info(f"  Test intrusions: {df_test['is_intrusion'].sum()} ({df_test['is_intrusion'].mean()*100:.1f}%)")
    
    # Track novel attacks
    train_attacks = set(df_train[label_col].unique())
    test_attacks = set(df_test[label_col].unique())
    novel_attacks = test_attacks - train_attacks
    
    logger.info(f"  Novel attacks in test: {len(novel_attacks)}")
    if verbose >= 2:
        logger.info(f"  Novel attack types: {sorted(novel_attacks)}")
    
    # 3. Prepare Features
    logger.info("\n[3/5] Preparing features...")
    X_train_full = df_train.drop(columns=["is_intrusion"], errors='ignore')
    y_train_full = df_train["is_intrusion"]
    X_test = df_test.drop(columns=["is_intrusion"], errors='ignore')
    y_test = df_test["is_intrusion"]
    
    # Store test attack labels for later analysis
    test_attack_labels = df_test["attack_class"].copy()
    
    # 4. Train/Val Split
    logger.info("\n[4/5] Creating train/validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )
    
    logger.info(f"  Train: {len(X_train)}")
    logger.info(f"  Val: {len(X_val)}")
    logger.info(f"  Test: {len(X_test)}")
    
    # 5. Train Pipeline
    logger.info("\n[5/5] Training pipeline...")
    
    # Create configuration
    if config_dict:
        config = IDSConfig(config_dict)
    else:
        config = IDSConfig()
    
    # Initialize pipeline
    pipeline = HybridIDSPipeline(config)
    
    # Fit pipeline
    pipeline.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    
    # 6. Save Pipeline
    logger.info("\nSaving pipeline...")
    model_manager = ModelManager(artifacts_dir)
    model_manager.save_pipeline(pipeline, version="latest")
    
    # Prepare data for return
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'test_attack_labels': test_attack_labels,
        'novel_attacks': novel_attacks
    }
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    
    return pipeline, data


if __name__ == "__main__":
    # Train with default configuration
    # Use parent directory's artifacts folder (where API loads from)
    artifacts_path = str(Path(__file__).parent.parent / "artifacts")
    pipeline, data = train_hybrid_ids(artifacts_dir=artifacts_path, verbose=2)
    
    print("\nPipeline trained successfully!")
    print(f"Threshold: {pipeline.threshold_optimizer.best_threshold:.3f}")
    print(f"Supervised weight: {pipeline.config.supervised_weight:.3f}")
