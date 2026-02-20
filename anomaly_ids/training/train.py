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


def train_hybrid_ids(train_csv="UNSW-NB15_Test_File_20thFeb.csv", 
                     test_csv=None,
                     artifacts_dir="artifacts",
                     config_dict=None,
                     test_size=0.2,
                     verbose=1):
    """
    Train Hybrid IDS pipeline
    
    Args:
        train_csv: Path to training CSV (can be the only file - will be auto-split)
        test_csv: Path to separate test CSV (optional - if None, train_csv is split)
        artifacts_dir: Directory to save models
        config_dict: Optional configuration dictionary
        test_size: Fraction of data to use as test set when auto-splitting (default: 0.2)
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
    if test_csv is not None and Path(test_csv).exists():
        # Two separate files provided (e.g., KDDTrain.csv + KDDTest.csv)
        # Load them separately
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        # Normalize column names (strip whitespace) - fixes e.g. 'ct_src_ ltm' in UNSW-NB15 Dataset
        df_train.columns = df_train.columns.str.strip()
        df_test.columns = df_test.columns.str.strip()
        logger.info(f"  Mode: Two separate files")
        logger.info(f"  Train samples: {len(df_train)}")
        logger.info(f"  Test samples: {len(df_test)}")
    else:
        # Single file - auto-split into train and test
        df_full = pd.read_csv(train_csv)
        # Normalize column names (strip whitespace) - fixes e.g. 'ct_src_ ltm' in UNSW-NB15
        df_full.columns = df_full.columns.str.strip()
        logger.info(f"  Mode: Single file (auto-split {int((1-test_size)*100)}/{int(test_size*100)})")
        logger.info(f"  Total samples: {len(df_full)}")
        logger.info(f"  Total columns: {len(df_full.columns)}")
        df_train, df_test = train_test_split(
            df_full,
            test_size=test_size,
            random_state=42,
            shuffle=True
        )
        # Resetting the indexes
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        logger.info(f"  Train samples: {len(df_train)}")
        logger.info(f"  Test samples: {len(df_test)}")
    
    # Irrespective of whether only one file was supplied or 2 files(Separate Train and Test), the final output 
    # uptill this point will be 2 files (df_train and df_test)

    # 2. Create Binary Target
    logger.info("\n[2/5] Creating binary targets...")
    
    # Auto-detect label column
    # IMPORTANT NOTE => THE DATASET (IF IT HAS A TARGET VARIABLE) SUPPLIED MUST HAVE ANY ONE OF THE NAMES GIVEN BELOW AS ITS COLUMN NAME(TARGET VARIABLE)
    # IF NO COLUMN IS FOUND LIKE THIS, THE PIPELINE CAN THROW AN ERROR
    # SO, ALWAYS SEE WHAT THE TARGET VARIABLE(S) NAME IS. IF IT IS ALREADY PRESENT, NO ERROR WILL COME.
    # IF NOT, ADD THAT NAME IN THE 'label_col_candidates' LIST ACCORDINGLY
    label_col_candidates = ["attack_class", "label", "class", "target", "attack", "intrusion"]
    label_col = None
    normal_label = "normal"  # Default normal class identifier
    
    for col in label_col_candidates:
        if col in df_train.columns:
            label_col = col
            # Auto-detect what represents "normal"
            unique_vals = df_train[col].astype(str).str.lower().unique() # This part makes it case insensitive
            if "normal" in unique_vals:
                normal_label = "normal"
            # ASSUMING "0" represents "Normal" class    
            elif "0" in unique_vals:
                # Check if the column is actually integer (e.g., UNSW-NB15 label=0/1)
                # Use integer 0 for comparison, not the string "0"
                if pd.api.types.is_integer_dtype(df_train[col]):
                    normal_label = 0
                else:
                    normal_label = "0"
            break
    
    if label_col is None:
        raise ValueError(f"No label column found. Expected one of: {label_col_candidates}")
    
    logger.info(f"  Using label column: '{label_col}' (normal class: '{normal_label}')")
    
    # Create binary intrusion target
    if isinstance(normal_label, int):
        # Integer label column (e.g., UNSW-NB15: 0=normal, 1=attack)
        # Below part converts a multi classification problem into binary classification problem
        df_train["is_intrusion"] = (df_train[label_col] != normal_label).astype(int) 
        df_test["is_intrusion"] = (df_test[label_col] != normal_label).astype(int) # Same thing applied to test
    else:
        # String label column (e.g., KDD: 'normal', UNSW-Dataset2: attack names)
        df_train["is_intrusion"] = (df_train[label_col].astype(str).str.lower() != normal_label).astype(int) # Logic stays the same as above
        df_test["is_intrusion"] = (df_test[label_col].astype(str).str.lower() != normal_label).astype(int)
    
    logger.info(f"  Train intrusions: {df_train['is_intrusion'].sum()} ({df_train['is_intrusion'].mean()*100:.1f}%)")
    logger.info(f"  Test intrusions: {df_test['is_intrusion'].sum()} ({df_test['is_intrusion'].mean()*100:.1f}%)")
    
    # Track novel(never seen before) attacks(If they exist)
    train_attacks = set(df_train[label_col].unique())
    test_attacks = set(df_test[label_col].unique())
    novel_attacks = test_attacks - train_attacks
    
    logger.info(f"Novel attacks in test: {len(novel_attacks)}")
    if verbose >= 2:
        logger.info(f"Novel attack types: {sorted(novel_attacks)}")
    
    # 3. Prepare Features
    logger.info("\n[3/5] Preparing features...")
    X_train_full = df_train.drop(columns=["is_intrusion"], errors='ignore') # Target column will be dropped from the feature data
    y_train_full = df_train["is_intrusion"] # Target variable
    X_test = df_test.drop(columns=["is_intrusion"], errors='ignore')
    y_test = df_test["is_intrusion"] # Target variable
    
    # Store test attack labels for later analysis
    test_attack_labels = df_test[label_col].copy() if label_col in df_test.columns else pd.Series(dtype=str)
    
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
    # Auto-detect available CSV files in the training directory
    training_dir = Path(__file__).parent
    artifacts_path = str(Path(__file__).parent.parent / "artifacts")

    # Check for KDD two-file setup first, then fall back to any single CSV
    train_csv = "KDDTrain.csv"
    test_csv = None
    if (training_dir / "KDDTrain.csv").exists() and (training_dir / "KDDTest.csv").exists():
        train_csv = "KDDTrain.csv"
        test_csv = "KDDTest.csv"
        print("Using KDD two-file setup: KDDTrain.csv + KDDTest.csv")
    else:
        # Find any CSV in the training directory
        csv_files = list(training_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {training_dir}")
        train_csv = str(csv_files[0])
        print(f"Using single file (auto-split): {csv_files[0].name}")

    pipeline, data = train_hybrid_ids(
        train_csv=train_csv,
        test_csv=test_csv,
        artifacts_dir=artifacts_path,
        verbose=2
    )
    
    # Compute log-loss on test set
    from sklearn.metrics import log_loss
    X_test = data['X_test']
    y_test = data['y_test']
    test_probs = pipeline.predict_proba(X_test) # This step can take some time to process, especially if the dataset is huge
    test_log_loss = log_loss(y_test, test_probs)
    
    print("\nPipeline trained successfully!")
    print(f"Threshold: {pipeline.threshold_optimizer.best_threshold:.3f}")
    print(f"Supervised weight: {pipeline.config.supervised_weight:.3f}")
    print(f"Test Log-Loss: {test_log_loss:.6f}")
