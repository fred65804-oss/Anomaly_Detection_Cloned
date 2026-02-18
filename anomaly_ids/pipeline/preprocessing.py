"""
    This file contains class, functions for preprocessing the data, drops correlated features,
    and sclaes the data using RobustScaler
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

class Preprocessor:
    def __init__(self, categorical_cols=None, label_cols=None, max_categories=50):
        """
        Dynamic preprocessor that auto-detects feature types
        
        Args:
            categorical_cols: List of categorical column names (auto-detected if None)
            label_cols: List of label column names to drop (e.g., ['label', 'attack_class'])
            max_categories: Max unique values to consider a column categorical (default: 50)
        """
        self.categorical_cols = categorical_cols  # Will be auto-detected if None
        self.label_cols = label_cols or ["attack_class", "attack_class_category", "label", "attack"] # Common label column names across datasets (KDD: attack_class, UNSW: attack)
        self.max_categories = max_categories
        self.drop_cols = None
        self.training_columns = None
    
    def fit(self, df):
        """
            Fit on training data to learn which columns to drop and detect categorical columns
            This step will not perform any steps on the data, but rather define operations to be done by the 'transform' function
        """
        # Backward compatibility: ensure label_cols exists.
        if not hasattr(self, 'label_cols'):
            self.label_cols = ["attack_class", "attack_class_category", "label", "attack"]

        # Normalize column names (strip whitespace) - fixes e.g. 'ct_src_ ltm' in UNSW-NB15
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Auto-detect categorical columns if not provided
        if self.categorical_cols is None:
            # For KDD dataset only, the original hardcoded columns will be used for backward compatibility
            kdd_categorical = ["protocol_type", "service", "flag"]
            if all(col in df.columns for col in kdd_categorical):
                self.categorical_cols = kdd_categorical
            else:
                # For other datasets, auto-detect
                # Auto-detection logic => If the column type is object, treat it as 'categorical'
                self.categorical_cols = []
                for col in df.columns:
                    if col in self.label_cols:
                        continue
                    if df[col].dtype == 'object':
                        self.categorical_cols.append(col)
        
        # Find columns to drop (constant or all-unique in terms of variance)
        self.drop_cols = [
            col for col in df.columns 
            if df[col].nunique() == 1 or df[col].nunique() == len(df)
        ]
        return self

    def transform(self, df):
        """
            Transform data (drop columns, one-hot encode)
            Transformations defined in 'fit' function will be used here to apply on the dataset
        """
        df = df.copy()

        # Normalize column names (strip whitespace) - fixes e.g. 'ct_src_ ltm' in UNSW-NB15
        df.columns = df.columns.str.strip()

        # Backward compatibility: ensure label_cols exists (for old saved models)
        if not hasattr(self, 'label_cols'):
            self.label_cols = ["attack_class", "attack_class_category", "label"]
        
        # Drop label columns if present
        df = df.drop(columns=self.label_cols, errors="ignore")
        
        # Drop constant/unique columns
        df = df.drop(columns=self.drop_cols, errors="ignore")
        
        # One-hot encoding for categorical columns
        if self.categorical_cols:
            # Only encode columns that exist in current dataframe
            cols_to_encode = [col for col in self.categorical_cols if col in df.columns]
            if cols_to_encode:
                df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=False) # Will increase dimensionality, but will retain more attack information. Useful for the models
        
        # Align columns with training data if this is a transformation on test/val
        if self.training_columns is not None:
            df = df.reindex(columns=self.training_columns, fill_value=0)
        else:
            # Store training columns
            self.training_columns = df.columns.tolist()
        
        return df
    
    def fit_transform(self, df):
        """
        Fit and transform in one step
        """
        return self.fit(df).transform(df)


def drop_correlated_features(X, threshold=0.95, sample_size=50_000):
    # Computing corr() on millions of rows is very slow (O(n * f^2)).
    # Correlation structure stabilises well before 50K rows, so we sample
    # to find which columns to drop, then apply the drop to the full dataset.
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    corr = X_sample.corr().abs() # Correlation data (using absolute value)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)) # Upper triangle mask (k=1 shifts off the main diagonal so self-correlations of 1.0 are excluded)
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)] # Drop one column from every highly-correlated pair
    return X.drop(columns=to_drop), to_drop

# Scaling
class ScalerWrapper:
    def __init__(self):
        self.scaler = RobustScaler() # Loading scaler

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

