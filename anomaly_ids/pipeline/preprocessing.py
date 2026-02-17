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
        self.label_cols = label_cols or ["attack_class", "attack_class_category", "label"] # Either they will be obtained or will be set to the default KDD dataset labels
        self.max_categories = max_categories
        self.drop_cols = None
        self.training_columns = None
    
    def fit(self, df):
        """Fit on training data to learn which columns to drop and detect categorical columns"""
        # Backward compatibility: ensure label_cols exists. Done in order so that the model does not crash while loading the default values in the pipeline
        if not hasattr(self, 'label_cols'):
            self.label_cols = ["attack_class", "attack_class_category", "label"]
        
        # Auto-detect categorical columns if not provided
        if self.categorical_cols is None:
            # For KDD dataset, use the original hardcoded columns for backward compatibility
            kdd_categorical = ["protocol_type", "service", "flag"]
            if all(col in df.columns for col in kdd_categorical):
                self.categorical_cols = kdd_categorical
            else:
                # For other datasets, auto-detect
                self.categorical_cols = []
                for col in df.columns:
                    if col in self.label_cols:
                        continue
                    # Only consider string/object types as categorical (not numeric)
                    if df[col].dtype == 'object':
                        self.categorical_cols.append(col)
        
        # Find columns to drop (constant or all-unique)
        self.drop_cols = [
            col for col in df.columns 
            if df[col].nunique() == 1 or df[col].nunique() == len(df)
        ]
        return self

    def transform(self, df):
        """Transform data (drop columns, one-hot encode)"""
        df = df.copy()
        
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
                df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=False)
        
        # Align columns with training data if this is a transformation on test/val
        if self.training_columns is not None:
            df = df.reindex(columns=self.training_columns, fill_value=0)
        else:
            # Store training columns
            self.training_columns = df.columns.tolist()
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        return self.fit(df).transform(df)


def drop_correlated_features(X, threshold = 0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)] # Drop one column out of every pair of columns which have correlation greater than 0.95
    return X.drop(columns = to_drop), to_drop

# Scaling
class ScalerWrapper:
    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

