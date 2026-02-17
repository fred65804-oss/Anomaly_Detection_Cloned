"""Pipeline package initialization"""

from .config import IDSConfig, DEFAULT_CONFIG
from .preprocessing import Preprocessor, ScalerWrapper, drop_correlated_features
from .feature_engineering import add_statistical_features, PCATransformer
from .autoencoder import AutoencoderIDS
from .anomaly_models import IsolationForestDetector, LocalOutlierFactorDetector
from .supervised import SupervisedRF
from .normalization import ScoreNormalizer
from .ensemble import HybridEnsemble, optimize_supervised_weight
from .threshold import ThresholdOptimizer
from .pipeline import HybridIDSPipeline

__all__ = [
    'IDSConfig',
    'DEFAULT_CONFIG',
    'Preprocessor',
    'ScalerWrapper',
    'drop_correlated_features',
    'add_statistical_features',
    'PCATransformer',
    'AutoencoderIDS',
    'IsolationForestDetector',
    'LocalOutlierFactorDetector',
    'SupervisedRF',
    'ScoreNormalizer',
    'HybridEnsemble',
    'optimize_supervised_weight',
    'ThresholdOptimizer',
    'HybridIDSPipeline',
]
