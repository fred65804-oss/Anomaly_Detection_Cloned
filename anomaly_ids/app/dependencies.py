"""
Dependency injection for FastAPI
"""

import sys
from pathlib import Path
from functools import lru_cache

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import ModelManager


# Global pipeline instance
_pipeline = None
_model_version = "latest"


@lru_cache()
def get_model_manager():
    """Get model manager singleton"""
    return ModelManager("artifacts")


def get_pipeline():
    """
    Get loaded pipeline (lazy loading)
    
    Returns:
        Loaded HybridIDSPipeline
    """
    global _pipeline, _model_version
    
    if _pipeline is None:
        model_manager = get_model_manager()
        _pipeline = model_manager.load_pipeline(_model_version)
    
    return _pipeline


def reload_pipeline(version="latest"):
    """
    Reload pipeline from disk
    
    Args:
        version: Model version to load
    """
    global _pipeline, _model_version
    
    _model_version = version
    model_manager = get_model_manager()
    _pipeline = model_manager.load_pipeline(version)
    
    return _pipeline
