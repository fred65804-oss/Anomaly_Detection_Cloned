import numpy as np
import pytest

from pipeline.normalization import ScoreNormalizer

def test_transform_before_fit_raises():
    normalizer = ScoreNormalizer()
    with pytest.raises(ValueError):
        normalizer.transform({"detector_a": np.array([1.0,2.0,3.0])})

def test_normalized_scores_stay_in_unit_range():
    scores = {"detector_a": np.array([0.0, 1.0, 5.0, 20.0, 100.0])}
    
    normalizer = ScoreNormalizer().fit(scores)
    result = normalizer.transform(scores)

    assert (result["detector_a"] >= 0).all()
    assert (result["detector_a"] <= 1).all()