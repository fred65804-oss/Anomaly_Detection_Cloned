# tests/test_inference.py
def test_probs_between_0_and_1(hybrid_probs):
    assert (hybrid_probs >= 0).all()
    assert (hybrid_probs <= 1).all()

