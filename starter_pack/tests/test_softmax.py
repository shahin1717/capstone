"""
test_softmax.py
---------------
Basic correctness tests for math_utils.py and softmax.py.

Run from starter_pack/src/:
    python ../tests/test_softmax.py

Tests
-----
1. Softmax output matches Section 3.6 worked example
2. Softmax rows sum to 1
3. One-hot encoding is correct
4. Cross-entropy matches Section 3.6 worked example
5. SoftmaxRegression forward pass shapes are correct
6. SoftmaxRegression predicted probabilities sum to 1
7. SoftmaxRegression backward pass shapes are correct
8. Loss decreases after a few gradient steps on tiny data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
from math_utils import softmax, one_hot, cross_entropy_loss
from softmax import SoftmaxRegression


def test_softmax_values():
    """Section 3.6 worked example: s = [1.2, 0.2, -0.4] -> p ~ [0.64, 0.23, 0.13]"""
    S = np.array([[1.2, 0.2, -0.4]])
    P = softmax(S)
    expected = np.array([[0.64, 0.23, 0.13]])
    assert np.allclose(P, expected, atol=1e-2), f"Expected ~{expected}, got {P}"
    print("PASS  test_softmax_values")


def test_softmax_sums_to_one():
    """Each row of softmax output must sum to exactly 1."""
    rng = np.random.default_rng(0)
    S = rng.standard_normal((50, 10))
    P = softmax(S)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10), "Rows do not sum to 1"
    print("PASS  test_softmax_sums_to_one")


def test_softmax_numerical_stability():
    """Softmax must not produce NaN or Inf on large inputs."""
    S = np.array([[1000.0, 1000.0, 1000.0],
                  [-1000.0, -1000.0, -1000.0]])
    P = softmax(S)
    assert not np.any(np.isnan(P)), "NaN detected in softmax output"
    assert not np.any(np.isinf(P)), "Inf detected in softmax output"
    print("PASS  test_softmax_numerical_stability")


def test_one_hot():
    """One-hot matrix must have exactly one 1 per row in the correct column."""
    y = np.array([0, 2, 1])
    Y_oh = one_hot(y, k=3)
    expected = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]], dtype=np.float64)
    assert np.array_equal(Y_oh, expected), f"Expected\n{expected}\ngot\n{Y_oh}"
    assert np.all(Y_oh.sum(axis=1) == 1), "Each row must sum to 1"
    print("PASS  test_one_hot")


def test_cross_entropy_value():
    """Section 3.6: true class=0, p~0.64 -> loss ~ 0.45"""
    P    = np.array([[0.64, 0.23, 0.13]])
    Y_oh = np.array([[1.0, 0.0, 0.0]])
    loss = cross_entropy_loss(P, Y_oh, weights=[], lam=0.0)
    assert abs(loss - 0.45) < 0.01, f"Expected ~0.45, got {loss:.4f}"
    print("PASS  test_cross_entropy_value")


def test_cross_entropy_second_class():
    """Section 3.6: true class=1, p~0.23 -> loss ~ 1.47"""
    P    = np.array([[0.64, 0.23, 0.13]])
    Y_oh = np.array([[0.0, 1.0, 0.0]])
    loss = cross_entropy_loss(P, Y_oh, weights=[], lam=0.0)
    assert abs(loss - 1.47) < 0.01, f"Expected ~1.47, got {loss:.4f}"
    print("PASS  test_cross_entropy_second_class")


def test_forward_shapes():
    """Forward pass must return correct shapes."""
    model = SoftmaxRegression(d=4, k=3, seed=0)
    X = np.random.default_rng(0).standard_normal((10, 4))
    S, P = model.forward(X)
    assert S.shape == (10, 3), f"S shape: expected (10,3), got {S.shape}"
    assert P.shape == (10, 3), f"P shape: expected (10,3), got {P.shape}"
    print("PASS  test_forward_shapes")


def test_forward_probs_sum_to_one():
    """Predicted probabilities must sum to 1 for every example."""
    model = SoftmaxRegression(d=4, k=3, seed=0)
    X = np.random.default_rng(1).standard_normal((20, 4))
    _, P = model.forward(X)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10), "P rows do not sum to 1"
    print("PASS  test_forward_probs_sum_to_one")


def test_backward_shapes():
    """Gradient shapes must match parameter shapes."""
    d, k, n = 4, 3, 10
    model = SoftmaxRegression(d=d, k=k, seed=0)
    X = np.random.default_rng(2).standard_normal((n, d))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    Y_oh = one_hot(y, k)
    _, P = model.forward(X)
    grads = model.backward(X, P, Y_oh)
    assert grads["W"].shape == (k, d), f"dW shape: expected ({k},{d}), got {grads['W'].shape}"
    assert grads["b"].shape == (k,),   f"db shape: expected ({k},), got {grads['b'].shape}"
    print("PASS  test_backward_shapes")


def test_loss_decreases():
    """Loss must decrease after a few gradient steps on tiny data."""
    d, k, n = 2, 2, 8
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, d))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    Y_oh = one_hot(y, k)

    model = SoftmaxRegression(d=d, k=k, seed=0)
    lr = 0.1

    _, P = model.forward(X)
    loss_before = model.loss(P, Y_oh, lam=0.0)

    for _ in range(20):
        _, P = model.forward(X)
        grads = model.backward(X, P, Y_oh, lam=0.0)
        model.W -= lr * grads["W"]
        model.b -= lr * grads["b"]

    _, P = model.forward(X)
    loss_after = model.loss(P, Y_oh, lam=0.0)

    assert loss_after < loss_before, (
        f"Loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"
    )
    print(f"PASS  test_loss_decreases  ({loss_before:.4f} -> {loss_after:.4f})")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n── math_utils tests ──")
    test_softmax_values()
    test_softmax_sums_to_one()
    test_softmax_numerical_stability()
    test_one_hot()
    test_cross_entropy_value()
    test_cross_entropy_second_class()

    print("\n── SoftmaxRegression tests ──")
    test_forward_shapes()
    test_forward_probs_sum_to_one()
    test_backward_shapes()
    test_loss_decreases()

    print("\nAll tests passed.")