import numpy as np
from lite_torch import Tensor
from softmax_regression import SoftmaxRegression

def test_tensor_softmax_logic():
    """Verify softmax activation on a Tensor matches expected math."""
    data = np.array([[1.2, 0.2, -0.4]])
    t = Tensor(data)
    p = t.softmax(axis=-1)
    
    expected = np.array([[0.64, 0.23, 0.13]])
    assert np.allclose(p.data, expected, atol=1e-2)
    assert np.allclose(np.sum(p.data, axis=-1), 1.0)
    print("PASS  test_tensor_softmax_logic")

def test_model_forward_shapes():
    """Verify the SoftmaxRegression call returns expected shapes."""
    input_dim, hidden_dim, batch_size = 4, 3, 10
    model = SoftmaxRegression(input_dim, hidden_dim, seed=42)
    X = Tensor(np.random.randn(batch_size, input_dim))
    
    P = model(X)
    assert P.data.shape == (batch_size, hidden_dim)
    print("PASS  test_model_forward_shapes")

def test_autodiff_flow():
    """Verify that calling .backward() populates gradients in model parameters."""
    model = SoftmaxRegression(input_dim=2, hidden_dim=2, seed=0)
    X = Tensor([[1.0, 2.0]])
    Y_true = Tensor([[1.0, 0.0]]) # Target
    
    # Forward pass
    P = model(X)
    # Compute loss (using the crossentropy method in your Tensor class)
    loss = P.crossentropy(Y_true)
    
    # Backward pass
    loss.backward()
    
    assert model.W.grad is not None
    assert model.b.grad is not None
    assert not np.all(model.W.grad == 0)
    print("PASS  test_autodiff_flow")

def test_loss_decreases():
    """Optimization test: ensure manual SGD steps reduce the loss."""
    input_dim, hidden_dim = 2, 2
    model = SoftmaxRegression(input_dim, hidden_dim, seed=1)
    
    # Simple XOR-like tiny data
    X_data = np.array([[0.5, 0.1], [0.1, 0.5]])
    Y_data = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    X = Tensor(X_data)
    Y = Tensor(Y_data)
    
    # Initial loss
    P_initial = model(X)
    loss_initial = P_initial.crossentropy(Y).data.sum()
    
    lr = 0.1
    for _ in range(10):
        # Reset gradients (lite_torch doesn't have zero_grad, so we manual reset)
        model.W.grad = np.zeros_like(model.W.data)
        model.b.grad = np.zeros_like(model.b.data)
        
        P = model(X)
        current_loss = P.crossentropy(Y)
        current_loss.backward()
        
        # SGD Update
        model.W.data -= lr * model.W.grad
        model.b.data -= lr * model.b.grad
        
    P_final = model(X)
    loss_final = P_final.crossentropy(Y).data.sum()
    
    assert loss_final < loss_initial
    print(f"PASS  test_loss_decreases ({loss_initial:.4f} -> {loss_final:.4f})")

if __name__ == "__main__":
    print("--- Running LiteTorch Softmax Tests ---")
    test_tensor_softmax_logic()
    test_model_forward_shapes()
    test_autodiff_flow()
    test_loss_decreases()
    print("--- All Tests Passed ---")