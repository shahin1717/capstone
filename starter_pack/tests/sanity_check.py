import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from lite_torch import Tensor
from softmax import SoftmaxRegression
from neural_network import Linear
from data_loader import load_moons
from onehot import one_hot
from optimizers import MomentumOptimizer
from config import Config as cfg



def grad_check(model, x, y, num_classes, eps=1e-6):
    """
    @brief Perform gradient checking on the model's parameters using a small subset of data.
    @param model The model to check.
    @param x Input data (numpy array).
    @param y True labels (numpy array).
    @param num_classes Number of classes for one-hot encoding.
    @param eps Perturbation value for numerical gradient approximation.
    """    
    x_t = Tensor(x)
    y_t = Tensor(one_hot(y, num_classes))
    out = model(x_t)
    loss = out.crossentropy(y_t)
    
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)
        
    loss.backward()
    all_passed = True
    
    for p in model.parameters():
        ana_grad = p.grad.copy()
        num_grad = np.zeros_like(p.data)
        view = p.data.ravel()
        grad_view = num_grad.ravel()
        
        for i in range(len(view)):
            old_val = view[i]
            view[i] = old_val + eps
            l_plus = model(x_t).crossentropy(y_t).data
            view[i] = old_val - eps
            l_minus = model(x_t).crossentropy(y_t).data
            grad_view[i] = np.sum(l_plus - l_minus) / (2 * eps)
            view[i] = old_val
            
        diff = np.linalg.norm(ana_grad - num_grad) / (np.linalg.norm(ana_grad) + np.linalg.norm(num_grad) + 1e-10)
        
        if diff > eps:
            print(f"FAILED: Shape {p.data.shape} | error: {diff:.10f}")
            all_passed = False

    if all_passed:
        print("Grad check passed.")
    else:
        print("WARNING: Grad check failed.")
        

def train_few_batches(model, optimizer, x_train, y_train, num_classes, model_name):
    """
    @brief Train the model on a few batches of data and plot the loss curve.
    @param model The model to train.
    @param optimizer The optimizer to use for training.
    @param x_train Training input data (numpy array).
    @param y_train Training labels (numpy array).
    @param num_classes Number of classes for one-hot encoding.
    @param model_name Name of the model (for plotting and saving results).
    """
    
    batch_index = 0
    epoch_losses = []
    
    for i in range(0, x_train.shape[0], cfg.BATCH_SIZE):
        batch_index += 1
        x_batch, y_batch = x_train[i:i+cfg.BATCH_SIZE], y_train[i:i+cfg.BATCH_SIZE]
        
        optimizer.zero_grad()
        predictions = model(Tensor(x_batch))
        
        assert not np.isnan(predictions.data).any(), f"NaN detected in {model_name} predictions"
        
        prob_sum = np.sum(predictions.data, axis=1)
        assert np.allclose(prob_sum, 1.0, atol=1e-5), f"Probs do not sum to 1 in {model_name}"
        
        loss = predictions.crossentropy(Tensor(one_hot(y_batch, num_classes)))
        loss.backward()
        optimizer.step()
        
        current_loss = float(loss.data.mean())
        epoch_losses.append(current_loss)
        print(f"Batch {batch_index} loss: {current_loss:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title(f"Training Loss - {model_name}")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.grid(True)
    
    save_path = os.path.join(os.getcwd(), f"{model_name}_loss.png")
    plt.savefig(save_path)
    plt.close()

def overfit_on_small_batch(model, optimizer, x_batch, y_batch, num_classes, num_runs=100):
    """
    @brief Overfit the model on a small batch of data.
    @param model The model to overfit.
    @param optimizer The optimizer to use for training.
    @param x_batch Input data (numpy array).
    @param y_batch True labels (numpy array).
    @param num_classes Number of classes for one-hot encoding.
    @param num_runs Number of training runs.
    """
    losses = []
    for i in range(num_runs):
        optimizer.zero_grad()
        predictions = model(Tensor(x_batch))
        loss = predictions.crossentropy(Tensor(one_hot(y_batch, num_classes)))
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            losses.append(float(loss.data.mean()))

    print(f"Overfit history (every 20th): {losses}")

x_train_lg, y_train_lg, x_val_lg, y_val_lg, x_test_lg, y_test_lg = load_moons()
num_classes = int(y_train_lg.max()) + 1

models_to_run = [
    {
        "name": "SoftmaxRegression",
        "model": SoftmaxRegression(x_train_lg.shape[1], num_classes),
        "lr": cfg.MOMENTUM_LR
    },
    {
        "name": "LinearModel",
        "model": Linear(x_train_lg.shape[1], cfg.CAPACITY_WIDTHS[0], num_classes),
        "lr": cfg.MOMENTUM_LR
    }
]

for item in models_to_run:
    name = item["name"]
    model = item["model"]
    opt = MomentumOptimizer(model.parameters(), item["lr"])
    
    print("\n" + "="*50)
    print(f" RUNNING: {name} ")
    print("="*50)
    
    print(f"\n[1] Gradient Checking - {name}")
    grad_check(model, x_val_lg[:10], y_val_lg[:10], num_classes, eps=1e-3)
    
    print(f"\n[2] Training Few Batches - {name}")
    train_few_batches(model, opt, x_train_lg, y_train_lg, num_classes, name)
    
    print(f"\n[3] Overfitting Test - {name}")
    overfit_on_small_batch(model, opt, x_train_lg[:10], y_train_lg[:10], num_classes)
    
    print(f"\n{name} tasks completed. Plot saved to: {os.path.join(os.getcwd(), name + '_loss.png')}")