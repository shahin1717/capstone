# From Linear Scores to a Single Hidden Layer
### Math4AI Capstone — National AI Center

**Jellyfish team:** Shahin Alakparov, Murad Asgarov, Gurban Burjuzada, Davud Gurbanov

---

## Project Overview

This project investigates a single question: *when does a one-hidden-layer neural network genuinely improve on a linear softmax classifier, and when is the added complexity unnecessary?*

Both models are implemented entirely from scratch in NumPy — including forward passes, backpropagation, three optimizers, and a lightweight automatic differentiation engine (`lite_torch.py`). No PyTorch, TensorFlow, JAX, or scikit-learn model classes were used.

The models are evaluated across three datasets of increasing difficulty: a linearly separable Gaussian task, a nonlinear moons task, and a real-world handwritten digits benchmark.

---

## Requirements

Python 3.8 or higher is required. All dependencies are from the standard scientific Python stack:

```
numpy
matplotlib
```

Install them with:

```bash
pip install numpy matplotlib
```

No other packages are needed.

---

## How to Reproduce All Experiments

All scripts must be run from inside the `starter_pack/src/` directory.

**Step 1 — Clone the repository**

```bash
git clone https://github.com/shahin1717/capstone
cd capstone
```

**Step 2 — Run all core experiments**

```bash
cd starter_pack/src
python run_experiments.py
```

This single command runs all five experiments in order:
- Core model comparisons on all three datasets
- Capacity ablation on moons (hidden widths 2, 8, 32)
- Optimizer study on digits (SGD, Momentum, Adam)
- Repeated-seed evaluation with 95% confidence intervals (5 seeds)
- Failure case analysis (width-2 network on moons)

All figures are saved to `starter_pack/figures/` and a JSON summary of all numerical results is saved to `starter_pack/results/summary.json`.

**Step 3 — Run Track B confidence and reliability analysis**

```bash
python track_b_analysis.py
```

This trains both models on digits and produces the reliability diagram and correct-vs-wrong confidence analysis. The figure is saved to `starter_pack/figures/reliability_diagram.png`.

**Step 4 — Run implementation sanity checks (optional)**

```bash
python sanity_check.py
```

This runs gradient checking via finite differences, loss-decrease verification on small batches, and an overfitting test on a tiny subset. It confirms that backpropagation and parameter updates are correct.

---

## Dataset Files

All datasets are provided as `.npz` files in `starter_pack/data/` and are loaded automatically by the scripts. Do not move or rename them.

| File | Description |
|---|---|
| `linear_gaussian.npz` | Synthetic two-class Gaussian blobs |
| `moons.npz` | Synthetic two-class nonlinear crescents |
| `digits_data.npz` | 8×8 handwritten digit images (64-dimensional, scaled to [0,1]) |
| `digits_split_indices.npz` | Fixed train / validation / test split indices for digits |

---

## Hyperparameters and Configuration

All hyperparameters are defined in `config.py` and match the experimental protocol exactly as specified in the project handout. No hyperparameter search was performed beyond what the protocol requires.

| Parameter | Value |
|---|---|
| Hidden width (default) | 32 |
| L2 regularization λ | 1e-4 |
| Batch size | 64 |
| Max epochs | 200 |
| SGD learning rate | 0.05 |
| Momentum learning rate | 0.05 |
| Momentum coefficient | 0.9 |
| Adam learning rate | 0.001 |
| Adam β1, β2, ε | 0.9, 0.999, 1e-8 |
| Number of seeds | 5 |
| 95% CI t-critical (df=4) | 2.776 |

---

## Source Files

| File | Purpose |
|---|---|
| `lite_torch.py` | Custom autograd engine — Tensor class with dynamic computation graph and reverse-mode autodiff |
| `neural_network.py` | One-hidden-layer network with tanh activations and softmax output |
| `softmax.py` | Softmax regression baseline model |
| `optimizers.py` | SGD, Momentum, and Adam optimizers with L2 weight decay |
| `train.py` | Training loop, evaluation, checkpointing |
| `data_loader.py` | Dataset loading utilities |
| `onehot.py` | One-hot encoding helper |
| `config.py` | All hyperparameters in one place |
| `run_experiments.py` | Orchestrates all five experiments and saves figures and results |
| `track_b_analysis.py` | Track B: confidence, entropy, and reliability analysis |
| `sanity_check.py` | Gradient checking and implementation verification |

---

## Output Files

After running the scripts, the following files are produced:

**Figures (`starter_pack/figures/`)**
- `decision_boundary_linear_gaussian.png` — decision boundaries on the Gaussian task
- `decision_boundary_moons.png` — decision boundaries on the moons task
- `digits_training_curves.png` — training and validation loss/accuracy curves for digits
- `capacity_ablation_moons.png` — decision boundaries for widths 2, 8, 32
- `optimizer_study_digits.png` — training curves for SGD, Momentum, and Adam
- `repeated_seeds_digits.png` — bar chart of 5-seed means with 95% CI
- `failure_case_moons.png` — width-2 failure case with boundary and loss curve
- `reliability_diagram.png` — calibration plot for both models on digits

**Results (`starter_pack/results/`)**
- `summary.json` — all numerical results from every experiment in JSON format

---

## Key Results Summary

| Dataset | Softmax Accuracy | NN Accuracy | Notes |
|---|---|---|---|
| Linear Gaussian | **95.0%** | 93.75% | Linear model is sufficient |
| Moons | 85.0% | **93.75%** | Hidden layer necessary |
| Digits (Adam, 5 seeds) | 91.6% ± 9.8% | **95.3% ± 0.5%** | NN more accurate and more stable |

The neural network outperforms softmax regression on nonlinear tasks but requires an adaptive optimizer (Adam) to train reliably on the digits benchmark. SGD and Momentum fail on digits due to poor conditioning in the 64-dimensional input space.

---

## Notes on Reproducibility

All random seeds are set explicitly via `np.random.seed()` before each model initialization and training run. Results should be fully reproducible across machines given the same NumPy version. The digits train/validation/test split is fixed by the provided index file and is never modified by any script.
