"""
run_experiments.py
------------------
Experiment orchestration for the Math4AI capstone.
Runs every required experiment, saves all figures to figures/,
and dumps a results summary to results/summary.json.

Usage (from inside starter_pack/src/):
    python run_experiments.py

All paths are relative to starter_pack/, so run from src/ or adjust
FIG_DIR / RES_DIR below if your layout differs.
"""

import os
import json
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config as cfg
from data_loader import load_linear_gaussian, load_moons, load_digits
from softmax import SoftmaxRegression
from neural_network import Linear
from optimizers import SGD, MomentumOptimizer, Adam
from train import train, evaluate
from lite_torch import Tensor

FIG_DIR = "../figures"
RES_DIR = "../results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"softmax": "#5177b8", "nn": "#e07b39"}



def _make_optimizer(name, params):
    """
    @brief Factory function to create optimizers by name. Uses hyperparameters from config.py.
    @param name: string, one of "SGD", "Momentum", "Adam"
    @param params: list of model parameters to optimize (from model.parameters())
    @return: an instance of the requested optimizer initialized with the given parameters
    """
    
    if name == "SGD":
        return SGD(params, cfg.SGD_LR)
    if name == "Momentum":
        return MomentumOptimizer(params, cfg.MOMENTUM_LR, cfg.MOMENTUM_COEF)
    if name == "Adam":
        return Adam(params, cfg.ADAM_LR, cfg.ADAM_B1, cfg.ADAM_B2, cfg.ADAM_EPS)
    raise ValueError(f"Unknown optimizer: {name}")


def _predict(model, X):
    return model(Tensor(X)).data.argmax(axis=1)


def _ci95(values):
    """
    @brief Computes the 95% confidence interval half-width for a list of values.
    @param values: list of numeric values (e.g., accuracies from multiple seeds)
    @return: the half-width of the 95% confidence interval for the mean of the values, using the sample standard deviation and the critical t-value for 95% confidence with n-
    """
    
    arr = np.array(values, dtype=float)
    return cfg.T_CRITICAL * arr.std(ddof=1) / np.sqrt(len(arr))


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {path}")


def plot_boundary(ax, model, X, y, title, h=0.02):
    x0_min, x0_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    x1_min, x1_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h),
                         np.arange(x1_min, x1_max, h))
    zz = _predict(model, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="RdBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu",
               edgecolors="k", linewidths=0.4, s=20, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


# ---------------------------------------------------------------------------
# EXPERIMENT 1 — CORE COMPARISONS
# ---------------------------------------------------------------------------

def run_core_comparisons(datasets):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — Core comparisons")
    print("=" * 60)
    results = {}

    for ds_name, (X_tr, y_tr, X_v, y_v, X_te, y_te) in datasets.items():
        k = int(y_tr.max()) + 1
        d = X_tr.shape[1]
        results[ds_name] = {}

        for mname in ("softmax", "nn"):
            np.random.seed(42)
            if mname == "softmax":
                model = SoftmaxRegression(d, k, seed=42)
                opt   = SGD(model.parameters(), cfg.SGD_LR)
            else:
                model = Linear(d, cfg.HIDDEN_WIDTH, k, seed=42)
                opt   = SGD(model.parameters(), cfg.SGD_LR)

            tr_loss, tr_acc, v_loss, v_acc = train(
                model, opt, X_tr, y_tr, X_v, y_v, k)
            te_ce, te_acc = evaluate(model, X_te, y_te, k)

            results[ds_name][mname] = dict(
                model=model,
                tr_loss=tr_loss, tr_acc=tr_acc,
                val_loss=v_loss, val_acc=v_acc,
                test_ce=te_ce,   test_acc=te_acc,
            )
            print(f"  {ds_name:15s} | {mname:7s} | "
                  f"test ce={te_ce:.4f}  acc={te_acc:.4f}")

    return results


def plot_core_boundaries(core_results, datasets):
    for ds_name in ("linear_gaussian", "moons"):
        X_te, y_te = datasets[ds_name][4], datasets[ds_name][5]
        fig, axes  = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (mname, label) in zip(
                axes, [("softmax", "Softmax regression"),
                       ("nn",     "Neural network (h=32)")]):
            plot_boundary(ax, core_results[ds_name][mname]["model"],
                          X_te, y_te, label)
        fig.suptitle(
            f"Decision boundaries — {ds_name.replace('_', ' ').title()}",
            fontweight="bold")
        plt.tight_layout()
        savefig(f"decision_boundary_{ds_name}.png")


def plot_digits_curves(core_results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mname, label, color in [
            ("softmax", "Softmax regression", COLORS["softmax"]),
            ("nn",      "Neural network",     COLORS["nn"])]:
        res = core_results["digits"][mname]
        axes[0].plot(res["tr_loss"],  color=color, alpha=0.4, linewidth=1.0)
        axes[0].plot(res["val_loss"], color=color, linewidth=1.6,
                     label=f"{label} val")
        axes[1].plot(res["tr_acc"],   color=color, alpha=0.4, linewidth=1.0)
        axes[1].plot(res["val_acc"],  color=color, linewidth=1.6,
                     label=f"{label} val")

    for ax, yl, title in zip(
            axes,
            ["Cross-entropy loss", "Accuracy"],
            ["Training dynamics — loss (faded=train, solid=val)",
             "Training dynamics — accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    savefig("digits_training_curves.png")


# ---------------------------------------------------------------------------
# EXPERIMENT 2 — CAPACITY ABLATION
# ---------------------------------------------------------------------------

def run_capacity_ablation(datasets):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — Capacity ablation on moons")
    print("=" * 60)
    X_tr, y_tr, X_v, y_v, X_te, y_te = datasets["moons"]
    k = int(y_tr.max()) + 1
    d = X_tr.shape[1]
    results = {}

    for width in cfg.CAPACITY_WIDTHS:
        accs, val_ces = [], []
        for seed in range(cfg.N_SEEDS):
            np.random.seed(seed)
            model = Linear(d, width, k, seed=seed)
            train(model, SGD(model.parameters(), cfg.SGD_LR),
                  X_tr, y_tr, X_v, y_v, k)
            v_ce, _      = evaluate(model, X_v,  y_v,  k)
            _,    te_acc = evaluate(model, X_te, y_te, k)
            val_ces.append(v_ce); accs.append(te_acc)

        m_acc=float(np.mean(accs));   ci_acc=float(_ci95(accs))
        m_ce =float(np.mean(val_ces));ci_ce =float(_ci95(val_ces))
        results[width] = dict(test_acc_mean=m_acc, test_acc_ci=ci_acc,
                              val_ce_mean=m_ce,    val_ce_ci=ci_ce)
        print(f"  width={width:2d} | val ce={m_ce:.4f}+-{ci_ce:.4f} | "
              f"test acc={m_acc:.4f}+-{ci_acc:.4f}")
    return results


def plot_capacity_ablation(ablation_results, datasets):
    X_tr, y_tr, X_v, y_v, X_te, y_te = datasets["moons"]
    k = int(y_tr.max()) + 1
    d = X_tr.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, width in zip(axes, cfg.CAPACITY_WIDTHS):
        np.random.seed(0)
        model = Linear(d, width, k, seed=0)
        train(model, SGD(model.parameters(), cfg.SGD_LR),
              X_tr, y_tr, X_v, y_v, k)
        res   = ablation_results[width]
        title = (f"Hidden width = {width}\n"
                 f"test acc = {res['test_acc_mean']:.3f} "
                 f"+- {res['test_acc_ci']:.3f}")
        plot_boundary(ax, model, X_te, y_te, title)
    fig.suptitle("Capacity ablation — moons", fontweight="bold")
    plt.tight_layout()
    savefig("capacity_ablation_moons.png")


# ---------------------------------------------------------------------------
# EXPERIMENT 3 — OPTIMIZER STUDY
# ---------------------------------------------------------------------------

def run_optimizer_study(datasets):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 — Optimizer study on digits (neural net only)")
    print("=" * 60)
    X_tr, y_tr, X_v, y_v, X_te, y_te = datasets["digits"]
    k = int(y_tr.max()) + 1
    d = X_tr.shape[1]
    results = {}

    for opt_name in ("SGD", "Momentum", "Adam"):
        np.random.seed(42)
        model = Linear(d, cfg.HIDDEN_WIDTH, k, seed=42)
        opt   = _make_optimizer(opt_name, model.parameters())
        tr_loss, tr_acc, v_loss, v_acc = train(
            model, opt, X_tr, y_tr, X_v, y_v, k)
        te_ce, te_acc = evaluate(model, X_te, y_te, k)
        results[opt_name] = dict(
            tr_loss=tr_loss, val_loss=v_loss,
            tr_acc=tr_acc,   val_acc=v_acc,
            test_ce=te_ce,   test_acc=te_acc)
        print(f"  {opt_name:8s} | test ce={te_ce:.4f}  acc={te_acc:.4f}")
    return results


def plot_optimizer_study(opt_results):
    opt_colors = {"SGD": "#5177b8", "Momentum": "#e07b39", "Adam": "#4caf7d"}
    fig, axes  = plt.subplots(1, 2, figsize=(11, 4))
    for opt_name, res in opt_results.items():
        c = opt_colors[opt_name]
        axes[0].plot(res["tr_loss"],  color=c, alpha=0.4, linewidth=1.0)
        axes[0].plot(res["val_loss"], color=c, linewidth=1.6, label=opt_name)
        axes[1].plot(res["tr_acc"],   color=c, alpha=0.4, linewidth=1.0)
        axes[1].plot(res["val_acc"],  color=c, linewidth=1.6, label=opt_name)
    for ax, yl, title in zip(
            axes,
            ["Cross-entropy loss", "Accuracy"],
            ["Optimizer study — loss (faded=train, solid=val)",
             "Optimizer study — accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    savefig("optimizer_study_digits.png")


# ---------------------------------------------------------------------------
# EXPERIMENT 4 — REPEATED-SEED EVALUATION
# ---------------------------------------------------------------------------

def run_repeated_seeds(datasets):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4 — Repeated-seed evaluation on digits (5 seeds)")
    print("=" * 60)
    X_tr, y_tr, X_v, y_v, X_te, y_te = datasets["digits"]
    k = int(y_tr.max()) + 1
    d = X_tr.shape[1]
    results = {}

    for mname in ("softmax", "nn"):
        accs, ces = [], []
        for seed in range(cfg.N_SEEDS):
            np.random.seed(seed)
            if mname == "softmax":
                model = SoftmaxRegression(d, k, seed=seed)
                opt   = SGD(model.parameters(), cfg.SGD_LR)
            else:
                model = Linear(d, cfg.HIDDEN_WIDTH, k, seed=seed)
                opt   = Adam(model.parameters(), cfg.ADAM_LR, cfg.ADAM_B1, cfg.ADAM_B2, cfg.ADAM_EPS)
            train(model, opt, X_tr, y_tr, X_v, y_v, k)
            te_ce, te_acc = evaluate(model, X_te, y_te, k)
            accs.append(te_acc); ces.append(te_ce)

        m_acc=float(np.mean(accs)); ci_acc=float(_ci95(accs))
        m_ce =float(np.mean(ces));  ci_ce =float(_ci95(ces))
        results[mname] = dict(acc_per_seed=accs, ce_per_seed=ces,
                              acc_mean=m_acc, acc_ci=ci_acc,
                              ce_mean=m_ce,   ce_ci=ci_ce)
        print(f"  {mname:7s} | acc={m_acc:.4f}+-{ci_acc:.4f} | "
              f"ce={m_ce:.4f}+-{ci_ce:.4f}")
    return results


def plot_repeated_seeds(seed_results):
    fig, axes  = plt.subplots(1, 2, figsize=(9, 4))
    labels     = ["Softmax regression", "Neural network"]
    keys       = ["softmax", "nn"]
    bar_colors = [COLORS["softmax"], COLORS["nn"]]

    for ax, metric, yl, title in zip(
            axes,
            ["acc",       "ce"],
            ["Accuracy",  "Cross-entropy loss"],
            ["Test accuracy — 5 seeds, 95% CI",
             "Test cross-entropy — 5 seeds, 95% CI"]):
        means = [seed_results[k][f"{metric}_mean"] for k in keys]
        cis   = [seed_results[k][f"{metric}_ci"]   for k in keys]
        bars  = ax.bar(labels, means, color=bar_colors,
                       width=0.45, edgecolor="k", linewidth=0.6)
        ax.errorbar(labels, means, yerr=cis, fmt="none",
                    color="black", capsize=5, linewidth=1.4)
        ax.set_title(title); ax.set_ylabel(yl)
        for bar, m, ci in zip(bars, means, cis):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 0.003,
                    f"{m:.3f}+-{ci:.3f}",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    savefig("repeated_seeds_digits.png")


# ---------------------------------------------------------------------------
# EXPERIMENT 5 — FAILURE CASE
# ---------------------------------------------------------------------------

def run_failure_case(datasets):
    """
    Width-2 network on moons: the hypothesis space is too small to fit a
    curved boundary, so the model stalls at a near-linear cut regardless
    of how long it trains.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 — Failure case: width-2 network on moons")
    print("=" * 60)
    X_tr, y_tr, X_v, y_v, X_te, y_te = datasets["moons"]
    k = int(y_tr.max()) + 1
    d = X_tr.shape[1]

    np.random.seed(0)
    model_fail = Linear(d, 2, k, seed=0)
    tr_loss_f, _, v_loss_f, _ = train(
        model_fail, SGD(model_fail.parameters(), cfg.SGD_LR),
        X_tr, y_tr, X_v, y_v, k)
    te_ce_f, te_acc_f = evaluate(model_fail, X_te, y_te, k)

    np.random.seed(0)
    model_good = Linear(d, 32, k, seed=0)
    tr_loss_g, _, v_loss_g, _ = train(
        model_good, SGD(model_good.parameters(), cfg.SGD_LR),
        X_tr, y_tr, X_v, y_v, k)
    te_ce_g, te_acc_g = evaluate(model_good, X_te, y_te, k)

    print(f"  width= 2 | test ce={te_ce_f:.4f}  acc={te_acc_f:.4f}  <- failure")
    print(f"  width=32 | test ce={te_ce_g:.4f}  acc={te_acc_g:.4f}  <- reference")

    fig = plt.figure(figsize=(14, 4))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    plot_boundary(ax0, model_fail, X_te, y_te,
                  f"Width 2  (acc={te_acc_f:.3f}) — failure")
    plot_boundary(ax1, model_good, X_te, y_te,
                  f"Width 32 (acc={te_acc_g:.3f}) — reference")
    ax2.plot(v_loss_f, color="#e07b39", linewidth=1.6, label="width=2 val")
    ax2.plot(v_loss_g, color="#5177b8", linewidth=1.6, label="width=32 val")
    ax2.plot(tr_loss_f, color="#e07b39", linewidth=1.0, alpha=0.4)
    ax2.plot(tr_loss_g, color="#5177b8", linewidth=1.0, alpha=0.4)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title("Validation loss: failure vs reference")
    ax2.legend()

    fig.suptitle("Failure case: under-capacity on moons", fontweight="bold")
    savefig("failure_case_moons.png")

    return dict(fail=dict(test_ce=float(te_ce_f), test_acc=float(te_acc_f)),
                good=dict(test_ce=float(te_ce_g), test_acc=float(te_acc_g)))


# ---------------------------------------------------------------------------
# SAVE RESULTS JSON
# ---------------------------------------------------------------------------

def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "model"}
    if isinstance(obj, list):
        return [float(x) for x in obj]
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def save_summary(core, ablation, optimizer, seeds, failure):
    summary = {
        "core_comparisons":        _strip({ds: dict(models) for ds, models in core.items()}),
        "capacity_ablation_moons": _strip({str(w): v for w, v in ablation.items()}),
        "optimizer_study_digits":  _strip(optimizer),
        "repeated_seeds_digits":   _strip(seeds),
        "failure_case":            failure,
    }
    path = os.path.join(RES_DIR, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  results saved -> {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading datasets ...")
    datasets = {
        "linear_gaussian": load_linear_gaussian(),
        "moons":           load_moons(),
        "digits":          load_digits(),
    }

    core_results    = run_core_comparisons(datasets)
    plot_core_boundaries(core_results, datasets)
    plot_digits_curves(core_results)

    ablation_results = run_capacity_ablation(datasets)
    plot_capacity_ablation(ablation_results, datasets)

    opt_results      = run_optimizer_study(datasets)
    plot_optimizer_study(opt_results)

    seed_results     = run_repeated_seeds(datasets)
    plot_repeated_seeds(seed_results)

    failure_results  = run_failure_case(datasets)

    save_summary(core_results, ablation_results,
                 opt_results,  seed_results, failure_results)

    print("\nAll experiments complete.")
    print(f"  Figures -> {FIG_DIR}/")
    print(f"  Results -> {RES_DIR}/summary.json")
