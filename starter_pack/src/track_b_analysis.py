import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_digits
from neural_network import Linear
from softmax import SoftmaxRegression
from train import train, evaluate
from optimizers import SGD, Adam
from config import Config as cfg
from lite_torch import Tensor


def compute_confidence_entropy(probs):
    """
    @brief Compute confidence and entropy for each prediction.
    @param probs: (N, k) array of predicted probabilities for each class.
    @return confidence: (N,) array of max predicted probability for each sample.
    @return entropy: (N,) array of entropy of the predicted distribution for each sample.
    """
    
    confidence = np.max(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    return confidence, entropy


def reliability_diagram(confidence, preds, y_true, model_name):
    """
    @brief Plot the reliability diagram for a given model.
    @param confidence: (N,) array of max predicted probabilities for each sample.
    @param preds: (N,) array of predicted labels for each sample.
    @param y_true: (N,) array of true labels for each sample.
    @param model_name: Name of the model (for plotting).
    """
    bins = np.linspace(0, 1, 6)
    bin_indices = np.digitize(confidence, bins) - 1

    accs = []
    confs = []

    for i in range(5):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            acc = np.mean(preds[mask] == y_true[mask])
            conf_avg = np.mean(confidence[mask])
        else:
            acc = 0
            conf_avg = 0

        accs.append(acc)
        confs.append(conf_avg)

        print(f"{model_name} | Bin {i}: count={np.sum(mask)}, acc={acc:.3f}, conf={conf_avg:.3f}")

    plt.plot(confs, accs, marker='o', label=model_name)


def correct_vs_wrong(confidence, entropy, preds, y_true, model_name):
    """
    @brief Analyze confidence and entropy for correct vs wrong predictions.
    @param confidence: (N,) array of max predicted probabilities for each sample.
    @param entropy: (N,) array of entropy of the predicted distribution for each sample.
    @param preds: (N,) array of predicted labels for each sample.
    @param y_true: (N,) array of true labels for each sample.
    @param model_name: Name of the model (for printing).
    """
    correct = preds == y_true

    print(f"\n{model_name} analysis:")
    print(f"Correct confidence mean: {confidence[correct].mean():.3f}")
    print(f"Wrong confidence mean:   {confidence[~correct].mean():.3f}")
    print(f"Correct entropy mean:    {entropy[correct].mean():.3f}")
    print(f"Wrong entropy mean:      {entropy[~correct].mean():.3f}")


if __name__ == "__main__":
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_digits()
    k = int(y_tr.max()) + 1
    d = X_tr.shape[1]

    # ===== SOFTMAX MODEL =====
    softmax = SoftmaxRegression(d, k)
    train(softmax, SGD(softmax.parameters(), cfg.SGD_LR),
          X_tr, y_tr, X_v, y_v, k)

    probs_sm = softmax(Tensor(X_te)).data
    preds_sm = softmax.predict(Tensor(X_te))

    conf_sm, ent_sm = compute_confidence_entropy(probs_sm)

    # ===== NN MODEL =====
    nn = Linear(d, cfg.HIDDEN_WIDTH, k)
    train(nn, Adam(nn.parameters(), cfg.ADAM_LR, cfg.ADAM_B1, cfg.ADAM_B2, cfg.ADAM_EPS),
          X_tr, y_tr, X_v, y_v, k)

    probs_nn = nn(Tensor(X_te)).data
    preds_nn = nn.predict(Tensor(X_te))

    conf_nn, ent_nn = compute_confidence_entropy(probs_nn)

    # ===== RELIABILITY DIAGRAM =====
    plt.figure(figsize=(6, 6))
    reliability_diagram(conf_sm, preds_sm, y_te, "Softmax")
    reliability_diagram(conf_nn, preds_nn, y_te, "Neural Network")

    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.savefig("../figures/reliability_diagram.png")
    plt.close()

    # ===== CORRECT VS WRONG =====
    correct_vs_wrong(conf_sm, ent_sm, preds_sm, y_te, "Softmax")
    correct_vs_wrong(conf_nn, ent_nn, preds_nn, y_te, "Neural Network")