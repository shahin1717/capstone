import numpy as np
from softmax import SoftmaxRegression
from neural_network import Linear
from config import Config as cfg
from optimizers import SGD, MomentumOptimizer, Adam
from data_loader import load_linear_gaussian, load_moons, load_digits
from lite_torch import Tensor
from onehot import one_hot


def _save_params(model):
    return [parameter.data.copy() for parameter in model.parameters()]


def _restore_params(model, saved_parameters):
    for parameter, data in zip(model.parameters(), saved_parameters):
        parameter.data = data.copy()


def evaluate(model, x_data, y_labels, num_classes):
    """
    @brief Evaluates the model on the given data and labels, computing the average cross-entropy loss and accuracy.
    @param model: the model to evaluate (e.g., SoftmaxRegression or Linear)
    @param x_data: input data as a numpy array of shape (num_samples, num_features)
    @param y_labels: true class labels as a numpy array of shape (num_samples
    @param num_classes: the number of classes for one-hot encoding the labels
    @return: a tuple (loss, accuracy) where loss is the average cross-entropy loss over the dataset and accuracy is the fraction of correctly predicted samples.
    """
    
    predictions = model(Tensor(x_data))
    loss = predictions.crossentropy(Tensor(one_hot(y_labels, num_classes)))
    accuracy = float(np.mean(np.argmax(predictions.data, axis=1) == y_labels))
    return float(loss.data.mean()), accuracy


def train_epoch(model, optimizer, x_train, y_train, num_classes):
    """
    @brief Trains the model for one epoch on the given training data and labels, using the specified optimizer.
    @param model: the model to train
    @param optimizer: the optimizer to use for updating the model parameters
    @param x_train: training input data as a numpy array of shape (num_samples, num_features)
    @param y_train: training class labels as a numpy array of shape (num_samples)
    @param num_classes: the number of classes for one-hot encoding the labels
    @return: a tuple (average_loss, accuracy) where average_loss is the average cross-entropy loss over the training data for this epoch and accuracy is the fraction of correctly predicted samples in the training data for this epoch.
    """
    
    permutation = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[permutation], y_train[permutation]
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for i in range(0, x_train.shape[0], cfg.BATCH_SIZE):
        x_batch, y_batch = x_train[i:i+cfg.BATCH_SIZE], y_train[i:i+cfg.BATCH_SIZE]
        optimizer.zero_grad()
        predictions = model(Tensor(x_batch))
        loss = predictions.crossentropy(Tensor(one_hot(y_batch, num_classes)))
        loss.backward()
        optimizer.step()
        total_correct += int(np.sum(np.argmax(predictions.data, axis=1) == y_batch))
        total_loss += float(loss.data.mean())
        total_samples += x_batch.shape[0]

    num_batches = int(np.ceil(x_train.shape[0] / cfg.BATCH_SIZE))
    return total_loss / num_batches, total_correct / total_samples


def train(model, optimizer, x_train, y_train, x_val, y_val, num_classes):
    """
    @brief Trains the model using the specified optimizer on the training data, while evaluating on the validation data to track performance and save the best model parameters based on validation loss.
    @param model: the model to train 
    @param optimizer: the optimizer to use for updating the model parameters 
    @param x_train: training input data as a numpy array of shape (num_samples, num_features)
    @param y_train: training class labels as a numpy array of shape (num_samples)
    @param x_val: validation input data as a numpy array of shape (num_val_samples, num_features)
    @param y_val: validation class labels as a numpy array of shape (num_val_samples)
    @param num_classes: the number of classes for one-hot encoding the labels
    """
    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []
    best_val_loss, best_params = np.inf, None

    for _ in range(cfg.MAX_EPOCHS):
        epoch_loss, epoch_acc = train_epoch(model, optimizer, x_train, y_train, num_classes)
        val_loss, val_acc = evaluate(model, x_val, y_val, num_classes)
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = _save_params(model)

    _restore_params(model, best_params)
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def run_softmax_baseline(dataset_name, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    @brief Runs the softmax regression baseline experiment on the given dataset, training the model and evaluating its performance on the validation and test sets, then printing the results.
    @param dataset_name: a string name for the dataset
    @param x_train: training input data as a numpy array of shape (num_samples, num_features)
    @param y_train: training class labels as a numpy array of shape (num_samples)
    @param x_val: validation input data as a numpy array of shape (num_val_samples, num_features)
    @param y_val: validation class labels as a numpy array of shape (num_val_samples)
    @param x_test: test input data as a numpy array of shape (num_test_samples, num_features)
    @param y_test: test class labels as a numpy array of shape (num_test_samples)
    @return: the trained SoftmaxRegression model after training and evaluation
    """
    num_classes = int(y_train.max()) + 1
    model = SoftmaxRegression(x_train.shape[1], num_classes)
    optimizer = SGD(model.parameters(), cfg.SGD_LR)
    train(model, optimizer, x_train, y_train, x_val, y_val, num_classes)
    
    val_loss, val_acc = evaluate(model, x_val, y_val, num_classes)
    test_loss, test_acc = evaluate(model, x_test, y_test, num_classes)
    
    print(f"\n=== Softmax | {dataset_name} ===")
    print(f"val   ce={val_loss:.4f} acc={val_acc:.4f}")
    print(f"test  ce={test_loss:.4f} acc={test_acc:.4f}")
    return model


def run_optimizer_study(dataset_name, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    @brief Runs the optimizer study experiment on the given dataset, training a linear model with different optimizers and evaluating their performance on the validation and test sets, then printing the results for each optimizer.
    @param dataset_name: a string name for the dataset
    @param x_train: training input data as a numpy array of shape 
    @param y_train: training class labels as a numpy array of shape (num_samples)
    @param x_val: validation input data as a numpy array of shape (num_val_samples, num_features)
    @param y_val: validation class labels as a numpy array of shape (num_val_samples)
    @param x_test: test input data as a numpy array of shape (num_test_samples, num_features)
    @param y_test: test class labels as a numpy array of shape (num_test_samples)
    @return: a dictionary containing the results for each optimizer, where the keys are the optimizer names and the values are dictionaries with validation and test loss and accuracy.
    """
    
    num_classes = int(y_train.max()) + 1
    optimizer_configs = [
        ("SGD",      lambda p: SGD(p, cfg.SGD_LR)),
        ("Momentum", lambda p: MomentumOptimizer(p, cfg.MOMENTUM_LR, cfg.MOMENTUM_COEF)),
        ("Adam",     lambda p: Adam(p, cfg.ADAM_LR, cfg.ADAM_B1, cfg.ADAM_B2, cfg.ADAM_EPS)),
    ]
    print(f"\n=== Optimizer Study | {dataset_name} ===")
    results = {}
    for opt_name, opt_factory in optimizer_configs:
        np.random.seed(0)
        model = Linear(x_train.shape[1], cfg.HIDDEN_WIDTH, num_classes)
        train(model, opt_factory(model.parameters()), x_train, y_train, x_val, y_val, num_classes)
        
        val_loss, val_acc = evaluate(model, x_val, y_val, num_classes)
        test_loss, test_acc = evaluate(model, x_test, y_test, num_classes)
        
        print(f"{opt_name}: val ce={val_loss:.4f} acc={val_acc:.4f} | test ce={test_loss:.4f} acc={test_acc:.4f}")
        results[opt_name] = dict(val_ce=val_loss, val_acc=val_acc, test_ce=test_loss, test_acc=test_acc)
    return results


def run_capacity_ablation(dataset_name, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    @brief Runs the capacity ablation experiment on the given dataset, training linear models with different hidden layer widths and evaluating their performance on the validation and test sets, then printing the results for each width along with confidence intervals.
    @param dataset_name: a string name for the dataset
    @param x_train: training input data as a numpy array of shape (num_samples, num_features)
    @param y_train: training class labels as a numpy array of shape (num_samples)
    @param x_val: validation input data as a numpy array of shape (num_val_samples, num_features)
    @param y_val: validation class labels as a numpy array of shape (num_val_samples)
    @param x_test: test input data as a numpy array of shape (num_test_samples, num_features)
    @param y_test: test class labels as a numpy array of shape (num_test_samples)
    @return: a dictionary containing the results for each hidden layer width, where the keys are the widths and the values are dictionaries with mean and confidence interval for validation loss and test accuracy.
    """
    num_classes = int(y_train.max()) + 1
    print(f"\n=== Capacity Ablation | {dataset_name} ===")
    results = {}
    for width in cfg.CAPACITY_WIDTHS:
        val_losses, test_accuracies = [], []
        for seed in range(cfg.N_SEEDS):
            np.random.seed(seed)
            model = Linear(x_train.shape[1], width, num_classes)
            optimizer = SGD(model.parameters(), cfg.SGD_LR)
            train(model, optimizer, x_train, y_train, x_val, y_val, num_classes)
            
            v_loss, _ = evaluate(model, x_val, y_val, num_classes)
            _, te_acc = evaluate(model, x_test, y_test, num_classes)
            
            val_losses.append(v_loss)
            test_accuracies.append(te_acc)
            
        val_losses, test_accuracies = np.array(val_losses), np.array(test_accuracies)
        val_mean = val_losses.mean()
        val_ci = cfg.T_CRITICAL * val_losses.std() / np.sqrt(cfg.N_SEEDS)
        test_mean = test_accuracies.mean()
        test_ci = cfg.T_CRITICAL * test_accuracies.std() / np.sqrt(cfg.N_SEEDS)
        
        print(f"width={width}: val ce={val_mean:.4f}±{val_ci:.4f} | test acc={test_mean:.4f}±{test_ci:.4f}")
        results[width] = dict(val_ce_mean=float(val_mean), val_ce_ci=float(val_ci),
                              test_acc_mean=float(test_mean), test_acc_ci=float(test_ci))
    return results


if __name__ == "__main__":
    x_train_lg, y_train_lg, x_val_lg, y_val_lg, x_test_lg, y_test_lg = load_linear_gaussian()
    x_train_mn, y_train_mn, x_val_mn, y_val_mn, x_test_mn, y_test_mn = load_moons()
    x_train_dg, y_train_dg, x_val_dg, y_val_dg, x_test_dg, y_test_dg = load_digits()

    run_softmax_baseline("linear_gaussian", x_train_lg, y_train_lg, x_val_lg, y_val_lg, x_test_lg, y_test_lg)
    run_softmax_baseline("moons", x_train_mn, y_train_mn, x_val_mn, y_val_mn, x_test_mn, y_test_mn)
    run_softmax_baseline("digits", x_train_dg, y_train_dg, x_val_dg, y_val_dg, x_test_dg, y_test_dg)

    run_optimizer_study("digits", x_train_dg, y_train_dg, x_val_dg, y_val_dg, x_test_dg, y_test_dg)

    run_capacity_ablation("moons", x_train_mn, y_train_mn, x_val_mn, y_val_mn, x_test_mn, y_test_mn)