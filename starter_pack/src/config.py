class Config:
    HIDDEN_WIDTH = 32  # default hidden layer width for the neural network

    LAMBDA = 1e-4  # L2 regularization coefficient for both models
    BATCH_SIZE = 64  # mini-batch size
    MAX_EPOCHS = 200  # maximum training epochs

    # Softmax regression baseline
    SGD_LR = 0.05

    # Neural network optimizer study
    MOMENTUM_LR = 0.05
    MOMENTUM_COEF = 0.9

    ADAM_LR = 0.001
    ADAM_B1 = 0.9
    ADAM_B2 = 0.999
    ADAM_EPS = 1e-8

    CAPACITY_WIDTHS = [2, 8, 32]  # moons
    N_SEEDS = 5
    T_CRITICAL = 2.776  # 95% CI critical value, t-distribution df=4
