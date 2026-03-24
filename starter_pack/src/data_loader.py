import numpy as np
import os

 
def _load_npz(path: str) -> dict:
    """Load an .npz file and return its contents as a plain dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: '{path}'\n")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
 

def load_linear_gaussian(path:str="../data/linear_gaussian.npz"):
    """Load the linear Gaussian dataset."""
    
    data=_load_npz(path)
    return (
        data["X_train"].astype(np.float64), data["y_train"].astype(np.int64),
        data["X_val"].astype(np.float64),   data["y_val"].astype(np.int64),
        data["X_test"].astype(np.float64),  data["y_test"].astype(np.int64),
    )

def load_moons(path:str="../data/moons.npz"):
    """Load the moons dataset."""
    data=_load_npz(path)
    return (
        data["X_train"].astype(np.float64), data["y_train"].astype(np.int64),
        data["X_val"].astype(np.float64),   data["y_val"].astype(np.int64),
        data["X_test"].astype(np.float64),  data["y_test"].astype(np.int64),
    )               
    
def load_digits(data_path:str="../data/digits_data.npz", split_data_path:str="../data/digits_split_indices.npz"):
    """Load the digits dataset."""
    data  = _load_npz(data_path)
    split = _load_npz(split_data_path)
 
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
 
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]
 
    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )
    
    
def data_info(name: str, X: np.ndarray, y: np.ndarray): # Used for debugging.  
    """Print a compact summary of a dataset."""
    classes, counts = np.unique(y, return_counts=True)
    print(f"  [{name}]")
    print(f"    X : {X.shape}  dtype={X.dtype}  range=[{X.min():.3f}, {X.max():.3f}]")
    print(f"    y : {y.shape}  dtype={y.dtype}")
    print(f"    classes : {classes.tolist()}")
    print(f"    counts  : {counts.tolist()}")
 

if __name__ == "__main__": 
    print("\n-- Linear Gaussian --")
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_linear_gaussian()
    data_info("train", X_tr, y_tr)
    data_info("val",   X_v,  y_v)
    data_info("test",  X_te, y_te)
 
    print("\n-- Moons --")
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_moons()
    data_info("train", X_tr, y_tr)
    data_info("val",   X_v,  y_v)
    data_info("test",  X_te, y_te)
 
    print("\n-- Digits --")
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_digits()
    data_info("train", X_tr, y_tr)
    data_info("val",   X_v,  y_v)
    data_info("test",  X_te, y_te)
 
    print("\nAll datasets loaded successfully.")