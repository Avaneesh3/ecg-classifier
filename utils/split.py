import numpy as np
from sklearn.model_selection import train_test_split

def create_validation_split(X, y, test_size=0.2, seed=42):
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)
    return train_idx, val_idx
