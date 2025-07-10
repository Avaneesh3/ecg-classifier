import numpy as np

def extract_features(X):
    features = []
    for x in X:
        x_np = np.array(x)
        features.append([
            np.mean(x_np), np.std(x_np), np.min(x_np), np.max(x_np),
            np.median(x_np), np.percentile(x_np, 25), np.percentile(x_np, 75)
        ])
    return np.array(features)
