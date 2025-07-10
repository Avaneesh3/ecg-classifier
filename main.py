from data.io import read_binary_file
from analysis.descriptive import summarize_class_distribution, analyze_lengths, compute_descriptive_stats
from analysis.plots import export_and_visualize_class_statistics, plot_sample_ecg_signals
from features.feature_extraction import extract_features
from models.cnn import ECGDataset, SimpleCNN
from models.train import train_cnn
from utils.split import create_validation_split

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def main():
    data_dir = "data"
    output_dir = "analysis_outputs"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    X_train = read_binary_file(os.path.join(data_dir, "X_train.bin"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"), header=None)[0]
    X_test = read_binary_file(os.path.join(data_dir, "X_test.bin"))

    summarize_class_distribution(y_train, output_dir)
    analyze_lengths(X_train, y_train, output_dir)
    compute_descriptive_stats(X_train, y_train, output_dir)
    export_and_visualize_class_statistics(X_train, y_train, output_dir)
    plot_sample_ecg_signals(X_train, y_train, samples_per_class=3, output_dir=output_dir)

    train_idx, val_idx = create_validation_split(X_train, y_train)
    X_tr = [X_train[i] for i in train_idx]
    X_val = [X_train[i] for i in val_idx]
    y_tr = y_train.iloc[train_idx].values
    y_val = y_train.iloc[val_idx].values

    # Random Forest
    X_tr_feat = extract_features(X_tr)
    X_val_feat = extract_features(X_val)
    scaler = StandardScaler().fit(X_tr_feat)
    X_tr_scaled = scaler.transform(X_tr_feat)
    X_val_scaled = scaler.transform(X_val_feat)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr_scaled, y_tr)
    rf_f1 = f1_score(y_val, rf.predict(X_val_scaled), average='macro')

    # CNN
    from torch.utils.data import DataLoader
    train_ds = ECGDataset(X_tr, y_tr)
    val_ds = ECGDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    cnn = SimpleCNN()
    cnn, cnn_f1 = train_cnn(cnn, train_loader, val_loader, epochs=10)

    # Compare
    plt.bar(["RandomForest", "CNN"], [rf_f1, cnn_f1])
    plt.ylabel("Validation F1 Score (Macro)")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.clf()

    # Final predictions
    test_loader = DataLoader(ECGDataset(X_test), batch_size=32)
    preds = []
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.eval().to(device)
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            preds.extend(cnn(xb).argmax(dim=1).cpu().numpy())

    pd.DataFrame(preds).to_csv(os.path.join(data_dir, "base.csv"), index=False, header=False)

if __name__ == "__main__":
    main()
