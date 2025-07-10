import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

def train_cnn(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(y_batch.numpy())

        f1 = f1_score(val_labels, val_preds, average='macro')
        acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{epochs}: Val F1 = {f1:.4f}, Accuracy = {acc:.4f}")

        if epoch == epochs - 1:
            print("\nFinal Classification Report:")
            print(classification_report(val_labels, val_preds, zero_division=0))

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model, best_val_f1
