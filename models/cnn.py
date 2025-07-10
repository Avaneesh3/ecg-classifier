import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, signals, labels=None, max_len=3000):
        self.X = [np.array(x[:max_len]) if len(x) >= max_len else np.pad(x, (0, max_len - len(x))) for x in signals]
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float).unsqueeze(0)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.long)
            return x, y
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 750, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
