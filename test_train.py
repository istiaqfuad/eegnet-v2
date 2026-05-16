import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def load_bci_iva_dataset():
    from moabb.paradigms import MotorImagery
    from moabb.datasets import BNCI2014_001
    paradigm = MotorImagery(n_classes=4, channels=None)
    dataset = BNCI2014_001()
    dataset.download()
    X, y, meta = paradigm.get_data(dataset)
    return X, y, meta


class BCIDataset(Dataset):
    def __init__(self, X, y):
        X = X[:, :, :1000].copy()
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BestEEGNet(nn.Module):
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.35)

        self.separable = nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.35)

        self.conv4 = nn.Conv2d(32, 64, (1, 7), padding=(0, 3), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d((1, 4))
        self.drop3 = nn.Dropout(0.35)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


print("Loading dataset...", flush=True)
X, y, meta = load_bci_iva_dataset()
label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
y = np.array([label_map[l] for l in y])

# Train only on subject 1
subject = 1
subject_idx = meta['subject'].values == subject
X_subject = X[subject_idx].copy()
y_subject = y[subject_idx].copy()
sessions = np.unique(meta.loc[subject_idx, 'session'].values)

if len(sessions) >= 2:
    s1 = meta.loc[subject_idx, 'session'].values == sessions[0]
    s2 = meta.loc[subject_idx, 'session'].values == sessions[1]
    all_X = np.concatenate([X_subject[s1], X_subject[s2]], axis=0)
    all_y = np.concatenate([y_subject[s1], y_subject[s2]], axis=0)
    idx = np.random.permutation(len(all_y))
    n_train = int(0.9 * len(all_y))
    X_train, X_test = all_X[idx[:n_train]], all_X[idx[n_train:]]
    y_train, y_test = all_y[idx[:n_train]], all_y[idx[n_train:]]
else:
    n_train = int(0.9 * len(y_subject))
    X_train, X_test = X_subject[:n_train].copy(), X_subject[n_train:].copy()
    y_train, y_test = y_subject[:n_train], y_subject[n_train:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

mean, std = X_train.mean(), X_train.std() + 1e-8
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

train_dataset = BCIDataset(X_train_norm, y_train)
val_dataset = BCIDataset(X_test_norm, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Train model
torch.manual_seed(42)
np.random.seed(42)

model = BestEEGNet(n_classes=4, n_channels=22).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.0035, epochs=100, steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos'
)

print("Starting training...", flush=True)
for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            _, predicted = model(X_batch).max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    val_acc = correct / total
    print(f"Epoch {epoch+1}/100 - Val Acc: {val_acc:.4f}", flush=True)

print("Done!", flush=True)