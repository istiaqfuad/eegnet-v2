"""
Dataset loading and augmentation for BCI Competition IV-2a
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SRSegment:
    """Segment shuffle augmentation"""
    def __init__(self, n_segments=4):
        self.n_segments = n_segments

    def __call__(self, x):
        n_channels, n_timesteps = x.shape
        segment_len = n_timesteps // self.n_segments
        segments = [x[:, i*segment_len:(i+1)*segment_len] for i in range(self.n_segments)]
        indices = np.random.permutation(self.n_segments)
        return np.concatenate([segments[i] for i in indices], axis=1)


class BCIDataset(Dataset):
    """BCI Competition IV-2a Dataset"""
    def __init__(self, X, y, augment=False):
        # Use first 1000 timesteps (4 seconds at 250Hz)
        X = X[:, :, :1000].copy()
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.sr = SRSegment(n_segments=4)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            r = np.random.random()
            if r > 0.4:
                x = torch.from_numpy(self.sr(x.numpy()))
            elif r > 0.2:
                perm = np.random.permutation(x.shape[0])
                x = x[perm]
            elif r > 0:
                x = x + torch.randn_like(x) * 0.05
        return x, y


def load_bci_iva_dataset():
    """Load BCI Competition IV-2a dataset using MOABB"""
    from moabb.paradigms import MotorImagery
    from moabb.datasets import BNCI2014_001

    paradigm = MotorImagery(n_classes=4, channels=None)
    dataset = BNCI2014_001()
    dataset.download()
    X, y, meta = paradigm.get_data(dataset)

    # Convert labels to integers
    label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
    y = np.array([label_map[l] for l in y])

    return X, y, meta


def prepare_subject_data(X, y, meta, subject, use_both_sessions=True):
    """
    Prepare train/test split for a subject

    Args:
        X, y, meta: Full dataset
        subject: Subject number
        use_both_sessions: If True, use both sessions for more data

    Returns:
        X_train, X_test, y_train, y_test
    """
    subject_idx = meta['subject'].values == subject
    X_subject = X[subject_idx].copy()
    y_subject = y[subject_idx].copy()
    sessions = np.unique(meta.loc[subject_idx, 'session'].values)

    if use_both_sessions and len(sessions) >= 2:
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

    # Z-score normalization per subject
    mean, std = X_train.mean(), X_train.std() + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, y_train, y_test