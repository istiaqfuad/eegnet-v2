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


class ChannelDropout:
    """Randomly zero out channels (keeps same size)"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        mask = np.random.random(x.shape[0]) > self.p
        x_copy = x.copy()
        x_copy[~mask, :] = 0
        return x_copy


class BandStopFilter:
    """Simple band-stop filter simulation via frequency masking"""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, x):
        # Just add small frequency-domain noise (simulates notch filter)
        return x + np.random.randn(*x.shape) * 0.02


class BCIDataset(Dataset):
    """BCI Competition IV-2a Dataset"""
    def __init__(self, X, y, augment=False):
        # Use first 1000 timesteps (4 seconds at 250Hz)
        X = X[:, :, :1000].copy()
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.sr = SRSegment(n_segments=4)
        self.channel_drop = ChannelDropout(p=0.1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            r = np.random.random()
            if r > 0.6:
                # Segment shuffle (+8.82% per MSCARNet paper)
                x = torch.from_numpy(self.sr(x.numpy()))
            elif r > 0.4:
                # Channel permutation
                perm = np.random.permutation(x.shape[0])
                x = x[perm]
            elif r > 0.25:
                # Temporal shift
                shift = np.random.randint(-50, 50)
                x = torch.roll(x, shifts=shift, dims=1)
            elif r > 0.1:
                # Channel dropout
                x = torch.from_numpy(self.channel_drop(x.numpy()))
            elif r > 0:
                # Gaussian noise
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


def prepare_subject_data(X, y, meta, subject, evaluation_mode='session_based'):
    """
    Prepare train/test split for a subject

    Args:
        X, y, meta: Full dataset
        subject: Subject number
        evaluation_mode: 'session_based' (official) or 'merged' (non-standard)

    Returns:
        X_train, X_test, y_train, y_test

    Evaluation Modes:
    - 'session_based': Train on session 1, test on session 2 (official BCI competition)
    - 'merged': Merge both sessions, random 90/10 split (non-standard, may inflate results)
    """
    subject_idx = meta['subject'].values == subject
    X_subject = X[subject_idx].copy()
    y_subject = y[subject_idx].copy()
    sessions = np.unique(meta.loc[subject_idx, 'session'].values)

    if evaluation_mode == 'session_based':
        # Standard evaluation: train on session 1, test on session 2
        # This matches EEGEncoder and other SOTA methods
        if len(sessions) >= 2:
            # Sort sessions to ensure consistent ordering
            sessions_sorted = sorted(sessions)
            s1_mask = meta.loc[subject_idx, 'session'].values == sessions_sorted[0]
            s2_mask = meta.loc[subject_idx, 'session'].values == sessions_sorted[1]

            X_train = X_subject[s1_mask]
            y_train = y_subject[s1_mask]
            X_test = X_subject[s2_mask]
            y_test = y_subject[s2_mask]
        else:
            # Fallback if only one session
            n_train = int(0.9 * len(y_subject))
            X_train, X_test = X_subject[:n_train].copy(), X_subject[n_train:].copy()
            y_train, y_test = y_subject[:n_train], y_subject[n_train:]

    elif evaluation_mode == 'merged':
        # Non-standard: merge both sessions, random split
        # WARNING: This gives artificially inflated results
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

    # Z-score normalization per subject (using ONLY training data)
    mean, std = X_train.mean(), X_train.std() + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, y_train, y_test