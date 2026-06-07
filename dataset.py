"""
Dataset loading and augmentation for BCI Competition IV-2a
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SRSegment:
    """Segmentation and Recombination (S&R) augmentation - critical for CTNet
    From CTNet paper: adds +7.21% accuracy improvement
    """
    def __init__(self, n_segments=4):
        self.n_segments = n_segments

    def __call__(self, x):
        n_channels, n_timesteps = x.shape
        segment_len = n_timesteps // self.n_segments
        remainder = n_timesteps % self.n_segments
        segments = []
        for i in range(self.n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len
            if i == self.n_segments - 1:
                end += remainder
            segments.append(x[:, start:end])
        indices = np.random.permutation(self.n_segments)
        return np.concatenate([segments[i] for i in indices], axis=1)


class FrequencyMasking:
    """Random frequency masking - masks random frequency bands
    Simulates band-stop filtering effects for robustness
    """
    def __init__(self, mask_prob=0.15, max_mask_ratio=0.2):
        self.mask_prob = mask_prob
        self.max_mask_ratio = max_mask_ratio

    def __call__(self, x):
        if np.random.random() > self.mask_prob:
            return x
        # Apply in time domain by smoothing random windows
        # Simulates frequency masking by temporal smoothing
        x = x.copy()
        n_channels, n_timesteps = x.shape
        mask_len = int(n_timesteps * self.max_mask_ratio * np.random.random())
        mask_start = np.random.randint(0, max(1, n_timesteps - mask_len))
        x[:, mask_start:mask_start + mask_len] *= float(np.random.uniform(0.2, 0.8))
        return x


class ChannelDropout:
    """Randomly zero out channels (keeps same size)"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        mask = np.random.random(x.shape[0]) > self.p
        x_copy = x.copy()
        x_copy[~mask, :] = 0
        return x_copy


class GaussianNoise:
    """Add Gaussian noise for robustness"""
    def __init__(self, p=0.2, std=0.05):
        self.p = p
        self.std = std

    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        return x + (np.random.randn(*x.shape) * self.std).astype(np.float32)


class TemporalShift:
    """Random temporal shift"""
    def __init__(self, p=0.2, max_shift=50):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        shift = np.random.randint(-self.max_shift, self.max_shift)
        return np.roll(x, shift, axis=1)


class ChannelPermutation:
    """Random channel permutation"""
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        perm = np.random.permutation(x.shape[0])
        return x[perm]


class SlidingWindowDataset(Dataset):
    """Sliding window augmentation - from MSCARNet paper (+8.82% improvement)
    Split EEG signal into 2 overlapping windows to expand dataset
    """
    def __init__(self, X, y, augment=False):
        X = X[:, :, :1000].copy()
        self.base_X = X.astype(np.float32)
        self.y = y.copy()
        self.augment = augment

        self.window_length = 750
        self.step = 375

        self.windows_X = []
        self.windows_y = []

        for i in range(len(X)):
            self.windows_X.append(X[i, :, :self.window_length])
            self.windows_y.append(y[i])
            w2 = X[i, :, self.step:]
            if w2.shape[1] < self.window_length:
                pad_size = self.window_length - w2.shape[1]
                w2 = np.pad(w2, ((0, 0), (0, pad_size)), mode='edge')
            self.windows_X.append(w2)
            self.windows_y.append(y[i])

        self.windows_X = np.array(self.windows_X)
        self.windows_y = np.array(self.windows_y)

        self.X = torch.FloatTensor(self.windows_X.astype(np.float32))
        self.y = torch.LongTensor(self.windows_y)

        self.sr = SRSegment(n_segments=4)

        if augment:
            self.sr_samples = {}
            for label in np.unique(y):
                class_idx = np.where(self.windows_y == label)[0]
                class_X = self.windows_X[class_idx].astype(np.float32)
                self.sr_samples[label] = class_X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            r = np.random.random()
            if r > 0.5 and self.sr_samples:
                sr_X = self.sr_samples.get(y.item())
                if sr_X is not None and len(sr_X) > 1:
                    rand_idx = np.random.randint(len(sr_X))
                    x = torch.from_numpy(sr_X[rand_idx].copy())
            elif r > 0.3:
                x = torch.from_numpy(self.sr(x.numpy()))
            elif r > 0.2:
                perm = np.random.permutation(x.shape[0])
                x = x[perm]
        return x, y


class BCIDataset(Dataset):
    """BCI Competition IV-2a Dataset with comprehensive augmentation pipeline"""
    def __init__(self, X, y, augment=False, use_sr=True, use_sliding_window=False):
        X = X[:, :, :1000].copy()

        if use_sliding_window:
            self.use_sliding = True
            self.sw_dataset = SlidingWindowDataset(X, y, augment=augment)
            self.X = self.sw_dataset.X
            self.y = self.sw_dataset.y
            self.augment = augment
            return

        self.use_sliding = False
        self.X = torch.FloatTensor(X.astype(np.float32))
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.use_sr = use_sr and augment

        # Augmentation pipeline
        self.sr = SRSegment(n_segments=4)
        self.freq_mask = FrequencyMasking(mask_prob=0.15)
        self.channel_drop = ChannelDropout(p=0.1)
        self.gaussian_noise = GaussianNoise(p=0.15, std=0.03)
        self.temporal_shift = TemporalShift(p=0.1, max_shift=40)
        self.channel_perm = ChannelPermutation(p=0.1)

        if self.use_sr:
            self.sr_samples = {}
            for label in np.unique(y):
                class_idx = np.where(y == label)[0]
                class_X = X[class_idx].astype(np.float32)
                self.sr_samples[label] = class_X

    def __len__(self):
        if self.use_sliding:
            return len(self.sw_dataset)
        return len(self.y)

    def __getitem__(self, idx):
        if self.use_sliding:
            return self.sw_dataset[idx]

        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            r = np.random.random()

            if self.use_sr and r > 0.55:
                # S&R augmentation (CTNet, +7.21%)
                sr_X = self.sr_samples.get(y.item())
                if sr_X is not None and len(sr_X) > 1:
                    rand_idx = np.random.randint(len(sr_X))
                    x = torch.from_numpy(sr_X[rand_idx].copy())
            elif r > 0.4:
                # Segment shuffle
                x = torch.from_numpy(self.sr(x.numpy()))
            elif r > 0.3:
                # Frequency masking
                x = torch.from_numpy(self.freq_mask(x.numpy()))
            elif r > 0.2:
                # Gaussian noise
                x = torch.from_numpy(self.gaussian_noise(x.numpy()))
            elif r > 0.1:
                # Channel permutation
                x_np = x.numpy().astype(np.float32)
                x_np = self.channel_perm(x_np)
                x = torch.from_numpy(x_np)
            elif r > 0:
                # Temporal shift
                x_np = x.numpy()
                x_np = self.temporal_shift(x_np)
                x = torch.from_numpy(x_np)

        return x, y


def load_bci_iva_dataset():
    """Load BCI Competition IV-2a dataset using MOABB
    Uses full frequency range (0.5-100 Hz) to match EEGEncoder paper preprocessing.
    """
    from moabb.paradigms import MotorImagery
    from moabb.datasets import BNCI2014_001

    # Use full frequency range (0.5-100 Hz) instead of default 7-35 Hz
    # This preserves gamma band (30-100 Hz) which is important for MI
    paradigm = MotorImagery(n_classes=4, channels=None, fmin=0.5, fmax=100.0)
    dataset = BNCI2014_001()
    dataset.download()
    X, y, meta = paradigm.get_data(dataset)

    label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
    y = np.array([label_map[l] for l in y])

    return X, y, meta


def prepare_pretrain_data(X, y, meta):
    """
    Prepare cross-subject pre-training data.
    Uses session 1 from ALL subjects. NO data leakage because session 2 is held out.
    """
    sessions = meta['session'].values
    session_1_mask = sessions == sorted(np.unique(sessions))[0]
    X_pretrain = X[session_1_mask].copy()
    y_pretrain = y[session_1_mask].copy()

    # Truncate to 1000 timesteps
    X_pretrain = X_pretrain[:, :, :1000].copy()

    # Normalize globally
    mean = X_pretrain.mean().astype(np.float32)
    std = X_pretrain.std().astype(np.float32) + 1e-8
    X_pretrain = ((X_pretrain.astype(np.float32) - mean) / std).astype(np.float32)

    print(f"    Pretrain data: {X_pretrain.shape} trials, {len(np.unique(y_pretrain))} classes")
    return X_pretrain, y_pretrain


def prepare_subject_data(X, y, meta, subject, evaluation_mode='session_based'):
    """
    Prepare train/test split for a subject with NO data leakage

    session_based: Train on session 1, test on session 2 (official BCI competition)
    """
    subject_idx = meta['subject'].values == subject
    X_subject = X[subject_idx].copy()
    y_subject = y[subject_idx].copy()
    sessions = np.unique(meta.loc[subject_idx, 'session'].values)

    if evaluation_mode == 'session_based':
        if len(sessions) >= 2:
            sessions_sorted = sorted(sessions)
            s1_mask = meta.loc[subject_idx, 'session'].values == sessions_sorted[0]
            s2_mask = meta.loc[subject_idx, 'session'].values == sessions_sorted[1]

            X_train = X_subject[s1_mask]
            y_train = y_subject[s1_mask]
            X_test = X_subject[s2_mask]
            y_test = y_subject[s2_mask]
        else:
            n_train = int(0.9 * len(y_subject))
            X_train, X_test = X_subject[:n_train].copy(), X_subject[n_train:].copy()
            y_train, y_test = y_subject[:n_train], y_subject[n_train:]

    elif evaluation_mode == 'merged':
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

    # Z-score normalization per subject (using ONLY training data - NO LEAKAGE)
    # Ensure float32 to avoid dtype mismatch on GPU
    mean = X_train.mean().astype(np.float32)
    std = X_train.std().astype(np.float32) + 1e-8
    X_train_norm = ((X_train.astype(np.float32) - mean) / std).astype(np.float32)
    X_test_norm = ((X_test.astype(np.float32) - mean) / std).astype(np.float32)

    return X_train_norm, X_test_norm, y_train, y_test
