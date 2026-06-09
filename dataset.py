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
    def __init__(self, X, y, augment=False, use_sr=True, use_sliding_window=False, expand=1):
        X = X[:, :, :1000].copy()
        self.expand = max(1, int(expand)) if augment else 1

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
        return len(self.y) * self.expand

    def __getitem__(self, idx):
        if self.use_sliding:
            return self.sw_dataset[idx]

        idx = idx % len(self.y)  # expand>1 reuses trials with fresh random augmentation
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


# Dataset registry: name -> (MOABB dataset class name, n_classes)
DATASETS = {
    'iv2a': ('BNCI2014_001', 4),   # BCI IV-2a: 22 ch, 4 class, 2 sessions
    'iv2b': ('BNCI2014_004', 2),   # BCI IV-2b: 3 ch,  2 class, 5 sessions
}

LABEL_MAP = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}


def load_dataset(name='iv2a', fmin=0.5, fmax=100.0):
    """Load a MOABB MotorImagery dataset by registry name.

    Band-pass [fmin, fmax] Hz is configurable. Returns
    (X, y, meta, n_classes, n_channels); n_channels is inferred from the data.
    """
    import moabb.datasets as mds
    from moabb.paradigms import MotorImagery

    assert name in DATASETS, f"unknown dataset {name}; choices={list(DATASETS)}"
    ds_name, n_classes = DATASETS[name]
    paradigm = MotorImagery(n_classes=n_classes, channels=None, fmin=fmin, fmax=fmax)
    dataset = getattr(mds, ds_name)()
    dataset.download()
    X, y, meta = paradigm.get_data(dataset)

    y = np.array([LABEL_MAP[l] for l in y])
    n_channels = X.shape[1]
    print(f"    Dataset {name} ({ds_name}): X={X.shape}, classes={n_classes}, channels={n_channels}")
    return X, y, meta, n_classes, n_channels


def load_bci_iva_dataset(fmin=0.5, fmax=100.0):
    """Backwards-compatible alias for the IV-2a loader (returns X, y, meta)."""
    X, y, meta, _, _ = load_dataset('iv2a', fmin, fmax)
    return X, y, meta


def prepare_pretrain_data(X, y, meta, align='none'):
    """
    Prepare cross-subject pre-training data.
    Uses session 1 from ALL subjects. NO data leakage because session 2 is held out.

    align='ea': whiten each subject's session-1 trials by that subject's own mean
    spatial covariance (label-free), matching the within-subject EA distribution.
    """
    sessions = meta['session'].values
    s1 = sorted(np.unique(sessions))[0]
    session_1_mask = sessions == s1
    X_pretrain = X[session_1_mask][:, :, :1000].copy().astype(np.float32)
    y_pretrain = y[session_1_mask].copy()

    subj_p = meta['subject'].values[session_1_mask]
    if align in ('ea', 'ra'):
        for s in np.unique(subj_p):
            m = subj_p == s
            X_pretrain[m] = _ea_apply(X_pretrain[m], _whitener(X_pretrain[m], align))
    elif align == 'dual':
        Xd = np.zeros((X_pretrain.shape[0], 2 * X_pretrain.shape[1], X_pretrain.shape[2]), dtype=np.float32)
        for s in np.unique(subj_p):
            m = subj_p == s
            Xs = X_pretrain[m]
            Xd[m] = _dual_views(Xs, _inv_sqrt(_ea_reference(Xs)), _ca_whitener(Xs))
        X_pretrain = Xd

    # Global scalar z-score
    mean = X_pretrain.mean().astype(np.float32)
    std = X_pretrain.std().astype(np.float32) + 1e-8
    X_pretrain = ((X_pretrain - mean) / std).astype(np.float32)

    print(f"    Pretrain data: {X_pretrain.shape} trials, {len(np.unique(y_pretrain))} classes, align={align}")
    return X_pretrain, y_pretrain


def _stratified_split(y, val_frac, seed):
    """Class-stratified index split. Returns (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def _per_trial_standardize(X):
    """Standardize each trial per channel using its own statistics.
    No cross-trial / cross-set statistics -> zero normalization leakage.
    X: [N, C, T]
    """
    X = X.astype(np.float32)
    m = X.mean(axis=2, keepdims=True)
    s = X.std(axis=2, keepdims=True) + 1e-8
    return ((X - m) / s).astype(np.float32)


def _ea_reference(X):
    """Euclidean-Alignment reference = mean spatial covariance over trials.
    X: [N, C, T] -> R: [C, C]. Label-free.
    """
    X = X.astype(np.float64)
    covs = np.einsum('nct,ndt->ncd', X, X) / X.shape[2]
    return covs.mean(axis=0)


def _inv_sqrt(R):
    """Symmetric inverse square root of an SPD matrix."""
    evals, evecs = np.linalg.eigh(R)
    evals = np.clip(evals, 1e-6, None)
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T


def _ea_apply(X, R_inv_sqrt):
    """Whiten each trial: X' = R^{-1/2} X. X: [N, C, T]. Works for any whitener."""
    return np.einsum('cd,ndt->nct', R_inv_sqrt, X.astype(np.float64)).astype(np.float32)


def _ca_whitener(X):
    """Centroid / Riemannian Alignment whitener = invsqrtm(Riemannian mean covariance).

    Uses the SPD geometric (Frechet) mean of per-trial OAS covariances as the
    reference, instead of EA's arithmetic mean. Label-free. X: [N, C, T] -> [C, C].
    Recent MI transfer-learning literature lists this as Centroid Alignment (CA),
    alongside EA / Riemannian Alignment / Parallel Transport.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.base import invsqrtm
    covs = Covariances("oas").fit_transform(X.astype(np.float64))
    return invsqrtm(mean_riemann(covs)).astype(np.float64)


def _whitener(X, align):
    """Per-set whitener for the requested alignment ('ea' arithmetic / 'ra' Riemannian)."""
    if align == 'ea':
        return _inv_sqrt(_ea_reference(X))
    if align == 'ra':
        return _ca_whitener(X)
    raise ValueError(f"unknown align {align}")


def _dual_views(X, W_ea, W_ra):
    """Concatenate EA-aligned and RA-aligned views along channels: [N,C,T] -> [N,2C,T]."""
    return np.concatenate([_ea_apply(X, W_ea), _ea_apply(X, W_ra)], axis=1)


def prepare_subject_data(X, y, meta, subject, val_frac=0.2, seed=42, align='none'):
    """
    Within-subject split, leak-free (official BCI IV-2a protocol):

      train / val = session 1, stratified split (val_frac held out for model selection)
      test        = session 2 (evaluated exactly once on the val-selected checkpoint)

    align='ea': Euclidean Alignment — whiten each session by its own mean spatial
    covariance (label-free) before z-score. session-1 R is fit on the TRAIN portion
    only and applied to train+val; session-2 R is fit on its own trials. This directly
    counters cross-session covariance shift and leaks no labels.

    Z-score statistics are fit on the TRAIN portion only and applied to val and test.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    subject_idx = meta['subject'].values == subject
    X_subject = X[subject_idx][:, :, :1000].copy()
    y_subject = y[subject_idx].copy()
    subj_sessions = meta.loc[subject_idx, 'session'].values
    sessions_sorted = sorted(np.unique(subj_sessions))
    assert len(sessions_sorted) >= 2, \
        f"Subject {subject}: need >=2 sessions for session-based split, got {sessions_sorted}"

    s1_mask = subj_sessions == sessions_sorted[0]
    s2_mask = subj_sessions == sessions_sorted[1]
    X_s1, y_s1 = X_subject[s1_mask], y_subject[s1_mask]
    X_test, y_test = X_subject[s2_mask], y_subject[s2_mask]

    tr_idx, va_idx = _stratified_split(y_s1, val_frac, seed)
    X_train, y_train = X_s1[tr_idx], y_s1[tr_idx]
    X_val, y_val = X_s1[va_idx], y_s1[va_idx]

    if align in ('ea', 'ra'):
        Ris1 = _whitener(X_train, align)          # session-1 ref from train only
        X_train, X_val = _ea_apply(X_train, Ris1), _ea_apply(X_val, Ris1)
        X_test = _ea_apply(X_test, _whitener(X_test, align))  # session-2 ref, label-free
    elif align == 'dual':
        # Stack EA-aligned and RA-aligned views along the channel axis -> [N, 2C, T].
        We1, Wr1 = _inv_sqrt(_ea_reference(X_train)), _ca_whitener(X_train)   # session-1 refs (train)
        We2, Wr2 = _inv_sqrt(_ea_reference(X_test)), _ca_whitener(X_test)     # session-2 refs (own)
        X_train = _dual_views(X_train, We1, Wr1)
        X_val = _dual_views(X_val, We1, Wr1)
        X_test = _dual_views(X_test, We2, Wr2)

    mean = X_train.mean().astype(np.float32)
    std = X_train.std().astype(np.float32) + 1e-8

    def norm(a):
        return ((a.astype(np.float32) - mean) / std).astype(np.float32)

    X_train, X_val, X_test = norm(X_train), norm(X_val), norm(X_test)

    print(f"    [no-leakage] Subject {subject}: train={len(y_train)} val={len(y_val)} "
          f"(session {sessions_sorted[0]}), test={len(y_test)} (session {sessions_sorted[1]}); "
          f"align={align}; z-score from train only", flush=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_subject_data_cv(X, y, meta, subject, n_splits=5, fold=0, val_frac=0.2,
                            seed=42, align='none'):
    """
    Within-subject k-fold CV split (sessions pooled), leak-free. Uniform protocol for
    datasets without a clean 2-session train/test layout (e.g. IV-2b, 5 sessions).

      All of a subject's trials -> StratifiedKFold(n_splits); fold `fold` = test,
      the rest = train; a stratified val_frac is held out of train for selection.

    Alignment (EA/RA/dual) references are fit on the TRAIN portion (applied to train+val)
    and on the test fold's own trials (label-free). Z-score from train only. Test fold is
    evaluated exactly once. Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    from sklearn.model_selection import StratifiedKFold

    subject_idx = meta['subject'].values == subject
    Xs = X[subject_idx][:, :, :1000].copy()
    ys = y[subject_idx].copy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(Xs, ys))
    tr_full, te = splits[fold]
    X_te, y_test = Xs[te], ys[te]

    ytf = ys[tr_full]
    rel_tr, rel_va = _stratified_split(ytf, val_frac, seed)
    X_train, y_train = Xs[tr_full[rel_tr]], ytf[rel_tr]
    X_val, y_val = Xs[tr_full[rel_va]], ytf[rel_va]

    if align in ('ea', 'ra'):
        Wtr = _whitener(X_train, align)
        X_train, X_val = _ea_apply(X_train, Wtr), _ea_apply(X_val, Wtr)
        X_te = _ea_apply(X_te, _whitener(X_te, align))
    elif align == 'dual':
        We, Wr = _inv_sqrt(_ea_reference(X_train)), _ca_whitener(X_train)
        Wee, Wrr = _inv_sqrt(_ea_reference(X_te)), _ca_whitener(X_te)
        X_train = _dual_views(X_train, We, Wr)
        X_val = _dual_views(X_val, We, Wr)
        X_te = _dual_views(X_te, Wee, Wrr)

    mean = X_train.mean().astype(np.float32)
    std = X_train.std().astype(np.float32) + 1e-8

    def norm(a):
        return ((a.astype(np.float32) - mean) / std).astype(np.float32)

    X_train, X_val, X_te = norm(X_train), norm(X_val), norm(X_te)

    print(f"    [no-leakage] Subject {subject} fold {fold+1}/{n_splits}: train={len(y_train)} "
          f"val={len(y_val)} test={len(y_test)}; align={align}; z-score from train only", flush=True)
    return X_train, X_val, X_te, y_train, y_val, y_test


def prepare_loso_data(X, y, meta, test_subject, val_frac=0.1, seed=42, align='none'):
    """
    Leave-one-subject-out (cross-subject) split, leak-free:

      train / val = all trials (both sessions) of every OTHER subject,
                    stratified split (val_frac held out for model selection)
      test        = all trials (both sessions) of the held-out subject
                    (evaluated exactly once)

    align='none': per-trial channel-wise standardization (no cross-set statistic).
    align='ea'/'ra': per-SUBJECT alignment — whiten each subject's trials by that
    subject's own EA/Riemannian reference (label-free), then z-score. This is the
    standard cross-subject alignment that maps every subject's centroid to identity.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    subj = meta['subject'].values
    Xa = X[:, :, :1000]
    train_mask = subj != test_subject
    test_mask = subj == test_subject

    if align in ('ea', 'ra'):
        Xal = Xa.astype(np.float32).copy()
        for s in np.unique(subj):
            m = subj == s
            Xal[m] = _ea_apply(Xal[m], _whitener(Xal[m], align))
        # global scalar z-score from the training pool only
        mtr = Xal[train_mask].mean().astype(np.float32)
        std = Xal[train_mask].std().astype(np.float32) + 1e-8
        X_pool = ((Xal[train_mask] - mtr) / std).astype(np.float32)
        X_test = ((Xal[test_mask] - mtr) / std).astype(np.float32)
    else:
        X_pool = _per_trial_standardize(Xa[train_mask])
        X_test = _per_trial_standardize(Xa[test_mask])
    y_pool = y[train_mask].copy()
    y_test = y[test_mask].copy()

    tr_idx, va_idx = _stratified_split(y_pool, val_frac, seed)
    X_train, y_train = X_pool[tr_idx], y_pool[tr_idx]
    X_val, y_val = X_pool[va_idx], y_pool[va_idx]

    others = sorted(int(s) for s in np.unique(subj[train_mask]))
    print(f"    [no-leakage] LOSO test_subject={test_subject}: train={len(y_train)} "
          f"val={len(y_val)} from subjects {others}, test={len(y_test)} (subject {test_subject}); "
          f"align={align}", flush=True)
    return X_train, X_val, X_test, y_train, y_val, y_test
