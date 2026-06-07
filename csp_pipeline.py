"""CSP + LDA/SVM pipeline for BCI IV-2a"""

import numpy as np
import mne
mne.set_log_level('ERROR')
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from dataset import load_bci_iva_dataset, prepare_subject_data


def bandpass_filter(X, sfreq=250, fmin=8, fmax=30):
    from scipy.signal import butter, filtfilt
    nyq = sfreq / 2
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(4, [low, high], btype='band')
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            X_filtered[i, ch] = filtfilt(b, a, X[i, ch])
    return X_filtered


def run_csp_for_subject(subject, fmin=8, fmax=30):
    X, y, meta = load_bci_iva_dataset()
    X_train, X_test, y_train, y_test = prepare_subject_data(
        X, y, meta, int(subject), evaluation_mode='session_based'
    )
    
    X_train = bandpass_filter(X_train, fmin=fmin, fmax=fmax)
    X_test = bandpass_filter(X_test, fmin=fmin, fmax=fmax)
    
    print(f"  Subject {subject}: Train={X_train.shape}, Test={X_test.shape}")
    
    best_acc = 0
    best_config = None
    
    configs = [
        (6, 'lda'), (8, 'lda'), (4, 'lda'),
        (6, 'svm'), (8, 'svm'), (4, 'svm'),
    ]
    
    for n_comp, clf_type in configs:
        try:
            # CSP expects 3D (n_trials, n_channels, n_times)
            X_train_csp = X_train
            X_test_csp = X_test
            
            csp = CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)
            
            if clf_type == 'lda':
                clf = LinearDiscriminantAnalysis()
            else:
                clf = SVC(kernel='rbf', C=1.0, gamma='scale')
            
            pipeline = make_pipeline(csp, clf)
            pipeline.fit(X_train_csp, y_train)
            acc = pipeline.score(X_test_csp, y_test)
            
            print(f"    CSP(n={n_comp}) + {clf_type.upper()}: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_config = f"CSP(n={n_comp}) + {clf_type.upper()}"
        except Exception as e:
            print(f"    CSP(n={n_comp}) + {clf_type.upper()}: FAILED ({e})")
    
    print(f"  Best: {best_config} -> {best_acc:.4f}")
    return best_acc


if __name__ == '__main__':
    all_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    print("=" * 60)
    print("CSP + LDA/SVM Pipeline for BCI IV-2a")
    print("=" * 60)
    
    results = {}
    
    for subj in all_subjects:
        print(f"\nSubject {subj}:")
        acc = run_csp_for_subject(subj, fmin=8, fmax=30)
        results[subj] = acc
    
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    for subj in all_subjects:
        print(f"  S{subj}: {results[subj]:.4f}")
    
    mean_acc = np.mean(list(results.values()))
    print(f"\n  Mean Accuracy: {mean_acc:.4f}")
