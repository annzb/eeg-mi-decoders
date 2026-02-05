import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def compute_cov(sample):
    """sample: (n_channels, n_samples) -> normalized covariance (n_channels, n_channels)"""
    C = sample @ sample.T
    tr = np.trace(C)
    return C / tr if tr > 0 else C


def fit_csp(X_train, y_train, n_components=4, reg=1e-10):
    """
    Classic 2-class CSP.
    Returns spatial filters W of shape (n_channels, n_components).
    """
    classes = np.unique(y_train)
    if len(classes) != 2:
        raise ValueError("CSP requires exactly 2 classes.")

    c0, c1 = classes[0], classes[1]
    X0 = X_train[y_train == c0]
    X1 = X_train[y_train == c1]

    C0 = np.mean([compute_cov(x) for x in X0], axis=0)
    C1 = np.mean([compute_cov(x) for x in X1], axis=0)

    # Regularize slightly for stability
    n_ch = C0.shape[0]
    C0 = C0 + reg * np.eye(n_ch)
    C1 = C1 + reg * np.eye(n_ch)

    Cc = C0 + C1

    # Solve generalized eigenvalue problem: C0 w = λ (C0 + C1) w
    # Use standard eig on inv(Cc) @ C0
    evals, evecs = np.linalg.eig(np.linalg.solve(Cc, C0))
    evals = np.real(evals)
    evecs = np.real(evecs)

    # Sort by eigenvalue (most discriminative at extremes)
    idx = np.argsort(evals)
    evecs = evecs[:, idx]

    # Pick extremes: first k/2 and last k/2
    k2 = n_components // 2
    W = np.concatenate([evecs[:, :k2], evecs[:, -k2:]], axis=1)
    return W


def csp_logvar_features(X, W):
    """
    Project: Z = W^T X, then log-variance per component.
    X: (n_trials, n_channels, n_samples)
    W: (n_channels, n_components)
    Returns F: (n_trials, n_components)
    """
    Z = np.einsum("tcj,ck->tkj", X, W)  # (trials, comps, samples)
    var = np.var(Z, axis=-1, ddof=0)
    # normalize variance per trial (optional but common)
    var = var / (np.sum(var, axis=1, keepdims=True) + 1e-12)
    return np.log(var + 1e-12)


def eval_subject_csp_lda(X_subj, y_subj, n_repeats=120, n_subsets=10, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y_subj)
    if len(classes) != 2:
        raise ValueError("Expected exactly 2 classes for MI (left/right).")

    idx_by_class = {c: np.where(y_subj == c)[0] for c in classes}
    subsets = {}
    for c in classes:
        idx = idx_by_class[c].copy()
        rng.shuffle(idx)
        subsets[c] = np.array_split(idx, n_subsets)

    accs = []
    subset_ids = np.arange(n_subsets)
    for _ in range(n_repeats):
        test_subset_ids = rng.choice(subset_ids, size=3, replace=False)
        test_idx = []
        train_idx = []
        for c in classes:
            for k in subset_ids:
                if k in test_subset_ids:
                    test_idx.append(subsets[c][k])
                else:
                    train_idx.append(subsets[c][k])
        test_idx = np.concatenate(test_idx)
        train_idx = np.concatenate(train_idx)
        X_train, y_train = X_subj[train_idx], y_subj[train_idx]
        X_test, y_test = X_subj[test_idx], y_subj[test_idx]
        W = fit_csp(X_train, y_train, n_components=4)
        F_train = csp_logvar_features(X_train, W)    
        F_test = csp_logvar_features(X_test, W)
        clf = LinearDiscriminantAnalysis()
        clf.fit(F_train, y_train)
        accs.append(clf.score(F_test, y_test))

    accs = np.asarray(accs, dtype=float)
    return float(accs.mean()), float(accs.std(ddof=0))


def evaluate_all_subjects(X, Y, groups, fs=512, n_repeats=120, test_frac=0.30, seed=0):
    subj_ids = np.unique(groups)
    results = {}
    for sid in subj_ids:
        mask = (groups == sid)
        Xs, ys = X[mask], Y[mask]
        mean_acc, std_acc = eval_subject_csp_lda(Xs, ys, fs=fs, n_repeats=n_repeats, test_frac=test_frac, seed=seed)
        results[int(sid)] = (mean_acc, std_acc)
        print(f"s{int(sid):02d}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    all_means = np.array([v[0] for v in results.values()], dtype=float)
    print("\nOverall (mean across subjects): "
          f"{all_means.mean()*100:.2f}% ± {all_means.std()*100:.2f}% (std across subjects)")
    return results


def summarize_eval(acc_per_subject):
    a = np.asarray(list(acc_per_subject), dtype=float)
    if a.size == 0:
        raise ValueError("Need at least one subject accuracy.")
    mean = float(a.mean())
    std = float(a.std(ddof=0))       
    sem = float(std / np.sqrt(a.size))
    ci95_half = float(1.96 * sem)
    return {
        "n_subjects": int(a.size),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half": ci95_half,
    }
