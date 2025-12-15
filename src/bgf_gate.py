import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.metrics import nll


def fit_cluster_reliability(
    Xtr, Xv, Xt, yv,
    base_val,
    experts_val,          # list of (name, P_val)
    k_clust=12,
    seed=42,
    min_cluster_val=3
):
    """
    Build a DMAP-space reliability map:
    - Standardize DMAP embeddings using TRAIN stats
    - Fit KMeans on TRAIN
    - Assign clusters for VAL and TEST
    - Estimate expected NLL per cluster for each expert using VAL labels

    Returns:
      cluster_nll: (K, E) where E = 1 + len(experts_val) [base first]
      ct_test:     (N_test,) cluster assignment for each test sample
    """
    # Standardize using TRAIN distribution
    scaler = StandardScaler().fit(Xtr)
    Xtr_z = scaler.transform(Xtr)
    Xv_z  = scaler.transform(Xv)
    Xt_z  = scaler.transform(Xt)

    # Cluster in standardized DMAP space (fit on TRAIN)
    km = KMeans(n_clusters=int(k_clust), n_init=8, random_state=int(seed))
    km.fit(Xtr_z)

    cv = km.predict(Xv_z)   # VAL clusters
    ct = km.predict(Xt_z)   # TEST clusters

    # mats: base + experts (VAL probs)
    mats = [base_val] + [P for (_, P) in experts_val]
    E = len(mats)

    # Global fallback NLL if a cluster has too few VAL samples
    global_nll = np.array([nll(mats[e], yv) for e in range(E)], dtype=float)

    cluster_nll = np.zeros((int(k_clust), E), dtype=float)

    for k in range(int(k_clust)):
        idx = np.where(cv == k)[0]
        if idx.size >= int(min_cluster_val):
            for e in range(E):
                cluster_nll[k, e] = nll(mats[e][idx], yv[idx])
        else:
            cluster_nll[k, :] = global_nll

    return cluster_nll, ct


def budgeted_gate_strict(
    base_test,            # (N,C)
    experts_test,         # list of (name, P_test)
    cluster_nll,          # (K,E) from fit_cluster_reliability
    ct_test,              # (N,) cluster ids for test
    budget=0.05,
    margin_min_nll=0.0,
    delta_conf=0.0
):
    """
    Budgeted reliability gate (strict-lite):
    - Start with base predictions
    - For each sample, choose expert with lowest expected cluster NLL
    - Consider flipping only if:
        * chosen expert is not base
        * expected NLL improvement > margin_min_nll
        * (optional) confidence improvement > delta_conf
    - Apply top budget fraction by priority score

    Returns:
      fused_test probabilities (N,C)
    """
    base_test = np.asarray(base_test, dtype=float)
    N, C = base_test.shape
    k_clust, E = cluster_nll.shape

    # Arrange TEST probs list in same order as cluster_nll columns:
    # col0=base, col1..=experts in the same order passed to fit_cluster_reliability
    mats_test = [base_test] + [P for (_, P) in experts_test]
    assert len(mats_test) == E, "Mismatch: cluster_nll expects E=1+num_experts"

    base_idx = 0
    P_base = mats_test[base_idx].copy()

    # Best expert index per sample (by expected cluster NLL)
    ct_test = np.asarray(ct_test, dtype=int)
    ct_test = np.clip(ct_test, 0, k_clust - 1)

    best_e = cluster_nll[ct_test].argmin(axis=1)      # (N,)
    best_nll = cluster_nll[ct_test, best_e]
    base_nll = cluster_nll[ct_test, base_idx]
    delta_improve = base_nll - best_nll               # positive => expected improvement

    # Confidence checks (optional)
    base_pred = P_base.argmax(1)
    base_conf = P_base.max(1)

    preds_all = [M.argmax(1) for M in mats_test]
    confs_all = [M.max(1) for M in mats_test]

    best_pred = np.array([preds_all[e][i] for i, e in enumerate(best_e)], dtype=int)
    best_conf = np.array([confs_all[e][i] for i, e in enumerate(best_e)], dtype=float)

    conf_margin = best_conf - base_conf

    # Candidate mask
    cand = (best_e != base_idx) & (delta_improve > float(margin_min_nll))
    if float(delta_conf) > 0.0:
        cand = cand & (conf_margin > float(delta_conf))

    cand_idx = np.where(cand)[0]
    if cand_idx.size == 0:
        return P_base

    # Priority: larger expected gain, larger base uncertainty
    priority = delta_improve * (1.0 - base_conf)
    cand_scores = priority[cand_idx]

    # Pick top budget fraction
    budget_n = int(np.floor(float(budget) * N))
    budget_n = max(0, min(budget_n, cand_idx.size))
    if budget_n == 0:
        return P_base

    order = np.argsort(-cand_scores)
    chosen = cand_idx[order[:budget_n]]

    # Fuse: replace chosen samples with their best expert probs
    P_fused = P_base.copy()
    for e in range(1, E):
        take = chosen[best_e[chosen] == e]
        if take.size:
            P_fused[take] = mats_test[e][take]

    return P_fused
