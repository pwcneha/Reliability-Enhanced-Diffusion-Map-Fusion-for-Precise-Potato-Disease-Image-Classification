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
    Build DMAP-space reliability map:
      - Standardize DMAP embeddings using TRAIN stats
      - Fit KMeans on TRAIN
      - Assign clusters for VAL and TEST
      - Estimate expected NLL per cluster for each expert using VAL labels

    Returns:
      cluster_nll: (K, E) where E = 1 + len(experts_val) [base first]
      ct_test:     (N_test,) cluster assignment for each test sample
    """
    scaler = StandardScaler().fit(Xtr)
    Xtr_z = scaler.transform(Xtr)
    Xv_z  = scaler.transform(Xv)
    Xt_z  = scaler.transform(Xt)

    km = KMeans(n_clusters=int(k_clust), n_init=8, random_state=int(seed))
    km.fit(Xtr_z)

    cv = km.predict(Xv_z)
    ct = km.predict(Xt_z)

    mats = [base_val] + [P for (_, P) in experts_val]  # VAL probs
    E = len(mats)

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
    experts_test,         # list of (name, P_test) (order must match experts_val order used in fit_cluster_reliability)
    cluster_nll,          # (K,E) base is col0, experts are col1.. in same order
    ct_test,              # (N,)
    budget=0.05,
    margin_min_nll=0.005,
    delta_conf=0.08
):
    """
    Strict budget-gated override:
      - Choose best expert by expected cluster NLL
      - Candidate flip only if:
          * best expert != base
          * predicted class changes (label-change)
          * expected NLL improvement > margin_min_nll
          * best expert confidence - base confidence > delta_conf
      - Select top budget fraction by priority:
          priority = delta_improve * max(conf_margin,0) * (1 - base_conf)

    Returns:
      fused_test (N,C)
      gate_info  (dict) for report/debug
    """
    base_test = np.asarray(base_test, dtype=float)
    N, C = base_test.shape
    k_clust, E = cluster_nll.shape

    mats_test = [base_test] + [P for (_, P) in experts_test]
    if len(mats_test) != E:
        raise ValueError(f"Mismatch: cluster_nll expects E={E} probs (base+experts), got {len(mats_test)}")

    ct_test = np.asarray(ct_test, dtype=int)
    ct_test = np.clip(ct_test, 0, k_clust - 1)

    base_idx = 0
    P_base = mats_test[base_idx].copy()

    # best expert index per sample by expected cluster NLL
    best_e = cluster_nll[ct_test].argmin(axis=1)      # (N,)
    best_nll = cluster_nll[ct_test, best_e]
    base_nll = cluster_nll[ct_test, base_idx]
    delta_improve = base_nll - best_nll               # positive => expected improvement

    base_pred = P_base.argmax(1)
    base_conf = P_base.max(1)

    preds_all = [M.argmax(1) for M in mats_test]
    confs_all = [M.max(1) for M in mats_test]

    best_pred = np.array([preds_all[e][i] for i, e in enumerate(best_e)], dtype=int)
    best_conf = np.array([confs_all[e][i] for i, e in enumerate(best_e)], dtype=float)

    conf_margin = best_conf - base_conf
    class_change = (best_pred != base_pred)

    cand = (
        (best_e != base_idx)
        & class_change
        & (delta_improve > float(margin_min_nll))
        & (conf_margin > float(delta_conf))
    )

    cand_idx = np.where(cand)[0]
    budget_n = int(np.floor(float(budget) * N))

    # priority score
    priority = delta_improve * np.maximum(conf_margin, 0.0) * (1.0 - base_conf)

    chosen = np.array([], dtype=int)
    if cand_idx.size > 0 and budget_n > 0:
        budget_n = min(budget_n, cand_idx.size)
        order = np.argsort(-priority[cand_idx])  # descending
        chosen = cand_idx[order[:budget_n]]

    # apply fusion
    P_fused = P_base.copy()
    for e in range(1, E):
        take = chosen[best_e[chosen] == e]
        if take.size:
            P_fused[take] = mats_test[e][take]

    # gate debug info
    expert_names = ["base"] + [n for (n, _) in experts_test]
    chosen_counts = {expert_names[e]: int(np.sum(best_e[chosen] == e)) for e in range(E)} if chosen.size else {}

    gate_info = {
        "N": int(N),
        "C": int(C),
        "K": int(k_clust),
        "E": int(E),
        "budget_n": int(budget_n),
        "candidates": int(cand_idx.size),
        "chosen": int(chosen.size),
        "chosen_counts": chosen_counts,
    }

    return P_fused, gate_info

