from __future__ import annotations
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
    scaler = StandardScaler().fit(Xtr)
    Xtr_z = scaler.transform(Xtr)
    Xv_z  = scaler.transform(Xv)
    Xt_z  = scaler.transform(Xt)

    km = KMeans(n_clusters=int(k_clust), n_init=8, random_state=int(seed))
    km.fit(Xtr_z)

    cv = km.predict(Xv_z)
    ct = km.predict(Xt_z)

    mats = [base_val] + [P for (_, P) in experts_val]
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
    base_test,
    experts_test,         # list of (name, P_test)
    cluster_nll,
    ct_test,
    budget=0.05,
    margin_min_nll=0.0,
    delta_conf=0.0,
    return_info: bool = False
):
    base_test = np.asarray(base_test, dtype=float)
    N, C = base_test.shape
    k_clust, E = cluster_nll.shape

    mats_test = [base_test] + [np.asarray(P, dtype=float) for (_, P) in experts_test]
    if len(mats_test) != E:
        raise ValueError(f"Mismatch: cluster_nll expects E={E}, got {len(mats_test)}")

    ct_test = np.asarray(ct_test, dtype=int)
    ct_test = np.clip(ct_test, 0, k_clust - 1)

    base_idx = 0
    P_base = mats_test[base_idx].copy()

    best_e = cluster_nll[ct_test].argmin(axis=1)
    best_nll = cluster_nll[ct_test, best_e]
    base_nll = cluster_nll[ct_test, base_idx]
    delta_improve = base_nll - best_nll

    base_conf = P_base.max(axis=1)
    confs_all = [M.max(axis=1) for M in mats_test]
    best_conf = np.array([confs_all[e][i] for i, e in enumerate(best_e)], dtype=float)
    conf_margin = best_conf - base_conf

    cand = (best_e != base_idx) & (delta_improve > float(margin_min_nll))
    if float(delta_conf) > 0.0:
        cand &= (conf_margin > float(delta_conf))

    cand_idx = np.where(cand)[0]
    if cand_idx.size == 0:
        info = {"flips": 0, "helped": 0, "hurt": 0}
        return (P_base, info) if return_info else P_base

    priority = delta_improve * (1.0 - base_conf)
    order = np.argsort(-priority[cand_idx])

    budget_n = int(np.floor(float(budget) * N))
    budget_n = max(0, min(budget_n, cand_idx.size))
    if budget_n == 0:
        info = {"flips": 0, "helped": 0, "hurt": 0}
        return (P_base, info) if return_info else P_base

    chosen = cand_idx[order[:budget_n]]

    P_fused = P_base.copy()
    for e in range(1, E):
        take = chosen[best_e[chosen] == e]
        if take.size:
            P_fused[take] = mats_test[e][take]

    info = {"flips": int(chosen.size), "helped": None, "hurt": None}
    return (P_fused, info) if return_info else P_fused



