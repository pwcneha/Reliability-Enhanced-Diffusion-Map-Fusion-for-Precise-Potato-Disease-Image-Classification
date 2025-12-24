import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.metrics import nll


def fit_cluster_reliability(
    Xtr, Xv, Xt, yv,
    base_val,
    experts_val,
    k_clust=12,
    seed=42,
    min_cluster_val=3
):
    scaler = StandardScaler().fit(Xtr)
    Xtr_z = scaler.transform(Xtr)
    Xv_z  = scaler.transform(Xv)
    Xt_z  = scaler.transform(Xt)

    km = KMeans(n_clusters=k_clust, n_init=8, random_state=seed)
    km.fit(Xtr_z)

    cv = km.predict(Xv_z)
    ct = km.predict(Xt_z)

    mats = [base_val] + [P for (_, P) in experts_val]
    E = len(mats)

    global_nll = np.array([nll(mats[e], yv) for e in range(E)])
    cluster_nll = np.zeros((k_clust, E))

    for k in range(k_clust):
        idx = np.where(cv == k)[0]
        if len(idx) >= min_cluster_val:
            for e in range(E):
                cluster_nll[k, e] = nll(mats[e][idx], yv[idx])
        else:
            cluster_nll[k] = global_nll

    return cluster_nll, ct


def budgeted_gate_strict(
    base_test,
    experts_test,
    cluster_nll,
    ct_test,
    budget=0.05,
    margin_min_nll=0.0,
    delta_conf=0.0
):
    base_test = np.asarray(base_test)
    mats = [base_test] + [P for (_, P) in experts_test]

    N = len(base_test)
    base_idx = 0

    best_e = cluster_nll[ct_test].argmin(1)
    delta_improve = cluster_nll[ct_test, base_idx] - cluster_nll[ct_test, best_e]

    base_conf = base_test.max(1)
    best_conf = np.array([mats[e][i].max() for i, e in enumerate(best_e)])

    cand = (best_e != base_idx) & (delta_improve > margin_min_nll)
    if delta_conf > 0:
        cand &= (best_conf - base_conf > delta_conf)

    cand_idx = np.where(cand)[0]
    budget_n = min(int(budget * N), len(cand_idx))

    if budget_n == 0:
        return base_test

    priority = delta_improve[cand_idx] * (1 - base_conf[cand_idx])
    chosen = cand_idx[np.argsort(-priority)[:budget_n]]

    fused = base_test.copy()
    for e in range(1, len(mats)):
        take = chosen[best_e[chosen] == e]
        if len(take):
            fused[take] = mats[e][take]

    return fused
