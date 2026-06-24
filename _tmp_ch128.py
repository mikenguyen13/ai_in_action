import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score

rng = np.random.RandomState(0)

# Variable-density data: a tight blob, a diffuse blob, a second tight blob,
# plus uniform background noise that neither true cluster should absorb.
centers = [[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]]
stds = [0.35, 1.4, 0.5]
X_blobs, y_true = make_blobs(
    n_samples=[150, 150, 150], centers=centers,
    cluster_std=stds, random_state=0,
)
noise = rng.uniform(low=-4, high=11, size=(60, 2))
X = np.vstack([X_blobs, noise])
y = np.concatenate([y_true, np.full(60, -1)])  # -1 marks true noise
print(f"dataset: {X.shape[0]} points, {X.shape[1]} features, "
      f"3 clusters at stds {stds} + 60 noise points")

# --- HDBSCAN: automatic stability-based extraction ---
hdb = HDBSCAN(min_cluster_size=15, min_samples=5,
              cluster_selection_method="eom")
hdb_labels = hdb.fit_predict(X)
n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
hdb_noise = int(np.sum(hdb_labels == -1))
print("\nHDBSCAN")
print(f"  clusters found:      {n_hdb}")
print(f"  points labeled noise:{hdb_noise:4d}")
print(f"  ARI vs truth:        {adjusted_rand_score(y, hdb_labels):.3f}")
print(f"  outlier score range: [{hdb.outlier_scores_.min():.3f}, "
      f"{hdb.outlier_scores_.max():.3f}]")

# --- OPTICS: ordering + xi extraction at variable density ---
opt = OPTICS(min_samples=5, xi=0.05, min_cluster_size=15)
opt_labels = opt.fit_predict(X)
n_opt = len(set(opt_labels)) - (1 if -1 in opt_labels else 0)
opt_noise = int(np.sum(opt_labels == -1))
reach = opt.reachability_[opt.ordering_]
reach_finite = reach[np.isfinite(reach)]
print("\nOPTICS (xi method)")
print(f"  clusters found:      {n_opt}")
print(f"  points labeled noise:{opt_noise:4d}")
print(f"  ARI vs truth:        {adjusted_rand_score(y, opt_labels):.3f}")
print(f"  reachability valleys vs peaks: min={reach_finite.min():.3f}, "
      f"median={np.median(reach_finite):.3f}, max={reach_finite.max():.3f}")
