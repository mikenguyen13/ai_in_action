import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(0)

# Normal regime: two compact clusters of "healthy" sensor readings.
X_normal, _ = make_blobs(
    n_samples=400, centers=[[-2.0, 0.0], [2.0, 1.5]],
    cluster_std=0.45, random_state=0,
)

# Held-out stream: mostly normal, plus injected anomalies far from both modes.
X_test_normal, _ = make_blobs(
    n_samples=100, centers=[[-2.0, 0.0], [2.0, 1.5]],
    cluster_std=0.45, random_state=1,
)
X_anom = rng.uniform(low=-6.0, high=6.0, size=(20, 2))
X_test = np.vstack([X_test_normal, X_anom])
y_test = np.r_[np.ones(len(X_test_normal)), -np.ones(len(X_anom))]

# Standardize on the training (normal) data only, then reuse the transform.
scaler = StandardScaler().fit(X_normal)
Xn = scaler.transform(X_normal)
Xt = scaler.transform(X_test)

# nu encodes a ~5% expected contamination; gamma="scale" sets RBF bandwidth.
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xn)

train_pred = ocsvm.predict(Xn)
train_outlier_frac = float(np.mean(train_pred == -1))
n_sv = int(ocsvm.support_vectors_.shape[0])

test_pred = ocsvm.predict(Xt)
test_scores = ocsvm.decision_function(Xt)   # > 0 inside boundary, < 0 outside

tp = int(np.sum((test_pred == -1) & (y_test == -1)))
fn = int(np.sum((test_pred == 1) & (y_test == -1)))
fp = int(np.sum((test_pred == -1) & (y_test == 1)))

np.set_printoptions(precision=3, suppress=True)
print(f"training points:            {Xn.shape[0]}  (features: {Xn.shape[1]})")
print(f"support vectors:            {n_sv}  ({n_sv / Xn.shape[0]:.1%} of training data)")
print(f"flagged-outlier fraction:   {train_outlier_frac:.1%}  (nu = 0.05 upper bound)")
print()
print(f"test points:                {Xt.shape[0]}  ({int((y_test==-1).sum())} true anomalies)")
print(f"anomalies caught:           {tp}/{tp + fn}")
print(f"false alarms on normal:     {fp}/{int((y_test==1).sum())}")
print()
print("decision_function on 3 normal and 3 anomalous test points:")
print("  normal   :", test_scores[:3])
print("  anomalous:", test_scores[-3:])
