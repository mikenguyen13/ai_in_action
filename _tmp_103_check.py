import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score

rng = np.random.RandomState(0)

# --- Part 1: Extra-Trees classifier on a synthetic tabular problem ---
X, y = make_classification(
    n_samples=600, n_features=20, n_informative=6, n_redundant=4,
    n_classes=3, class_sep=1.2, random_state=0,
)

et = ExtraTreesClassifier(
    n_estimators=300, max_features="sqrt", bootstrap=False, random_state=0, n_jobs=-1,
)
cv = cross_val_score(et, X, y, cv=5, scoring="accuracy")
et.fit(X, y)

print("Extra-Trees on make_classification (600 x 20, 3 classes)")
print(f"  5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}")
top = np.argsort(et.feature_importances_)[::-1][:5]
print(f"  top-5 feature indices : {top.tolist()}")
print(f"  their importances     : {np.round(et.feature_importances_[top], 4).tolist()}")

# --- Part 2: Isolation Forest for unsupervised anomaly detection ---
n_inliers, n_outliers = 480, 20
inliers = rng.normal(loc=0.0, scale=1.0, size=(n_inliers, 2))
outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 2))
Xa = np.vstack([inliers, outliers])
is_outlier = np.r_[np.zeros(n_inliers), np.ones(n_outliers)]

iso = IsolationForest(
    n_estimators=200, max_samples=256, contamination=0.04, random_state=0,
)
iso.fit(Xa)
pred = iso.predict(Xa)                 # +1 inlier, -1 outlier
scores = -iso.score_samples(Xa)        # higher => more anomalous
flagged = (pred == -1)

ap = average_precision_score(is_outlier, scores)
recall = flagged[is_outlier == 1].mean()
fp_rate = flagged[is_outlier == 0].mean()

print("\nIsolation Forest on 480 inliers + 20 injected outliers")
print(f"  data shape            : {Xa.shape}")
print(f"  points flagged anomaly: {int(flagged.sum())}")
print(f"  recall on true outliers: {recall:.2f}")
print(f"  false-positive rate    : {fp_rate:.3f}")
print(f"  average precision (AP) : {ap:.3f}")
print(f"  score range [min, max] : [{scores.min():.3f}, {scores.max():.3f}]")
