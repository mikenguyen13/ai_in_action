import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

rng = 42

# A self-contained tabular problem: 6 informative features, 4 redundant,
# 10 pure noise. This setup is where random forests shine and where the
# two importance measures visibly disagree.
X, y = make_classification(
    n_samples=1200, n_features=20, n_informative=6, n_redundant=4,
    n_repeated=0, n_classes=2, flip_y=0.03, class_sep=1.1, random_state=rng,
)
feat = [f"x{j:02d}" for j in range(X.shape[1])]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=rng
)

forest = RandomForestClassifier(
    n_estimators=400, max_features="sqrt", min_samples_leaf=1,
    oob_score=True, n_jobs=-1, random_state=rng,
)
forest.fit(X_tr, y_tr)

proba = forest.predict_proba(X_te)[:, 1]
pred = forest.predict(X_te)

print(f"train / test shapes:    {X_tr.shape} / {X_te.shape}")
print(f"OOB accuracy:           {forest.oob_score_:.4f}")
print(f"test accuracy:          {accuracy_score(y_te, pred):.4f}")
print(f"test ROC AUC:           {roc_auc_score(y_te, proba):.4f}")

# Mean decrease in impurity (free, but biased toward high-cardinality features).
mdi = forest.feature_importances_
top_mdi = np.argsort(mdi)[::-1][:5]
print("\ntop-5 features by MDI (impurity):")
for j in top_mdi:
    print(f"  {feat[j]}  {mdi[j]:.4f}")

# Permutation importance on held-out data (slower, but honest).
perm = permutation_importance(
    forest, X_te, y_te, n_repeats=10, random_state=rng, n_jobs=-1
)
top_perm = np.argsort(perm.importances_mean)[::-1][:5]
print("\ntop-5 features by permutation importance (test set):")
for j in top_perm:
    print(f"  {feat[j]}  {perm.importances_mean[j]:.4f} +/- {perm.importances_std[j]:.4f}")
