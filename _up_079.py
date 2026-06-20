import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, cross_validate,
)

RNG = 0
np.random.seed(RNG)

# A modest, noisy dataset so that selection optimism is visible.
X, y = make_classification(
    n_samples=200, n_features=20, n_informative=5, n_redundant=2,
    class_sep=0.8, flip_y=0.10, random_state=RNG,
)

# --- Part 1: variance of the CV estimate as k varies -----------------------
print("k-fold accuracy estimate vs k (SVC, RBF kernel)")
print(f"{'k':>4} {'mean_acc':>10} {'std_across_folds':>18}")
for k in (2, 3, 5, 10, 20):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RNG)
    scores = cross_val_score(SVC(kernel="rbf", C=1.0, gamma="scale"), X, y, cv=skf)
    print(f"{k:>4} {scores.mean():>10.4f} {scores.std():>18.4f}")

# --- Part 2: optimism of non-nested selection vs nested CV -----------------
param_grid = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [1e-3, 1e-2, 1e-1, 1]}

inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG + 1)

# Non-nested: tune and report on the SAME folds (the optimistic protocol).
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=inner)
grid.fit(X, y)
non_nested = grid.best_score_

# Nested: outer folds never see the tuning, so the estimate is honest.
nested = cross_validate(
    GridSearchCV(SVC(kernel="rbf"), param_grid, cv=inner),
    X, y, cv=outer,
)["test_score"]

print("\nNested vs non-nested cross-validation")
print(f"non-nested (tuned and reported on same folds): {non_nested:.4f}")
print(f"nested     (honest outer estimate)           : {nested.mean():.4f}"
      f"  +/- {nested.std():.4f}")
print(f"optimism of non-nested selection             : {non_nested - nested.mean():+.4f}")
