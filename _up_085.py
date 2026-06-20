import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# Synthetic regression: only 3 of 10 features are truly relevant.
n, p = 120, 10
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[[0, 3, 6]] = [3.0, -2.0, 1.5]
y = X @ true_beta + 0.5 * rng.standard_normal(n)

# Standardize features and center the response.
X = StandardScaler().fit_transform(X)
y = y - y.mean()

# Compute the Lasso path over a grid of penalties.
alphas, coefs, _ = lasso_path(X, y, n_alphas=30)

print("Truly nonzero features:", np.flatnonzero(true_beta).tolist())
print(f"{'lambda':>10} {'n_nonzero':>10}  selected")
for k in range(0, len(alphas), 6):
    beta = coefs[:, k]
    sel = np.flatnonzero(np.abs(beta) > 1e-8).tolist()
    print(f"{alphas[k]:10.4f} {len(sel):10d}  {sel}")

# Show coefficients hitting exactly zero as lambda grows.
strong = coefs[:, 3]   # small penalty
weak = coefs[:, -4]    # large penalty
print("\nFeature 0 coef at small lambda:", round(float(strong[0]), 4))
print("Feature 0 coef at large lambda:", round(float(weak[0]), 4))
print("Exact zeros at large lambda:", int(np.sum(np.abs(weak) < 1e-8)), "of", p)
