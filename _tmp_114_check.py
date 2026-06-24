import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics import r2_score
import time

RNG = 0
np.set_printoptions(precision=4, suppress=True)

# Friedman #1: a nonlinear regression target with sin/quadratic structure
# y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise, plus noise features.
X, y = make_friedman1(n_samples=4000, n_features=10, noise=1.0, random_state=RNG)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RNG)
print(f"train {X_tr.shape}, test {X_te.shape}")

gamma = 0.1
lam = 1.0  # ridge / KRR regularization

# 1. Exact kernel ridge regression (dense n x n solve, O(n^3)).
t0 = time.perf_counter()
krr = make_pipeline(StandardScaler(),
                    KernelRidge(kernel="rbf", gamma=gamma, alpha=lam))
krr.fit(X_tr, y_tr)
t_krr = time.perf_counter() - t0
r2_krr = r2_score(y_te, krr.predict(X_te))

# 2. Nystroem: data-dependent landmark sketch -> explicit m-dim features, then Ridge.
D = 200
t0 = time.perf_counter()
nys = make_pipeline(StandardScaler(),
                    Nystroem(kernel="rbf", gamma=gamma,
                             n_components=D, random_state=RNG),
                    Ridge(alpha=lam))
nys.fit(X_tr, y_tr)
t_nys = time.perf_counter() - t0
Z_nys = nys[:-1].transform(X_tr)
r2_nys = r2_score(y_te, nys.predict(X_te))

# 3. Random Fourier features (RBFSampler): data-independent map -> Ridge.
t0 = time.perf_counter()
rff = make_pipeline(StandardScaler(),
                    RBFSampler(gamma=gamma, n_components=D, random_state=RNG),
                    Ridge(alpha=lam))
rff.fit(X_tr, y_tr)
t_rff = time.perf_counter() - t0
Z_rff = rff[:-1].transform(X_tr)
r2_rff = r2_score(y_te, rff.predict(X_te))

# Linear ridge baseline (no kernel) to show the nonlinear gain.
lin = make_pipeline(StandardScaler(), Ridge(alpha=lam)).fit(X_tr, y_tr)
r2_lin = r2_score(y_te, lin.predict(X_te))

print(f"\nfeature dim used by approximations: D = {D}")
print(f"Nystroem feature matrix shape:  {Z_nys.shape}")
print(f"RFF feature matrix shape:       {Z_rff.shape}")
print("\nmethod            test R^2     fit time (s)")
print(f"linear ridge      {r2_lin:7.4f}      {0.0:6.3f}")
print(f"exact KRR (rbf)   {r2_krr:7.4f}      {t_krr:6.3f}")
print(f"Nystroem + ridge  {r2_nys:7.4f}      {t_nys:6.3f}")
print(f"RFF + ridge       {r2_rff:7.4f}      {t_rff:6.3f}")
print(f"\napprox kept {100*r2_nys/r2_krr:.1f}% (Nystroem) and "
      f"{100*r2_rff/r2_krr:.1f}% (RFF) of exact-KRR R^2 "
      f"with a {X_tr.shape[0]}x{D} feature map.")
