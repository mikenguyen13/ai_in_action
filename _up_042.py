import numpy as np
from scipy.optimize import minimize

np.random.seed(42)

# Problem: minimize 1/2 x^T Q x + c^T x s.t. Ax = b and x >= 0 (box-ish)
# We solve two ways: (1) closed-form equality-constrained KKT system,
# (2) scipy.optimize.minimize with inequality + equality constraints.

# --- Part 1: equality-constrained quadratic via the KKT linear system ---
Q = np.array([[2.0, 0.0], [0.0, 2.0]])
c = np.array([0.0, 0.0])
A = np.array([[1.0, 1.0]])
b = np.array([1.0])

n = Q.shape[0]
p = A.shape[0]
KKT = np.block([[Q, A.T], [A, np.zeros((p, p))]])
rhs = np.concatenate([-c, b])
sol = np.linalg.solve(KKT, rhs)
x_eq, nu = sol[:n], sol[n:]
print("Equality-constrained quadratic (KKT linear solve):")
print(f"  x* = {x_eq}")
print(f"  nu* = {nu}")
print(f"  objective = {0.5 * x_eq @ Q @ x_eq + c @ x_eq:.6f}")

# --- Part 2: inequality-constrained quadratic via scipy ---
# minimize 1/2||x - p0||^2 s.t. x1 + x2 <= 1, x >= 0  (projection onto a simplex-like region)
p0 = np.array([0.9, 0.8])

def obj(x):
    return 0.5 * np.sum((x - p0) ** 2)

def obj_grad(x):
    return x - p0

cons = [
    {"type": "ineq", "fun": lambda x: 1.0 - (x[0] + x[1])},  # x1+x2 <= 1
]
bnds = [(0.0, None), (0.0, None)]
res = minimize(obj, x0=np.array([0.0, 0.0]), jac=obj_grad,
               bounds=bnds, constraints=cons, method="SLSQP")
print("\nInequality-constrained QP (scipy SLSQP):")
print(f"  success = {res.success}")
print(f"  x* = {res.x}")
print(f"  active sum constraint residual = {1.0 - (res.x[0] + res.x[1]):.6f}")
print(f"  objective = {res.fun:.6f}")

# Sanity checks
assert res.success
assert abs(x_eq[0] - 0.5) < 1e-9 and abs(x_eq[1] - 0.5) < 1e-9
assert res.x[0] + res.x[1] <= 1.0 + 1e-6
print("\nAll checks passed.")
