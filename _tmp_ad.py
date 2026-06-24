import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)
np.random.seed(0)

# A small, self-contained regression problem.
X_np, y_np, true_w = make_regression(
    n_samples=200, n_features=5, noise=8.0,
    coef=True, random_state=0,
)
X_np = StandardScaler().fit_transform(X_np)
y_np = (y_np - y_np.mean()) / y_np.std()

X = torch.tensor(X_np, dtype=torch.float64)
y = torch.tensor(y_np, dtype=torch.float64)

# ---- 1. torch.autograd: reverse-mode gradient of a ridge loss ----
w = torch.zeros(5, dtype=torch.float64, requires_grad=True)
b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
lam = 0.1

def loss_fn(w, b):
    pred = X @ w + b
    mse = ((pred - y) ** 2).mean()
    return mse + lam * (w @ w)

# One backward pass yields gradients wrt all parameters at once.
L = loss_fn(w, b)
L.backward()
print("initial loss:        ", round(L.item(), 6))
print("grad wrt w (reverse):", np.round(w.grad.numpy(), 4))
print("grad wrt b (reverse):", np.round(b.grad.numpy(), 4))

# Fit by gradient descent driven entirely by autograd.
opt = torch.optim.LBFGS([w, b], max_iter=50)
def closure():
    opt.zero_grad()
    val = loss_fn(w, b)
    val.backward()
    return val
opt.step(closure)
with torch.no_grad():
    final = loss_fn(w, b).item()
print("fitted loss:         ", round(final, 6))
print("fitted w:            ", np.round(w.detach().numpy(), 4))

# ---- 2. Forward mode for free: a directional (Jacobian-vector) derivative ----
v = torch.ones(5, dtype=torch.float64)  # seed direction
_, jvp = torch.autograd.functional.jvp(
    lambda ww: loss_fn(ww, b.detach()), w.detach(), v,
)
print("directional deriv (fwd-mode JVP):", round(jvp.item(), 6))

# ---- 3. A tiny from-scratch reverse-mode engine on a scalar graph ----
class Var:
    def __init__(self, value, parents=(), local=()):
        self.value = float(value)
        self.parents = parents      # upstream Vars
        self.local = local          # d(self)/d(parent) for each parent
        self.grad = 0.0
    def __add__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        return Var(self.value + o.value, (self, o), (1.0, 1.0))
    def __mul__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        return Var(self.value * o.value, (self, o), (o.value, self.value))

def sin(x):
    return Var(np.sin(x.value), (x,), (np.cos(x.value),))

def backward(node):
    topo, seen = [], set()
    def build(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n.parents:
            build(p)
        topo.append(n)
    build(node)
    node.grad = 1.0
    for n in reversed(topo):
        for p, d in zip(n.parents, n.local):
            p.grad += n.grad * d   # accumulate adjoints

# y = x1*x2 + sin(x1), the chapter's running example, at (3, 4).
x1, x2 = Var(3.0), Var(4.0)
y_out = x1 * x2 + sin(x1)
backward(y_out)
print("from-scratch y:      ", round(y_out.value, 6))
print("from-scratch dy/dx1: ", round(x1.grad, 6),
      "(closed form x2+cos x1 =", round(4 + np.cos(3.0), 6), ")")
print("from-scratch dy/dx2: ", round(x2.grad, 6), "(closed form x1 = 3.0)")
