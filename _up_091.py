import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# Part 1: SymPy symbolic check of the posterior proportionality, the
# conditional-independence log factorisation, and Laplace smoothing as
# the Dirichlet posterior mean.
# ----------------------------------------------------------------------
Px_given_y, Py, Px = sp.symbols("P_x_given_y P_y P_x", positive=True)
posterior = Px_given_y * Py / Px
# The evidence P(x) does not depend on y, so the argmax uses P(x|y) P(y).
unnormalized = sp.simplify(posterior * Px)
print("posterior * P(x) =", unnormalized, " (= P(x|y) P(y))")

# Conditional independence: log of the joint factorises into a sum.
p1, p2, p3, pri = sp.symbols("p1 p2 p3 pi_y", positive=True)
log_joint = sp.expand_log(sp.log(pri * p1 * p2 * p3), force=True)
print("log P(y, x) =", log_joint)

# Laplace smoothing equals the Dirichlet(alpha) posterior mean; the
# MLE is recovered as alpha -> 0.
N_wy, total, alpha_s, V = sp.symbols("N_wy N_total alpha V", positive=True)
smoothed = (N_wy + alpha_s) / (total + alpha_s * V)
print("smoothed theta =", smoothed, " limit(alpha->0) =",
      sp.limit(smoothed, alpha_s, 0))

# ----------------------------------------------------------------------
# Part 2: Gaussian NB decision boundary on a synthetic 2D, 3-class set.
# ----------------------------------------------------------------------
X, y = make_blobs(n_samples=600, centers=3, cluster_std=1.8, random_state=42)
gnb = GaussianNB().fit(X, y)

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300),
)
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
print(f"\nGaussian NB training accuracy: "
      f"{accuracy_score(y, gnb.predict(X)):.3f}")

# ----------------------------------------------------------------------
# Part 3: Self-contained text classification. We generate a synthetic
# bag-of-words corpus: each of three "topics" has its own word
# distribution, and documents are multinomial draws from their topic.
# ----------------------------------------------------------------------
vocab_size, n_topics, docs_per_topic = 60, 3, 300
# Each topic favours an overlapping block of the vocabulary, so the
# topics are genuinely confusable and short documents are noisy.
topic_word = np.full((n_topics, vocab_size), 1.0)
for t in range(n_topics):
    block = slice(t * 16, t * 16 + 22)
    topic_word[t, block] += 1.5
topic_word /= topic_word.sum(axis=1, keepdims=True)

X_txt, y_txt = [], []
for t in range(n_topics):
    for _ in range(docs_per_topic):
        length = rng.integers(8, 18)
        X_txt.append(rng.multinomial(length, topic_word[t]))
        y_txt.append(t)
X_txt = np.array(X_txt)
y_txt = np.array(y_txt)

Xtr, Xte, ytr, yte = train_test_split(
    X_txt, y_txt, test_size=0.3, random_state=42, stratify=y_txt)

rows = []
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
    clf = MultinomialNB(alpha=alpha).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rows.append({
        "alpha": alpha,
        "test_accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro"),
    })
results = pd.DataFrame(rows)
print("\nMultinomial NB: Laplace smoothing sweep (synthetic 3-topic corpus)")
print(results.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
best = results.loc[results["test_accuracy"].idxmax()]
print(f"\nBest alpha = {best['alpha']} -> accuracy {best['test_accuracy']:.4f}")

# ----------------------------------------------------------------------
# Figures: decision boundary + smoothing curve.
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].contourf(xx, yy, Z, alpha=0.25, cmap="viridis", levels=2)
ax[0].scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="viridis",
              edgecolor="k", linewidth=0.2)
ax[0].set_title("Gaussian NB decision regions")
ax[0].set_xlabel("feature 1"); ax[0].set_ylabel("feature 2")

ax[1].plot(results["alpha"], results["test_accuracy"], marker="o", label="accuracy")
ax[1].plot(results["alpha"], results["macro_f1"], marker="s", label="macro F1")
ax[1].set_xscale("log")
ax[1].set_title("Multinomial NB vs. Laplace alpha")
ax[1].set_xlabel("smoothing alpha (log)"); ax[1].set_ylabel("score")
ax[1].legend()
fig.tight_layout()
plt.show()
print("\nAll checks passed.")
