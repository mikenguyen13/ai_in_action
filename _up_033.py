import numpy as np
import scipy.stats as st

rng = np.random.default_rng(42)

# True population: exponential with rate lambda=1, so mean mu = 1, variance = 1
true_mu = 1.0
n = 30          # sample size
reps = 20000    # number of simulated samples

# --- Sampling distribution, bias, and variance of two estimators of the mean ---
# Estimator A: sample mean (unbiased)
# Estimator B: shrinkage estimator mu_hat * n / (n + k) (biased, lower variance)
k = 1.0

means = np.empty(reps)
shrunk = np.empty(reps)
for i in range(reps):
    x = rng.exponential(scale=1.0 / true_mu, size=n)  # scale = 1/lambda = mean
    m = x.mean()
    means[i] = m
    shrunk[i] = m * n / (n + k)

def report(name, est, target):
    bias = est.mean() - target
    var = est.var()
    mse = ((est - target) ** 2).mean()
    print(f"{name:18s} bias={bias:+.4f}  var={var:.4f}  mse={mse:.4f}  (bias^2+var={bias**2+var:.4f})")

print("Sampling distribution over", reps, "samples of size", n)
report("sample mean", means, true_mu)
report("shrinkage", shrunk, true_mu)

# --- Bootstrap confidence interval for the mean from ONE sample ---
sample = rng.exponential(scale=1.0 / true_mu, size=n)
theta_hat = sample.mean()
B = 10000
boot = np.empty(B)
for b in range(B):
    resample = rng.choice(sample, size=n, replace=True)
    boot[b] = resample.mean()

se_boot = boot.std(ddof=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nOne sample: theta_hat={theta_hat:.4f}")
print(f"Bootstrap SE={se_boot:.4f}")
print(f"95% percentile bootstrap CI=[{lo:.4f}, {hi:.4f}]  (true mu={true_mu})")

# Normal-theory CI for comparison
se_normal = sample.std(ddof=1) / np.sqrt(n)
t = st.t.ppf(0.975, df=n - 1)
print(f"95% t-interval=[{theta_hat - t*se_normal:.4f}, {theta_hat + t*se_normal:.4f}]")
