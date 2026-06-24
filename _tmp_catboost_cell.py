import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

rng = np.random.default_rng(42)
n = 4000

# High-cardinality categorical: 600 "merchants", each with a latent risk.
n_merchants = 600
merchant_risk = rng.normal(0.0, 1.3, size=n_merchants)
merchant = rng.integers(0, n_merchants, size=n)

# Low-cardinality categoricals.
countries = np.array(["US", "CA", "GB", "DE", "FR", "IN", "BR"])
country = countries[rng.integers(0, len(countries), size=n)]
device = np.array(["ios", "android", "web"])[rng.integers(0, 3, size=n)]

# Two numeric features.
amount = rng.gamma(2.0, 50.0, size=n)
account_age = rng.uniform(0, 2000, size=n)

# Latent logit: merchant risk dominates, with a country/device interaction.
country_eff = {"US": -0.2, "CA": -0.1, "GB": 0.0, "DE": 0.1,
               "FR": 0.1, "IN": 0.6, "BR": 0.7}
ce = np.array([country_eff[c] for c in country])
device_eff = np.where(device == "web", 0.5, -0.1)
logit = (merchant_risk[merchant]
         + ce
         + ce * (device == "web") * 1.2          # interaction
         + 0.0008 * (amount - 100)
         - 0.0006 * account_age
         + rng.normal(0, 0.5, size=n))
prob = 1.0 / (1.0 + np.exp(-logit))
y = (rng.uniform(size=n) < prob).astype(int)

# Assemble feature matrix as object array (mixed types), CatBoost handles raw strings.
import pandas as pd
X = pd.DataFrame({
    "merchant": merchant.astype(str),
    "country": country,
    "device": device,
    "amount": amount,
    "account_age": account_age,
})
cat_features = ["merchant", "country", "device"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
test_pool = Pool(X_te, y_te, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    boosting_type="Ordered",   # ordered boosting, the chapter's subject
    random_seed=42,
    verbose=False,
)
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

proba = model.predict_proba(test_pool)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"positive rate (train): {y_tr.mean():.3f}")
print(f"best iteration:        {model.get_best_iteration()}")
print(f"test AUC:              {roc_auc_score(y_te, proba):.4f}")
print(f"test accuracy:         {accuracy_score(y_te, pred):.4f}")

imp = model.get_feature_importance(train_pool)
order = np.argsort(imp)[::-1]
print("feature importances (PredictionValuesChange):")
for i in order:
    print(f"  {X.columns[i]:<12} {imp[i]:6.2f}")
