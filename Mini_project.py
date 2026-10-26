#!/usr/bin/env python
# coding: utf-8

# Data Exploration

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests


# In[2]:


path = "spy_data.csv"

df = pd.read_csv(path, index_col=0, parse_dates=True)
df.index.name = "date"
df = df[["log_ret", "rv5", "bv"]]

print(df.head())
print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDate range:", df.index.min(), "to", df.index.max())
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate dates:", df.index.duplicated().sum())

assert not df.index.hasnans
assert not df.index.duplicated().any()
assert np.isfinite(df.to_numpy(dtype=float)).all()
assert (df[["rv5", "bv"]] > 0).all().all()

df = df.sort_index()


# In[3]:


# test parts of the data.

TEST_START = "2007-01-01"
eda = df.loc[df.index < TEST_START].copy()
r = eda["log_ret"]

print("Exploration observations:", len(eda))
print("Exploration period:", eda.index.min(), "to", eda.index.max())

summary = eda.describe(
    percentiles=[0.01, 0.05, 0.10, 0.50, 0.95, 0.99]
).T

summary["skewness"] = eda.skew()
summary["excess_kurtosis"] = eda.kurt()

print(summary.round(6))


# In[ ]:





# In[4]:


# Variation

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

axes[0].plot(r.index, 100 * r, linewidth=0.6)
axes[0].set_title("Daily Returns")
axes[0].set_ylabel("Return (%)")

axes[1].plot(r.index, 100 * r.rolling(63).std())
axes[1].set_title("Trailing 63-Day Standard Deviation")
axes[1].set_ylabel("Daily SD (%)")

plt.tight_layout()
plt.show()


# In[5]:


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(100 * r, bins=50, density=True, alpha=0.7)
axes[0].set_xlabel("Daily Return (%)")
axes[0].set_ylabel("Density")

stats.probplot(100 * r, dist="norm", plot=axes[1])
axes[1].set_title("Normal QQ Plot")

plt.tight_layout()
plt.show()


# In[ ]:





# In[6]:


# test whether there is self-correlation and ARCH effect

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

plot_acf(r, lags=30, zero=False, ax=axes[0])
plot_acf(r.abs(), lags=30, zero=False, ax=axes[1])

axes[0].set_title("Return ACF")
axes[1].set_title("Absolute Return ACF")

plt.tight_layout()
plt.show()

for name, series in {"Return": r, "Absolute Return": r.abs()}.items():
    print(f"\n{name}: Ljung-Box test")
    print(acorr_ljungbox(series, lags=[10], return_df=True))

lm_stat, lm_pvalue, _, _ = het_arch(
    r - r.mean(),
    nlags=10,
    ddof=1
)

print("\nARCH-LM statistic:", lm_stat)
print("ARCH-LM p-value:", lm_pvalue)


# In[ ]:





# In[7]:


realized = pd.DataFrame({
    "log_rv5": np.log(eda["rv5"]),
    "log_bv": np.log(eda["bv"])
})

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for row, column in enumerate(realized.columns):
    axes[row, 0].plot(
        realized.index,
        realized[column],
        linewidth=0.7
    )
    axes[row, 0].set_title(column)

    plot_acf(
        realized[column],
        lags=30,
        zero=False,
        ax=axes[row, 1]
    )
    axes[row, 1].set_title(f"{column} ACF")

plt.tight_layout()
plt.show()


# In[8]:


features = pd.DataFrame({
    "abs_return_t": r.abs(),
    "log_rv5_t": realized["log_rv5"],
    "log_bv_t": realized["log_bv"]
})

# Pair today's features with next-day absolute returns.
aligned = features.assign(
    abs_return_next=r.abs().shift(-1)
).dropna()

next_day_corr = aligned.corr(method="spearman").loc[
    features.columns,
    "abs_return_next"
]

print("Next-day absolute return: Spearman correlations")
print(next_day_corr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Model Implementation

# In[16]:


# Historic Simulation

def rolling_hs(data, start_date, end_date, window=1260):
    returns = data["log_ret"].sort_index()
    alphas = np.array([0.01, 0.05, 0.10])

    # Convert ranks to zero-based positions.
    positions = np.ceil(window * alphas).astype(int) - 1

    start = max(
        window,
        returns.index.searchsorted(pd.Timestamp(start_date)),
    )
    stop = returns.index.searchsorted(
        pd.Timestamp(end_date), side="right"
    )

    rows = []

    for i in range(start, stop):
        # Exclude the forecast date.
        history = returns.iloc[i - window:i].to_numpy()
        quantiles = np.sort(history)[positions]

        rows.append({
            "date": returns.index[i],
            "origin": returns.index[i - 1],
            "model": "HS",
            "actual": returns.iloc[i],
            "VaR_01": quantiles[0],
            "VaR_05": quantiles[1],
            "VaR_10": quantiles[2],
        })

    return pd.DataFrame(rows)




# In[17]:


hs_validation = rolling_hs(
    df,
    start_date="2005-01-01",
    end_date="2006-12-31",
)

hs_validation.to_csv("hs_validation.csv", index=False)

print(hs_validation.head())
print("Forecast dates:", len(hs_validation))


# In[ ]:





# In[18]:


# GARCH
from scipy.stats import t as student_t
from arch.univariate import ConstantMean, GARCH, StudentsT


# In[20]:


class StationaryGARCH(GARCH):
    def constraints(self):
        matrix, bound = super().constraints()
        # Keep alpha + beta strictly below one.
        bound[-1] = -(1 - 1e-6)
        return matrix, bound
    
    

def garch_forecast(returns):
    model = ConstantMean(
        100 * returns,
        volatility=StationaryGARCH(p=1, q=1),
        distribution=StudentsT(),
        rescale=False,
    )

    fitted = model.fit(
        disp="off",
        show_warning=False,
        options={"maxiter": 1000},
    )

    if fitted.convergence_flag != 0:
        raise RuntimeError("Optimization did not converge.")

    forecast = fitted.forecast(horizon=1, reindex=False)

    mean = forecast.mean.iloc[-1, 0]
    variance = forecast.residual_variance.iloc[-1, 0]
    nu = fitted.params["nu"]

    if not (nu > 2 and np.isfinite(variance) and variance > 0):
        raise RuntimeError("Invalid variance or degrees of freedom.")

    alphas = np.array([0.01, 0.05, 0.10])
    quantiles = (
        np.sqrt((nu - 2) / nu)
        * student_t.ppf(alphas, df=nu)
    )

    return (mean + np.sqrt(variance) * quantiles) / 100
    


# In[21]:


def rolling_garch(data, start_date, end_date, window=1260):
    returns = data["log_ret"].sort_index()

    start = max(
        window,
        returns.index.searchsorted(pd.Timestamp(start_date)),
    )
    stop = returns.index.searchsorted(
        pd.Timestamp(end_date), side="right"
    )

    rows = []

    for i in range(start, stop):
        history = returns.iloc[i - window:i]
        quantiles = np.full(3, np.nan)
        status = "ok"

        try:
            quantiles = garch_forecast(history)

            if not np.isfinite(quantiles).all():
                raise RuntimeError("Invalid quantile forecasts.")

        except (RuntimeError, ValueError, FloatingPointError) as error:
            quantiles = np.full(3, np.nan)
            status = str(error)

        rows.append({
            "date": returns.index[i],
            "origin": returns.index[i - 1],
            "model": "GARCH-t",
            "actual": returns.iloc[i],
            "VaR_01": quantiles[0],
            "VaR_05": quantiles[1],
            "VaR_10": quantiles[2],
            "status": status,
        })

    return pd.DataFrame(rows)


# In[22]:


garch_validation = rolling_garch(
    df,
    start_date="2005-01-01",
    end_date="2006-12-31",
)

garch_validation.to_csv("garch_validation.csv", index=False)

print(garch_validation.head())
print("Forecast dates:", len(garch_validation))
print(garch_validation["status"].value_counts())


# In[ ]:





# In[23]:


# MLP
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# In[24]:


# Network Structure and Loss Function
LOOKBACK = 20
HIDDEN = 16
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SEED = 42

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class QuantileMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LOOKBACK * 3, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 3),
        )

    def forward(self, x):
        return self.network(x)


def quantile_loss(prediction, target):
    alpha = prediction.new_tensor([0.01, 0.05, 0.10])
    error = target - prediction

    return torch.maximum(
        alpha * error,
        (alpha - 1) * error,
    ).mean()


# In[25]:


def mlp_forecast(history):
    torch.manual_seed(SEED)

    returns = history["log_ret"].to_numpy()

    features = np.column_stack([
        returns,
        np.log(history["rv5"]),
        np.log(history["bv"]),
    ])

    # Fit scaling on training input dates.
    mean = features[:-1].mean(axis=0)
    scale = features[:-1].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    features = (features - mean) / scale

    X = np.stack([
        features[j - LOOKBACK:j]
        for j in range(LOOKBACK, len(history))
    ])
    y = 100 * returns[LOOKBACK:, None]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = QuantileMLP()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    model.train()

    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = quantile_loss(model(xb), yb)

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")

            loss.backward()
            optimizer.step()

    # Predict from the latest observed sequence.
    X_next = torch.tensor(
        features[-LOOKBACK:][None, :, :],
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        return model(X_next).numpy()[0] / 100


# In[26]:


def rolling_mlp(
    data,
    start_date,
    end_date,
    window=1260,
    save_path="mlp_validation.csv",
):
    data = data.sort_index()
    assert window > LOOKBACK

    start = max(
        window,
        data.index.searchsorted(pd.Timestamp(start_date)),
    )
    stop = data.index.searchsorted(
        pd.Timestamp(end_date), side="right"
    )

    rows = []

    for step, i in enumerate(range(start, stop), start=1):
        history = data.iloc[i - window:i]
        quantiles = np.full(3, np.nan)
        status = "ok"

        try:
            quantiles = mlp_forecast(history)

            if not np.isfinite(quantiles).all():
                raise RuntimeError("Invalid quantile forecasts.")

        except (RuntimeError, ValueError, FloatingPointError) as error:
            quantiles = np.full(3, np.nan)
            status = str(error)

        rows.append({
            "date": data.index[i],
            "origin": data.index[i - 1],
            "model": "MLP",
            "actual": data["log_ret"].iloc[i],
            "VaR_01": quantiles[0],
            "VaR_05": quantiles[1],
            "VaR_10": quantiles[2],
            "status": status,
            "crossing": (
                bool(np.any(np.diff(quantiles) < 0))
                if status == "ok" else np.nan
            ),
        })

        if step == 1 or step % 25 == 0 or i == stop - 1:
            pd.DataFrame(rows).to_csv(save_path, index=False)
            print(f"{step}/{stop-start}: {data.index[i].date()}")

    return pd.DataFrame(rows)


# In[27]:


mlp_validation = rolling_mlp(
    df,
    start_date="2005-01-01",
    end_date="2006-12-31",
    save_path="mlp_validation.csv",
)

print(mlp_validation.head())
print("Forecast dates:", len(mlp_validation))
print(mlp_validation["status"].value_counts())
print("Quantile crossings:", mlp_validation["crossing"].sum())


# In[ ]:





# In[28]:


# LSTM

class QuantileLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        self.output = nn.Linear(HIDDEN, 3)

    def forward(self, x):
        states, _ = self.lstm(x)
        return self.output(states[:, -1, :])




# In[29]:


def lstm_forecast(history):
    torch.manual_seed(SEED)

    returns = history["log_ret"].to_numpy()

    features = np.column_stack([
        returns,
        np.log(history["rv5"]),
        np.log(history["bv"]),
    ])

    # Fit scaling on training input dates.
    mean = features[:-1].mean(axis=0)
    scale = features[:-1].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    features = (features - mean) / scale

    X = np.stack([
        features[j - LOOKBACK:j]
        for j in range(LOOKBACK, len(history))
    ])
    y = 100 * returns[LOOKBACK:, None]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = QuantileLSTM()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    model.train()

    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = quantile_loss(model(xb), yb)

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")

            loss.backward()
            optimizer.step()

    # Predict from the latest observed sequence.
    X_next = torch.tensor(
        features[-LOOKBACK:][None, :, :],
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        return model(X_next).numpy()[0] / 100


# In[30]:


def rolling_lstm(
    data,
    start_date,
    end_date,
    window=1260,
    save_path="lstm_validation.csv",
):
    data = data.sort_index()
    assert window > LOOKBACK

    start = max(
        window,
        data.index.searchsorted(pd.Timestamp(start_date)),
    )
    stop = data.index.searchsorted(
        pd.Timestamp(end_date), side="right"
    )

    rows = []

    for step, i in enumerate(range(start, stop), start=1):
        history = data.iloc[i - window:i]
        quantiles = np.full(3, np.nan)
        status = "ok"

        try:
            quantiles = lstm_forecast(history)

            if not np.isfinite(quantiles).all():
                raise RuntimeError("Invalid quantile forecasts.")

        except (RuntimeError, ValueError, FloatingPointError) as error:
            quantiles = np.full(3, np.nan)
            status = str(error)

        rows.append({
            "date": data.index[i],
            "origin": data.index[i - 1],
            "model": "LSTM",
            "actual": data["log_ret"].iloc[i],
            "VaR_01": quantiles[0],
            "VaR_05": quantiles[1],
            "VaR_10": quantiles[2],
            "status": status,
            "crossing": (
                bool(np.any(np.diff(quantiles) < 0))
                if status == "ok" else np.nan
            ),
        })

        if step == 1 or step % 25 == 0 or i == stop - 1:
            pd.DataFrame(rows).to_csv(save_path, index=False)
            print(f"{step}/{stop-start}: {data.index[i].date()}")

    return pd.DataFrame(rows)


# In[31]:


lstm_validation = rolling_lstm(
    df,
    start_date="2005-01-01",
    end_date="2006-12-31",
    save_path="lstm_validation.csv",
)

print(lstm_validation.head())
print("Forecast dates:", len(lstm_validation))
print(lstm_validation["status"].value_counts())
print("Quantile crossings:", lstm_validation["crossing"].sum())


# In[ ]:





# In[32]:


# Read the validation results

files = {
    "HS": "hs_validation.csv",
    "GARCH-t": "garch_validation.csv",
    "MLP": "mlp_validation.csv",
    "LSTM": "lstm_validation.csv",
}

frames = []

for model, path in files.items():
    frame = pd.read_csv(path, parse_dates=["date", "origin"])
    assert frame["model"].eq(model).all()

    if "status" not in frame.columns:
        frame["status"] = "ok"

    frames.append(frame)

validation = pd.concat(frames, ignore_index=True)


# In[33]:


var_columns = ["VaR_01", "VaR_05", "VaR_10"]

assert not validation.duplicated(["date", "model"]).any()

validation["valid"] = (
    validation["status"].eq("ok")
    & np.isfinite(validation[var_columns].to_numpy()).all(axis=1)
)

validation["crossing"] = (
    (validation["VaR_01"] > validation["VaR_05"])
    | (validation["VaR_05"] > validation["VaR_10"])
)

quality = validation.groupby("model").agg(
    forecasts=("date", "size"),
    valid=("valid", "sum"),
    crossings=("crossing", "sum"),
    first_date=("date", "min"),
    last_date=("date", "max"),
)

print(quality)

assert quality["forecasts"].eq(488).all()
assert validation["valid"].all(), "Some forecasts failed."
assert validation.groupby("date")["model"].nunique().eq(4).all()
assert validation.groupby("date")["actual"].nunique().eq(1).all()


# In[34]:


alphas = [0.01, 0.05, 0.10]
rows = []

for model, group in validation.groupby("model"):
    actual = group["actual"].to_numpy()
    n = len(actual)

    for alpha, column in zip(alphas, var_columns):
        prediction = group[column].to_numpy()
        error = actual - prediction

        exceptions = int((error < 0).sum())
        failure_rate = exceptions / n

        loss = np.maximum(
            alpha * error,
            (alpha - 1) * error,
        )

        rows.append({
            "model": model,
            "alpha": alpha,
            "N": n,
            "exceptions": exceptions,
            "expected_exceptions": n * alpha,
            "failure_rate": failure_rate,
            "violation_ratio": failure_rate / alpha,
            "QL": loss.mean(),
        })

validation_metrics = pd.DataFrame(rows)
validation_metrics.to_csv("validation_metrics.csv", index=False)

print(
    validation_metrics
    .sort_values(["alpha", "QL"])
    .round(6)
)


# In[35]:


ALPHAS = [0.01, 0.05, 0.10]
QCOLS = ["VaR_01", "VaR_05", "VaR_10"]


def quantile_crossing(result):
    return (
        (result["VaR_01"] > result["VaR_05"])
        | (result["VaR_05"] > result["VaR_10"])
    )


def order_quantiles(result):
    result = result.copy()
    neural = result["model"].isin(["MLP", "LSTM"])

    result.loc[neural, QCOLS] = np.sort(
        result.loc[neural, QCOLS].to_numpy(), axis=1
    )
    result["crossing"] = quantile_crossing(result)
    return result


validation_raw = validation.copy()
validation_raw["crossing"] = quantile_crossing(validation_raw)
validation_final = order_quantiles(validation_raw)

print(pd.DataFrame({
    "raw_crossings": validation_raw.groupby("model")["crossing"].sum(),
    "final_crossings": validation_final.groupby("model")["crossing"].sum(),
}))

validation_raw.to_csv("validation_raw.csv", index=False)
validation_final.to_csv("validation_final.csv", index=False)


# In[36]:


def score_forecasts(result):
    rows = []

    for model, group in result.groupby("model"):
        actual = group["actual"].to_numpy()

        for alpha, column in zip(ALPHAS, QCOLS):
            error = actual - group[column].to_numpy()
            hits = error < 0
            n = len(error)

            rows.append({
                "model": model,
                "alpha": alpha,
                "N": n,
                "exceptions": int(hits.sum()),
                "expected_exceptions": n * alpha,
                "failure_rate": hits.mean(),
                "violation_ratio": hits.mean() / alpha,
                "QL": np.maximum(
                    alpha * error, (alpha - 1) * error
                ).mean(),
            })

    return pd.DataFrame(rows).sort_values(["alpha", "QL"])


validation_metrics = score_forecasts(validation_final)

print(validation_metrics.round(6))
validation_metrics.to_csv("validation_metrics_final.csv", index=False)


# In[37]:


# Fix the hyperparameters

df = df.sort_index()

WINDOW = 1260
LOOKBACK = 20
HIDDEN = 16
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SEED = 42

TEST_START = "2007-01-01"
TEST_END = "2018-06-27"

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


# In[38]:


# Implement on the test set

# HS and GARCH
hs_test = rolling_hs(
    df, TEST_START, TEST_END, window=WINDOW
)
hs_test.to_csv("hs_test.csv", index=False)

garch_test = rolling_garch(
    df, TEST_START, TEST_END, window=WINDOW
)
garch_test.to_csv("garch_test.csv", index=False)


# In[39]:


# MLP
mlp_test = rolling_mlp(
    df,
    TEST_START,
    TEST_END,
    window=WINDOW,
    save_path="mlp_test.csv",
)
mlp_test.to_csv("mlp_test.csv", index=False)


# In[40]:


# LSTM

lstm_test = rolling_lstm(
    df,
    TEST_START,
    TEST_END,
    window=WINDOW,
    save_path="lstm_test.csv",
)
lstm_test.to_csv("lstm_test.csv", index=False)


# In[41]:


# Merge

test_raw = pd.concat(
    [hs_test, garch_test, mlp_test, lstm_test],
    ignore_index=True,
).sort_values(["date", "model"])

test_raw.to_csv("test_raw.csv", index=False)
test_final = order_quantiles(test_raw)

expected_dates = df.loc[TEST_START:TEST_END].index

assert set(test_final["model"]) == set(validation_raw["model"])

for model, group in test_final.groupby("model"):
    group = group.sort_values("date")

    assert pd.DatetimeIndex(group["date"]).equals(expected_dates), model
    assert np.isfinite(group[["actual"] + QCOLS].to_numpy()).all(), model
    assert np.allclose(
        group["actual"].to_numpy(),
        df.loc[expected_dates, "log_ret"].to_numpy(),
    ), model

test_final.to_csv("test_final.csv", index=False)

print(test_final[["date", "model", "actual"] + QCOLS].head(12))
print("Forecast dates:", len(expected_dates))


# In[42]:


# Statstical Test

# Kupiec UC
# Christoffersen IND
# Christoffersen CC

from scipy.special import xlogy
from scipy.stats import chi2


def bernoulli_ll(successes, trials, probability):
    return (
        xlogy(successes, probability)
        + xlogy(trials - successes, 1 - probability)
    )


def var_tests(hit, alpha):
    hit = np.asarray(hit, dtype=int)
    n = len(hit)
    k = hit.sum()

    ll_null = bernoulli_ll(k, n, alpha)
    ll_fitted = bernoulli_ll(k, n, k / n)
    lr_uc = max(0.0, 2 * (ll_fitted - ll_null))

    previous, current = hit[:-1], hit[1:]
    n0 = np.sum(previous == 0)
    n1 = np.sum(previous == 1)
    lr_ind = np.nan

    if n0 > 0 and n1 > 0:
        n01 = np.sum((previous == 0) & (current == 1))
        n11 = np.sum((previous == 1) & (current == 1))

        ll_iid = bernoulli_ll(
            current.sum(), len(current), current.mean()
        )
        ll_markov = (
            bernoulli_ll(n01, n0, n01 / n0)
            + bernoulli_ll(n11, n1, n11 / n1)
        )
        lr_ind = max(0.0, 2 * (ll_markov - ll_iid))

    lr_cc = lr_uc + lr_ind

    return {
        "LR_UC": lr_uc,
        "p_UC": chi2.sf(lr_uc, 1),
        "LR_IND": lr_ind,
        "p_IND": chi2.sf(lr_ind, 1),
        "LR_CC": lr_cc,
        "p_CC": chi2.sf(lr_cc, 2),
    }


# In[43]:


# Test metrics

test_statistics = []

for model, group in test_final.groupby("model"):
    group = group.sort_values("date")

    for alpha, column in zip(ALPHAS, QCOLS):
        hit = (
            group["actual"].to_numpy()
            < group[column].to_numpy()
        )

        test_statistics.append({
            "model": model,
            "alpha": alpha,
            **var_tests(hit, alpha),
        })

test_metrics = score_forecasts(test_final).merge(
    pd.DataFrame(test_statistics),
    on=["model", "alpha"],
)

print(test_metrics.round(6))
test_metrics.to_csv("test_metrics.csv", index=False)


# In[45]:


# Visualization

for alpha, column in zip(ALPHAS, QCOLS):
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 9), sharex=True
    )

    for ax, (model, group) in zip(
        axes, test_final.groupby("model")
    ):
        group = group.sort_values("date")
        hit = group["actual"] < group[column]

        ax.plot(
            group["date"], 100 * group["actual"],
            color="lightgray", linewidth=0.6, label="Actual",
        )
        ax.plot(
            group["date"], 100 * group[column],
            linewidth=0.9, label="VaR",
        )
        ax.scatter(
            group.loc[hit, "date"],
            100 * group.loc[hit, "actual"],
            color="red", s=8, label="Exception",
        )
        ax.set_title(f"{model}: {alpha:.0%} VaR")
        ax.set_ylabel("Return (%)")

    if alpha == 0.01:
        axes[0].legend(loc="upper right")
    else:
        axes[0].legend()

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(f"VaR_{int(alpha * 100):02d}.pdf")
    plt.show()

