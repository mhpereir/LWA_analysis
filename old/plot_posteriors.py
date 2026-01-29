import glob
import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

import os
import argparse

OUTPUT_PLOTS_PATH = "/home/mhpereir/LWA_analysis/plots"
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

OUTPUT_POSTERIOR = "/home/mhpereir/LWA_analysis/pymc_member_fits"

LWA_var = "LWA_c"
REGION  = "pnw_bartusek"
SEASON  = "JJA"

# -----------------------------
# Paths (edit to match yours)
# -----------------------------
era_file = os.path.join(
    OUTPUT_POSTERIOR,
    f"pymc_ar1_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5.nc"
)

can_files = sorted(glob.glob(os.path.join(
    OUTPUT_POSTERIOR, f"pymc_ar1_{LWA_var}_tas_sm_{REGION}_{SEASON}_CanESM5_*.nc"
)))

print(can_files)
assert len(can_files) > 0, f"No CanESM posterior files found in {can_files}"

# Parameters you estimated (match what you saved)
param_names = ["b0", "b1", "b2", "b3", "rho", "sigma", "nu_minus_2"]

# -----------------------------
# Helpers
# -----------------------------
def posterior_samples(idata: az.InferenceData, var: str) -> np.ndarray:
    """Return 1D numpy array of posterior samples for var."""
    da = idata.posterior[var] #type: ignore
    return da.stack(sample=("chain", "draw")).values.astype("float64")

def thin(x: np.ndarray, max_n: int = 6000, seed: int = 0) -> np.ndarray:
    """Downsample to max_n to keep plotting fast."""
    if x.size <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]

def kde1d(x: np.ndarray, gridsize: int = 400, pad: float = 0.25):
    """
    Lightweight Gaussian KDE without scipy.
    Returns grid, density.
    """
    x = x[np.isfinite(x)]
    if x.size < 50:
        return None, None

    lo, hi = np.quantile(x, [0.005, 0.995])
    span = hi - lo
    lo -= pad * span
    hi += pad * span
    grid = np.linspace(lo, hi, gridsize)

    # Scott's rule bandwidth
    std = np.std(x)
    n = x.size
    bw = 1.06 * std * (n ** (-1/5)) if std > 0 else 1e-6

    # KDE
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.mean(np.exp(-0.5 * z * z), axis=1) / (bw * np.sqrt(2*np.pi))
    return grid, dens


def _stack_draws(da) -> np.ndarray:
    # (chain, draw, ...) -> (sample, ...)
    return da.stack(sample=("chain", "draw")).values

# -----------------------------
# Load posteriors
# -----------------------------
idata_era         = az.from_netcdf(era_file)
idata_can_members = [az.from_netcdf(f) for f in can_files]

# -----------------------------
# Plot PDFs
# -----------------------------
npar = len(param_names)
fig, axes = plt.subplots(npar, 1, figsize=(9, 2.2*npar), sharex=False)

if npar == 1:
    axes = [axes]

for ax, p in zip(axes, param_names):
    # ERA5
    era = thin(posterior_samples(idata_era, p), max_n=12000, seed=1)
    gx, gd = kde1d(era)
    if gx is not None:
        ax.plot(gx, gd, linewidth=2.5, label="ERA5")

    # CanESM members (faint)
    pooled = []
    for i, idm in enumerate(idata_can_members):
        xm = thin(posterior_samples(idm, p), max_n=4000, seed=10+i)
        pooled.append(xm)

        gx, gd = kde1d(xm)
        if gx is not None:
            ax.plot(gx, gd, linewidth=0.8, alpha=0.25)

    # CanESM pooled (thicker)
    pooled = np.concatenate(pooled)
    pooled = thin(pooled, max_n=25000, seed=2)
    gx, gd = kde1d(pooled)
    if gx is not None:
        ax.plot(gx, gd, linewidth=2.5, label=f"CanESM pooled (N={len(can_files)})")

    ax.set_ylabel("density")
    ax.set_title(p)

axes[0].legend(frameon=False)
axes[-1].set_xlabel("parameter value (posterior samples)")
plt.tight_layout()
out_png = os.path.join(OUTPUT_PLOTS_PATH, f"posterior_pdfs_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5_vs_CanESM.png")
plt.savefig(out_png, dpi=200)
plt.show()

print("Saved:", out_png)


# -----------------------------
# Joint posterior diagnostics
# -----------------------------

beta_vars = ["b1", "b2", "b3"]

# ERA5
az.plot_pair(
    idata_era,
    var_names=beta_vars,
    kind="kde",
    marginals=True,
    figsize=(7, 7),
)

out_png = os.path.join(
    OUTPUT_PLOTS_PATH,
    f"posterior_joint_betas_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5.png"
)
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Saved:", out_png)


# CanESM pooled
idata_pool = az.concat(idata_can_members, dim="draw")

az.plot_pair(
    idata_pool,
    var_names=beta_vars,
    kind="kde",
    marginals=True,
    figsize=(7, 7),
)

out_png = os.path.join(
    OUTPUT_PLOTS_PATH,
    f"posterior_joint_betas_{LWA_var}_tas_sm_{REGION}_{SEASON}_CanESM_pooled.png"
)
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Saved:", out_png)

# -----------------------------
# Bayesian R2 calculation
# -----------------------------

def bayes_r2_ar1_studentt(idata: az.InferenceData) -> np.ndarray:
    """
    Bayesian R2 per posterior draw for your AR(1) StudentT model.
    Returns: r2[sample]
    Assumes pm.Data were used for x1, x2, y, same_year (as in your model).
    """
    post = idata.posterior # type: ignore

    # posterior draws
    b0  = _stack_draws(post["b0"])             # (S,)
    b1  = _stack_draws(post["b1"])
    b2  = _stack_draws(post["b2"])
    b3  = _stack_draws(post["b3"])
    rho = _stack_draws(post["rho"])
    sig = _stack_draws(post["sigma"])
    nu_m2 = _stack_draws(post["nu_minus_2"])
    nu = nu_m2 + 2.0

    # data (time series)
    cd = idata.constant_data #type: ignore # created because you used pm.Data in the model
    x1 = cd["x1"].values.astype("float64")          # (T,)
    x2 = cd["x2"].values.astype("float64")
    y  = cd["y"].values.astype("float64")
    same_year = cd["same_year"].values.astype("float64")  # (T,)

    x12 = x1 * x2  # (T,)

    # mu: (S, T)
    mu = (b0[:, None]
          + b1[:, None] * x1[None, :]
          + b2[:, None] * x2[None, :]
          + b3[:, None] * x12[None, :])

    # build cond_mu consistent with your likelihood
    y_prev  = np.concatenate([y[:1], y[:-1]])
    mu_prev = np.concatenate([mu[:, :1], mu[:, :-1]], axis=1)

    cond_mu = mu + rho[:, None] * same_year[None, :] * (y_prev[None, :] - mu_prev)

    # variance over time (per draw)
    var_mu = np.var(cond_mu, axis=1, ddof=1)

    # StudentT residual variance (per draw)
    var_eps = (sig**2) * (nu / (nu - 2.0))

    r2 = var_mu / (var_mu + var_eps)
    return r2


def partial_r2(r2_full: np.ndarray, r2_reduced: np.ndarray) -> np.ndarray:
    # Cohen-style partial R2 (per draw)
    return (r2_full - r2_reduced) / (1.0 - r2_reduced)



r2_era = bayes_r2_ar1_studentt(idata_era)
print("ERA5 R2 median [p05,p95]:",
      np.median(r2_era),
      np.quantile(r2_era, 0.05),
      np.quantile(r2_era, 0.95))

for f, idm in zip(can_files, idata_can_members):
    r2_m = bayes_r2_ar1_studentt(idm)
    print(os.path.basename(f), "R2 median:", np.median(r2_m))




# example:
r2_full   = bayes_r2_ar1_studentt(idata_full)
r2_noInt  = bayes_r2_ar1_studentt(idata_noInt)
r2_lwaOnly = bayes_r2_ar1_studentt(idata_lwaOnly)
r2_smOnly  = bayes_r2_ar1_studentt(idata_smOnly)

pR2_interaction = partial_r2(r2_full, r2_noInt)
dR2_interaction = r2_full - r2_noInt
