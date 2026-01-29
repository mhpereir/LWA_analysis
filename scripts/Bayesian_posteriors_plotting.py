import os
import sys
import glob
import argparse

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, config_pymc, preprocess, data_io
from typing import Dict, List, Tuple, Any

import xarray as xr
import numpy as np

import pymc as pm
import arviz as az
import pytensor.tensor as pt
import scipy.stats as stats
import statsmodels.api as sm #type:ignore

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib import ticker as mticker


OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/bayesian_lwa_dt_sm_correlation")
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

OUTPUT_POSTERIOR = os.path.join(config.OUTPUT_PATH, "pymc_fits")


ENSEMBLE_LIST   = config.ENSEMBLE_LIST
MODEL_VARIATIONS = config_pymc.model_variations


# Parameters you estimated (match what you saved)
param_names = ["b0", "b1", "b2", "b3", "rho", "sigma", "nu_minus_2"]

# to do: (factorize existing code)
# make LWA_var, REGION, SEASON command-line args via argparse (new)
# find all posterior files (safety measures: ensure all ensemble members, model variants are present) (new)
# plot pair
# plot posteriors (overlaid ERA5 vs CanESM pooled, and faint individual members)
# calculate Bayesian R2 for all members, summarize
# calculate partial R2 for interaction term, LWA only, SM only (new)
# 


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot Bayesian posterior distributions and calculate Bayesian R2."
    )
    parser.add_argument(
        "--lwa_var",
        type=str,
        default="LWA_a",
        choices=config.LWA_VARS,
        help="LWA variable to analyze.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="pnw_bartusek",
        choices=list(config.REGIONS.keys()),
        help="Region to analyze.",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="JJA",
        choices=list(config.SEASON_NAMES),
        help="Season to analyze.",
    )
    return parser.parse_args()



# ----------------------------- I/O functions -----------------------------

def open_posterior_file(filepath: str) -> az.InferenceData:
    """Open a PyMC posterior InferenceData from NetCDF."""
    idata = az.from_netcdf(filepath)
    return idata


# ----------------------------- Helper functions -----------------------------

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

# ----------------------------- Bayesian R2 calculation -----------------------------

def bayes_r2_ar1_studentt(idata: az.InferenceData,
                          include_lwa: bool,
                          include_sm: bool,
                          include_interaction: bool) -> np.ndarray:
    """
    Bayesian R2 per posterior draw for your AR(1) StudentT model.
    Returns: r2[sample]
    Assumes pm.Data were used for x1, x2, y, same_year (as in your model).
    """
    post = idata.posterior # type: ignore

    # posterior draws
    b0    = _stack_draws(post["b0"])             # (S,)
    b1    = _stack_draws(post["b1"])
    b2    = _stack_draws(post["b2"])
    b3    = _stack_draws(post["b3"])
    rho   = _stack_draws(post["rho"])
    sig   = _stack_draws(post["sigma"])
    nu_m2 = _stack_draws(post["nu_minus_2"])
    nu = nu_m2 - 2.0

    # data (time series)
    cd = idata.constant_data #type: ignore # created because you used pm.Data in the model
    x1 = cd["x1"].values.astype("float64")          # (T,)
    x2 = cd["x2"].values.astype("float64")
    y  = cd["y"].values.astype("float64")
    same_year = cd["same_year"].values.astype("float64")  # (T,)

    x12 = x1 * x2  # (T,)

    S = b0.shape[0] # number of posterior samples
    T = x1.shape[0] # number of time points
    mu = np.zeros((S, T), dtype=np.float64)
    
    mu += b0[:, None]
    if include_lwa:
        mu += b1[:, None] * x1[None, :]
    if include_sm:
        mu += b2[:, None] * x2[None, :]
    if include_interaction:
        mu += b3[:, None] * x12[None, :]

    # # build cond_mu consistent with your likelihood
    # y_prev  = np.concatenate([y[:1], y[:-1]])
    # mu_prev = np.concatenate([mu[:, :1], mu[:, :-1]], axis=1)

    # cond_mu = mu + rho[:, None] * same_year[None, :] * (y_prev[None, :] - mu_prev)

    # variance over time (per draw)
    var_mu = np.var(mu, axis=1, ddof=1)

    # StudentT residual variance (per draw)
    var_eps = (sig**2) * (nu / (nu - 2.0))

    r2 = var_mu / (var_mu + var_eps)
    return r2


def partial_r2(r2_full: np.ndarray, r2_reduced: np.ndarray) -> np.ndarray:
    # Cohen-style partial R2 (per draw)
    return (r2_full - r2_reduced) / (1.0 - r2_reduced)


# ----------------------------- RMSE calculation -----------------------------

def rmse_per_draw(y:    np.ndarray,   # time series data
                  yhat: np.ndarray):  # yhat: (S,T) model predictions per draw
    
    err2 = (yhat - y[None, :])**2
    return np.sqrt(np.mean(err2, axis=1))  # (S,)

def build_mu(b0,b1,b2,b3,x1,x2, include_lwa, include_sm, include_interaction):
    S = b0.shape[0]
    T = x1.shape[0]
    mu = np.zeros((S, T), dtype=np.float64)
    mu += b0[:, None]
    if include_lwa:
        mu += b1[:, None] * x1[None, :]
    if include_sm:
        mu += b2[:, None] * x2[None, :]
    if include_interaction:
        mu += b3[:, None] * (x1 * x2)[None, :]
    return mu

# ----------------------------- Plotting -----------------------------

def plot_posterior_overlaid(idata_era: az.InferenceData,
                            idata_can_members: List[az.InferenceData],
                            N_members: int,
                            LWA_var: str,
                            REGION: str,
                            SEASON: str):

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
            ax.plot(gx, gd, linewidth=2.5, label=f"CanESM pooled (N={N_members})")

        ax.set_ylabel("density")
        ax.set_title(p)

    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Parameter Value (posterior samples)")
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_PLOTS_PATH, f"posterior_pdfs_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5_vs_CanESM.png")
    fig.savefig(out_png, dpi=300)

def joint_posterior_diagnostics(idata: az.InferenceData,
                                vars: List[str],
                                LWA_var: str,
                                REGION: str,
                                SEASON: str,
                                SIM_NAME: str):

    
    az.plot_pair(
        idata,
        var_names=vars,
        kind="kde",
        marginals=True,
        figsize=(7, 7),
    )

    out_png = os.path.join(
        OUTPUT_PLOTS_PATH,
        f"posterior_joint_betas_{LWA_var}_tas_sm_{REGION}_{SEASON}_{SIM_NAME}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("Saved:", out_png)

# ----------------------------- Main analysis -----------------------------

def run_analysis(LWA_var: str, REGION: str, SEASON: str):


    idata_era_models    = {}
    idata_canesm_models = {}
    
    print("Loading posteriors for all model variations...")
    for model_name in MODEL_VARIATIONS:
        idata_era_models[model_name]    = None  
        idata_canesm_models[model_name] = []

        era_file = os.path.join(OUTPUT_POSTERIOR,
            f"pymc_ar1_{model_name}_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5.nc"
        )

        can_files = sorted(glob.glob(os.path.join(
            OUTPUT_POSTERIOR, f"pymc_ar1_{model_name}_{LWA_var}_tas_sm_{REGION}_{SEASON}_CanESM5_*.nc"
        )))
        N_members = len(can_files)
        assert N_members == len(ENSEMBLE_LIST), f"Expected {len(ENSEMBLE_LIST)} CanESM members, found {N_members}"
        # Verify all ensemble members are present
        found_members = {os.path.basename(f).split('_')[-1].split('.')[0] for f in can_files}
        missing = set(ENSEMBLE_LIST) - found_members
        assert not missing, f"Missing ensemble members for {model_name}: {missing} \n {found_members}"

        # Load ERA5
        idata_era_models[model_name] = open_posterior_file(era_file)

        # Load CanESM members
        idata_canesm_models[model_name] = [open_posterior_file(f) for f in can_files]

    # -----------------------------

    # Validate that "full" model was loaded
    if "full" not in idata_era_models or idata_era_models["full"] is None:
        raise ValueError("Full model not found in MODEL_VARIATIONS or failed to load")
    if "full" not in idata_canesm_models or not idata_canesm_models["full"]:
        raise ValueError("Full model CanESM data not found or failed to load")

    # -----------------------------
    # Plot PDFs
    # -----------------------------
    
    idata_era         = idata_era_models["full"]
    idata_can_members = idata_canesm_models["full"]

    N_members=len(ENSEMBLE_LIST) #re-defined, should be the same
    plot_posterior_overlaid(
        idata_era,
        idata_can_members,
        N_members,
        LWA_var,
        REGION,
        SEASON
    )

    # -----------------------------
    # Joint posterior diagnostics
    # -----------------------------

    beta_vars = ["b1", "b2", "b3"]

    idata_canesm_pool = az.concat(idata_can_members, dim="draw")
    joint_posterior_diagnostics(
        idata_era,
        beta_vars,
        LWA_var,
        REGION,
        SEASON,
        "ERA5"
    )


    # CanESM pooled
    joint_posterior_diagnostics(
        idata_canesm_pool,  #type: ignore
        beta_vars,
        LWA_var,
        REGION,
        SEASON,
        "CanESM5_pooled"
    )    

    # -----------------------------
    # Bayesian R2 calculation
    # -----------------------------

    r2_partial_era = {} #model_name
    r2_partial_can = {} #model_name, member

    output_csv = os.path.join(
        config.OUTPUT_PATH,
        f"bayesian_r2_summary_{LWA_var}_tas_sm_{REGION}_{SEASON}.csv"
    )

    

    #loop for all model variations
    for model_name in MODEL_VARIATIONS:
        print(f"\nCalculating Bayesian R2 for model variation: {model_name}")

        idata_era            = idata_era_models[model_name]
        idata_canesm_members = idata_canesm_models[model_name]

        # Calculate R2 for ERA5
        r2_era = bayes_r2_ar1_studentt(idata_era, **MODEL_VARIATIONS[model_name])
        r2_partial_era[model_name] = r2_era

        for idm in idata_canesm_members:
            r2_m = bayes_r2_ar1_studentt(idm, **MODEL_VARIATIONS[model_name])

            if model_name not in r2_partial_can:
                r2_partial_can[model_name] = []
            r2_partial_can[model_name].append(r2_m)

    with open(output_csv, 'w') as fcsv:
        header = "dataset,R2_median,R2_p05,R2_p95,dR2_lwa_only,dR2_sm_only,dR2_interaction,pR2_lwa_only,pR2_sm_only,pR2_interaction\n"
        fcsv.write(header)

        # ERA5
        r2_era_median = np.median(r2_partial_era["full"])
        r2_era_p05    = np.quantile(r2_partial_era["full"], 0.05)
        r2_era_p95    = np.quantile(r2_partial_era["full"], 0.95)

        line = "ERA5,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            r2_era_median,
            r2_era_p05,
            r2_era_p95,
            r2_era_median - np.median(r2_partial_era["lwa_only"]),
            r2_era_median - np.median(r2_partial_era["sm_only"]),
            r2_era_median - np.median(r2_partial_era["no_int"]),
            np.median(partial_r2(r2_partial_era["full"], r2_partial_era["lwa_only"])),
            np.median(partial_r2(r2_partial_era["full"], r2_partial_era["sm_only"])),
            np.median(partial_r2(r2_partial_era["full"], r2_partial_era["no_int"])),
        )


        fcsv.write(
            line + "\n"
        )

        # CanESM5
        for i, emem in enumerate(ENSEMBLE_LIST):
            r2_can_member = r2_partial_can["full"][i]
            r2_can_median = np.median(r2_can_member)
            r2_can_p05    = np.quantile(r2_can_member, 0.05)
            r2_can_p95    = np.quantile(r2_can_member, 0.95)

            line = "{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                'CanESM5_'+emem,
                r2_can_median,
                r2_can_p05,
                r2_can_p95,
                r2_can_median - np.median(r2_partial_can["lwa_only"][i]),
                r2_can_median - np.median(r2_partial_can["sm_only"][i]),
                r2_can_median - np.median(r2_partial_can["no_int"][i]),
                np.median(partial_r2(r2_partial_can["full"][i], r2_partial_can["lwa_only"][i])),
                np.median(partial_r2(r2_partial_can["full"][i], r2_partial_can["sm_only"][i])),
                np.median(partial_r2(r2_partial_can["full"][i], r2_partial_can["no_int"][i])),
            )

            fcsv.write(
                line + "\n"
            )
    print("Saved Bayesian R2 summary to:", output_csv)




    
if __name__ == "__main__":
    args = arg_parser()

    LWA_var = args.lwa_var
    REGION  = args.region
    SEASON  = args.season

    run_analysis(LWA_var, REGION, SEASON)






    #     r2_era = bayes_r2_ar1_studentt(idata_era)
    #     print("ERA5 R2 median [p05,p95]:",
    #         np.median(r2_era),
    #         np.quantile(r2_era, 0.05),
    #         np.quantile(r2_era, 0.95))

    #     for f, idm in zip(can_files, idata_can_members):
    #         r2_m = bayes_r2_ar1_studentt(idm)
    #         print(os.path.basename(f), "R2 median:", np.median(r2_m))

    # r2_era = bayes_r2_ar1_studentt(idata_era)
    # print("ERA5 R2 median [p05,p95]:",
    #     np.median(r2_era),
    #     np.quantile(r2_era, 0.05),
    #     np.quantile(r2_era, 0.95))

    # for f, idm in zip(can_files, idata_can_members):
    #     r2_m = bayes_r2_ar1_studentt(idm)
    #     print(os.path.basename(f), "R2 median:", np.median(r2_m))


    # # example:
    # r2_full   = bayes_r2_ar1_studentt(idata_full)
    # r2_noInt  = bayes_r2_ar1_studentt(idata_noInt)
    # r2_lwaOnly = bayes_r2_ar1_studentt(idata_lwaOnly)
    # r2_smOnly  = bayes_r2_ar1_studentt(idata_smOnly)

    # pR2_interaction = partial_r2(r2_full, r2_noInt)
    # dR2_interaction = r2_full - r2_noInt
