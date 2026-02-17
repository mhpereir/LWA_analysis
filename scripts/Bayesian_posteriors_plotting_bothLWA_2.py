import os
import sys
import glob
import argparse

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, config_pymc, preprocess, data_io
from src.pymc_models import Variation, ModelSpec
from typing import Dict, List, Optional, Tuple, Any

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



ENSEMBLE_LIST        = config.ENSEMBLE_LIST
MODEL_VARIATIONS     = config_pymc.model_variations
BAYESIAN_MODEL_SPECS = config_pymc.bayesian_model_specs

# Parameters you estimated (match what you saved)
# param_names = ["b0", "b1", "b2", "b3", "rho", "sigma", "nu_minus_2"]
param_names: List[str] = []




def arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot Bayesian posterior distributions and calculate Bayesian R2."
    )
    parser.add_argument(
        "--bayesian_model",
        type=str,
        choices=list(config_pymc.bayesian_model_specs.keys()),
        default="both_lwa_noar_normal",
        help="Bayesian model to use.",
    )
    parser.add_argument(
        "--lwa_var",
        type=str,
        default="LWA_a",
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

def bayes_r2_generic(
    idata: az.InferenceData,
    spec: ModelSpec,
    variation: Variation,
) -> np.ndarray:
    """
    Compute Bayesian R² for an arbitrary model specification and variation.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior samples and constant data from a fitted PyMC model.
    spec : ModelSpec
        Description of the model structure (e.g. AR1 vs no-AR, normal vs Student-t).
    variation : Variation
        Flags indicating which terms are active for this model variation.

    Returns
    -------
    np.ndarray
        R² per posterior draw. Shape (S,), where S is number of posterior samples.
    """
    # Extract posterior draws
    post = idata.posterior  # type: ignore
    cd = idata.constant_data  # type: ignore

    # Intercept
    if spec.intercept_name not in post:
        raise KeyError(f"Intercept parameter '{spec.intercept_name}' not found in posterior")
    b0 = _stack_draws(post[spec.intercept_name])  # (S,)

    # Initialize mu array for all draws and time points
    y_vals = cd["y"].values.astype("float64")
    T = y_vals.shape[0]
    S = b0.shape[0]
    mu = np.zeros((S, T), dtype=np.float64)
    mu += b0[:, None]

    # Build a dictionary of predictor arrays keyed by data_names (excluding same_year)
    data_dict: Dict[str, Any] = {}
    for name in spec.data_names:
        if name == "same_year":
            continue
        arr = cd[name].values
        data_dict[name] = arr.astype("float64")

    # Add contributions from each term if active for this variation
    for term in spec.terms:
        if term.active(variation):
            coef_name = term.coef_name
            if coef_name not in post:
                # Coefficient missing for this spec; skip
                continue
            coef = _stack_draws(post[coef_name])  # (S,)
            # Evaluate the term expression on the constant data
            try:
                x_term = term.expr(data_dict)
            except Exception:
                # Fallback: use first used variable
                x_term = data_dict[term.uses[0]]
            if isinstance(x_term, xr.DataArray):
                x_term = x_term.values
            x_term = np.asarray(x_term, dtype="float64")
            # Ensure shape matches time dimension
            if x_term.ndim != 1 or x_term.shape[0] != T:
                raise ValueError(f"Term expression for {coef_name} did not return a vector of length {T}")
            # Broadcast and accumulate
            mu += coef[:, None] * x_term[None, :]

    # Compute variance of mu over time (per draw)
    var_mu = np.var(mu, axis=1, ddof=1)

    # Residual variance depends on likelihood
    sigma_draws = _stack_draws(post[spec.sigma_name])  # (S,)
    if spec.likelihood == "studentt":
        # Degrees of freedom parameter stored as nu_minus_2; recover actual nu
        nu_m2 = _stack_draws(post[spec.nu_name])  # (S,)
        nu = nu_m2 + 2.0
        # Var of Student-t noise: sigma^2 * nu / (nu - 2)
        var_eps = (sigma_draws**2) * (nu / (nu - 2.0))
    else:
        # Normal likelihood
        var_eps = sigma_draws**2

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

def build_mu(b0,b1,b2,b3,x1,x2, include_lwa, include_sm):
    S = b0.shape[0]
    T = x1.shape[0]
    mu = np.zeros((S, T), dtype=np.float64)
    mu += b0[:, None]
    if include_lwa:
        mu += b1[:, None] * x1[None, :]
    if include_sm:
        mu += b2[:, None] * x2[None, :]
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

def run_analysis(BAYESIAN_MODEL: str, LWA_var: str, REGION: str, SEASON: str):

    # Determine the model specification from the provided key.  This spec describes
    # the likelihood type and whether AR1 is included.  We keep it constant
    # across all variations when computing R² and joint posterior diagnostics.
    spec = BAYESIAN_MODEL_SPECS[BAYESIAN_MODEL]

    # Construct Variation objects for each model variation.  A Variation
    # encapsulates flags controlling which predictors are active.  Keys in
    # MODEL_VARIATIONS map to dictionaries of boolean flags.
    variations: Dict[str, Variation] = {
        name: Variation(
            name=name,
            include_lwa=flags.get("include_lwa", True),
            include_sm=flags.get("include_sm", True),
            include_interaction=flags.get("include_interaction", False),
        )
        for name, flags in MODEL_VARIATIONS.items()
    }

    # Storage for inference data per variation
    idata_era_models: Dict[str, Optional[az.InferenceData]] = {}
    idata_canesm_models: Dict[str, List[az.InferenceData]] = {}
    
    print("Loading posteriors for all model variations...")
    for model_variation in MODEL_VARIATIONS:
        idata_era_models[model_variation]    = None  
        idata_canesm_models[model_variation] = []

        # era_file = os.path.join(OUTPUT_POSTERIORS,
        #     f"pymc_ar1_{model_name}_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5.nc"
        # )

        era_file = os.path.join(OUTPUT_POSTERIORS_PATH,
            f"ERA5_{REGION}_{SEASON}_{BAYESIAN_MODEL}_{model_variation}.nc"
        )
        
        can_files = sorted(glob.glob(os.path.join(OUTPUT_POSTERIORS_PATH, 
            f"CanESM5_*_{REGION}_{SEASON}_{BAYESIAN_MODEL}_{model_variation}.nc"
        )))
        N_members = len(can_files)

        print(can_files)
        print()

        assert N_members == len(ENSEMBLE_LIST), f"Expected {len(ENSEMBLE_LIST)} CanESM members, found {N_members}"
        # Verify all ensemble members are present
        found_members = {os.path.basename(f).split('_')[1] for f in can_files}
        missing = set(ENSEMBLE_LIST) - found_members
        assert not missing, f"Missing ensemble members for {model_variation}: {missing} \n {found_members}"

        # Load ERA5
        idata_era_models[model_variation] = open_posterior_file(era_file)

        # Load CanESM members
        idata_canesm_models[model_variation] = [open_posterior_file(f) for f in can_files]

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

    # Determine coefficient names from the model specification.  Only
    # coefficients appear in spec.terms.  We use these for joint posterior
    # diagnostics.
    beta_vars = [term.coef_name for term in spec.terms]

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
        OUTPUT_POSTERIORS_PATH,
        f"bayesian_r2_summary_{LWA_var}_tas_sm_{REGION}_{SEASON}.csv"
    )

    

    # Loop for all model variations and compute R² using the generic routine
    for model_name, variation_flags in MODEL_VARIATIONS.items():
        print(f"\nCalculating Bayesian R² for model variation: {model_name}")
        var_obj = variations[model_name]
        idata_era_var = idata_era_models[model_name]
        idata_canesm_var = idata_canesm_models[model_name]

        # Compute R² for ERA5 (only if data was loaded)
        if idata_era_var is not None:
            r2_era = bayes_r2_generic(idata_era_var, spec, var_obj)
            r2_partial_era[model_name] = r2_era
        else:
            print(f"Warning: ERA5 data not loaded for {model_name}")

        # Compute R² for each CanESM member
        for idm in idata_canesm_var:
            r2_m = bayes_r2_generic(idm, spec, var_obj)
            if model_name not in r2_partial_can:
                r2_partial_can[model_name] = []
            r2_partial_can[model_name].append(r2_m)

    with open(output_csv, 'w') as fcsv:
        header = "dataset,R2_median,R2_p05,R2_p95,dR2_lwa_only,dR2_sm_only,pR2_lwa_only,pR2_sm_only\n"
        fcsv.write(header)

        # ERA5
        r2_era_median = np.median(r2_partial_era["full"])
        r2_era_p05    = np.quantile(r2_partial_era["full"], 0.05)
        r2_era_p95    = np.quantile(r2_partial_era["full"], 0.95)

        line = "ERA5,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            r2_era_median,
            r2_era_p05,
            r2_era_p95,
            r2_era_median - np.median(r2_partial_era["lwa_only"]),
            r2_era_median - np.median(r2_partial_era["sm_only"]),
            np.median(partial_r2(r2_partial_era["full"], r2_partial_era["lwa_only"])),
            np.median(partial_r2(r2_partial_era["full"], r2_partial_era["sm_only"])),
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

            line = "{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                'CanESM5_'+emem,
                r2_can_median,
                r2_can_p05,
                r2_can_p95,
                r2_can_median - np.median(r2_partial_can["lwa_only"][i]),
                r2_can_median - np.median(r2_partial_can["sm_only"][i]),
                np.median(partial_r2(r2_partial_can["full"][i], r2_partial_can["lwa_only"][i])),
                np.median(partial_r2(r2_partial_can["full"][i], r2_partial_can["sm_only"][i])),
            )

            fcsv.write(
                line + "\n"
            )
    print("Saved Bayesian R2 summary to:", output_csv)




    
if __name__ == "__main__":
    args = arg_parser()

    BAYESIAN_MODEL = args.bayesian_model
    LWA_var = args.lwa_var
    REGION  = args.region
    SEASON  = args.season

    param_names = BAYESIAN_MODEL_SPECS[BAYESIAN_MODEL].posterior_param_names()

    OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, f"plots/{BAYESIAN_MODEL}")
    os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

    OUTPUT_POSTERIORS_PATH = os.path.join(config.OUTPUT_PATH, f"pymc_fits/{BAYESIAN_MODEL}/idata")
    

    run_analysis(BAYESIAN_MODEL, LWA_var, REGION, SEASON)






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
