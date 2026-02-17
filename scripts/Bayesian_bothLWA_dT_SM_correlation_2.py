import os
import sys
import argparse

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, config_pymc, preprocess, data_io
from src.pymc_models import Variation, Term, ModelSpec, build_pymc_model, save_idata

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

# Define script specific constants

# Model variations
_full     = {'include_lwa': True, 'include_sm': True, 'include_interaction': True}
_sm_only  = {'include_lwa': False, 'include_sm': True, 'include_interaction': False}
_lwa_only = {'include_lwa': True, 'include_sm': False, 'include_interaction': False}

model_variations = {
    "full":     _full,
    "sm_only":  _sm_only,
    "lwa_only": _lwa_only,
}



TEMP_VAR      = config.TEMP_VAR
ENSEMBLE_LIST = config.ENSEMBLE_LIST

PYMC_NDRAW:int           = config_pymc.PYMC_NDRAW
PYMC_NTUNE:int           = config_pymc.PYMC_NTUNE
PYMC_TARGET_ACCEPT:float = config_pymc.PYMC_TARGET_ACCEPT
PYMC_CHAINS:int          = config_pymc.PYMC_CHAINS

BAYESIAN_MODEL_SPECS:Dict[str, ModelSpec] = config_pymc.bayesian_model_specs

# Define argument parser

def arg_parser():
    parser = argparse.ArgumentParser(
        description="Bayesian analysis of LWA vs deltaT correlation."
    )

    parser.add_argument(
        "--bayesian_model",
        type=str,
        choices=list(config_pymc.bayesian_model_specs.keys()),
        default="both_lwa_noar_normal",
        help="Bayesian model to use.",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=list(config.REGIONS.keys()),
        default="west_south",
        help="Region to analyze.",
    )
    parser.add_argument(
        "--season",
        type=str,
        choices=config.SEASON_NAMES,
        default="JJA",
        help="Season to analyze.",
    )
    parser.add_argument(
        "--zg",
        type=int,
        choices=[250, 500],
        default=500,
        help="Geopotential height level for LWA.",
    )
    return parser.parse_args()



# ------------------------------ Helper functions ----------------------------------


def _prep_design_matrix(
    lwa_a: xr.DataArray,
    lwa_c: xr.DataArray,
    sm: xr.DataArray,
    dt: xr.DataArray,
    norm_lwaa: tuple[float, float] | None= None,
    norm_lwac: tuple[float, float] | None = None,
    norm_sm: tuple[float, float] | None = None,
    norm_dt: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:#, tuple, tuple, tuple, tuple]:
    """
    Return centered+standardized (X1, X2, y) as numpy arrays, aligned on time.
    Assumes inputs are 1D with coord 'time' and no 'member' dimension.
    """
    # Align on time
    lwa_a, lwa_c, sm, dt = xr.align(lwa_a, lwa_c, sm, dt, join="inner")

    # Drop NaNs
    good = np.isfinite(lwa_a.values) & np.isfinite(lwa_c.values) & np.isfinite(sm.values) & np.isfinite(dt.values)
    lwa_a = lwa_a.isel(time=good)
    lwa_c = lwa_c.isel(time=good)
    sm  = sm.isel(time=good)
    dt  = dt.isel(time=good)

    # Center + standardize (within the already-cropped season)
    x1 = lwa_a.values.astype("float64")
    x2 = lwa_c.values.astype("float64")
    x3 = sm.values.astype("float64")
    y  = dt.values.astype("float64")


    if norm_lwaa==None or norm_lwac==None or norm_sm==None or norm_dt==None:
        
        norm_lwaa = (x1.mean(), x1.std())
        norm_lwac = (x2.mean(), x2.std())
        norm_sm   = (x3.mean(), x3.std())
        norm_dt   = (y.mean(), y.std())


    x1 = (x1 - norm_lwaa[0]) / norm_lwaa[1]
    x2 = (x2 - norm_lwac[0]) / norm_lwac[1]
    x3 = (x3 - norm_sm[0]) / norm_sm[1]
    y  = (y  - norm_dt[0]) / norm_dt[1]

    return x1, x2, x3, y#, norm_lwaa, norm_lwac, norm_sm, norm_dt



# def _year_segments(time: xr.DataArray) -> list[np.ndarray]:
#     """
#     For a seasonal-cropped daily series, return a list of integer index arrays,
#     one per year (contiguous segments).
#     """
#     years = time.dt.year.values
#     segs = []
#     for yr in np.unique(years):
#         idx = np.where(years == yr)[0]
#         # sanity: require at least a few days
#         if idx.size >= 10:
#             segs.append(idx)
#     return segs


def load_lwa_data(ds_lwa_var, region):
    ds_lwa_reg = preprocess.compute_region_mean(ds_lwa_var, region).chunk({"time": 365}).compute()
    
    ds_sqrt_lwa_reg = xr.apply_ufunc(np.sqrt, ds_lwa_reg)
   
    ds_sqrt_lwa_reg = ds_sqrt_lwa_reg.assign_coords(time=ds_sqrt_lwa_reg.time.dt.floor("D"))

    return ds_sqrt_lwa_reg


# ------------------------------ Stats Model ----------------------------------


# def fit_pymc_ar1_interaction(
#     time: xr.DataArray,
#     x1: np.ndarray,  # standardized LWA
#     x2: np.ndarray,  # standardized SM
#     y: np.ndarray,   # standardized deltaT or SM depending on your target
#     draws: int = 2000,
#     tune: int = 2000,
#     target_accept: float = 0.9,
#     chains: int = 4,
# ) -> az.InferenceData:
#     """
#     Fit: y = b0 + b1*x1 + b2*x2 + b3*(x1*x2) + AR1 residual (rho, sigma)
#     using a whitened AR(1) likelihood within each year-segment.

#     Returns arviz InferenceData.
#     """
#     segs = _year_segments(time)
#     x12 = x1 * x2

#     with pm.Model() as model:
#         # Priors (standardized inputs => N(0,1) priors are reasonable defaults)
#         b0 = pm.Normal("b0", mu=0.0, sigma=1.0)
#         b1 = pm.Normal("b1", mu=0.0, sigma=1.0)
#         b2 = pm.Normal("b2", mu=0.0, sigma=1.0)
#         b3 = pm.Normal("b3", mu=0.0, sigma=1.0)

#         # AR(1) coefficient
#         rho = pm.Uniform("rho", lower=-0.99, upper=0.99)

#         # Robust error model (helpful for tails / heatwave days)
#         nu = pm.Exponential("nu_minus_2", 1/10) + 2.0
#         sigma = pm.HalfNormal("sigma", sigma=1.0)

#         mu = b0 + b1*x1 + b2*x2 + b3*x12

#         # Whitened likelihood: product over year segments
#         for k, idx in enumerate(segs):
#             # within-segment series
#             yk = y[idx]
#             muk = mu[idx]

#             # Use generative AR(1) form: y_t = mu_t + rho*(y_{t-1}-mu_{t-1}) + eps_t
#             # First point in each segment (no lag info)
#             pm.StudentT(
#                 f"y_like_init_seg{k}",
#                 nu=nu,
#                 mu=muk[0],
#                 sigma=sigma,
#                 observed=yk[0]
#             )

#             # Remaining points conditional on previous observation
#             pm.StudentT(
#                 f"y_like_seg{k}",
#                 nu=nu,
#                 mu=muk[1:] + rho * (yk[:-1] - muk[:-1]),
#                 sigma=sigma,
#                 observed=yk[1:]
#             )

#         idata = pm.sample(
#             draws=draws,
#             tune=tune,
#             chains=chains,
#             target_accept=target_accept,
#             progressbar=True,
#         )

#     return idata



# def fit_pymc_ar1_interaction(
#     time: xr.DataArray,
#     x1: np.ndarray,
#     x2: np.ndarray,
#     y: np.ndarray,
#     draws: int = 2000,
#     tune: int = 2000,
#     target_accept: float = 0.9,
#     chains: int = 4,
# ) -> az.InferenceData:

#     years     = time.dt.year.values.astype(np.int32)
#     same_year = np.concatenate([[0], (years[1:] == years[:-1]).astype(np.int8)]) # 1 if same year as previous, else 0

#     with pm.Model() as model:
#         # data containers (optional but good practice)
#         x1d = pm.Data("x1", x1)
#         x2d = pm.Data("x2", x2)
#         yd  = pm.Data("y",  y)
#         syd = pm.Data("same_year", same_year) #mask of same year (1 if same year as previous, else 0)
#                                               #resets the conditional mean at year boundaries

#         x12 = x1d * x2d #type: ignore

#         b0 = pm.Normal("b0", 0.0, 1.0) # intercept
#         b1 = pm.Normal("b1", 0.0, 1.0) # LWA coeff
#         b2 = pm.Normal("b2", 0.0, 1.0) # SM coeff
#         b3 = pm.Normal("b3", 0.0, 1.0) # interaction term

#         rho = pm.Uniform("rho", lower=-0.99, upper=0.99)

#         nu = pm.Exponential("nu_minus_2", 1/10) + 2.0
#         sigma = pm.HalfNormal("sigma", 1.0)

#         mu = b0 + b1*x1d + b2*x2d + b3*x12

#         # previous observed y and previous mu
#         y_prev  = pt.concatenate([yd[:1], yd[:-1]]) # type: ignore
#         mu_prev = pt.concatenate([mu[:1], mu[:-1]]) # type: ignore

#         # reset at year boundary via mask (0 at first obs of each year)
#         cond_mu = mu + rho * syd * (y_prev - mu_prev) # type: ignore

#         pm.StudentT("y_like", nu=nu, mu=cond_mu, sigma=sigma, observed=yd)

#         idata = pm.sample(
#             draws=draws, tune=tune, chains=chains,
#             target_accept=target_accept, progressbar=True
#         )

#     return idata

def fit_from_spec(
    *,
    spec: ModelSpec,
    variation: Variation,
    time: xr.DataArray,
    data: dict[str, np.ndarray],
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    attrs: dict[str, object] | None = None,
) -> az.InferenceData:
    model = build_pymc_model(spec=spec, variation=variation, time=time, data=data)

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
        )

    # always attach core metadata
    idata.attrs["model_key"] = spec.key
    idata.attrs["variation"] = variation.name
    idata.attrs["include_lwa"] = variation.include_lwa
    idata.attrs["include_sm"] = variation.include_sm
    idata.attrs["include_interaction"] = variation.include_interaction
    idata.attrs["ar1"] = spec.ar1
    idata.attrs["likelihood"] = spec.likelihood

    if attrs:
        idata.attrs.update(attrs)

    return idata



def compute_mu_hat(
    x1: np.ndarray, # LWA_a standardised
    x2: np.ndarray, # LWA_c standardised
    x3: np.ndarray, # SM  standardised
    idata: az.InferenceData, #model posteriors
    include_lwa: bool = True,
    include_sm: bool = True,
    stat: str = "median",  # or "mean"
) -> np.ndarray:
    post = idata.posterior # type: ignore

    def _pt(var: str) -> float:
        arr = post[var].values.reshape(-1)
        return float(np.median(arr) if stat == "median" else np.mean(arr))

    b0 = _pt("b0")
    b1 = _pt("b1")
    b2 = _pt("b2")
    b3 = _pt("b3")

    mu = np.full_like(x1, b0, dtype="float64")
    if include_lwa:
        mu += b1 * x1 + b2 * x2
    if include_sm:
        mu += b3 * x3
    return mu


# ------------------------------ Plotting functions ----------------------------------



def plot_lwa_sm_correlation(masked_lwa_era: xr.DataArray,
                            masked_mrsos_era: xr.DataArray, 
                            masked_lwa_can: xr.DataArray, 
                            masked_mrsos_can: xr.DataArray, 
                            LWA_var: str, 
                            REGION: str, 
                            SEASON: str):
    

    fig,ax = plt.subplots(figsize=(12,6), ncols=2, nrows=1, constrained_layout=True, sharey=True, sharex=True)
    ax[0].scatter(masked_lwa_era, masked_mrsos_era, color='blue', alpha=0.5, label='ERA5')
    ax[1].scatter(masked_lwa_can, masked_mrsos_can, color='red', alpha=0.2, label='CanESM')

    ax[0].set_xlabel(f"{LWA_var} (sqrt hPa m)")
    ax[0].set_ylabel("Soil Moisture Anomaly (kg/m2)")

    # ax[1].set_xlabel(f"{LWA_var} (sqrt hPa m)")
    ax[1].set_ylabel("Soil Moisture Anomaly (kg/m2)")
    ax[1].set_title(f"CanESM")

    ax[0].legend()
    ax[1].legend()

    fig.suptitle(f"{LWA_var} vs Soil Moisture Anomaly Correlation - {REGION} - {SEASON}", fontsize=16)

    fig_name = f"{OUTPUT_PLOTS_PATH}/{LWA_var}_vs_SoilMoisture_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')



def plot_residual_lag1(
    time: xr.DataArray,
    eps: np.ndarray,
    out_png: str,
    drop_year_boundaries: bool = True,
) -> None:
    years = time.dt.year.values.astype(np.int32)
    same_year = np.concatenate([[0], (years[1:] == years[:-1]).astype(np.int8)])

    # pairs: (t-1, t)
    e0 = eps[:-1]
    e1 = eps[1:]

    if drop_year_boundaries:
        ok = same_year[1:] == 1  # t has same year as t-1
        e0 = e0[ok]
        e1 = e1[ok]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(e0, e1, s=8, alpha=0.25)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(r"$\epsilon_{t-1}$")
    ax.set_ylabel(r"$\epsilon_{t}$")
    ax.set_title("Residual lag-1 scatter")

    # quick visual reference line y=x
    lo = np.nanmin([e0.min(), e1.min()])
    hi = np.nanmax([e0.max(), e1.max()])
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------ Main Analysis --------------------------------


def run_analysis(BAYESIAN_MODEL: str, REGION: str, SEASON: str, ZG_COORD: int):
    
    # TEMP VAR

    ds_tas_canesm = data_io.open_canesm_temperature(TEMP_VAR, ENSEMBLE_LIST)
    ds_tas_canesm = ds_tas_canesm.chunk({"time": 365})
    ds_tas_canesm = preprocess.compute_region_mean(ds_tas_canesm, REGION).compute()
    
    ds_tas_canesm_clim = ds_tas_canesm.groupby("time.dayofyear").mean("time")
    ds_tas_canesm_anom = ds_tas_canesm.groupby("time.dayofyear") - ds_tas_canesm_clim

    ds_tas_era5 = data_io.open_era5_temperature(TEMP_VAR)
    ds_tas_era5 = ds_tas_era5.chunk({"time": 365})
    ds_tas_era5 = preprocess.compute_region_mean(ds_tas_era5, REGION).compute()
    
    ds_tas_era5_clim = ds_tas_era5.groupby("time.dayofyear").mean("time")
    ds_tas_era5_anom = ds_tas_era5.groupby("time.dayofyear") - ds_tas_era5_clim

    ds_tas_canesm_anom = ds_tas_canesm_anom.assign_coords(time=ds_tas_canesm_anom.time.dt.floor("D"))
    ds_tas_era5_anom = ds_tas_era5_anom.assign_coords(time=ds_tas_era5_anom.time.dt.floor("D")) #ensure time coords match
    
    # MRSOS

    ds_mrsos_canesm = data_io.open_canesm_mrsos(var='mrsos', ensemble_list=ENSEMBLE_LIST)
    ds_mrsos_canesm = preprocess.compute_region_mean(ds_mrsos_canesm, REGION).compute()

    ds_mrsos_canesm_clim = ds_mrsos_canesm.groupby("time.dayofyear").mean("time")
    ds_mrsos_canesm_anom = ds_mrsos_canesm.groupby("time.dayofyear") - ds_mrsos_canesm_clim

    ds_mrsos_era5 = data_io.open_era5_mrsos(var='swvl1')
    ds_mrsos_era5 = preprocess.compute_region_mean(ds_mrsos_era5, REGION).compute()

    ds_mrsos_era5_clim = ds_mrsos_era5.groupby("time.dayofyear").mean("time")
    ds_mrsos_era5_anom = ds_mrsos_era5.groupby("time.dayofyear") - ds_mrsos_era5_clim
    
    ds_mrsos_canesm_anom = preprocess.floor_daily_time(ds_mrsos_canesm_anom)
    ds_mrsos_era5_anom   = preprocess.floor_daily_time(ds_mrsos_era5_anom)

    ## loading LWA data
    # 1) Open CanESM and ERA5 for all VARiables
    ds_canesm_lwas: Dict[str, xr.DataArray] = data_io.open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas: Dict[str, xr.DataArray]   = data_io.open_era5_lwa(ZG_COORD)

    ds_canesm_lwa_a = ds_canesm_lwas["LWA_a"]#.sel(MEMBER=MEMBER)
    ds_canesm_lwa_c = ds_canesm_lwas["LWA_c"]#.sel(MEMBER=MEMBER)

    ds_era5_lwa_a   = ds_era5_lwas["LWA_a"]
    ds_era5_lwa_c   = ds_era5_lwas["LWA_c"]

    ds_canesm_lwa_a_reg = load_lwa_data(ds_canesm_lwa_a, REGION)
    ds_canesm_lwa_c_reg = load_lwa_data(ds_canesm_lwa_c, REGION)

    ds_era5_lwa_a_reg   = load_lwa_data(ds_era5_lwa_a, REGION)
    ds_era5_lwa_c_reg   = load_lwa_data(ds_era5_lwa_c, REGION)
    

    season_mask_era = ds_era5_lwa_a_reg.time.dt.season == SEASON
    season_mask_can = ds_canesm_lwa_a_reg.time.dt.season == SEASON

    masked_lwa_a_era = ds_era5_lwa_a_reg.where(season_mask_era,drop=True)
    masked_lwa_a_can = ds_canesm_lwa_a_reg.where(season_mask_can,drop=True)

    masked_lwa_c_era = ds_era5_lwa_c_reg.where(season_mask_era,drop=True)
    masked_lwa_c_can = ds_canesm_lwa_c_reg.where(season_mask_can,drop=True)

    masked_tas_era = ds_tas_era5_anom.where(season_mask_era,drop=True)
    masked_tas_can = ds_tas_canesm_anom.where(season_mask_can,drop=True)

    masked_mrsos_era = ds_mrsos_era5_anom.where(season_mask_era,drop=True)
    masked_mrsos_can = ds_mrsos_canesm_anom.where(season_mask_can,drop=True)
    
    # plot_lwa_sm_correlation(
    #     masked_lwa_era=masked_lwa_a_era,
    #     masked_mrsos_era=masked_mrsos_era,
    #     masked_lwa_can=masked_lwa_a_can,
    #     masked_mrsos_can=masked_mrsos_can,
    #     LWA_var="LWA_a",
    #     REGION=REGION,
    #     SEASON=SEASON
    # )


    x_a_lo = np.min([masked_lwa_a_era.min().item(), masked_lwa_a_can.min().item()])
    x_a_hi = np.max([masked_lwa_a_era.max().item(), masked_lwa_a_can.max().item()])
    x_c_lo = np.min([masked_lwa_c_era.min().item(), masked_lwa_c_can.min().item()])
    x_c_hi = np.max([masked_lwa_c_era.max().item(), masked_lwa_c_can.max().item()])
    y_lo = np.min([masked_tas_era.min().item(), masked_tas_can.min().item()])
    y_hi = np.max([masked_tas_era.max().item(), masked_tas_can.max().item()])
    z_lo = np.min([masked_mrsos_era.min().item(), masked_mrsos_can.min().item()])
    z_hi = np.max([masked_mrsos_era.max().item(), masked_mrsos_can.max().item()])

    z_max = np.max([np.abs(z_lo), np.abs(z_hi)])

    # --- PyMC fit for ERA5: deltaT ~ LWA + SM + LWA*SM with AR(1) residual ---
    # prep standardized design arrays (align, drop NaNs, standardize)

    _lwa_a_align, _lwa_c_align, _sm_align, _dt_align = xr.align(
            masked_lwa_a_era, masked_lwa_c_era, masked_mrsos_era, masked_tas_era, join="inner"
    )
    
    # Convert to numpy and create mask for finite values across all variables
    _lwa_a_vals = _lwa_a_align.values.astype("float64")
    _lwa_c_vals = _lwa_c_align.values.astype("float64")
    _sm_vals    = _sm_align.values.astype("float64")
    _dt_vals    = _dt_align.values.astype("float64")
    _good = (
        np.isfinite(_lwa_a_vals)
        & np.isfinite(_lwa_c_vals)
        & np.isfinite(_sm_vals)
        & np.isfinite(_dt_vals)
    )
    # Extract good rows
    _lwa_a_good = _lwa_a_vals[_good]
    _lwa_c_good = _lwa_c_vals[_good]
    _sm_good    = _sm_vals[_good]
    _dt_good    = _dt_vals[_good]
    # Compute norm tuples as (mean, std) for each variable
    norm_lwaa_era = (float(_lwa_a_good.mean()), float(_lwa_a_good.std()))
    norm_lwac_era = (float(_lwa_c_good.mean()), float(_lwa_c_good.std()))
    norm_sm_era   = (float(_sm_good.mean()),   float(_sm_good.std()))
    norm_dt_era   = (float(_dt_good.mean()),   float(_dt_good.std()))

    x1, x2, x3, y = _prep_design_matrix( 
        lwa_a=masked_lwa_a_era,
        lwa_c=masked_lwa_c_era,
        sm=masked_mrsos_era,
        dt=masked_tas_era, 
        norm_lwaa=norm_lwaa_era, 
        norm_lwac=norm_lwac_era,
        norm_sm=norm_sm_era, 
        norm_dt=norm_dt_era
    )

    # loop for different model variations
    for model_name in model_variations:
        model_ = model_variations[model_name]
        spec   = BAYESIAN_MODEL_SPECS[BAYESIAN_MODEL]
        variation = Variation(model_name, 
                               include_lwa=model_['include_lwa'],
                               include_sm=model_['include_sm'],
                               include_interaction=model_.get('include_interaction', False))


        print(f"\n=== Fitting ERA5 model: {model_name} ===")

        idata_model_era = fit_from_spec(
            spec=spec,
            variation=variation,
            time=masked_tas_era.time,
            data={
                "x_lwa_a": x1,
                "x_lwa_c": x2,
                "x_sm":    x3,
                "y":       y,
            },
            draws=PYMC_NDRAW, tune=PYMC_NTUNE, chains=PYMC_CHAINS, target_accept=PYMC_TARGET_ACCEPT,
            attrs={"region": REGION, "season": SEASON, "dataset": "ERA5"},
        )

        if model_name == 'full': #only produce residual lag plots for full model
            mu_hat = compute_mu_hat(
                x1=x1, x2=x2, x3=x3, idata=idata_model_era,
                include_lwa=model_["include_lwa"],
                include_sm=model_["include_sm"],
                stat="median",
            )
            eps = y - mu_hat

            out_png = os.path.join(
                OUTPUT_PLOTS_PATH,
                f"resid_lag1_{model_name}_bothLWA_tas_sm_{REGION}_{SEASON}_ERA5.png"
            )
            plot_residual_lag1(masked_tas_era.time, eps, out_png)

        # Attach standardisation parameters into idata attributes so they can
        # be recovered later for de‑standardisation of model predictions
        idata_model_era.attrs["norm_lwa_a_mean"] = norm_lwaa_era[0]
        idata_model_era.attrs["norm_lwa_a_std"]  = norm_lwaa_era[1]
        idata_model_era.attrs["norm_lwa_c_mean"] = norm_lwac_era[0]
        idata_model_era.attrs["norm_lwa_c_std"]  = norm_lwac_era[1]
        idata_model_era.attrs["norm_sm_mean"]   = norm_sm_era[0]
        idata_model_era.attrs["norm_sm_std"]    = norm_sm_era[1]
        idata_model_era.attrs["norm_dt_mean"]   = norm_dt_era[0]
        idata_model_era.attrs["norm_dt_std"]    = norm_dt_era[1]

        context_args = {
            "dataset": "ERA5",
            "region": REGION,
            "season": SEASON,
            "lwa_var": "bothLWA"
        }

        out_path = (
            f"{OUTPUT_POSTERIORS_PATH}/"
            f"ERA5_{REGION}_{SEASON}_{spec.key}_{variation.name}.nc"
        )

        save_idata(idata=idata_model_era, 
                   out_nc=out_path,
                   spec=spec, 
                   variation=variation,
                   context=context_args)

        
    # --- PyMC loop-per-member for CanESM5 ---
    members_to_run = masked_tas_can.member.values

    norm_lwaa = (float(masked_lwa_a_can.mean()), float(masked_lwa_a_can.std()))
    norm_lwac = (float(masked_lwa_c_can.mean()), float(masked_lwa_c_can.std()))
    norm_sm   = (float(masked_mrsos_can.mean()), float(masked_mrsos_can.std()))
    norm_dt   = (float(masked_tas_can.mean()),   float(masked_tas_can.std()))

    for m in members_to_run:
        print(f"\n=== Fitting member {m} ===")

        # Select 1D time series for this member
        lwa_a_m = masked_lwa_a_can.sel(member=m)
        lwa_c_m = masked_lwa_c_can.sel(member=m)
        sm_m    = masked_mrsos_can.sel(member=m)
        dt_m    = masked_tas_can.sel(member=m)

        # Prep standardized design arrays (align, drop NaNs, standardize)
        x1, x2, x3, y = _prep_design_matrix(lwa_a=lwa_a_m, lwa_c=lwa_c_m, 
                                            sm=sm_m, dt=dt_m,
                                            norm_lwaa=norm_lwaa,
                                            norm_lwac=norm_lwac,
                                            norm_sm=norm_sm,
                                            norm_dt=norm_dt
        )

        # Loop for different model variations
        for model_name in model_variations:
            model_ = model_variations[model_name]
            spec = BAYESIAN_MODEL_SPECS[BAYESIAN_MODEL]
            variation = Variation(model_name, 
                                include_lwa=model_['include_lwa'],
                                include_sm=model_['include_sm'],
                                include_interaction=model_.get('include_interaction', False))
            
            print(f"\n--- Fitting member {m} model: {model_name} ---")
            idata_model_m = fit_from_spec(
                spec=spec,
                variation=variation,
                time=masked_tas_can.time,
                data={
                    "x_lwa_a": x1,
                    "x_lwa_c": x2,
                    "x_sm":    x3,
                    "y":       y,
                },
                draws=PYMC_NDRAW, tune=PYMC_NTUNE, chains=PYMC_CHAINS,
                target_accept=PYMC_TARGET_ACCEPT,
                attrs={"region": REGION, "season": SEASON, "dataset": f"CanESM5_{str(m)}"},
        )

            if model_name == 'full': #only produce residual lag plots for full model
                mu_hat = compute_mu_hat(
                    x1=x1, x2=x2, x3=x3, idata=idata_model_m,
                    include_lwa=model_["include_lwa"],
                    include_sm=model_["include_sm"],
                    stat="median",
                )
                eps = y - mu_hat

                out_png = os.path.join(
                    OUTPUT_PLOTS_PATH,
                    f"resid_lag1_{model_name}_bothLWA_tas_sm_{REGION}_{SEASON}_CanESM5_{str(m)}.png"
                )
                plot_residual_lag1(masked_tas_era.time, eps, out_png)


            # Attach standardisation parameters into the idata attributes so that
            # de‑standardisation can later be performed.  These norms are
            # computed over all members (global) and used for standardising
            # the design matrix.
            idata_model_m.attrs["norm_lwa_a_mean"] = norm_lwaa[0]
            idata_model_m.attrs["norm_lwa_a_std"]  = norm_lwaa[1]
            idata_model_m.attrs["norm_lwa_c_mean"] = norm_lwac[0]
            idata_model_m.attrs["norm_lwa_c_std"]  = norm_lwac[1]
            idata_model_m.attrs["norm_sm_mean"]    = norm_sm[0]
            idata_model_m.attrs["norm_sm_std"]     = norm_sm[1]
            idata_model_m.attrs["norm_dt_mean"]    = norm_dt[0]
            idata_model_m.attrs["norm_dt_std"]     = norm_dt[1]

            context_args = {
                "dataset": f"CanESM5_{str(m)}",
                "region": REGION,
                "season": SEASON,
                "lwa_var": "bothLWA"
            }

            out_path = (
                f"{OUTPUT_POSTERIORS_PATH}/"
                f"CanESM5_{str(m)}_{REGION}_{SEASON}_{spec.key}_{variation.name}.nc"
            )

            save_idata(idata=idata_model_m, 
                       out_nc=out_path,
                       spec=spec, 
                       variation=variation,
                       context=context_args
            )

    print("\nDone. Member-level summaries collected.")

if __name__ == "__main__":
    args = arg_parser()
    BAYESIAN_MODEL = args.bayesian_model
    REGION         = args.region
    SEASON         = args.season
    ZG_LEVEL       = args.zg

    OUTPUT_POSTERIORS_PATH = os.path.join(config.OUTPUT_PATH, f"pymc_fits/{BAYESIAN_MODEL}/idata")
    os.makedirs(OUTPUT_POSTERIORS_PATH, exist_ok=True)

    OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, f"plots/{BAYESIAN_MODEL}")
    os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

    run_analysis(BAYESIAN_MODEL, REGION, SEASON, ZG_LEVEL)
