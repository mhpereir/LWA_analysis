import os
import sys
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

# Define script specific constants

# Model variations
_full     = {'include_lwa': True, 'include_sm': True, 'include_interaction': True}
_no_int   = {'include_lwa': True, 'include_sm': True, 'include_interaction': False}
_sm_only  = {'include_lwa': False, 'include_sm': True, 'include_interaction': False}
_lwa_only = {'include_lwa': True, 'include_sm': False, 'include_interaction': False}

model_variations = {
    "full":     _full,
    "no_int":   _no_int,
    "sm_only":  _sm_only,
    "lwa_only": _lwa_only,
}

OUTPUT_POSTERIORS_PATH = os.path.join(config.OUTPUT_PATH, "pymc_fits")
os.makedirs(OUTPUT_POSTERIORS_PATH, exist_ok=True)

OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/bayesian_lwa_dt_sm_correlation")
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)


TEMP_VAR     = config.TEMP_VAR
ENSEMBLE_LIST = config.ENSEMBLE_LIST

PYMC_NDRAW:int           = config_pymc.PYMC_NDRAW
PYMC_NTUNE:int           = config_pymc.PYMC_NTUNE
PYMC_TARGET_ACCEPT:float = config_pymc.PYMC_TARGET_ACCEPT
PYMC_CHAINS:int          = config_pymc.PYMC_CHAINS


# Define argument parser

def arg_parser():
    parser = argparse.ArgumentParser(
        description="LWA vs deltaT correlation analysis and plotting."
    )
    parser.add_argument(
        "--lwa_var",
        type=str,
        choices=config.LWA_VARS,
        default="LWA",
        help="LWA variable to analyze.",
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
    lwa: xr.DataArray,
    sm: xr.DataArray,
    dt: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return centered+standardized (X1, X2, y) as numpy arrays, aligned on time.
    Assumes inputs are 1D with coord 'time' and no 'member' dimension.
    """
    # Align on time
    lwa, sm, dt = xr.align(lwa, sm, dt, join="inner")

    # Drop NaNs
    good = np.isfinite(lwa.values) & np.isfinite(sm.values) & np.isfinite(dt.values)
    lwa = lwa.isel(time=good)
    sm  = sm.isel(time=good)
    dt  = dt.isel(time=good)

    # Center + standardize (within the already-cropped season)
    x1 = lwa.values.astype("float64")
    x2 = sm.values.astype("float64")
    y  = dt.values.astype("float64")

    x1 = (x1 - x1.mean()) / x1.std()
    x2 = (x2 - x2.mean()) / x2.std()
    y  = (y  - y.mean())  / y.std()

    return x1, x2, y


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


def fit_pymc_ar1_model(
    time: xr.DataArray,
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    include_lwa: bool = True,
    include_sm: bool = True,
    include_interaction: bool = True,
    draws: int = PYMC_NDRAW,
    tune: int = PYMC_NTUNE,
    target_accept: float = PYMC_TARGET_ACCEPT,
    chains: int = PYMC_CHAINS,
) -> az.InferenceData:

    years     = time.dt.year.values.astype(np.int32)
    same_year = np.concatenate([[0], (years[1:] == years[:-1]).astype(np.int8)])

    with pm.Model() as model:
        x1d = pm.Data("x1", x1)
        x2d = pm.Data("x2", x2)
        yd  = pm.Data("y",  y)
        syd = pm.Data("same_year", same_year)

        # Always define params, but optionally gate their contribution
        b0 = pm.Normal("b0", 0.0, 1.0)
        b1 = pm.Normal("b1", 0.0, 1.0)
        b2 = pm.Normal("b2", 0.0, 1.0)
        b3 = pm.Normal("b3", 0.0, 1.0)

        rho = pm.Uniform("rho", lower=-0.99, upper=0.99)
        nu = pm.Exponential("nu_minus_2", 1/10) + 2.0
        sigma = pm.HalfNormal("sigma", 1.0)

        x12 = x1d * x2d # type: ignore

        # build deterministic mean based on included terms
        mu = b0
        if include_lwa:
            mu += b1 * x1d
        if include_sm:
            mu += b2 * x2d
        if include_interaction:
            mu += b3 * x12

        # ------------------------------------------------------------------
        # Build lag-1 versions of the observed data and the deterministic mean
        # ------------------------------------------------------------------
        # y_prev[t]  = y[t-1]   (with a dummy value at t=0)
        # mu_prev[t] = mu[t-1]  (same dummy initialization)
        #
        # These are used to introduce AR(1) structure on the *residuals*,
        # not on the raw data or on the deterministic mean itself.
        y_prev  = pt.concatenate([yd[:1], yd[:-1]])  # type: ignore
        mu_prev = pt.concatenate([mu[:1], mu[:-1]])  # type: ignore


        # ------------------------------------------------------------------
        # Conditional mean of y_t given y_{t-1}
        # ------------------------------------------------------------------
        # cond_mu[t] = mu_t + rho * same_year[t] * (y_{t-1} - mu_{t-1})
        #
        # Interpretation:
        #   - mu_t is the deterministic, physics-based expectation at time t
        #   - (y_{t-1} - mu_{t-1}) is the realized residual at time t-1
        #   - rho controls persistence of residual anomalies
        #   - same_year prevents persistence across year boundaries
        #
        # Importantly:
        #   - mu_t contains NO time dependence
        #   - all temporal correlation lives in the residual term
        cond_mu = mu + rho * syd * (y_prev - mu_prev)  # type: ignore


        # ------------------------------------------------------------------
        # Likelihood: data model
        # ------------------------------------------------------------------
        # This line DOES NOT define a new noise variable.
        # It states that the observed data yd are distributed as a Student-t
        # random variable centered on cond_mu, with scale sigma and dof nu.
        #
        # In equation form:
        #   y_t = cond_mu_t + eta_t
        #   eta_t ~ StudentT(0, sigma, nu)
        #
        # The per-time-step noise eta_t is *implicit* in the likelihood.
        # PyMC uses this distribution to evaluate the log-probability of the
        # observed data given the model parameters.
        pm.StudentT(
            "y_like",
            nu=nu,
            mu=cond_mu,
            sigma=sigma,
            observed=yd
        )


        # ------------------------------------------------------------------
        # Posterior sampling
        # ------------------------------------------------------------------
        # pm.sample() draws samples from the *joint posterior* of all model
        # parameters (b0, b1, b2, b3, rho, sigma, nu), conditioned on:
        #   - the priors specified earlier in the model
        #   - the likelihood defined above
        #
        # No sampling of y_like occurs here: yd is observed and fixed.
        # The sampler explores parameter values that make the observed data
        # likely under the Student-t likelihood centered on cond_mu.
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            progressbar=True
        )


    return idata


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


# ------------------------------ Main Analysis --------------------------------


def run_analysis(REGION: str, LWA_var: str, SEASON: str, ZG_COORD: int):
    
    #load in data

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
    
    ds_mrsos_canesm = data_io.open_canesm_mrsos(var='mrsos', ensemble_list=ENSEMBLE_LIST)
    ds_mrsos_canesm = preprocess.compute_region_mean(ds_mrsos_canesm, REGION).compute()

    ds_mrsos_canesm_clim = ds_mrsos_canesm.groupby("time.dayofyear").mean("time")
    ds_mrsos_canesm_anom = ds_mrsos_canesm.groupby("time.dayofyear") - ds_mrsos_canesm_clim

    ds_mrsos_era5 = data_io.open_era5_mrsos(var='swvl1')
    ds_mrsos_era5 = preprocess.compute_region_mean(ds_mrsos_era5, REGION).compute()

    ds_mrsos_era5_clim = ds_mrsos_era5.groupby("time.dayofyear").mean("time")
    ds_mrsos_era5_anom = ds_mrsos_era5.groupby("time.dayofyear") - ds_mrsos_era5_clim

    ## loading LWA data
    # 1) Open CanESM and ERA5 for all VARiables
    ds_canesm_lwas: Dict[str, xr.DataArray] = data_io.open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas: Dict[str, xr.DataArray]   = data_io.open_era5_lwa(ZG_COORD)

    ds_canesm_lwa = ds_canesm_lwas[LWA_var]#.sel(MEMBER=MEMBER)
    ds_era5_lwa   = ds_era5_lwas[LWA_var]

    ds_canesm_lwa_reg = preprocess.compute_region_mean(ds_canesm_lwa, REGION).chunk({"time": 365}).compute()
    ds_era5_lwa_reg   = preprocess.compute_region_mean(ds_era5_lwa, REGION).chunk({"time": 365}).compute()

    ds_canesm_lwa_reg = xr.apply_ufunc(np.sqrt, ds_canesm_lwa_reg)
    ds_era5_lwa_reg   = xr.apply_ufunc(np.sqrt, ds_era5_lwa_reg)

    ds_canesm_lwa_reg = ds_canesm_lwa_reg.assign_coords(time=ds_canesm_lwa_reg.time.dt.floor("D"))
    ds_era5_lwa_reg = ds_era5_lwa_reg.assign_coords(time=ds_era5_lwa_reg.time.dt.floor("D"))

    ds_mrsos_canesm_anom = ds_mrsos_canesm_anom.assign_coords(time=ds_mrsos_canesm_anom.time.dt.floor("D"))
    ds_mrsos_era5_anom   = ds_mrsos_era5_anom.assign_coords(time=ds_mrsos_era5_anom.time.dt.floor("D"))

    season_mask_era = ds_era5_lwa_reg.time.dt.season == SEASON
    season_mask_can = ds_canesm_lwa_reg.time.dt.season == SEASON

    masked_lwa_era = ds_era5_lwa_reg.where(season_mask_era,drop=True)
    masked_lwa_can = ds_canesm_lwa_reg.where(season_mask_can,drop=True)

    masked_tas_era = ds_tas_era5_anom.where(season_mask_era,drop=True)
    masked_tas_can = ds_tas_canesm_anom.where(season_mask_can,drop=True)

    masked_mrsos_era = ds_mrsos_era5_anom.where(season_mask_era,drop=True)
    masked_mrsos_can = ds_mrsos_canesm_anom.where(season_mask_can,drop=True)

    
    plot_lwa_sm_correlation(
        masked_lwa_era=masked_lwa_era,
        masked_mrsos_era=masked_mrsos_era,
        masked_lwa_can=masked_lwa_can,
        masked_mrsos_can=masked_mrsos_can,
        LWA_var=LWA_var,
        REGION=REGION,
        SEASON=SEASON
    )


    x_lo = np.min([masked_lwa_era.min().item(), masked_lwa_can.min().item()])
    x_hi = np.max([masked_lwa_era.max().item(), masked_lwa_can.max().item()])
    y_lo = np.min([masked_tas_era.min().item(), masked_tas_can.min().item()])
    y_hi = np.max([masked_tas_era.max().item(), masked_tas_can.max().item()])
    z_lo = np.min([masked_mrsos_era.min().item(), masked_mrsos_can.min().item()])
    z_hi = np.max([masked_mrsos_era.max().item(), masked_mrsos_can.max().item()])

    z_max = np.max([np.abs(z_lo), np.abs(z_hi)])

    # --- PyMC fit for ERA5: deltaT ~ LWA + SM + LWA*SM with AR(1) residual ---
    # prep standardized design arrays (align, drop NaNs, standardize)
    x1, x2, y = _prep_design_matrix( 
        lwa=masked_lwa_era,
        sm=masked_mrsos_era,
        dt=masked_tas_era
    )

    # loop for different model variations
    for model_name in model_variations:
        model_ = model_variations[model_name]
        print(f"\n=== Fitting ERA5 model: {model_name} ===")
        idata_model_era = fit_pymc_ar1_model(
            time=masked_tas_era.time,
                x1=x1, x2=x2, y=y,
                include_lwa=model_['include_lwa'],
                include_sm=model_['include_sm'],
                include_interaction=model_['include_interaction'],
            draws=2000, tune=2000,
            target_accept=0.9
        )

        print(az.summary(idata_model_era, var_names=["b0","b1","b2","b3","rho","sigma","nu_minus_2"]))
        az.to_netcdf(idata_model_era, f"{OUTPUT_POSTERIORS_PATH}/pymc_ar1_{model_name}_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5.nc")


    # --- PyMC loop-per-member for CanESM5 ---

    # # Ensure consistent dim order (member, time) for easy indexing
    # def _ensure_member_time(da: xr.DataArray) -> xr.DataArray:
    #     if "member" not in da.dims or "time" not in da.dims:
    #         raise ValueError(f"Expected dims to include ('member','time'), got {da.dims}")
    #     # transpose if needed
    #     if da.dims != ("member", "time"):
    #         da = da.transpose("member", "time")
    #     return da

    # masked_lwa_can   = _ensure_member_time(masked_lwa_can)
    # masked_mrsos_can = _ensure_member_time(masked_mrsos_can)
    # masked_tas_can   = _ensure_member_time(masked_tas_can)

    # Optional: choose a subset of members for testing
    # members_to_run = masked_tas_can.member.values[:5]
    members_to_run = masked_tas_can.member.values

    summaries = []

    for m in members_to_run:
        print(f"\n=== Fitting member {m} ===")

        # Select 1D time series for this member
        lwa_m = masked_lwa_can.sel(member=m)
        sm_m  = masked_mrsos_can.sel(member=m)
        dt_m  = masked_tas_can.sel(member=m)

        # Prep standardized design arrays (align, drop NaNs, standardize)
        x1, x2, y = _prep_design_matrix(lwa=lwa_m, sm=sm_m, dt=dt_m)

        # Loop for different model variations
        for model_name in model_variations:
            model_ = model_variations[model_name]
            print(f"\n--- Fitting member {m} model: {model_name} ---")
            idata_model_m = fit_pymc_ar1_model(
                time=dt_m.time,   # use the member’s time coord (same as others after align)
                x1=x1, x2=x2, y=y,
                include_lwa=model_['include_lwa'],
                include_sm=model_['include_sm'],
                include_interaction=model_['include_interaction'],
                draws=2000, tune=2000,
                target_accept=0.9,
                chains=4
            )

            print(az.summary(idata_model_m, var_names=["b0","b1","b2","b3","rho","sigma","nu_minus_2"]))
            az.to_netcdf(
                idata_model_m,
                os.path.join(
                    OUTPUT_POSTERIORS_PATH,
                    f"pymc_ar1_{model_name}_{LWA_var}_tas_sm_{REGION}_{SEASON}_CanESM5_{str(m)}.nc"
                )
            )

            # Collect a compact summary row
            if model_name == 'full':
                s = az.summary(idata_model_m, var_names=["b0","b1","b2","b3","rho","sigma","nu_minus_2"])
                s["member"] = str(m)
                summaries.append(s.reset_index().rename(columns={"index": "param"})) #type: ignore
            else:
                pass

        # # Fit PyMC AR(1) interaction model (whitened likelihood by year segments)
        # idata_full_m = fit_pymc_ar1_model(
        #     time=dt_m.time,   # IMPORTANT: use the member’s time coord (same as others after align)
        #     x1=x1, x2=x2, y=y,
        #     include_lwa=True,
        #     include_sm=True,
        #     include_interaction=True,
        #     draws=2000, tune=2000,
        #     target_accept=0.9,
        #     chains=4
        # )

        # # Save posterior to disk (NetCDF)
        # out_nc = os.path.join(
        #     OUTPUT_POSTERIOR,
        #     f"pymc_ar1_{LWA_var}_tas_sm_{REGION}_{SEASON}_CanESM5_{str(m)}.nc"
        # )
        # az.to_netcdf(idata_full_m, out_nc)

        

    # Combine summaries into a single table (easy to plot / compare)
    summary_df = xr.Dataset.from_dataframe(
        xr.concat([xr.Dataset.from_dataframe(df.set_index(["member","param"])) for df in summaries], dim="dummy")
        .to_dataframe()
    ).to_dataframe()  # optional; you may prefer pandas directly


    summary_outfile = os.path.join(
        OUTPUT_POSTERIORS_PATH,
        f"pymc_ar1_full_model_summaries_{LWA_var}_tas_sm_{REGION}_{SEASON}_CanESM5.csv"
    )
    summary_df.to_csv(summary_outfile)

    print("\nDone. Member-level summaries collected.")




if __name__ == "__main__":
    args = arg_parser()
    REGION   = args.region
    LWA_var  = args.lwa_var
    SEASON   = args.season
    ZG_LEVEL = args.zg

    run_analysis(REGION, LWA_var, SEASON, ZG_LEVEL)
