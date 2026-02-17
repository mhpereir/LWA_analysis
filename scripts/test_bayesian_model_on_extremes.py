import os
import sys
import glob
import argparse


# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = "/home/mhpereir/LWA_analysis"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import pymc as pm
import arviz as az

from src import config, config_pymc, preprocess, data_io
from typing import Dict, List, Optional


MODEL_VARIATIONS  = config_pymc.model_variations
OUTPUT_POSTERIORS = os.path.join(config.OUTPUT_PATH, "pymc_fits_noAR_bothLWA")

ENSEMBLE_LIST = config.ENSEMBLE_LIST
TEMP_VAR      = config.TEMP_VAR

# Create output directory for diagnostic plots
diag_dir = os.path.join(config.OUTPUT_PATH, "plots", "bayesian_extreme_diagnostics")
os.makedirs(diag_dir, exist_ok=True)

# ----------------------------- Data loading functions -----------------------------

def open_posterior_file(filepath: str) -> az.InferenceData:
    """Open a PyMC posterior InferenceData from NetCDF."""
    idata = az.from_netcdf(filepath)
    return idata

# -----------------------------------------------------------------------------
# Event/composite helper functions

def _process_mask_1d(arr: np.ndarray) -> np.ndarray:
    """Return an array of event IDs for a 1D boolean array.

    Contiguous True segments are assigned sequential IDs starting from 1; False
    values are given 0.  This helper does not consider season and is
    vectorised over the time dimension only.
    """
    event_ids = np.zeros_like(arr, dtype=int)
    in_event = False
    event_id = 0
    for i in range(len(arr)):
        if arr[i] and not in_event:
            in_event = True
            event_id += 1
        elif not arr[i] and in_event:
            in_event = False
        if in_event:
            event_ids[i] = event_id
    return event_ids


def mask_to_events(mask: xr.DataArray) -> xr.DataArray:
    """Convert a boolean mask (True=event) into event IDs.

    For each contiguous True segment along the 'time' dimension, assign a
    sequential integer event ID. Works for both single‑time‑series arrays and
    arrays with a 'member' dimension. Non‑event points are given ID 0.
    """
    if "time" not in mask.dims:
        raise ValueError("mask_to_events expects a 'time' dimension")

    # Robustness: keep this step eager to avoid dask gufunc edge cases with
    # zero-length chunks after seasonal subsetting.
    mask = mask.fillna(False).astype(bool).compute()

    if mask.sizes["time"] == 0:
        return xr.zeros_like(mask, dtype=int).rename("event_id")

    if "member" in mask.dims:
        out_datarray = xr.apply_ufunc(
            _process_mask_1d,
            mask,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
        ).rename("event_id")
        return out_datarray

    event_ids = _process_mask_1d(np.asarray(mask))
    out_datarray = xr.DataArray(
        event_ids,
        coords=mask.coords,
        dims=mask.dims,
        name="event_id",
    )
    return out_datarray


def _filter_events_1d(event_ids: np.ndarray, seasons: np.ndarray, target_season: str) -> np.ndarray:
    """Filter events so that only those entirely within target_season are kept.

    Returns a new array of event IDs where events not fully contained within
    `target_season` are set to 0.  This helper operates on a single 1D array of
    event IDs and a parallel array of season labels.  It is vectorised over
    time; non‑matching events are removed.
    """
    raw_ids = np.unique(event_ids)
    
    valid_ids = []
    for eid in raw_ids:
        if eid <= 0:
            continue
        uniq = np.unique(seasons[event_ids == eid])
        if (len(uniq) == 1) and (uniq[0] == target_season):
            valid_ids.append(eid)
    valid_ids = np.asarray(valid_ids, dtype=int)
    return np.where(np.isin(event_ids, valid_ids), event_ids, 0)


def filter_events_by_season(events: xr.DataArray, target_season: str) -> xr.DataArray:
    """Keep only those events whose entire lifetime occurs within `target_season`.

    Works for arrays with or without a 'member' dimension.  Events with any
    component outside the specified season are dropped (set to 0).
    """

    event_season = events.time.dt.season
    filtered = xr.apply_ufunc(
        _filter_events_1d,
        events,
        event_season,
        kwargs={"target_season": target_season},
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
    ).rename(events.name or "event_id")
    return filtered


def _composite_1d(
    tas_series: xr.DataArray,
    lwa_series: xr.DataArray,
    event_ids: xr.DataArray,
    half_window: int,
    n_day_min: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute composite windows centred on maximum LWA within each event.

    For each event ID in `event_ids`, find the peak day within that event in
    `lwa_series`, then extract a window of length `2*half_window+1` around
    that day from `tas_series` and `lwa_series`.  Windows that would extend
    beyond the boundaries of the data are discarded.  Only events with a
    duration longer than `n_day_min` days are considered.  Returns the mean
    composite for the temperature series and for the LWA series respectively.
    If no valid windows exist, returns (None, None).
    """
    tas_vals = np.asarray(tas_series)
    lwa_vals = np.asarray(lwa_series)
    event_vals = np.asarray(event_ids, dtype=int)
    if tas_vals.size == 0:
        return None, None
    tas_windows: List[np.ndarray] = []
    lwa_windows: List[np.ndarray] = []
    for event_id in np.unique(event_vals):
        if event_id <= 0:
            continue
        event_idx = np.where(event_vals == event_id)[0]
        if event_idx.size <= n_day_min:
            continue
        peak_idx = event_idx[np.argmax(lwa_vals[event_idx])]
        start = peak_idx - half_window
        end = peak_idx + half_window + 1
        if start < 0 or end > tas_vals.size:
            continue  # skip events that cannot supply a full window
        tas_windows.append(tas_vals[start:end])
        lwa_windows.append(lwa_vals[start:end])
    if not tas_windows:
        return None, None
    tas_mean = np.stack(tas_windows, axis=0).mean(axis=0)
    lwa_mean = np.stack(lwa_windows, axis=0).mean(axis=0)
    return tas_mean, lwa_mean


def _process_dataset(
    tas_da: xr.DataArray,
    lwa_da: xr.DataArray,
    event_da: xr.DataArray,
    lag_coord: np.ndarray,
    half_window: int,
    n_day_min: int,
) -> tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Compute composites for datasets with or without a 'member' dimension.

    This function wraps `_composite_1d` to operate on `xarray.DataArray`
    objects.  If a 'member' dimension is present, composites are computed
    independently for each member.  The resulting composites are returned as
    DataArrays with a 'lag_day' coordinate (and a 'member' dimension where
    appropriate).
    """
    if "member" in tas_da.dims:
        tas_member_composites = []
        lwa_member_composites = []
        member_labels = []
        for member in tas_da.member.values:
            tas_comp, lwa_comp = _composite_1d(
                tas_da.sel(member=member),
                lwa_da.sel(member=member),
                event_da.sel(member=member),
                half_window,
                n_day_min,
            )
            if tas_comp is None or lwa_comp is None:
                continue
            tas_member_composites.append(tas_comp)
            lwa_member_composites.append(lwa_comp)
            member_labels.append(member)
        if not tas_member_composites:
            return None, None
        tas_data = np.stack(tas_member_composites, axis=0)
        lwa_data = np.stack(lwa_member_composites, axis=0)
        tas_da_out = xr.DataArray(
            tas_data,
            coords={"member": member_labels, "lag_day": lag_coord},
            dims=("member", "lag_day"),
        )
        lwa_da_out = xr.DataArray(
            lwa_data,
            coords={"member": member_labels, "lag_day": lag_coord},
            dims=("member", "lag_day"),
        )
        return tas_da_out, lwa_da_out
    else:
        tas_comp, lwa_comp = _composite_1d(
            tas_da,
            lwa_da,
            event_da,
            half_window,
            n_day_min,
        )
        if tas_comp is None or lwa_comp is None:
            return None, None
        tas_da_out = xr.DataArray(
            tas_comp,
            coords={"lag_day": lag_coord},
            dims=("lag_day",),
        )
        lwa_da_out = xr.DataArray(
            lwa_comp,
            coords={"lag_day": lag_coord},
            dims=("lag_day",),
        )
        return tas_da_out, lwa_da_out


# ----------------------------- Helper functions -----------------------------

def posterior_samples(idata: az.InferenceData, var: str) -> np.ndarray:
    """Return 1D numpy array of posterior samples for var."""
    da = idata.posterior[var] #type: ignore
    return da.stack(sample=("chain", "draw")).values.astype("float64")


def load_lwa_data(ds_lwa_var, region):
    ds_lwa_reg = preprocess.compute_region_mean(ds_lwa_var, region).chunk({"time": 365}).compute()
    
    ds_sqrt_lwa_reg = xr.apply_ufunc(np.sqrt, ds_lwa_reg)
   
    ds_sqrt_lwa_reg = ds_sqrt_lwa_reg.assign_coords(time=ds_sqrt_lwa_reg.time.dt.floor("D"))

    return ds_sqrt_lwa_reg


# SELECT MEDIAN OF POSTERIORS
# -------------------------------------------------------------------------
# Compute model predictions and diagnostic plots
#
# We want to compare the observed temperature anomaly (dT_data) during
# extreme hot events against the predictions from our Bayesian regression
# models (dT_model).  To do so we:
#   1. Compute the posterior median coefficients from each fitted model.
#   2. Construct non‑AR predictions (mu) using masked LWA_a, LWA_c and
#      soil moisture anomalies.  For now we use dummy means and standard
#      deviations (0 and 1) to de‑standardise the predictors; these should
#      be replaced with the actual values stored in the ArviZ files once
#      available.
#   3. Identify extreme hot events using the masks created above, convert
#      them to event IDs and filter by season.
#   4. For each model variation, build composites of dT_data and dT_model
#      centred on the peak LWA_a day within each event, and plot the
#      results.
#   5. Produce scatter plots of dT_model vs dT_data for all hot days.



# Define helper to compute posterior medians of coefficients
def posterior_median(idata: az.InferenceData, var: str) -> float:
    vals = idata.posterior[var].stack(sample=("chain", "draw")).values.astype("float64") # type: ignore
    return float(np.median(vals))

# We will extract de‑standardisation parameters (means and standard
# deviations) from the model's ArviZ InferenceData attributes.  These
# parameters were saved during model training and allow us to transform
# standardized model predictions back into physical units.  If for some
# reason these attributes are missing, sensible defaults (mean=0,
# std=1) will be used.

# Function to compute dT_model = mu (non‑AR component) for a given
# inference data object and predictor arrays.  If a coefficient is not
# present in the posterior, its contribution is set to zero.
def compute_dt_model(idata: az.InferenceData,
                        lwa_a: xr.DataArray,
                        lwa_c: xr.DataArray,
                        sm: xr.DataArray) -> xr.DataArray:
    """Compute the non‑AR component of the modelled temperature anomaly.

    This function extracts posterior median coefficients from `idata` and
    applies them to the predictor arrays.  Predictor variables are
    standardized using the mean and standard deviation stored in the
    InferenceData attributes (saved during training).  The predicted
    temperature anomaly is then de‑standardised using the stored target
    mean and standard deviation.
    """
    # Extract medians if they exist
    coeffs: Dict[str, float] = {}
    for name in ["b0", "b1", "b2", "b3"]:
        if name in idata.posterior:  # type: ignore
            coeffs[name] = posterior_median(idata, name)
    # Base intercept
    mu = xr.zeros_like(lwa_a)  # inherit coords/dims
    b0 = coeffs.get("b0", 0.0)
    mu = mu + b0
    # Retrieve required standardisation parameters from idata attributes.
    required_attrs = [
        "norm_lwa_a_mean",
        "norm_lwa_a_std",
        "norm_lwa_c_mean",
        "norm_lwa_c_std",
        "norm_sm_mean",
        "norm_sm_std",
        "norm_dt_mean",
        "norm_dt_std",
    ]
    missing = [k for k in required_attrs if k not in idata.attrs]  # type: ignore
    if missing:
        raise KeyError(
            f"Missing required idata.attrs keys for de-standardisation: {missing}"
        )

    mean_lwa_a = float(idata.attrs["norm_lwa_a_mean"])  # type: ignore
    std_lwa_a  = float(idata.attrs["norm_lwa_a_std"])  # type: ignore
    mean_lwa_c = float(idata.attrs["norm_lwa_c_mean"])  # type: ignore
    std_lwa_c  = float(idata.attrs["norm_lwa_c_std"])  # type: ignore
    mean_sm = float(idata.attrs["norm_sm_mean"])  # type: ignore
    std_sm  = float(idata.attrs["norm_sm_std"])  # type: ignore
    mean_dt = float(idata.attrs["norm_dt_mean"])  # type: ignore
    std_dt  = float(idata.attrs["norm_dt_std"])  # type: ignore
    # Contributions from predictors
    if "b1" in coeffs:
        # Standardise LWA_a
        x = (lwa_a - mean_lwa_a) / std_lwa_a
        mu = mu + coeffs["b1"] * x
    if "b2" in coeffs:
        x2 = (lwa_c - mean_lwa_c) / std_lwa_c
        mu = mu + coeffs["b2"] * x2
    # Contribution from soil moisture
    if "b3" in coeffs:
        x3 = (sm - mean_sm) / std_sm
        mu = mu + coeffs["b3"] * x3
    # Finally de‑standardise predicted dT
    dt = mu * std_dt + mean_dt
    return dt

# ------------------------------ Thresholds functions ------------------------


def smooth_doy_threshold(da_doy: xr.DataArray, win: int = 7) -> xr.DataArray:
    return da_doy.rolling(dayofyear=win, center=True).mean()

def build_extreme_mask(series: xr.DataArray, thresh_doy: xr.DataArray, is_hot: bool) -> xr.DataArray:
    if is_hot:
        return series.groupby("time.dayofyear") > thresh_doy
    else:
        return series.groupby("time.dayofyear") < thresh_doy


def align_mask_to_da(mask: xr.DataArray, da: xr.DataArray) -> xr.DataArray:
    mask = preprocess.floor_daily_time(mask)
    da = preprocess.floor_daily_time(da)
    mask = mask.reindex(time=da["time"]).astype(bool)
    return mask


def composite_on_mask(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    mask = align_mask_to_da(mask, da)
    ntrue = int(mask.sum().compute() if hasattr(mask.data, "compute") else mask.sum())
    if ntrue == 0:
        print("[warn] composite_on_mask: mask has zero True after time alignment")
    return da.where(mask).mean("time")


def prep_doy_threshold(thr: xr.DataArray, series: xr.DataArray) -> xr.DataArray:
    # Drop singleton time dim if present (your CanESM thresh has time:1)
    if "time" in thr.dims:
        if thr.sizes["time"] != 1:
            raise ValueError(f"Expected threshold time dim to be size 1, got {thr.sizes['time']}")
        thr = thr.squeeze("time", drop=True)

    # Align members if both have member dim
    if "member" in series.dims and "member" in thr.dims:
        thr = thr.reindex(member=series["member"])

    return thr


# ------------------------------- Main analysis ---------------------------

def run_analysis(BAYESIAN_MODEL, LWA_var, REGION, SEASON, ZG_COORD, Q_HOT):

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

    ds_tas_canesm_anom = preprocess.floor_daily_time(ds_tas_canesm_anom)
    ds_tas_era5_anom = preprocess.floor_daily_time(ds_tas_era5_anom)  #ensure time coords match
    

    # TEMP VAR THRESHOLD

    thr_hot_can = smooth_doy_threshold(
        data_io.open_canesm_hw_thresh(config.TEMP_VAR, Q_HOT, REGION, ENSEMBLE_LIST)
    )
    thr_hot_era = smooth_doy_threshold(data_io.open_era5_hw_thresh(config.TEMP_VAR, Q_HOT, REGION))

    thr_hot_can = prep_doy_threshold(thr_hot_can, ds_tas_canesm)
    thr_hot_era = prep_doy_threshold(thr_hot_era, ds_tas_era5)  # not needed, but consistent

    tmean_canesm = preprocess.select_season_time(ds_tas_canesm, SEASON)
    tmean_era5   = preprocess.select_season_time(ds_tas_era5, SEASON)

    mask_hot_can = build_extreme_mask(tmean_canesm, thr_hot_can, is_hot=True) # mask of hot days
    mask_hot_era = build_extreme_mask(tmean_era5, thr_hot_era, is_hot=True)

    # xarray does not allow boolean dask indexers with drop=True; materialize once.
    mask_hot_can = mask_hot_can.compute()
    mask_hot_era = mask_hot_era.compute()

    mask_hot_can = preprocess.floor_daily_time(mask_hot_can)
    mask_hot_era = preprocess.floor_daily_time(mask_hot_era)

    print(np.sum(mask_hot_can.values), len(tmean_canesm.time), "hot days in CanESM")
    print(np.sum(mask_hot_era.values), len(tmean_era5.time), "hot days in ERA5")


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

    season_lwa_a_era = ds_era5_lwa_a_reg.where(season_mask_era,drop=True)
    season_lwa_a_can = ds_canesm_lwa_a_reg.where(season_mask_can,drop=True)

    season_lwa_c_era = ds_era5_lwa_c_reg.where(season_mask_era,drop=True)
    season_lwa_c_can = ds_canesm_lwa_c_reg.where(season_mask_can,drop=True)

    season_dt_era = ds_tas_era5_anom.where(season_mask_era,drop=True) #seasonal masked [season]
    season_dt_can = ds_tas_canesm_anom.where(season_mask_can,drop=True)

    season_mrsos_era = ds_mrsos_era5_anom.where(season_mask_era,drop=True)
    season_mrsos_can = ds_mrsos_canesm_anom.where(season_mask_can,drop=True)

    idata_era_models    = {}
    idata_canesm_models = {}

    print("Loading posteriors for all model variations...")
    for model_variation in MODEL_VARIATIONS:
        idata_era_models[model_variation]    = None  
        idata_canesm_models[model_variation] = []


        era_file = os.path.join(OUTPUT_POSTERIORS,
            f"ERA5_{REGION}_{SEASON}_{BAYESIAN_MODEL}_{model_variation}.nc"
        )
        
        can_files = sorted(glob.glob(os.path.join(OUTPUT_POSTERIORS, 
            f"CanESM5_*_{REGION}_{SEASON}_{BAYESIAN_MODEL}_{model_variation}.nc"
        )))
        N_members = len(can_files)

        if len(ENSEMBLE_LIST) < N_members and config.FAST_IO:
            #running fewer ensemble members than expected;
            # need to select only the ensemble members that match the files we have
            found_members = {os.path.basename(f).split('_')[1] for f in can_files}
            valid_members = set(ENSEMBLE_LIST) & found_members
            if not valid_members:
                raise ValueError(f"No valid ensemble members found for {model_variation}. Found files for members: {found_members}, expected at least one of: {ENSEMBLE_LIST}")
            can_files = [f for f in can_files if os.path.basename(f).split('_')[1] in valid_members]
        else:

            assert N_members == len(ENSEMBLE_LIST), f"Expected {len(ENSEMBLE_LIST)} CanESM members, found {N_members}"
            # Verify all ensemble members are present
            found_members = {os.path.basename(f).split('_')[1] for f in can_files}
            missing = set(ENSEMBLE_LIST) - found_members
            assert not missing, f"Missing ensemble members for {model_variation}: {missing} \n {found_members}"

        # Load ERA5
        idata_era_models[model_variation] = open_posterior_file(era_file)

        # Load CanESM members
        idata_canesm_models[model_variation] = [open_posterior_file(f) for f in can_files]


    # Mask hot days to season and convert to events
    events_hot_era = mask_to_events(mask_hot_era) # hot days mask
    events_hot_can = mask_to_events(mask_hot_can)
    # Filter events entirely within the season
    events_hot_era = filter_events_by_season(events_hot_era, SEASON) #
    events_hot_can = filter_events_by_season(events_hot_can, SEASON)

    # Number of days window either side of the peak LWA
    half_window = 5
    lag_coord = np.arange(-half_window, half_window + 1)
    # Minimum duration of events to consider (in days)
    n_day_min = 1

    for model_name in ["full"]:
        print(f"Processing diagnostics for model variation: {model_name}")
        # Retrieve idata objects
        idata_era = idata_era_models.get(model_name)
        idata_can_list = idata_canesm_models.get(model_name, [])
        if idata_era is None or not idata_can_list:
            print(f"[warn] Missing posterior data for model {model_name}; skipping.")
            continue
        # Compute model predictions for ERA5
        dt_model_era = compute_dt_model(
            idata_era,
            season_lwa_a_era,
            season_lwa_c_era,
            season_mrsos_era,
        )
        # Compute model predictions for each CanESM member and assemble into a
        # single DataArray with a 'member' dimension
        dt_model_can_list = []
        # Compute predictions for each CanESM ensemble member individually.  Each
        # inference data object corresponds to a single ensemble member, so we
        # select the appropriate predictor arrays for that member.
        for i, idata in enumerate(idata_can_list):
            member_name = ENSEMBLE_LIST[i]
            pred = compute_dt_model(
                idata,
                season_lwa_a_can.sel(member=member_name),
                season_lwa_c_can.sel(member=member_name),
                season_mrsos_can.sel(member=member_name),
            )
            # Add a new member dimension so we can concat later
            pred = pred.expand_dims({"member": [member_name]})
            dt_model_can_list.append(pred)
        # Concatenate along the new 'member' dimension
        dt_model_can = xr.concat(dt_model_can_list, dim="member")

        # Align dT_data arrays (observed anomalies)
        dt_data_can = season_dt_can
        dt_data_era = season_dt_era

        # Compute event‑centred composites for data and model
        comp_dt_data_era, comp_lwa_era = _process_dataset(
            season_dt_era,
            season_lwa_a_era,
            events_hot_era,
            lag_coord,
            half_window,
            n_day_min,
        )
        comp_dt_model_era, _ = _process_dataset(
            dt_model_era,
            season_lwa_a_era,
            events_hot_era,
            lag_coord,
            half_window,
            n_day_min,
        )
        comp_dt_data_can, comp_lwa_can = _process_dataset(
            season_dt_can,
            season_lwa_a_can,
            events_hot_can,
            lag_coord,
            half_window,
            n_day_min,
        )
        comp_dt_model_can, _ = _process_dataset(
            dt_model_can, #this is a datatree # type: ignore
            season_lwa_a_can,
            events_hot_can,
            lag_coord,
            half_window,
            n_day_min,
        )
        # Skip plotting if no composites were produced
        if comp_dt_data_era is None or comp_dt_model_era is None:
            print(f"[warn] No valid event composites for ERA5 in model {model_name}; skipping plots.")
            continue
        if comp_dt_data_can is None or comp_dt_model_can is None:
            print(f"[warn] No valid event composites for CanESM in model {model_name}; skipping plots.")
            continue
        # Average CanESM composites across members for plotting
        comp_dt_data_can_mean = comp_dt_data_can.mean("member")
        comp_dt_model_can_mean = comp_dt_model_can.mean("member")
        comp_lwa_can_mean = comp_lwa_can.mean("member") #type:ignore
        # Plot event‑centred composites
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        # Observed dT
        axes[0].plot(
            comp_dt_data_era.lag_day,
            comp_dt_data_era,
            label="ERA5",
            color="k",
        )
        axes[0].plot(
            comp_dt_data_can_mean.lag_day,
            comp_dt_data_can_mean,
            label="CanESM5 (mean)",
            color="C1",
        )
        axes[0].axvline(0, color="grey", linestyle="--")
        axes[0].set_ylabel("dT (K)")
        axes[0].set_title(f"Observed dT anomaly (hot events)")
        axes[0].legend(loc="best")
        # Modelled dT
        axes[1].plot(
            comp_dt_model_era.lag_day,
            comp_dt_model_era,
            label="ERA5 model",
            color="k",
        )
        axes[1].plot(
            comp_dt_model_can_mean.lag_day,
            comp_dt_model_can_mean,
            label="CanESM5 model (mean)",
            color="C1",
        )
        axes[1].axvline(0, color="grey", linestyle="--")
        axes[1].set_ylabel("Model dT (K)")
        axes[1].set_title(f"Modelled dT anomaly (hot events)")
        axes[1].legend(loc="best")
        # LWA evolution for context
        axes[2].plot(
            comp_lwa_era.lag_day, #type:ignore
            comp_lwa_era,
            label="ERA5 LWA_a",
            color="k",
        )
        axes[2].plot(
            comp_lwa_can_mean.lag_day,
            comp_lwa_can_mean,
            label="CanESM5 LWA_a (mean)",
            color="C1",
        )
        axes[2].axvline(0, color="grey", linestyle="--")
        axes[2].set_xlabel("Lag day (relative to peak LWA_a)")
        axes[2].set_ylabel("sqrt(LWA_a) (arbitrary units)")
        axes[2].set_title("LWA_a evolution")
        axes[2].legend(loc="best")
        fig.suptitle(
            f"Event‑centred composites for model '{model_name}'\n"
            f"Region: {REGION}, Season: {SEASON}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
        # Save composite figure
        composite_path = os.path.join(
            diag_dir,
            f"event_composite_{model_name}_{LWA_var}_{REGION}_{SEASON}.png",
        )
        fig.savefig(composite_path, dpi=150)
        plt.close(fig)
        print(f"Saved composite plot to {composite_path}")

        # Scatter plots of dT_model vs dT_data for hot days
        # Flatten arrays for ERA
        dt_data_era_hot  = dt_data_era.where(mask_hot_era, drop=True)
        dt_model_era_hot = dt_model_era.where(mask_hot_era, drop=True)
        x_era = dt_model_era_hot.values.ravel()
        y_era = dt_data_era_hot.values.ravel()
        # Remove NaNs
        valid = np.isfinite(x_era) & np.isfinite(y_era)
        x_era = x_era[valid]
        y_era = y_era[valid]
        # Flatten arrays for CanESM across member/time
        dt_data_can_hot  = dt_data_can.where(mask_hot_can, drop=True)
        dt_model_can_hot = dt_model_can.where(mask_hot_can, drop=True)
        # stack member and time to a single dimension
        dt_data_can_flat  = dt_data_can_hot.stack(z=("member", "time"))
        dt_model_can_flat = dt_model_can_hot.stack(z=("member", "time"))
        x_can = dt_model_can_flat.values
        y_can = dt_data_can_flat.values
        valid_can = np.isfinite(x_can) & np.isfinite(y_can)
        x_can = x_can[valid_can]
        y_can = y_can[valid_can]
        # Create scatter plot
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
        # ERA scatter
        ax2[0].scatter(x_era, y_era, s=10, alpha=0.5, color="k")
        min_val = min(x_era.min(), y_era.min())
        max_val = max(x_era.max(), y_era.max())
        ax2[0].plot([min_val, max_val], [min_val, max_val], "--", color="grey")
        ax2[0].set_xlabel("Model dT (K)")
        ax2[0].set_ylabel("Observed dT (K)")
        ax2[0].set_title("ERA5 hot days")
        ax2[0].set_xlim(min_val, max_val)
        ax2[0].set_ylim(min_val, max_val)
        # CanESM scatter
        ax2[1].scatter(x_can, y_can, s=10, alpha=0.5, color="C1")
        min_val = min(x_can.min(), y_can.min())
        max_val = max(x_can.max(), y_can.max())
        ax2[1].plot([min_val, max_val], [min_val, max_val], "--", color="grey")
        ax2[1].set_xlabel("Model dT (K)")
        ax2[1].set_ylabel("Observed dT (K)")
        ax2[1].set_title("CanESM5 hot days")
        ax2[1].set_xlim(min_val, max_val)
        ax2[1].set_ylim(min_val, max_val)
        fig2.suptitle(
            f"Model vs observed dT on hot days for model '{model_name}'\n"
            f"Region: {REGION}, Season: {SEASON}",
            fontsize=12,
        )
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
        scatter_path = os.path.join(
            diag_dir,
            f"scatter_dt_{model_name}_{LWA_var}_{REGION}_{SEASON}.png",
        )
        fig2.savefig(scatter_path, dpi=150)
        plt.close(fig2)
        print(f"Saved scatter plot to {scatter_path}")





if __name__ == '__main__':


    BAYESIAN_MODEL = "both_lwa_noar_studentt"

    LWA_var  = "bothLWA"
    REGION   = "pnw_bartusek"
    SEASON   = "JJA"
    ZG_COORD = 500

    Q_HOT    = 95

    run_analysis(BAYESIAN_MODEL, LWA_var, REGION, SEASON, ZG_COORD, Q_HOT)
