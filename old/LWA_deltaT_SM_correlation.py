import numpy as np
import xarray as xr
import statsmodels.api as sm #type:ignore
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pymc as pm
import arviz as az
import pytensor.tensor as pt

import glob
import os
import argparse

from typing import Dict, List, Tuple, Any
from scipy import stats

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

# ------------------------------ Configuration --------------------------------

plt.rcParams.update({"font.size": 16})

# Ensemble MEMBERs
ENSEMBLE_LIST: List[str] = [
    "r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1", "r5i1p1f1",
    "r6i1p1f1", "r7i1p1f1", "r8i1p1f1", "r9i1p1f1", "r10i1p1f1",
    "r1i1p2f1", "r2i1p2f1", "r3i1p2f1", "r4i1p2f1", "r5i1p2f1",
    "r6i1p2f1", "r7i1p2f1", "r8i1p2f1", "r9i1p2f1", "r10i1p2f1"
]

# LWA VARiables
VAR_NAMES: List[str] = ["LWA", "LWA_a", "LWA_c"]

# Period and latitude band (to match existing practice)
TIME_SLICE = slice("1970-01-01", "2014-12-31")
LAT_SLICE  = slice(20, 85)  # 20–90°N

# REGIONs (consistent with your threshold files)
REGIONS: dict[str, Tuple[slice, slice]] = {
    "canada":        (slice(40, 70), slice(-140,    -60)),
    "canada_north":  (slice(55, 70), slice(-140,    -60)),
    "canada_south":  (slice(40, 55), slice(-140,    -60)),
    "west":          (slice(40, 70), slice(-140,    -113.33)),
    "west_north":    (slice(55, 70), slice(-140,    -113.33)),
    "west_south":    (slice(40, 55), slice(-140,    -113.33)),
    "central":       (slice(40, 70), slice(-113.33, -88.66)),
    "central_north": (slice(55, 70), slice(-113.33, -88.66)),
    "central_south": (slice(40, 55), slice(-113.33, -88.66)),
    "east":          (slice(40, 70), slice(-88.66,  -60)),
    "east_north":    (slice(55, 70), slice(-88.66,  -60)),
    "east_south":    (slice(40, 55), slice(-88.66,  -60)),
    "pnw_bartusek":  (slice(40, 60), slice(-130.0,  -110.0)),  # Bartusek et al. 2023
}

SEASON_NAMES = {"DJF", "MAM", "JJA", "SON"}

# LWA_var:str = "LWA_c"

VAR:str      = "tas"
# REGION:str   = "west_south"
# SEASON:str   = "JJA"
# ZG_COORD:int = 500
    

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


# Root paths (adapt if needed)
CANESM_LWA_ROOT = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/CanESM5/historical"
ERA5_LWA_ROOT   = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/ERA5"

# HW_THRESH_ROOT  = "/space/hall5/sitestore/eccc/crd/ccrn/users/mpw000/development/HW_analysis/thresholds"
# LWA_THRESH_ROOT = "/space/hall5/sitestore/eccc/crd/ccrn/users/mpw000/development/HW_analysis/lwa_thresholds"
ERA5_TAS_ROOT   = "/home/mhpereir/data-mhpereir/standard_grid_daily/REANALYSIS/ERA5/tas"
CANESM_TAS_ROOT = "/home/mhpereir/data-mhpereir/standard_grid_daily/CMIP6/CanESM5/tas/historical"

ERA5_MRSOS_ROOT   = "/home/mhpereir/data-mhpereir/standard_grid_daily/REANALYSIS/ERA5/soil_moisture"
CANESM_MRSOS_ROOT = "/home/mhpereir/data-mhpereir/standard_grid_daily/CMIP6/CanESM5/mrsos/historical"

OUTPUT_PLOTS_PATH = "/home/mhpereir/LWA_analysis/plots/LWA_deltaT_correlation"
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

OUTPUT_POSTERIOR = "/home/mhpereir/LWA_analysis/pymc_member_fits"
os.makedirs(OUTPUT_POSTERIOR, exist_ok=True)

# Plot options
HATCH_K: float = 2.0
PROJ = ccrs.EqualEarth() 

def arg_parser():
    parser = argparse.ArgumentParser(
        description="LWA vs deltaT correlation analysis and plotting."
    )
    parser.add_argument(
        "--lwa_var",
        type=str,
        choices=VAR_NAMES,
        default="LWA",
        help="LWA variable to analyze.",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=list(REGIONS.keys()),
        default="west_south",
        help="Region to analyze.",
    )
    parser.add_argument(
        "--season",
        type=str,
        choices=SEASON_NAMES,
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



# ------------------------------ I/O Utilities --------------------------------

def open_temp_data_canesm(var, ensemble_list):
    path_to_data = CANESM_TAS_ROOT
    
    files_by_member = []
    for m in ensemble_list:
        filepat = f"{path_to_data}/{var}_daily_CanESM5_historical_{m}_1850_2014_2x2_bil.nc"
        files = sorted(glob.glob(filepat))
        if not files:
            raise FileNotFoundError(f"No files matched for {m}; pattern: {filepat}")
        files_by_member.append(files)

    # 2) Open all lazily and stack along new 'MEMBER' then 'time'
    ds_CanESM = xr.open_mfdataset(
        files_by_member,
        combine="nested",
        concat_dim=["member", "time"],   # first stacks MEMBERs, then concatenates their time files
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
    )

    ds_CanESM = ds_CanESM.sel(time=TIME_SLICE)
    # 3) Give the MEMBER axis meaningful labels
    ds_CanESM = ds_CanESM.assign_coords(member=("member", ensemble_list))
    var_name = f"{var}"
    return ds_CanESM[var_name]


def open_temp_data_era5(var):
    path_to_data = f"{ERA5_TAS_ROOT}"
    # {BASE_DIR}/${MEMBER}/day/${VAR}/gn/v20190429/${VAR}_day_CanESM5_historical_${MEMBER}_gn_18500101-20141231_2x2_bil.nc

    filepat = f"{path_to_data}/{var}_daily_ERA_*_2x2_bil.nc"
    files = sorted(glob.glob(filepat))
    if not files:
        raise FileNotFoundError(f"No files matched for ERA5 temperature {var}; pattern: {filepat}")


    # 2) Open all lazily and stack along new 'MEMBER' then 'time'
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
    )

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    ds = ds.sel(time=TIME_SLICE)
    #remove leap days
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    # crop to lat slice
    ds = ds.sel(lat=LAT_SLICE)
    # 3) Give the MEMBER axis meaningful labels
    var_name = f"{var}"
    return ds[var_name]



def open_mrsos_data_canesm(var, ensemble_list):
    path_to_data = CANESM_MRSOS_ROOT
    
    files_by_member = []
    for m in ensemble_list:
        filepat = f"{path_to_data}/{var}_daily_CanESM5_historical_{m}_1850_2014_2x2_bil.nc"
        files = sorted(glob.glob(filepat))
        if not files:
            raise FileNotFoundError(f"No files matched for {m}; pattern: {filepat}")
        files_by_member.append(files)

    # 2) Open all lazily and stack along new 'MEMBER' then 'time'
    ds_CanESM = xr.open_mfdataset(
        files_by_member,
        combine="nested",
        concat_dim=["member", "time"],   # first stacks MEMBERs, then concatenates their time files
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365},             # tune to your chunking (e.g., ~1 year per chunk)
    )

    ds_CanESM = ds_CanESM.sel(time=TIME_SLICE)
    # 3) Give the MEMBER axis meaningful labels
    ds_CanESM = ds_CanESM.assign_coords(member=("member", ensemble_list))
    var_name = f"{var}"  #MRSOS is in units of kg/m2
    return ds_CanESM[var_name]


def open_mrsos_data_era5(var):
    path_to_data = f"{ERA5_MRSOS_ROOT}"
    # {BASE_DIR}/${MEMBER}/day/${VAR}/gn/v20190429/${VAR}_day_CanESM5_historical_${MEMBER}_gn_18500101-20141231_2x2_bil.nc

    filepat = f"{path_to_data}/soil_moisture_daily_ERA5_*_2x2_bil.nc"
    files = sorted(glob.glob(filepat))
    if not files:
        raise FileNotFoundError(f"No files matched for ERA5 temperature {var}; pattern: {filepat}")


    # 2) Open all lazily and stack along new 'MEMBER' then 'time'
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365},             # tune to your chunking (e.g., ~1 year per chunk)
    )

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    ds = ds.sel(time=TIME_SLICE)
    #remove leap days
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    # crop to lat slice
    ds = ds.sel(lat=LAT_SLICE)
    # 3) Give the MEMBER axis meaningful labels
    var_name = f"{var}" 

    ds = ds.rename({var_name: "mrsos"})
    #ERA5 stores soil moisture in m3/m3, convert to kg/m2 assuming 10cm depth and density of water 1000kg/m3
    ds = ds["mrsos"] * 0.1 * 1000.0  #kg/m2  #CanESM has 0.1m depth, 1000kg/m3 density
    #ERA5 soil layer 1 is only 7cm deep, but we use 10cm for consistency with CanESM
    return ds

def open_canesm_lwa(ensemble_list: List[str], zg_coord: int) -> Dict[str, xr.DataArray]:
    """
    Open a CanESM5 VARiable across multiple ensemble MEMBERs and concatenate time.

    Parameters
    ----------
    VAR_name : str
        VARiable name (must match the VARiable inside files).
    ENSEMBLE_LIST : list[str]
        List of MEMBER IDs to stack on a 'MEMBER' axis.
    ZG_COORD : int
        Used only to construct the file pattern (e.g., '...*.500.nc').

    Returns
    -------
    xr.DataArray
        Shape: (member, time, lat, lon), subset to 20–90°N, with 'member' coord.
    """
    path_to_data = CANESM_LWA_ROOT

    files_by_member: List[List[str]] = []
    for m in ensemble_list:
        filepat = f"{path_to_data}/z{zg_coord}/LWA_day_CanESM5_historical_{m}_2deg.{zg_coord}.nc"
        files = sorted(glob.glob(filepat))
        if not files:
            raise FileNotFoundError(f"No files matched for {m}: {filepat}")
        files_by_member.append(files)

    ds = xr.open_mfdataset(
        files_by_member,
        combine="nested",
        cache=False,
        concat_dim=["member", "time"],
        parallel=True,  # True can cause issues with netCDF4/HDF5
        chunks={"time": 365, "member": 10, "lat": 35, "lon": 180},  # ~1 year per chunk;
        engine="h5netcdf"  # h5netcdf handles parallel better than netCDF4
    )
    ds = ds.sel(time=TIME_SLICE)
    ds = ds.assign_coords(member=("member", ensemble_list)).sel(lat=LAT_SLICE)

    ds_lwa  = ds[ "LWA"]
    ds_lwaa = ds["LWA_a"]
    ds_lwac = ds["LWA_c"]
    return {"LWA": ds_lwa, "LWA_a": ds_lwaa, "LWA_c": ds_lwac}


def open_era5_lwa(zg_coord: int) -> Dict[str, xr.DataArray]:
    """
    Open an ERA5 VARiable and trim to analysis period and latitude range.

    Parameters
    ----------
    ZG_COORD : int
        Unused except for clarity parity with CanESM function.

    Returns
    -------
    xr.DataArray
        Shape: (time, lat, lon), subset to 1970–2014 and 20–90°N.
    """
    filepat = f"{ERA5_LWA_ROOT}/z{zg_coord}/LWA_day_ERA5_2deg.{zg_coord}.nc"
    files = sorted(glob.glob(filepat))
    if not files:
        raise FileNotFoundError(f"No ERA5 files matched: {filepat}")

    ds = xr.open_mfdataset(
        files, 
        parallel=True,
        cache=False,
        chunks={"time": 3650, "lat":35, "lon":180},
        engine="h5netcdf"
    )  # ~10 years per chunk;

    ds = ds.sel(time=TIME_SLICE)
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    ds = ds.sel(lat=LAT_SLICE)

    ds_lwa  = ds["LWA"]
    ds_lwaa = ds["LWA_a"]
    ds_lwac = ds["LWA_c"]
    return {"LWA": ds_lwa, "LWA_a": ds_lwaa, "LWA_c": ds_lwac}



# ------------------------------ Computation ----------------------------------

def compute_region_mean(ds: xr.DataArray, region: str) -> xr.DataArray:
    """Compute area-weighted mean over region."""
    lat_slice, lon_slice = REGIONS[region]
    ds_region = ds.sel(lat=lat_slice, lon=lon_slice)

    weights = np.cos(np.deg2rad(ds_region.lat))
    return ds_region.weighted(weights).mean(dim=["lat", "lon"])


def quantile_fit(x, y, quantiles: List[float]) -> Dict[float, Any]:
    X = sm.add_constant(x)
    fits = {}
    for q in quantiles:
        model = sm.QuantReg(y, X)
        res = model.fit(q=q)
        fits[q] = res
    return fits



# ------------------------------ Helper Functions ----------------------------------

def _stack_clean(a: xr.DataArray, b: xr.DataArray, has_member: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper: flatten (member,time) or (time,) into 1D arrays and drop NaNs.
    """
    if has_member:
        a = a.stack(points=("member", "time"))
        b = b.stack(points=("member", "time"))
    else:
        pass

    xa = np.asarray(a).ravel()
    ya = np.asarray(b).ravel()

    good = np.isfinite(xa) & np.isfinite(ya)
    return xa[good], ya[good]


def _stack_clean_3(a: xr.DataArray, b: xr.DataArray, c: xr.DataArray, has_member: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper: flatten (member,time) or (time,) into 1D arrays and drop NaNs.
    """
    if has_member:
        a = a.stack(points=("member", "time"))
        b = b.stack(points=("member", "time"))
        c = c.stack(points=("member", "time"))
    else:
        pass

    xa = np.asarray(a).ravel()
    ya = np.asarray(b).ravel()
    ca = np.asarray(c).ravel()

    good = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(ca)
    return xa[good], ya[good], ca[good]

def _normalize(pdf, dx):
        area = np.trapezoid(pdf, dx=dx)
        return pdf / area if area > 0 else pdf, area


def _safe_gaussian_kde(x, y, bw='scott'):
    """
    Return a callable KDE(xgrid,ygrid) that is robust to singular covariance.
    On failure, returns a zero-density KDE.
    """
    pts = np.vstack([x, y])

    kde = stats.gaussian_kde(pts, bw_method=bw)

    bw  = kde.factor

    def eval_on(xx, yy):
        return kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    return eval_on, bw

def _relative_error_pct(value: float, error: float) -> float:
    """Return |error/value| in percent, guarding against division by zero."""
    if not np.isfinite(value) or value == 0 or not np.isfinite(error):
        return np.nan
    return abs(error / value) * 100.0


def _format_quantile_label(q: float, res: Any, prefix: str = "") -> str:
    """Format legend text with slope/intercept and relative errors."""
    intercept = float(res.params[0])
    slope = float(res.params[1])
    intercept_rel = _relative_error_pct(intercept, float(res.bse[0]))
    slope_rel = _relative_error_pct(slope, float(res.bse[1]))

    def _fmt(val: float) -> str:
        return f"{val:.1f}" if np.isfinite(val) else "nan"

    prefix_txt = f"{prefix} " if prefix else ""
    return (
        f"{prefix_txt}Q{int(q*100)} slope={slope:.2e} +/-{_fmt(slope_rel)}%, "
        f"b0={intercept:.2f} +/-{_fmt(intercept_rel)}%"
    )



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
    draws: int = 2000,
    tune: int = 2000,
    target_accept: float = 0.9,
    chains: int = 4,
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



# ------------------------------ Plotting ----------------------------------


def plot_joint_kde_with_marginals(
    x_in: xr.DataArray,
    y_in: xr.DataArray,
    
    title: str = "",
    sim: str = "",
    x_label: str = f"LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    point_alpha: float = 0.4,
    max_scatter: int = 5000,
    gridsize: int = 200,
    cmap: str = "afmhot_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) with marginal PDFs and a best-fit regression line.

    Layout:
        - main panel: 2D KDE background (+ scatter sample, + linear fit)
        - top panel:  1D PDF of x
        - right panel:1D PDF of y
    Returns slope, intercept, r-value for logging.

    Parameters
    ----------
    x_in, y_in : np.ndarray
        1D arrays of equal length. These should already be anomalies
        or whatever you want to compare.
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    if "member" in x_in.dims:
        x, y = _stack_clean(x_in, y_in, has_member=True)
    else:
        x, y = _stack_clean(x_in, y_in, has_member=False)

    if x.size < 3 or y.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    # ---------------------------------------------------------------------
    # 2. Fit linear regression using SciPy
    # ---------------------------------------------------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
 
    # x-range for plotting the regression line
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = slope * x_line + intercept

    # ---------------------------------------------------------------------
    # 2.1 Fit quantile regression lines (50,90,95%)
    lw = [2., 2. ,2.]
    if   SEASON == "JJA":
        quantile_list = [0.5, 0.95]
        q_color_list  = ["cyan", "magenta", "red"]
    elif SEASON == "DJF":
        quantile_list = [0.05, 0.5]
        q_color_list  = ["blue", "cyan"]
    else:
        quantile_list = [0.05, 0.5, 0.95]
        q_color_list  = ["blue", "cyan", "magenta"]
    quantile_fits = quantile_fit(x, y, quantiles=quantile_list)

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d = stats.gaussian_kde(np.vstack([x, y]))
    # grid
    xi = np.linspace(0, x_limits[1]*1.25, gridsize)
    yi = np.linspace(y_limits[0]*1.25, y_limits[1]*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # 1D KDEs for marginals ( PDFs )
    def kde_1d(arr: np.ndarray, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        k = stats.gaussian_kde(arr)
        # xs = np.linspace(np.min(arr), np.max(arr), npts)
        pdf = k(xs)

        # normalize pdf so area ~ 1
        dx = xs[1] - xs[0]
        area = np.trapezoid(pdf, dx=dx)
        if area > 0:
            pdf = pdf / area
        return xs, pdf

    xs_pdf, x_pdf = kde_1d(x, xs=np.linspace(x_limits[0], x_limits[1], gridsize))
    ys_pdf, y_pdf = kde_1d(y, xs=np.linspace(y_limits[0], y_limits[1], gridsize))

    # ---------------------------------------------------------------------
    # 4. Layout: Joint plot + marginals
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[6, 1, 0.25],
        height_ratios=[1, 6],
        wspace=0.0,
        hspace=0.0,
        figure=fig,
    )

    ax_top    = fig.add_subplot(gs[0, 0])
    ax_main   = fig.add_subplot(gs[1, 0])
    ax_right  = fig.add_subplot(gs[1, 1])
    ax_cbar   = fig.add_subplot(gs[1, 2])

    # --------------------
    # main panel (joint)
    # --------------------
    # KDE as background
    pcm = ax_main.pcolormesh(
        Xg, Yg, Z,
        shading="auto",
        cmap=cmap,
    )

    # light scatter sample: subselect if huge
    if x.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
    else:
        xs_sc = x
        ys_sc = y

    ax_main.scatter(
        xs_sc,
        ys_sc,
        s=6,
        c="gray",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    # # regression line
    # ax_main.plot(
    #     x_line,
    #     y_line,
    #     color="cyan",
    #     linewidth=2,
    #     label=f"fit: y = {slope:.2e} x + {intercept:.2f}\n" #type:ignore
    #           f"r = {r_value:.2f}",
    # )

    # quantile regression lines
    xs = np.linspace(x_limits[0], x_limits[1], 200)
    X_eval = sm.add_constant(xs)
    
    for ii,q in enumerate(quantile_list):
        yq_line = quantile_fits[q].predict(X_eval)
        ax_main.plot(
            xs,
            yq_line,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle="--",
            label=_format_quantile_label(q, quantile_fits[q]),
            zorder=30
        )

    # ax_main.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_main.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_main.set_ylabel(y_label + " (K)")
    ax_main.set_xlim(x_limits if x_limits else (np.min(Xg), np.max(Xg)))
    ax_main.set_ylim(y_limits if y_limits else (np.min(Yg), np.max(Yg)))
    ax_main.legend(loc="lower right", frameon=True, fontsize=9)

    # colorbar for KDE
    #cax = fig.add_axes([0.12, 0.04, 0.65, 0.015])  # [left, bottom, width, height] in fig coords #type: ignore
    cb = fig.colorbar(pcm, cax=ax_cbar, orientation="vertical")
    cb.set_label("Joint density (KDE)")

    # ax_cbar.xaxis.set_ticks_position("bottom")
    ax_cbar.xaxis.set_ticks([])

    # --------------------
    # top marginal (x PDF)
    # --------------------
    ax_top.plot(xs_pdf, x_pdf, color="black", lw=1.5)
    ax_top.fill_between(xs_pdf, 0, x_pdf, color="0.7", alpha=0.4)
    ax_top.set_ylabel("log(P(LWA))")
    #set y-axis to log scale
    ax_top.set_yscale("log")
    ax_top.set_ylim((1e-6, 1e-3))
    ax_top.set_xticks([])
    ax_top.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
        pad=0,
    )
    # sync x-lims with main panel's x
    ax_top.set_xlim(ax_main.get_xlim())
    # make top spine pretty
    ax_top.spines["right"].set_visible(True)
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["left"].set_visible(True)
    ax_top.spines["bottom"].set_visible(False)

    # --------------------
    # right marginal (y PDF)
    # --------------------
    ax_right.plot(y_pdf, ys_pdf, color="black", lw=1.5)
    ax_right.fill_betweenx(ys_pdf, 0, y_pdf, color="0.7", alpha=0.4)
    ax_right.set_xlabel("log(P(T))")
    #set x-axis to log scale
    ax_right.set_xscale("log")
    ax_right.set_yticks([])
    ax_right.set_xlim((1e-4, 1))
    ax_right.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        pad=0,
    )
    # sync y-lims with main panel's y
    ax_right.set_ylim(ax_main.get_ylim())
    # cosmetic spines
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(True)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(True)

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    # tighten layout a bit (but keep our manual colorbar placement)
    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    fig_name = f"{output_path}/{sim}_{LWA_var}_deltaT_kde_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None







def plot_joint_kde_with_marginals_mrsos(
    x_in: xr.DataArray,
    y_in: xr.DataArray,
    z_in: xr.DataArray,
    title: str = "",
    sim: str = "",
    x_label: str = f"LWA",
    y_label: str = r"$\Delta T$",
    z_label: str = r"$\Delta$MRSOS (kg/m2)",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    z_limits: Tuple[float, float] = None, #type:ignore
    point_alpha: float = 0.8,
    max_scatter: int = 5000,
    gridsize: int = 200,
    cmap: str = "afmhot_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) with marginal PDFs and a best-fit regression line.

    Layout:
        - main panel: 2D KDE background (+ scatter sample, + linear fit)
        - top panel:  1D PDF of x
        - right panel:1D PDF of y
    Returns slope, intercept, r-value for logging.

    Parameters
    ----------
    x_in, y_in : np.ndarray
        1D arrays of equal length. These should already be anomalies
        or whatever you want to compare.
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    if "member" in x_in.dims:
        x, y, z = _stack_clean_3(x_in, y_in, z_in, has_member=True)
    else:
        x, y, z = _stack_clean_3(x_in, y_in, z_in, has_member=False)

    if x.size < 3 or y.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    # ---------------------------------------------------------------------
    # 2. Fit linear regression using SciPy
    # ---------------------------------------------------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
 
    # x-range for plotting the regression line
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = slope * x_line + intercept

    # ---------------------------------------------------------------------
    # 2.1 Fit quantile regression lines (50,90,95%)
    lw = [2., 2. ,2.]
    if   SEASON == "JJA":
        quantile_list = [0.5, 0.95]
        q_color_list  = ["cyan", "magenta", "red"]
    elif SEASON == "DJF":
        quantile_list = [0.05, 0.5]
        q_color_list  = ["blue", "cyan"]
    else:
        quantile_list = [0.05, 0.5, 0.95]
        q_color_list  = ["blue", "cyan", "magenta"]
    quantile_fits = quantile_fit(x, y, quantiles=quantile_list)

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d = stats.gaussian_kde(np.vstack([x, y]))
    # grid
    xi = np.linspace(0, x_limits[1]*1.25, gridsize)
    yi = np.linspace(y_limits[0]*1.25, y_limits[1]*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # 1D KDEs for marginals ( PDFs )
    def kde_1d(arr: np.ndarray, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        k = stats.gaussian_kde(arr)
        # xs = np.linspace(np.min(arr), np.max(arr), npts)
        pdf = k(xs)

        # normalize pdf so area ~ 1
        dx = xs[1] - xs[0]
        area = np.trapezoid(pdf, dx=dx)
        if area > 0:
            pdf = pdf / area
        return xs, pdf

    xs_pdf, x_pdf = kde_1d(x, xs=np.linspace(x_limits[0], x_limits[1], gridsize))
    ys_pdf, y_pdf = kde_1d(y, xs=np.linspace(y_limits[0], y_limits[1], gridsize))

    # ---------------------------------------------------------------------
    # 4. Layout: Joint plot + marginals
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[6, 1, 0.25],
        height_ratios=[1, 6],
        wspace=0.0,
        hspace=0.0,
        figure=fig,
    )

    ax_top    = fig.add_subplot(gs[0, 0])
    ax_main   = fig.add_subplot(gs[1, 0])
    ax_right  = fig.add_subplot(gs[1, 1])
    ax_cbar   = fig.add_subplot(gs[1, 2])

    # --------------------
    # main panel (joint)
    # --------------------
    # KDE as background
    pcm = ax_main.pcolormesh(
        Xg, Yg, Z,
        shading="auto",
        cmap=cmap,
    )

    # light scatter sample: subselect if huge
    if x.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
        cs_sc = z[idx]

        scatter = ax_main.scatter(
            xs_sc,
            ys_sc,
            c=cs_sc,
            s=10,
            cmap="RdBu",
            edgecolors="k",
            linewidths=0.1,
            alpha=point_alpha,
            vmin=z_limits[0],
            vmax=z_limits[1],
        )

    else:
        xs_sc = x
        ys_sc = y
        cs_sc = z

        scatter = ax_main.scatter(
            xs_sc,
            ys_sc,
            c=cs_sc,
            s=10,
            cmap="RdBu",
            edgecolors="k",
            linewidths=0.1,
            alpha=point_alpha,
            vmin=z_limits[0],
            vmax=z_limits[1],
        )

    
    cbar_scatter = fig.colorbar(scatter, ax=ax_main, pad=0.02)
    cbar_scatter.set_label(f"{z_in.name if hasattr(z_in, 'name') else 'MRSOS'} (kg/m²)")

    # # regression line
    # ax_main.plot(
    #     x_line,
    #     y_line,
    #     color="cyan",
    #     linewidth=2,
    #     label=f"fit: y = {slope:.2e} x + {intercept:.2f}\n" #type:ignore
    #           f"r = {r_value:.2f}",
    # )

    # quantile regression lines
    xs = np.linspace(x_limits[0], x_limits[1], 200)
    X_eval = sm.add_constant(xs)
    
    for ii,q in enumerate(quantile_list):
        yq_line = quantile_fits[q].predict(X_eval)
        ax_main.plot(
            xs,
            yq_line,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle="--",
            label=_format_quantile_label(q, quantile_fits[q]),
            zorder=30
        )

    # ax_main.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_main.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_main.set_ylabel(y_label + " (K)")
    ax_main.set_xlim(x_limits if x_limits else (np.min(Xg), np.max(Xg)))
    ax_main.set_ylim(y_limits if y_limits else (np.min(Yg), np.max(Yg)))
    ax_main.legend(loc="lower right", frameon=True, fontsize=9)

    # colorbar for KDE
    #cax = fig.add_axes([0.12, 0.04, 0.65, 0.015])  # [left, bottom, width, height] in fig coords #type: ignore
    cb = fig.colorbar(pcm, cax=ax_cbar, orientation="vertical")
    cb.set_label("Joint density (KDE)")

    # ax_cbar.xaxis.set_ticks_position("bottom")
    ax_cbar.xaxis.set_ticks([])

    # --------------------
    # top marginal (x PDF)
    # --------------------
    ax_top.plot(xs_pdf, x_pdf, color="black", lw=1.5)
    ax_top.fill_between(xs_pdf, 0, x_pdf, color="0.7", alpha=0.4)
    ax_top.set_ylabel("log(P(LWA))")
    #set y-axis to log scale
    ax_top.set_yscale("log")
    ax_top.set_ylim((1e-6, 1e-3))
    ax_top.set_xticks([])
    ax_top.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
        pad=0,
    )
    # sync x-lims with main panel's x
    ax_top.set_xlim(ax_main.get_xlim())
    # make top spine pretty
    ax_top.spines["right"].set_visible(True)
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["left"].set_visible(True)
    ax_top.spines["bottom"].set_visible(False)

    # --------------------
    # right marginal (y PDF)
    # --------------------
    ax_right.plot(y_pdf, ys_pdf, color="black", lw=1.5)
    ax_right.fill_betweenx(ys_pdf, 0, y_pdf, color="0.7", alpha=0.4)
    ax_right.set_xlabel("log(P(T))")
    #set x-axis to log scale
    ax_right.set_xscale("log")
    ax_right.set_yticks([])
    ax_right.set_xlim((1e-4, 1))
    ax_right.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        pad=0,
    )
    # sync y-lims with main panel's y
    ax_right.set_ylim(ax_main.get_ylim())
    # cosmetic spines
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(True)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(True)

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    # tighten layout a bit (but keep our manual colorbar placement)
    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    fig_name = f"{output_path}/{sim}_{LWA_var}_deltaT_kde_MRSOS_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None





def plot_kde_comparison_model_vs_ref(
    x_mod,  # xr.DataArray, dims ('member','time') or ('time',) #model, CanESM
    y_mod,  # xr.DataArray, dims ('member','time') or ('time',)
    x_ref,  # xr.DataArray, dims ('time',) #reference, ERA5
    y_ref,  # xr.DataArray, dims ('time',)
    title: str = "",
    x_label: str = "LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    member_dim: str = "member",
    gridsize: int = 200,
    max_scatter: int = 4000,
    cmap: str = "RdBu",
    hatch_alpha: float = 0.15,
    output_path: str = "./",
) -> None:
    """
    Compare model joint density vs reference joint density in (x,y) space.

    Main panel:
      - Filled colormap: model *ensemble-mean* KDE
      - Contour lines:   reference (e.g. ERA5) KDE
      - Scatter:         reference points (optional thinning)
      - Regression lines: model-mean fit and ref fit
      - Hatching: where ref KDE lies inside the (p5,p95) band of
                  per-member KDE densities from the model.

    Inputs
    ------
    x_mod, y_mod : xr.DataArray
        Model quantities. Must share dims. If they have `member_dim`, we
        treat them as ensemble; otherwise treated as single-field model.
    x_ref, y_ref : xr.DataArray
        Reference quantities (e.g. ERA5). No member dim.
    member_dim : str
        Name of ensemble dimension in x_mod/y_mod. Ignored if absent.
    gridsize : int
        Resolution of KDE grid in both x and y directions.
    hatch_alpha : float
        Alpha for the hatched overlay mask.

    Output
    ------
    Saves a figure to `output_path`. Returns None.
    """

    if x_label == 'LWA_a' or x_label == 'LWA':
        leg_loc="upper left"
    elif x_label == 'LWA_c':
        leg_loc="lower left"
    else:
        Warning("x_label not recognized, setting legend location to 'best'")
        leg_loc="best"


    # ------------------------------------------------------------------
    # 1. Flatten data
    # ------------------------------------------------------------------
    has_member = (member_dim in x_mod.dims) and (member_dim in y_mod.dims)

    # model ALL points (for xrange/yrange)
    xm_all, ym_all = _stack_clean(x_mod, y_mod, has_member=has_member)
    # reference ALL points
    xr_all, yr_all = _stack_clean(x_ref, y_ref, has_member=False)

    if xm_all.size < 5 or xr_all.size < 5:
        raise ValueError("Not enough points to build KDEs.")

    # ------------------------------------------------------------------
    # 2. Build grid in (x,y) space
    #    Use combined 1st-99th percentiles of BOTH model+ref so both fit
    # ------------------------------------------------------------------
    x_lo = np.min(np.concatenate([xm_all, xr_all]))
    x_hi = np.max(np.concatenate([xm_all, xr_all]))
    y_lo = np.min(np.concatenate([ym_all, yr_all]))
    y_hi = np.max(np.concatenate([ym_all, yr_all]))

    xi = np.linspace(x_lo, x_hi, gridsize)
    yi = np.linspace(y_lo, y_hi, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)

    # ---------------------------------------------------------------------
    # 2.1 Fit quantile regression lines (50,90,95%)
    lw = [2., 2. ,2.]
    if   SEASON == "JJA":
        quantile_list = [0.5, 0.95]
        q_color_list  = ["cyan", "magenta", "red"]
    elif SEASON == "DJF":
        quantile_list = [0.05, 0.5]
        q_color_list  = ["blue", "cyan"]
    else:
        quantile_list = [0.05, 0.5, 0.95]
        q_color_list  = ["blue", "cyan", "magenta"]

    quantile_fits_ref = quantile_fit(xr_all, yr_all, quantiles=quantile_list)
    quantile_fits_mod = quantile_fit(xm_all, ym_all, quantiles=quantile_list)

    # ------------------------------------------------------------------
    # 3. KDEs:
    #    - ref KDE (single)
    #    - model member KDEs, then mean, then p05/p95
    # ------------------------------------------------------------------

    # reference KDE
    ref_eval, bw_ref = _safe_gaussian_kde(xr_all, yr_all)
    Z_ref = ref_eval(Xg, Yg)  # (gridsize, gridsize)

    Z_mean = None
    Z_p05  = None
    Z_p95  = None

    # model KDEs
    if has_member:
        Z_members = []
        for m in x_mod[member_dim].values:
            xm_m, ym_m = _stack_clean(x_mod.sel({member_dim: m}),
                                      y_mod.sel({member_dim: m}),
                                      has_member=False)
            if xm_m.size < 5:
                continue
            eval_m, _ = _safe_gaussian_kde(xm_m, ym_m, bw=bw_ref) #type:ignore
            Z_m = eval_m(Xg, Yg)
            Z_members.append(Z_m)

        if len(Z_members) == 0:
            raise ValueError("No valid member KDEs could be estimated.")

        Z_members = np.stack(Z_members, axis=0)  # (nmem, ny, nx)

        # ensemble mean density
        Z_mean = np.nanmean(Z_members, axis=0)

        # ensemble spread band (5th, 95th)
        Z_p05 = np.nanpercentile(Z_members, 5, axis=0)
        Z_p95 = np.nanpercentile(Z_members, 95, axis=0)
    else:
        SystemExit(f"Model data has no member dimension of name {member_dim}; ensemble spread hatching not possible.")
        # # no member dim: treat "model" as single field
        # mod_eval = _safe_gaussian_kde(xm_all, ym_all)
        # Z_mean = mod_eval(Xg, Yg)
        # Z_p05 = Z_mean.copy()
        # Z_p95 = Z_mean.copy()

    # ------------------------------------------------------------------
    # 4. Regression fits (ref vs model-mean)
    # ------------------------------------------------------------------
    def linfit(xarr, yarr):
        slope, intercept, r, p, se = stats.linregress(xarr, yarr)
        xline = np.linspace(x_lo, x_hi, 200)
        yline = slope * xline + intercept #type:ignore
        return slope, intercept, r, xline, yline

    slope_ref, intercept_ref, r_ref, xline_ref, yline_ref = linfit(xr_all, yr_all)
    slope_mod, intercept_mod, r_mod, xline_mod, yline_mod = linfit(xm_all, ym_all)

    # ------------------------------------------------------------------
    # 5. Hatching mask:
    #    hatch where ref KDE is within [p05, p95] of model members
    # ------------------------------------------------------------------
    within_band = (Z_ref >= Z_p05) & (Z_ref <= Z_p95)

    # ------------------------------------------------------------------
    # 6. Plot layout
    #    We'll imitate your current setup: main joint panel,
    #    plus top/right marginals using the reference data only
    #    (since that's your observed distribution anchor).
    #    Colorbar sits next to main panel.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[6, 1, 0.25],
        height_ratios=[1, 6],
        wspace=0.0,
        hspace=0.0,
        figure=fig,
    )

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_cbar  = fig.add_subplot(gs[1, 2])

    # -------------------------
    # 6a. MAIN PANEL
    # -------------------------

    # If you really want anomaly instead of model density:
    Z_plot = Z_mean - Z_ref
    vmax = np.nanmax(np.abs(Z_plot))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin,
                        vcenter=0.0,
                        vmax=vmax)
    pcm = ax_main.pcolormesh(Xg, Yg, Z_plot,
                             shading="auto",
                             cmap="RdBu_r",
                             norm=norm)

    # Otherwise: shaded model-mean density
    # pcm = ax_main.pcolormesh(
    #     Xg, Yg, Z_mean, #type: ignore
    #     shading="auto",
    #     cmap=cmap,
    # )

    # Hatching: where ERA5 KDE is within [p05,p95] of ensemble KDEs
    # We'll overlay a transparent contourf hatch layer.
    ax_main.contourf(
        Xg, Yg, within_band.astype(int),
        levels=[-0.5, 0.5, 1.5],
        hatches=["", ".."],
        colors="none",
        linewidth=0.1,
        alpha=hatch_alpha,
        linewidths=0.0,
    )

    # Ref KDE contours
    # Use a few log-spaced-ish levels so we see structure
    # (add small epsilon to avoid log(0)).
    eps = np.nanmax(Z_ref) * 1e-3
    levels = np.linspace(np.nanmax(Z_ref)*0.1, np.nanmax(Z_ref), 5)
    cs = ax_main.contour(
        Xg, Yg, Z_ref,
        levels=levels,
        colors="white",
        linewidths=1,
        alpha=0.9,
    )
    ax_main.clabel(cs, inline=True, fmt="%.1e", fontsize=8)

    # Scatter reference points (thin if huge)
    if xr_all.size > max_scatter:
        rng = np.random.default_rng(123)
        idx = rng.choice(xr_all.size, size=max_scatter, replace=False)
        xs_sc = xr_all[idx]
        ys_sc = yr_all[idx]
    else:
        xs_sc = xr_all
        ys_sc = yr_all

    ax_main.scatter(
        xs_sc, ys_sc,
        s=6,
        c="gray",
        edgecolors="none",
        linewidths=0.2,
        alpha=0.4,
        zorder=10,
    )

    # Regression lines
    # ax_main.plot(
    #     xline_ref, yline_ref,
    #     color="cyan",
    #     lw=2,
    #     label=f"ERA5: y={slope_ref:.2e}x+{intercept_ref:.2f}, r={r_ref:.2f}",
    #     zorder=11,
    # )
    # ax_main.plot(
    #     xline_mod, yline_mod,
    #     color="yellow",
    #     lw=2,
    #     label=f"CanESM: y={slope_mod:.2e}x+{intercept_mod:.2f}, r={r_mod:.2f}",
    #     zorder=11,
    # )

    # quantile regression lines
    xs = np.linspace(x_lo, x_hi, 200)
    X_eval = sm.add_constant(xs)
    
    for ii,q in enumerate(quantile_list):
        yq_line_ref = quantile_fits_ref[q].predict(X_eval)
        yq_line_mod = quantile_fits_mod[q].predict(X_eval)
        ax_main.plot(
            xs,
            yq_line_ref,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle="--",
            label=_format_quantile_label(q, quantile_fits_ref[q], prefix="ERA5"),
            zorder=30
        )

        ax_main.plot(
            xs,
            yq_line_mod,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle=":",
            label=_format_quantile_label(q, quantile_fits_mod[q], prefix="CanESM"),
            zorder=30
        )


    # axes labels / limits
    # ax_main.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_main.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_main.set_ylabel(y_label + ' [K]')
    ax_main.set_xlim(x_limits if x_limits else (x_lo, x_hi))
    ax_main.set_ylim(y_limits if y_limits else (y_lo, y_hi))

    ax_main.legend(
        loc=leg_loc,
        frameon=True,
        fontsize=8,
    )

    # -------------------------
    # 6b. TOP marginal (PDF of x_ref and x_mod-mean, optional)
    # -------------------------
    # We'll show reference in black/gray fill, and model-mean in a line.

    # --- reference KDE ---
    kx_ref = stats.gaussian_kde(xr_all, bw_method='scott')
    xs = np.linspace(x_lo, x_hi, 200)
    pdf_x_ref = kx_ref(xs)

    kx_ref_factor = kx_ref.factor

    # --- model: per-member KDEs ---
    pdfs_members = []
    for m in x_mod["member"].values:              # assumes x_mod has member dim
        xm = np.asarray(x_mod.sel(member=m)).ravel()
        xm = xm[np.isfinite(xm)]
        
        kde_m = stats.gaussian_kde(xm, bw_method=kx_ref_factor)
        pdf_m = kde_m(xs)
        pdfs_members.append(pdf_m)

    pdfs_members = np.stack(pdfs_members, axis=0)   # shape (nmem, nx)

    # ensemble mean + 5–95% range
    pdf_x_mod_mean = np.nanmean(pdfs_members, axis=0)
    pdf_x_mod_p05  = np.nanpercentile(pdfs_members, 5, axis=0)
    pdf_x_mod_p95  = np.nanpercentile(pdfs_members, 95, axis=0)

    # --- normalize all PDFs to integrate ≈1 ---
    dx = xs[1] - xs[0]

    pdf_x_ref, area_x_ref           = _normalize(pdf_x_ref, dx)
    pdf_x_mod_mean, area_x_mod_mean = _normalize(pdf_x_mod_mean, dx)
    pdf_x_mod_p05                   = pdf_x_mod_p05 / area_x_mod_mean
    pdf_x_mod_p95                   = pdf_x_mod_p95 / area_x_mod_mean

    # --- plotting ---
    ax_top.fill_between(xs, 0, pdf_x_ref, color="0.7", alpha=0.4)
    ax_top.plot(xs, pdf_x_ref, color="black", lw=1.5, label="ERA5")

    # ensemble-spread shading
    ax_top.fill_between(
        xs, pdf_x_mod_p05, pdf_x_mod_p95,
        color="gold", alpha=0.25, label="CanESM 5–95%"
    )

    # mean line
    ax_top.plot(xs, pdf_x_mod_mean, color="gold", lw=1.8, label="CanESM mean")

    # cosmetics
    ax_top.set_xlim(x_lo, x_hi)
    ax_top.set_xticks([])
    ax_top.set_ylabel("log(P(LWA))")
    #make ax_top y-axis log scale
    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-6, 1e-3)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False, pad=0)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.legend(frameon=False, fontsize=8)


    # -------------------------
    # 6c. RIGHT marginal (PDF of y_ref vs y_mod)
    # -------------------------
    # --- reference KDE (1D, y) ---
    ky_ref = stats.gaussian_kde(yr_all, bw_method='scott')
    ys = np.linspace(y_lo, y_hi, 200)
    pdf_y_ref = ky_ref(ys)
    ky_ref_factor = ky_ref.factor

    # --- model: per-member KDEs for y ---
    pdfs_members_y = []
    for m in y_mod["member"].values:                 # assumes y_mod has a 'member' dim
        ym = np.asarray(y_mod.sel(member=m)).ravel()
        ym = ym[np.isfinite(ym)]
        
        kde_m = stats.gaussian_kde(ym, bw_method=ky_ref_factor)
        pdf_m = kde_m(ys)

        pdfs_members_y.append(pdf_m)

    if len(pdfs_members_y) == 0:
        # Fallback: no valid members → use zeros so plotting still works
        pdfs_members_y = [np.zeros_like(ys)]

    pdfs_members_y = np.stack(pdfs_members_y, axis=0)  # (nmem, ny)

    # ensemble mean + 5–95% range
    pdf_y_mod_mean = np.nanmean(pdfs_members_y, axis=0)
    pdf_y_mod_p05  = np.nanpercentile(pdfs_members_y, 5, axis=0)
    pdf_y_mod_p95  = np.nanpercentile(pdfs_members_y, 95, axis=0)

    # --- normalize all PDFs to integrate ≈1 ---
    dy = ys[1] - ys[0]

    pdf_y_ref, area_y_ref           = _normalize(pdf_y_ref, dy)
    pdf_y_mod_mean, area_y_mod_mean = _normalize(pdf_y_mod_mean, dy)
    pdf_y_mod_p05                   = pdf_y_mod_p05 / area_y_mod_mean
    pdf_y_mod_p95                   = pdf_y_mod_p95 / area_y_mod_mean

    # --- plotting (RIGHT marginal) ---
    # REF as filled baseline
    ax_right.fill_betweenx(ys, 0, pdf_y_ref, color="0.7", alpha=0.4)
    ax_right.plot(pdf_y_ref, ys, color="black", lw=1.5, label="ERA5")

    # ensemble-spread shading (model 5–95%)
    ax_right.fill_betweenx(
        ys, pdf_y_mod_p05, pdf_y_mod_p95,
        color="gold", alpha=0.25, label="CanESM 5–95%"
    )

    # model mean line
    ax_right.plot(pdf_y_mod_mean, ys, color="gold", lw=1.8, label="CanESM mean")

    # cosmetics
    ax_right.set_ylim(y_lo, y_hi)
    ax_right.set_yticks([])
    ax_right.set_xlabel("log(P(T))")
    ax_right.set_xscale("log")
    ax_right.set_xlim(1e-4, 1)
    ax_right.tick_params(axis="y", left=False, labelleft=False, pad=0)
    ax_right.spines["left"].set_visible(False)
    # (optional) show legend here, or keep it in ax_top/ax_main
    # ax_right.legend(frameon=False, fontsize=8, loc="lower right")

    # -------------------------
    # 6d. Colorbar
    # -------------------------
    cb = fig.colorbar(pcm, cax=ax_cbar, orientation="vertical")
    cb.set_label("CanESM - ERA5 KDE Anomaly")
    ax_cbar.xaxis.set_ticks([])

    # hatch legend proxy
    # hatch_proxy = Patch(facecolor="none", edgecolor="0.25",
    #                     hatch="////",
    #                     label="ERA5 KDE within CanESM 5–95% band")
    # leg_2 = ax_main.legend(handles=[hatch_proxy], loc="upper left", frameon=True, fontsize=8)
    # ax_main.add_artist(leg_1)


    handles1, labels1 = ax_main.get_legend_handles_labels() #grabs existing legend handles

    # build proxies for contour and hatch
    contour_proxy = Line2D([], [], color="white", lw=1.5, label="ERA5 KDE contours")
    hatch_proxy = Patch(facecolor="none", edgecolor="0.25", hatch="////",
                        label="ERA5 KDE within CanESM 5–95% band")

    # combine all
    handles = handles1 + [contour_proxy, hatch_proxy]
    ax_main.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)

    # -------------------------
    # 6e. Title & layout
    # -------------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    fig_name = f"{output_path}/KDE_{LWA_var}_modelvref_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    plt.close(fig)


def plot_conditional_kdes(
    x_in: xr.DataArray,
    y_in: xr.DataArray,
    title: str = "",
    sim: str = "",
    x_label: str = "LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    point_alpha: float = 0.4,
    max_scatter: int = 5000,
    gridsize: int = 400,
    cmap: str = "afmhot_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) with marginal PDFs and a best-fit regression line.

    Layout:
        - main panel: 2D KDE background (+ scatter sample, + linear fit)
        - top panel:  1D PDF of x
        - right panel:1D PDF of y
    Returns slope, intercept, r-value for logging.

    Parameters
    ----------
    x_in, y_in : np.ndarray
        1D arrays of equal length. These should already be anomalies
        or whatever you want to compare.
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    if "member" in x_in.dims:
        x, y = _stack_clean(x_in, y_in, has_member=True)
    else:
        x, y = _stack_clean(x_in, y_in, has_member=False)

    if x.size < 3 or y.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    if x_label == 'LWA_a' or x_label == 'LWA':
        leg_loc="upper left"
    elif x_label == 'LWA_c':
        leg_loc="lower left"
    else:
        Warning("x_label not recognized, setting legend location to 'best'")
        leg_loc="best"

    if x_limits is None:
        xmin, xmax = np.min([x_era.min(), x_can.min()]), np.max([x_era.max(), x_can.max()])
    else:
        xmin, xmax = x_limits

    if y_limits is None:
        ymin, ymax = np.min([y_era.min(), y_can.min()]), np.max([y_era.max(), y_can.max()])
    else:
        ymin, ymax = y_limits

    # ---------------------------------------------------------------------
    # 2. Fit linear regression using SciPy
    # ---------------------------------------------------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
 
    # x-range for plotting the regression line
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), gridsize)
    y_line = slope * x_line + intercept

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d = stats.gaussian_kde(np.vstack([x, y]))
    # grid
    xi = np.linspace(0, xmax*1.25, gridsize+5)
    yi = np.linspace(ymin*1.25, ymax*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # 1D KDEs for marginals ( PDFs )
    def kde_1d(arr: np.ndarray, npts: int = gridsize) -> Tuple[np.ndarray, np.ndarray]:

        k = stats.gaussian_kde(arr)
        xs = np.linspace(np.min(arr), np.max(arr), npts)
        pdf = k(xs)

        # normalize pdf so area ~ 1
        dx = xs[1] - xs[0]
        area = np.trapezoid(pdf, dx=dx)
        if area > 0:
            pdf = pdf / area
        return xs, pdf

    xs_pdf, x_pdf = kde_1d(x) #PDF of LWA
    ys_pdf, y_pdf = kde_1d(y) #PDF of T

    # ---------------------------------------------------------------------
    # 4. Layout: two main conditional panels + cbar below
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig, axs = plt.subplots(ncols=2, figsize=(13, 8), tight_layout=True)
    ax_1     = axs[0]
    ax_2     = axs[1]

    x_marg = np.trapezoid(Z, x=yi, axis=0)   # p(LWA=x), shape (nx,)
    y_marg = np.trapezoid(Z, x=xi, axis=1)   # p(T=y),   shape (ny,)

    print(np.trapezoid(x_marg, x=xi))  # should be ~1
    print(np.trapezoid(y_marg, x=yi))  # should be ~1

    eps = np.finfo(float).eps
    Z_cond_LWA = Z / np.maximum(x_marg, eps)[None, :]   # p(T | LWA)  (normalize columns)
    Z_cond_T   = Z / np.maximum(y_marg, eps)[:, None]   # p(LWA | T)  (normalize rows)

    ## testing that the conditional densities integrate to 1
    arr_1 = np.trapezoid(Z_cond_LWA, x=xi, axis=1) # "Conditional density P(T | LWA) does not integrate to 1"
    arr_2 = np.trapezoid(Z_cond_T,   x=yi, axis=0)   # "Conditional density P(LWA | T) does not integrate to 1"

    print("range of conditional prob Z_cond_LWA", arr_1.min(), arr_1.max())
    print("range of conditional prob Z_cond_T", arr_2.min(), arr_2.max())

    # --------------------
    # Left main panel (x | y)
    # --------------------
    # KDE as background
    pcm = ax_1.pcolormesh(
        Xg, Yg, Z_cond_LWA,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=0.45
    )

    #conditional mean line
    bins = np.linspace(x.min(), x.max(), 30)
    digitized = np.digitize(x, bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cond_mean = np.array([y[digitized == i].mean() for i in range(1, len(bins))])
    
    cond_lowb = []
    cond_upb = []
    for i in range(1, len(bins)):
        vals = y[digitized == i]
        if vals.size == 0:
            cond_lowb.append(cond_mean[i-1])
            cond_upb.append(cond_mean[i-1])
        else:
            cond_lowb.append(np.percentile(vals, 25))
            cond_upb.append(np.percentile(vals, 75))
    cond_lowb = np.array(cond_lowb)
    cond_upb = np.array(cond_upb)

    ax_1.plot(bin_centers, cond_mean, color='cyan', lw=2, label='conditional mean')
    ax_1.fill_between(bin_centers, cond_lowb, cond_upb, color='cyan', alpha=0.3, label='IQR')

    # light scatter sample: subselect if huge
    if x.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
    else:
        xs_sc = x
        ys_sc = y

    ax_1.scatter(
        xs_sc,
        ys_sc,
        s=8,
        c="white",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    ax_1.set_title("P(T | LWA)")
    # ax_1.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_1.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_1.set_ylabel(y_label + " (K)")
    ax_1.set_xlim(x_limits if x_limits else (np.min(x), np.max(x)))
    ax_1.set_ylim(y_limits if y_limits else (np.min(y), np.max(y)))
    ax_1.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm, ax=ax_1, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # Right main panel (y | x)
    # --------------------
    pcm2 = ax_2.pcolormesh(
        Xg, Yg, Z_cond_T,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=4e-4
    )

    #conditional mean line
    bins = np.linspace(y.min(), y.max(), 30)
    digitized = np.digitize(y, bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cond_mean = np.array([x[digitized == i].mean() for i in range(1, len(bins))])
    
    cond_lowb = []
    cond_upb = []
    for i in range(1, len(bins)):
        vals = x[digitized == i]
        if vals.size == 0:
            cond_lowb.append(cond_mean[i-1])
            cond_upb.append(cond_mean[i-1])
        else:
            cond_lowb.append(np.percentile(vals, 25))
            cond_upb.append(np.percentile(vals, 75))
    cond_lowb = np.array(cond_lowb)
    cond_upb = np.array(cond_upb)

    #conditional mean line
    ax_2.plot(cond_mean, bin_centers, color='cyan', lw=2, label='conditional mean')
    ax_2.fill_betweenx(bin_centers, cond_lowb, cond_upb, color='cyan', alpha=0.3, label='IQR')

    # light scatter sample: subselect if huge
    if y.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(y.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
    else:
        xs_sc = x
        ys_sc = y

    ax_2.scatter(
        xs_sc,
        ys_sc,
        s=8,
        c="white",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    ax_2.set_title("P(LWA | T)")
    # ax_2.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_2.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_2.set_ylabel(y_label + " (K)")
    ax_2.set_xlim(x_limits if x_limits else (np.min(x), np.max(x)))
    ax_2.set_ylim(y_limits if y_limits else (np.min(y), np.max(y)))
    ax_2.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm2, ax=ax_2, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    fig_name = f"{output_path}/{sim}_{LWA_var}_deltaT_conditional_kde_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None


def plot_conditional_kdes_model_vs_ref(
    lwa_can: xr.DataArray,
    deltaT_can: xr.DataArray,
    lwa_era: xr.DataArray,
    deltaT_era: xr.DataArray,
    title: str = "",
    x_label: str = "LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    gridsize: int = 400,
    cmap: str = "RdBu_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) of conditional probability anomaly, between CanESM ensemble and ERA5.

    Layout:
        - LHS main panel: 2D KDE of P(T | LWA) for CanESM - ERA5
        - RHS main panel: 2D KDE of P(LWA | T) for CanESM - ERA5

    Parameters
    ----------
    lwa_can, deltaT_can : np.ndarray 
        dimensions (member, time, lat, lon).
    lwa_era, deltaT_era : np.ndarray
        dimensions (time, lat, lon).
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    x_era, y_era = _stack_clean(lwa_era, deltaT_era, has_member=False)

    if "member" in lwa_can.dims:
        x_can, y_can = _stack_clean(lwa_can, deltaT_can, has_member=True)
    else:
        x_can, y_can = _stack_clean(lwa_can, deltaT_can, has_member=False)

    if x_limits is None:
        xmin, xmax = np.min([x_era.min(), x_can.min()]), np.max([x_era.max(), x_can.max()])
    else:
        xmin, xmax = x_limits

    if y_limits is None:
        ymin, ymax = np.min([y_era.min(), y_can.min()]), np.max([y_era.max(), y_can.max()])
    else:
        ymin, ymax = y_limits

    if x_can.size < 3 or y_can.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    if x_label == 'LWA_a' or x_label == 'LWA':
        leg_loc="upper left"
    elif x_label == 'LWA_c':
        leg_loc="lower left"
    else:
        Warning("x_label not recognized, setting legend location to 'best'")
        leg_loc="best"

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d_era = stats.gaussian_kde(np.vstack([x_era, y_era]))
    kde2d_can = stats.gaussian_kde(np.vstack([x_can, y_can]))
    # grid
    xi = np.linspace(0, xmax*1.25, gridsize+5)
    yi = np.linspace(ymin*1.25, ymax*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    
    Z_era = kde2d_era(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    Z_can = kde2d_can(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    #1D Marginal KDEs for CanESM ( PDFs )

    x_marg_era = np.trapezoid(Z_era, x=yi, axis=0)   # p(LWA=x), shape (nx,)
    y_marg_era = np.trapezoid(Z_era, x=xi, axis=1)   # p(T=y),   shape (ny,)

    x_marg_can = np.trapezoid(Z_can, x=yi, axis=0)   # p(LWA=x), shape (nx,)
    y_marg_can = np.trapezoid(Z_can, x=xi, axis=1)   # p(T=y),   shape (ny,)

    eps = np.finfo(float).eps
    Z_cond_LWA_can = Z_can / np.maximum(x_marg_can, eps)[None, :]   # p(T | LWA)  (normalize columns)
    Z_cond_T_can   = Z_can / np.maximum(y_marg_can, eps)[:, None]   # p(LWA | T)  (normalize rows)

    Z_cond_LWA_era = Z_era / np.maximum(x_marg_era, eps)[None, :]   # p(T | LWA)  (normalize columns)
    Z_cond_T_era   = Z_era / np.maximum(y_marg_era, eps)[:, None]   # p(LWA | T)  (normalize rows)

    Z_cond_LWA_anom = Z_cond_LWA_can - Z_cond_LWA_era
    Z_cond_T_anom   = Z_cond_T_can - Z_cond_T_era

    # ---------------------------------------------------------------------
    # 4. Layout: two main conditional panels + cbar below
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig, axs = plt.subplots(ncols=2, figsize=(13, 8), tight_layout=True)
    ax_1     = axs[0]
    ax_2     = axs[1]

    # --------------------
    # Left main panel (x | y)
    # --------------------
    # KDE as background
    pcm = ax_1.pcolormesh(
        Xg, Yg, Z_cond_LWA_anom,
        shading="auto",
        cmap=cmap,
        vmin=-0.3,
        vmax=0.3
    )

    #conditional mean line
    bins = np.linspace(xmin, xmax, 30)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    digitized_can = np.digitize(x_can, bins)
    cond_mean_can = np.array([y_can[digitized_can == i].mean() for i in range(1, len(bins))])

    digitized_era = np.digitize(x_era, bins)
    cond_mean_era = np.array([y_era[digitized_era == i].mean() for i in range(1, len(bins))])
    
    cond_lowb_can = []
    cond_upb_can = []
    for i in range(1, len(bins)):
        vals = y_can[digitized_can == i]
        if vals.size == 0:
            cond_lowb_can.append(cond_mean_can[i-1])
            cond_upb_can.append(cond_mean_can[i-1])
        else:
            cond_lowb_can.append(np.percentile(vals, 25))
            cond_upb_can.append(np.percentile(vals, 75))
    cond_lowb_can = np.array(cond_lowb_can)
    cond_upb_can = np.array(cond_upb_can)

    cond_lowb_era = []
    cond_upb_era = []
    for i in range(1, len(bins)):
        vals = y_era[digitized_era == i]
        if vals.size == 0:
            cond_lowb_era.append(cond_mean_era[i-1])
            cond_upb_era.append(cond_mean_era[i-1])
        else:
            cond_lowb_era.append(np.percentile(vals, 25))
            cond_upb_era.append(np.percentile(vals, 75))
    cond_lowb_era = np.array(cond_lowb_era)
    cond_upb_era = np.array(cond_upb_era)

    ax_1.plot(bin_centers, cond_mean_can, color='yellow', lw=2)
    ax_1.fill_between(bin_centers, cond_lowb_can, cond_upb_can, color='yellow', alpha=0.3)
    
    ax_1.plot(bin_centers, cond_mean_era, color='cyan', lw=2, label='conditional mean (ERA5)')
    ax_1.fill_between(bin_centers, cond_lowb_era, cond_upb_era, color='cyan', alpha=0.3, label='IQR (ERA5)')

    ax_1.set_title("P(T | LWA)")
    # ax_1.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_1.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_1.set_ylabel(y_label + " (K)")
    ax_1.set_xlim(x_limits if x_limits else (xmin, xmax))
    ax_1.set_ylim(y_limits if y_limits else (ymin, ymax))
    ax_1.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm, ax=ax_1, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # Right main panel (y | x)
    # --------------------
    pcm2 = ax_2.pcolormesh(
        Xg, Yg, Z_cond_T_anom,
        shading="auto",
        cmap=cmap,
        vmin=-3e-4,
        vmax=3e-4
    )

    #conditional mean line
    bins = np.linspace(ymin, ymax, 30)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    digitized_can = np.digitize(y_can, bins)
    cond_mean_can = np.array([x_can[digitized_can == i].mean() for i in range(1, len(bins))])

    digitized_era = np.digitize(y_era, bins)
    cond_mean_era = np.array([x_era[digitized_era == i].mean() for i in range(1, len(bins))])
    
    cond_lowb_can = []
    cond_upb_can = []
    for i in range(1, len(bins)):
        vals = x_can[digitized_can == i]
        if vals.size == 0:
            cond_lowb_can.append(cond_mean_can[i-1])
            cond_upb_can.append(cond_mean_can[i-1])
        else:
            cond_lowb_can.append(np.percentile(vals, 25))
            cond_upb_can.append(np.percentile(vals, 75))
    cond_lowb_can = np.array(cond_lowb_can)
    cond_upb_can = np.array(cond_upb_can)

    cond_lowb_era = []
    cond_upb_era = []
    for i in range(1, len(bins)):
        vals = x_era[digitized_era == i]
        if vals.size == 0:
            cond_lowb_era.append(cond_mean_era[i-1])
            cond_upb_era.append(cond_mean_era[i-1])
        else:
            cond_lowb_era.append(np.percentile(vals, 25))
            cond_upb_era.append(np.percentile(vals, 75))
    cond_lowb_era = np.array(cond_lowb_era)
    cond_upb_era = np.array(cond_upb_era)

    ax_2.plot(cond_mean_can, bin_centers, color='yellow', lw=2)
    ax_2.fill_betweenx(bin_centers, cond_lowb_can, cond_upb_can, color='yellow', alpha=0.3)
    
    ax_2.plot(cond_mean_era, bin_centers, color='cyan', lw=2, label='conditional mean (ERA5)')
    ax_2.fill_betweenx(bin_centers, cond_lowb_era, cond_upb_era, color='cyan', alpha=0.3, label='IQR (ERA5)')

    ax_2.set_title("P(LWA | T)")
    # ax_2.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_2.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_2.set_ylabel(y_label + " (K)")
    ax_2.set_xlim(x_limits if x_limits else (xmin, xmax))
    ax_2.set_ylim(y_limits if y_limits else (ymin, ymax))
    ax_2.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm2, ax=ax_2, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    fig_name = f"{output_path}/KDE_{LWA_var}_conditional_modelvref_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None


# ------------------------------ Main ----------------------------------

def main(REGION, LWA_var, SEASON, ZG_COORD):

    #load in data

    ds_tas_canesm = open_temp_data_canesm(VAR, ENSEMBLE_LIST)
    ds_tas_canesm = ds_tas_canesm.chunk({"time": 365})
    ds_tas_canesm = compute_region_mean(ds_tas_canesm, REGION).compute()
    
    ds_tas_canesm_clim = ds_tas_canesm.groupby("time.dayofyear").mean("time")
    ds_tas_canesm_anom = ds_tas_canesm.groupby("time.dayofyear") - ds_tas_canesm_clim

    ds_tas_era5 = open_temp_data_era5(VAR)
    ds_tas_era5 = ds_tas_era5.chunk({"time": 365})
    ds_tas_era5 = compute_region_mean(ds_tas_era5, REGION).compute()
    
    ds_tas_era5_clim = ds_tas_era5.groupby("time.dayofyear").mean("time")
    ds_tas_era5_anom = ds_tas_era5.groupby("time.dayofyear") - ds_tas_era5_clim

    ds_tas_canesm_anom = ds_tas_canesm_anom.assign_coords(time=ds_tas_canesm_anom.time.dt.floor("D"))
    ds_tas_era5_anom = ds_tas_era5_anom.assign_coords(time=ds_tas_era5_anom.time.dt.floor("D")) #ensure time coords match
    
    ds_mrsos_canesm = open_mrsos_data_canesm(var='mrsos', ensemble_list=ENSEMBLE_LIST)
    ds_mrsos_canesm = compute_region_mean(ds_mrsos_canesm, REGION).compute()

    ds_mrsos_canesm_clim = ds_mrsos_canesm.groupby("time.dayofyear").mean("time")
    ds_mrsos_canesm_anom = ds_mrsos_canesm.groupby("time.dayofyear") - ds_mrsos_canesm_clim

    ds_mrsos_era5 = open_mrsos_data_era5(var='swvl1')
    ds_mrsos_era5 = compute_region_mean(ds_mrsos_era5, REGION).compute()

    ds_mrsos_era5_clim = ds_mrsos_era5.groupby("time.dayofyear").mean("time")
    ds_mrsos_era5_anom = ds_mrsos_era5.groupby("time.dayofyear") - ds_mrsos_era5_clim

    print(ds_mrsos_canesm)
    print(ds_mrsos_era5)



    # doy_all_canesm = ds_tas_canesm.time.dt.dayofyear
    # doy_all_era5   = ds_tas_era5.time.dt.dayofyear

    # ds_tas_yearly_canesm = ds_tas_canesm.groupby("time.year")
    # ds_tas_yearly_era5   = ds_tas_era5.groupby("time.year")

    # print(ds_tas_canesm)
    # print(ds_tas_era5)



    ## loading LWA data
    # 1) Open CanESM and ERA5 for all VARiables
    ds_canesm_lwas: Dict[str, xr.DataArray] = open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas: Dict[str, xr.DataArray]   = open_era5_lwa(ZG_COORD)

    ds_canesm_lwa = ds_canesm_lwas[LWA_var]#.sel(MEMBER=MEMBER)
    ds_era5_lwa   = ds_era5_lwas[LWA_var]

    ds_canesm_lwa_reg = compute_region_mean(ds_canesm_lwa, REGION).chunk({"time": 365}).compute()
    ds_era5_lwa_reg   = compute_region_mean(ds_era5_lwa, REGION).chunk({"time": 365}).compute()

    ds_canesm_lwa_reg = xr.apply_ufunc(np.sqrt, ds_canesm_lwa_reg)
    ds_era5_lwa_reg   = xr.apply_ufunc(np.sqrt, ds_era5_lwa_reg)

    ds_canesm_lwa_reg = ds_canesm_lwa_reg.assign_coords(time=ds_canesm_lwa_reg.time.dt.floor("D"))
    ds_era5_lwa_reg = ds_era5_lwa_reg.assign_coords(time=ds_era5_lwa_reg.time.dt.floor("D"))

    ds_mrsos_canesm_anom = ds_mrsos_canesm_anom.assign_coords(time=ds_mrsos_canesm_anom.time.dt.floor("D"))
    ds_mrsos_era5_anom   = ds_mrsos_era5_anom.assign_coords(time=ds_mrsos_era5_anom.time.dt.floor("D"))

    # ds_canesm_lwa_reg_yearly = ds_canesm_lwa_reg.groupby("time.year")
    # ds_era5_lwa_reg_yearly   = ds_era5_lwa_reg.groupby("time.year")

    # plot HW thresholds
    # plot_temp_thresh_canesm_era5(ds_thresh_canesm_smooth, ds_thresh_era5_smooth, ds_tas_era5, OUTPUT_PLOTS_PATH)
    # plot_lwa_thresh_canesm_era5(ds_lwa_thresh_canesm_smooth[LWA_var], ds_lwa_thresh_era5_smooth[LWA_var], ds_era5_lwa_reg, OUTPUT_PLOTS_PATH)

    # mask_era = (ds_era5_lwa_reg.groupby("time.dayofyear") >= ds_lwa_thresh_era5_smooth[LWA_var]).compute()
    # mask_can = (ds_canesm_lwa_reg.groupby("time.dayofyear") >= ds_lwa_thresh_canesm_smooth[LWA_var].isel(time=0)).compute()

    season_mask_era = ds_era5_lwa_reg.time.dt.season == SEASON
    season_mask_can = ds_canesm_lwa_reg.time.dt.season == SEASON

    masked_lwa_era = ds_era5_lwa_reg.where(season_mask_era,drop=True)
    masked_lwa_can = ds_canesm_lwa_reg.where(season_mask_can,drop=True)

    masked_tas_era = ds_tas_era5_anom.where(season_mask_era,drop=True)
    masked_tas_can = ds_tas_canesm_anom.where(season_mask_can,drop=True)

    masked_mrsos_era = ds_mrsos_era5_anom.where(season_mask_era,drop=True)
    masked_mrsos_can = ds_mrsos_canesm_anom.where(season_mask_can,drop=True)

    print(masked_mrsos_can, masked_lwa_can)
    print(masked_mrsos_era, masked_lwa_era)


    fig,ax = plt.subplots(figsize=(8,6))
    ax.scatter(masked_lwa_era, masked_mrsos_era, color='blue', alpha=0.5, label='ERA5')
    ax.scatter(masked_lwa_can, masked_mrsos_can, color='red', alpha=0.5, label='CanESM')
    ax.set_xlabel(f"{LWA_var} (sqrt hPa m)")
    ax.set_ylabel("Soil Moisture Anomaly (kg/m2)")
    ax.set_title(f"{LWA_var} vs Soil Moisture Anomaly, {REGION}, {SEASON}")
    ax.legend()
    fig_name = f"{OUTPUT_PLOTS_PATH}/{LWA_var}_vs_SoilMoisture_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')


    print(masked_lwa_era)
    print(masked_tas_era)

    x_lo = np.min([masked_lwa_era.min().item(), masked_lwa_can.min().item()])
    x_hi = np.max([masked_lwa_era.max().item(), masked_lwa_can.max().item()])
    y_lo = np.min([masked_tas_era.min().item(), masked_tas_can.min().item()])
    y_hi = np.max([masked_tas_era.max().item(), masked_tas_can.max().item()])
    z_lo = np.min([masked_mrsos_era.min().item(), masked_mrsos_can.min().item()])
    z_hi = np.max([masked_mrsos_era.max().item(), masked_mrsos_can.max().item()])

    z_max = np.max([np.abs(z_lo), np.abs(z_hi)])

    plot_joint_kde_with_marginals_mrsos(
        masked_lwa_era,
        masked_tas_era,
        masked_mrsos_era,
        title=f"ERA5 {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="ERA5",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        z_limits=(-z_max, z_max),
        output_path=OUTPUT_PLOTS_PATH
    )



    plot_joint_kde_with_marginals_mrsos(
        masked_lwa_can,
        masked_tas_can,
        masked_mrsos_can,
        title=f"CanESM {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="CanESM",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        z_limits=(-z_max, z_max),
        output_path=OUTPUT_PLOTS_PATH
    )


    # --- PyMC fit for ERA5: deltaT ~ LWA + SM + LWA*SM with AR(1) residual ---
    x1, x2, y = _prep_design_matrix(
        lwa=masked_lwa_era,
        sm=masked_mrsos_era,
        dt=masked_tas_era
    )

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
        az.to_netcdf(idata_model_era, f"{OUTPUT_POSTERIOR}/pymc_ar1_{model_name}_{LWA_var}_tas_sm_{REGION}_{SEASON}_ERA5.nc")


    # --- PyMC loop-per-member for CanESM5 ---


    # Ensure consistent dim order (member, time) for easy indexing
    def _ensure_member_time(da: xr.DataArray) -> xr.DataArray:
        if "member" not in da.dims or "time" not in da.dims:
            raise ValueError(f"Expected dims to include ('member','time'), got {da.dims}")
        # transpose if needed
        if da.dims != ("member", "time"):
            da = da.transpose("member", "time")
        return da

    masked_lwa_can   = _ensure_member_time(masked_lwa_can)
    masked_mrsos_can = _ensure_member_time(masked_mrsos_can)
    masked_tas_can   = _ensure_member_time(masked_tas_can)

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


        for model_name in model_variations:
            model_ = model_variations[model_name]
            print(f"\n--- Fitting member {m} model: {model_name} ---")
            idata_model_m = fit_pymc_ar1_model(
                time=dt_m.time,   # IMPORTANT: use the member’s time coord (same as others after align)
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
                    OUTPUT_POSTERIOR,
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
        OUTPUT_POSTERIOR,
        f"pymc_ar1_full_model_summaries_{LWA_var}_tas_sm_{REGION}_{SEASON}_CanESM5.csv"
    )
    summary_df.to_csv(summary_outfile)

    print("\nDone. Member-level summaries collected.")



    '''
    x_lo = np.min([masked_lwa_era.min().item(), masked_lwa_can.min().item()])
    x_hi = np.max([masked_lwa_era.max().item(), masked_lwa_can.max().item()])
    y_lo = np.min([masked_tas_era.min().item(), masked_tas_can.min().item()])
    y_hi = np.max([masked_tas_era.max().item(), masked_tas_can.max().item()])

    plot_joint_kde_with_marginals(
        masked_lwa_era,
        masked_tas_era,
        title=f"ERA5 {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="ERA5",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    plot_conditional_kdes(
        x_in=masked_lwa_era,
        y_in=masked_tas_era,
        title=f"ERA5 {LWA_var} vs TAS (conditional), {REGION}, {SEASON}",
        sim="ERA5",
        x_label=f"{LWA_var}", #just lwa_var
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )
    
    plot_joint_kde_with_marginals(
        masked_lwa_can,
        masked_tas_can,
        title=f"CanESM {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="CanESM",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    plot_conditional_kdes(
        x_in=masked_lwa_can,
        y_in=masked_tas_can,
        title=f"CanESM {LWA_var} vs TAS (conditional), {REGION}, {SEASON}",
        sim="CanESM",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    plot_kde_comparison_model_vs_ref(
        masked_lwa_can,  # xr.DataArray, dims ('member','time') or ('time',) #model, CanESM
        masked_tas_can,  # xr.DataArray, dims ('member','time') or ('time',)
        masked_lwa_era,  # xr.DataArray, dims ('time',) #reference, ERA5
        masked_tas_era,  # xr.DataArray, dims ('time',)
        title = "CanESM vs ERA5 KDE Comparison",
        x_label = f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )


    plot_conditional_kdes_model_vs_ref(
        lwa_can=masked_lwa_can,
        deltaT_can=masked_tas_can,
        lwa_era=masked_lwa_era,
        deltaT_era=masked_tas_era,
        title=f"CanESM vs ERA5 {LWA_var} vs TAS (conditional), {REGION}, {SEASON}",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    '''




if __name__ == "__main__":
    args = arg_parser()
    REGION   = args.region
    LWA_var  = args.lwa_var
    SEASON   = args.season
    ZG_LEVEL = args.zg

    main(REGION, LWA_var, SEASON, ZG_LEVEL)
