import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    "r1i1p1f1", "r2i1p1f1","r3i1p1f1", "r4i1p1f1", "r5i1p1f1",
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
    "canada":       (slice(40, 70), slice(-140,    -60)),
    "canada_north": (slice(55, 70), slice(-140, -60)),
    "canada_south": (slice(40, 55), slice(-140, -60)),
    "west":       (slice(40, 70), slice(-140,    -113.33)),
    "west_north": (slice(55, 70), slice(-140, -113.33)),
    "west_south": (slice(40, 55), slice(-140, -113.33)),
    "central":       (slice(40, 70), slice(-113.33, -88.66)),
    "central_north": (slice(55, 70), slice(-113.33, -88.66)),
    "central_south": (slice(40, 55), slice(-113.33, -88.66)),
    "east":       (slice(40, 70), slice(-88.66,  -60)),
    "east_north": (slice(55, 70), slice(-88.66, -60)),
    "east_south": (slice(40, 55), slice(-88.66, -60)),
    "pnw_bartusek":        (slice(40, 60), slice(-130.0, -110.0)),  # Bartusek et al. 2023
}

SEASON_NAMES = {"DJF", "MAM", "JJA", "SON"}

# LWA_var:str = "LWA_c"

VAR:str      = "tas"
# REGION:str   = "west_south"
# SEASON:str   = "JJA"
# ZG_COORD:int = 500


# Root paths (adapt if needed)
CANESM_LWA_ROOT = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/CanESM5/historical"
ERA5_LWA_ROOT   = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/ERA5"

# HW_THRESH_ROOT  = "/space/hall5/sitestore/eccc/crd/ccrn/users/mpw000/development/HW_analysis/thresholds"
LWA_THRESH_ROOT = "/home/mhpereir/LWA_analysis/lwa_thresholds"
ERA5_TAS_ROOT   = "/home/mhpereir/data-mhpereir/standard_grid_daily/REANALYSIS/ERA5/tas"
CANESM_TAS_ROOT = "/home/mhpereir/data-mhpereir/standard_grid_daily/CMIP6/CanESM5/tas/historical"

OUTPUT_PLOTS_PATH = "/home/mhpereir/LWA_analysis/plots/event_duration"
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

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
        "--lwa_quantile",
        type=int,
        default=90,
        help="LWA quantile to analyze.",
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
    parser.add_argument(
        "--event_duration_threshold",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=2,
        help="Minimum event duration (days) to include in dashed composite line.",
    )
    return parser.parse_args()



# ------------------------------ I/O Utilities --------------------------------

# def open_thresh_data_canesm(var, ensemble_list, q, region):
#     path_to_data = HW_THRESH_ROOT
    
#     files_by_member = []
#     for m in ensemble_list:
#         filepat = f"{path_to_data}/CanESM5_HWthresh_block_1970_2014_{var}_q{q}_{region}_{m}.nc"
#         files = sorted(glob.glob(filepat))
#         if not files:
#             raise FileNotFoundError(f"No files matched for {m}; pattern: {filepat}")
#         files_by_member.append(files)

#     # 2) Open all lazily and stack along new 'member' then 'time'
#     ds_CanESM = xr.open_mfdataset(
#         files_by_member,
#         combine="nested",
#         concat_dim=["member", "time"],   # first stacks member, then concatenates their time files
#         parallel=True,
#         engine="h5netcdf",
#         chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
#     )

#     # 3) Give the member axis meaningful labels
#     ds_CanESM = ds_CanESM.assign_coords(member=("member", ensemble_list))
#     var_name = f"{var}_thresh_p{q}_win31"
#     return ds_CanESM[var_name]


# def open_thresh_data_era5(var, q, region):
#     path_to_data = HW_THRESH_ROOT

#     filepat = f"{path_to_data}/ERA5_HWthresh_block_1970_2014_{var}_q{q}_{region}.nc"

#     # 2) Open all lazily and stack along new 'member' then 'time'
#     ds_ERA = xr.open_dataset(
#         filepat,
#         engine="h5netcdf",
#         chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
#     )

#     # 3) Give the member axis meaningful labels
#     var_name = f"{var}_thresh_p{q}_win31"
#     return ds_ERA[var_name]


def open_lwa_thresh_data_canesm(ensemble_list, q, region, zg_coord):
    path_to_data = LWA_THRESH_ROOT
    
    files_by_MEMBER = []
    for m in ensemble_list:
        filepat = f"{path_to_data}/CanESM5_LWAthresh_block_1970_2014_q{q}_{region}_{m}.{zg_coord}.nc"
        files = sorted(glob.glob(filepat))
        if not files:
            raise FileNotFoundError(f"No files matched for {m}; pattern: {filepat}")
        files_by_MEMBER.append(files)

    # 2) Open all lazily and stack along new 'MEMBER' then 'time'
    ds_CanESM = xr.open_mfdataset(
        files_by_MEMBER,
        combine="nested",
        concat_dim=["member", "time"],   # first stacks MEMBERs, then concatenates their time files
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
    )

    # 3) Give the MEMBER axis meaningful labels
    ds_CanESM = ds_CanESM.assign_coords(member=("member", ensemble_list))
    #VAR_name = f"{VAR}_thresh_p{Q}_win31"
    return ds_CanESM#[VAR_name]


def open_lwa_thresh_data_era5(q, region, zg_coord):
    path_to_data = LWA_THRESH_ROOT

    filepat = f"{path_to_data}/ERA5_LWAthresh_block_1970_2014_q{q}_{region}.{zg_coord}.nc"

    # 2) Open all lazily and stack along new 'MEMBER' then 'time'
    ds_ERA5 = xr.open_dataset(
        filepat,
        engine="h5netcdf",
        chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
    )

    # 3) Give the MEMBER axis meaningful labels
    return ds_ERA5



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

    print(ds)

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # # --- Ensure we have an indexable time axis ---
    # if "time" not in ds.dims:
    #     raise ValueError("Input Dataset must have a 'time' dimension.")

    # if "time" not in ds.coords:
    #     if "valid_time" in ds.coords and ds["valid_time"].dims == ("time",):
    #         # Make valid_time the dimension (indexable), then rename to time
    #         ds = ds.swap_dims({"time": "valid_time"}).rename({"valid_time": "time"})
    #         ds = ds.chunk({"time": 365})  # re-chunk after swap
    #     else:
    #         raise ValueError(
    #             "Dataset has a 'time' dimension but no indexable time coordinate "
    #             "(expected valid_time(time))."
    #         )

    ds = ds.sel(time=TIME_SLICE)
    #remove leap days
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    # crop to lat slice
    ds = ds.sel(lat=LAT_SLICE)
    # 3) Give the MEMBER axis meaningful labels
    var_name = f"{var}"
    return ds[var_name]


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


def mask_to_events(mask: xr.DataArray) -> xr.DataArray:
    """
    Convert a boolean mask (True=event) into event IDs.

    For each contiguous True segment along 'time', replace with event ID.
    E.g., [F, T, F, T, T, F, T, T] -> [0, 1, 0, 2, 2, 0, 3, 3]
    Works for DataArrays with or without 'member' dimension.
    """
    if "member" in mask.dims:
        # Apply along 'time' for each 'member'
        desired_chunks = {}
        desired_chunks["member"] = 1
        if "time" in mask.dims:
            desired_chunks["time"] = mask.sizes["time"]
        mask = mask.chunk(desired_chunks)
        
        out_datarray = xr.apply_ufunc(
            _process_mask_1d,
            mask,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int]
        ).rename("event_id")

        return out_datarray
    else:
        # Single time series
        event_ids = _process_mask_1d(np.asarray(mask))
        out_datarray = xr.DataArray(
            event_ids,
            coords=mask.coords,
            dims=mask.dims,
            name="event_id"
        )
        return out_datarray


def filter_events_by_season(events: xr.DataArray, target_season: str) -> xr.DataArray:
    """
    Keep only events whose entire lifetime occurs within `target_season`.

    Works for arrays with or without 'member' dimension.
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


def compute_doy_climatology(da: xr.DataArray) -> xr.DataArray:
    """Compute day-of-year climatology from daily time series."""
    doy = da.time.dt.dayofyear
    return da.groupby(doy).mean(dim="time")

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

# def _event_durations_1d(arr: np.ndarray) -> np.ndarray:
#         durations = np.zeros_like(arr, dtype=int)
#         in_event = False
#         event_start = 0

#         for i in range(len(arr)):
#             if arr[i] and not in_event:
#                 in_event = True
#                 event_start = i
#             elif not arr[i] and in_event:
#                 in_event = False
#                 event_length = i - event_start
#                 durations[event_start:i] = event_length

#         # Handle case where event goes to end of array
#         if in_event:
#             event_length = len(arr) - event_start
#             durations[event_start:] = event_length

#         return durations

def _process_mask_1d(arr: np.ndarray) -> np.ndarray:
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


def _filter_events_1d(event_ids: np.ndarray, seasons: np.ndarray, target_season: str) -> np.ndarray:
        """Return event_ids with only those entirely in target_season kept."""
        raw_ids = np.unique(event_ids)
        raw_ids = raw_ids[~np.isnan(raw_ids)]
        raw_ids = raw_ids.astype(int)

        valid_ids = []
        for eid in raw_ids:
            if eid <= 0:
                continue
            uniq = np.unique(seasons[event_ids == eid])
            if (len(uniq) == 1) and (uniq[0] == target_season):
                valid_ids.append(eid)

        valid_ids = np.asarray(valid_ids, dtype=int)
        return np.where(np.isin(event_ids, valid_ids), event_ids, 0)


def _durations_from_ids(arr: np.ndarray) -> np.ndarray:
        """Return counts for each non-zero event id."""
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            return np.array([], dtype=int)
        valid = arr[arr > 0]
        if valid.size == 0:
            return np.array([], dtype=int)
        _, counts = np.unique(valid, return_counts=True)
        return counts


def _composite_1d(
    tas_series: xr.DataArray,
    lwa_series: xr.DataArray,
    event_ids: xr.DataArray,
    half_window: int,
    n_day_min: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
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
    n_day_min: int
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
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
                n_day_min
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
        tas_comp, lwa_comp = _composite_1d(tas_da, lwa_da, event_da, half_window, n_day_min)
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

# ------------------------------ Plotting ----------------------------------

# def plot_temp_thresh_canesm_era5(ds_hw_thresh_canesm: xr.DataArray, #dim("member", "dayofyear")
#                                  ds_hw_thresh_era5: xr.DataArray,   #dim("dayofyear")
#                                  ds_tas_era5: xr.DataArray,  #grouped by year
#                                  output_path: str):
#     """Plot temperature thresholds for CanESM and ERA5."""

#     ds_tas_yearly_era5 = ds_tas_era5.groupby("time.year")

#     hw_thresh_canesm_min = ds_hw_thresh_canesm.quantile(0.05, dim="member")
#     hw_thresh_canesm_max = ds_hw_thresh_canesm.quantile(0.95, dim="member")

#     doy = ds_hw_thresh_canesm.dayofyear

#     fig,ax = plt.subplots()

#     ax.plot(doy, ds_hw_thresh_canesm.mean(dim="member"), color='k', label=f'CanESM q={Q_TEMP} mean')
#     ax.fill_between(doy, hw_thresh_canesm_min, hw_thresh_canesm_max, color='gray', alpha=0.5)

#     ax.plot(doy, ds_hw_thresh_era5, color='blue', label=f'ERA5 q={Q_TEMP}')

#     for year in ds_tas_yearly_era5.groups.keys():
#         if year == 1970:
#             ax.plot(
#                 doy,
#                 ds_tas_yearly_era5[year],
#                 color='tab:blue',
#                 alpha=0.5,
#                 label='ERA5 Yearly'
#             )
#             continue
#         else:
#             ax.plot(
#                 doy,
#                 ds_tas_yearly_era5[year],
#                 color='tab:blue',
#                 alpha=0.1
#             )

#     ax.set_xlabel('Day of Year')
#     ax.set_ylabel('Temperature (K)')
#     ax.legend()
#     outfile = f"{output_path}/HW_TAS_v_THRESH_Q{Q_TEMP}_{REGION}_{SEASON}.png"
#     fig.savefig(outfile, dpi=300, bbox_inches='tight')



# def plot_lwa_thresh_canesm_era5(ds_lwa_thresh_canesm: xr.DataArray, #dim("member", "dayofyear")
#                                 ds_lwa_thresh_era5: xr.DataArray,   #dim("dayofyear")
#                                 ds_lwa_era5: xr.DataArray,  #grouped by year
#                                 output_path: str):
#     """Plot temperature thresholds for CanESM and ERA5."""

#     ds_lwa_yearly_era5 = ds_lwa_era5.groupby("time.year")

#     ds_lwa_thresh_canesm = ds_lwa_thresh_canesm.isel(time=0)

#     lwa_thresh_canesm_min = ds_lwa_thresh_canesm.quantile(0.05, dim="member")
#     lwa_thresh_canesm_max = ds_lwa_thresh_canesm.quantile(0.95, dim="member")

#     doy = ds_lwa_thresh_canesm.dayofyear

#     fig,ax = plt.subplots()

#     ax.plot(doy, ds_lwa_thresh_canesm.mean(dim="member"), color='k', label=f'CanESM q={Q_TEMP} mean')
#     ax.fill_between(doy, lwa_thresh_canesm_min, lwa_thresh_canesm_max, color='gray', alpha=0.5)

#     ax.plot(doy, ds_lwa_thresh_era5, color='blue', label=f'ERA5 q={Q_TEMP}')

#     for year in ds_lwa_yearly_era5.groups.keys():
#         if year == 1970:
#             ax.plot(
#                 doy,
#                 ds_lwa_yearly_era5[year],
#                 color='tab:blue',
#                 alpha=0.5,
#                 label='ERA5 Yearly'
#             )
#             continue
#         else:
#             ax.plot(
#                 doy,
#                 ds_lwa_yearly_era5[year],
#                 color='tab:blue',
#                 alpha=0.1
#             )

#     ax.set_xlabel('Day of Year')
#     ax.set_ylabel(f'{LWA_var} (m)')
#     ax.legend()
#     outfile = f"{output_path}/{LWA_var}_v_THRESH_Q{Q_LWA}_{REGION}_{SEASON}.png"
#     fig.savefig(outfile, dpi=300, bbox_inches='tight')



def plot_histogram_of_duration(
    lwa_can: xr.DataArray, # CanESM list of event IDs
    lwa_era: xr.DataArray, # ERA5 list of event IDs
    title: str = "",
    sim: str = "",
    x_label: str = "Event duration [d]",
    y_label: str = "Frequency",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """Plot histogram of event durations for CanESM and ERA5."""

    #calculate event durations from IDs
    # duration = total number of days with the same event ID
    
    if "member" in lwa_can.dims:
        can_duration_lists: List[np.ndarray] = [
            _durations_from_ids(lwa_can.sel(member=member).values)
            for member in lwa_can.member.values
        ]
        can_duration_values = np.concatenate(
            [vals for vals in can_duration_lists if vals.size], axis=0
        ) if any(vals.size for vals in can_duration_lists) else np.array([], dtype=int)
    else:
        can_duration_values = _durations_from_ids(lwa_can.values)
        can_duration_lists = [can_duration_values]
    
    era_duration_values = _durations_from_ids(lwa_era.values)

    max_duration = 0
    if can_duration_values.size:
        max_duration = max(max_duration, int(can_duration_values.max()))
    if era_duration_values.size:
        max_duration = max(max_duration, int(era_duration_values.max()))
    max_duration =  30# max(max_duration, 1)

    bins = (np.arange(1, max_duration + 2) - 0.5).tolist()

    fig, ax = plt.subplots(figsize=(10,8), tight_layout=True)

    # --- CanESM: filled + outlined
    ax.hist(
        can_duration_values,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color="orange",
        alpha=0.35,
        label="CanESM",
    )
    ax.hist(
        can_duration_values,
        bins=bins,
        density=True,
        histtype="step",
        color="orange",
        linewidth=1.6,
    )

    # --- ERA5: black outline only
    ax.hist(
        era_duration_values,
        bins=bins,
        density=True,
        histtype="step",
        color="black",
        linewidth=2.0,
        label="ERA5",
    )

    # --- CanESM ensemble spread per bin (5th–95th of member-wise bin heights)
    # Build per-member histograms so spread reflects ensemble variability in bin frequency
    if "member" in lwa_can.dims:
        member_hists = []
        for vals in can_duration_lists:
            if vals.size == 0:
                # If a member had no events, contribute zeros (keeps alignment)
                member_hists.append(np.zeros(len(bins) - 1, dtype=float))
            else:
                h, _ = np.histogram(vals, bins=bins, density=True)
                member_hists.append(h)
        H = np.vstack(member_hists)  # (n_member, n_bins)
    else:
        # Single-field case: no spread, just compute one histogram
        H = np.histogram(can_duration_values, bins=bins, density=True)[0][None, :]

    p05 = np.percentile(H, 5, axis=0)
    p95 = np.percentile(H, 95, axis=0)

    # Draw vertical lines (whiskers) with T-caps.
    # Use errorbar to get proper "T" end caps; center at midpoint of [p05, p95]
    y_mid = 0.5 * (p05 + p95)
    yerr_lower = y_mid - p05
    yerr_upper = p95 - y_mid
    ax.errorbar(
        np.asarray(bins[1:])-0.5,
        y_mid,
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor="orange",
        elinewidth=1.8,
        capsize=6,
        capthick=1.8,
        zorder=5,
    )

    # --- cosmetics
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, linestyle="--")

    fig_name = f"{output_path}/{LWA_var}_Q{Q_LWA}_duration_histogram_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")


    return None


def _plot_panel(ax,  
                canesm_comp, 
                era_comp,
                canesm_comp_3d,
                era_comp_3d,
                ylabel,
                panel_title, 
                lag_coord, 
                legend_loc="best",
                event_duration_threshold: int | None = None):
    
    has_any = False
    if canesm_comp is not None:
        canesm_mean = canesm_comp.mean(dim="member")
        canesm_p05 = canesm_comp.quantile(0.05, dim="member")
        canesm_p95 = canesm_comp.quantile(0.95, dim="member")
        ax.plot(
            lag_coord,
            canesm_mean,
            color="tab:orange",
            label="CanESM5 mean",
            linewidth=2
        )
        ax.fill_between(
            lag_coord,
            canesm_p05,
            canesm_p95,
            color="tab:orange",
            alpha=0.3,
            label="CanESM5 5–95%",
        )
        has_any = True

    if canesm_comp_3d is not None:
        canesm_mean = canesm_comp_3d.mean(dim="member")
        ax.plot(
            lag_coord,
            canesm_mean,
            color="tab:red",
            linestyle="--",
            linewidth=2,
            label=f"CanESM5 ≥{event_duration_threshold} d" if event_duration_threshold is not None else "CanESM5 (duration filter)",
        )
        has_any = True

    if era_comp is not None:
        ax.plot(
            lag_coord,
            era_comp,
            color="tab:blue",
            linewidth=2,
            label="ERA5",
        )
        has_any = True

    if era_comp_3d is not None:
        ax.plot(
            lag_coord,
            era_comp_3d,
            color="tab:cyan",
            linewidth=2,
            linestyle="--",
            label=f"ERA5 ≥{event_duration_threshold} d" if event_duration_threshold is not None else "ERA5 (duration filter)",
        )
        has_any = True

    ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)
    if has_any:
        ax.legend(loc=legend_loc)
    ax.grid(alpha=0.2, linestyle="--")



def _plot_panel_all(ax,  
                era_comp,
                era_comp_3d,
                ylabel,
                panel_title, 
                lag_coord, 
                legend_loc="best",
                event_duration_threshold: int | None = None):
    
    has_any = False
   
    if era_comp is not None:
        ax.plot(
            lag_coord,
            era_comp,
            color="tab:blue",
            linewidth=2,
            label="ERA5",
        )
        has_any = True

    if era_comp_3d is not None:
        ax.plot(
            lag_coord,
            era_comp_3d,
            color="tab:cyan",
            linewidth=2,
            linestyle="--",
            label=f"ERA5 ≥{event_duration_threshold} d" if event_duration_threshold is not None else "ERA5 (duration filter)"
        )
        has_any = True

    ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)
    if has_any:
        ax.legend(loc=legend_loc)
    ax.grid(alpha=0.2, linestyle="--")


def plot_composite_TAS_during_LWA_events(
    ds_tas_canesm: xr.DataArray,  # dim("member", "time") regional average TAS
    ds_tas_era5: xr.DataArray,    # dim("time") regional average TAS
    lwa_can: xr.DataArray,        # dim("member", "time") regional average LWA
    lwa_era: xr.DataArray,        # dim("time") regional average LWA
    events_can: xr.DataArray,     # dim("member", "time") event IDs
    events_era: xr.DataArray,     # dim("time") event IDs
    half_window: int,
    event_duration_threshold: int,
    title: str = "",
    x_label: str = "Days relative to LWA peak",
    y_label: str = "TAS (K)",
    output_path: str = './',
) -> None: 
    """
    Plot composites of TAS and LWA centred on the peak LWA day for each event.

    For every contiguous event (from `events_*`), find the day with the largest
    LWA, extract ±half_window days for TAS and LWA, average per event, then:
      • For CanESM: average windows per member, then summarize ensemble mean and spread.
      • For ERA5: average across events once (single blue line).
    A dashed line shows the composite including only events with length
    ≥ `event_duration_threshold` days.
    """

    lag_coord = np.arange(-half_window, half_window + 1)

    canesm_tas_comp_all, canesm_lwa_comp_all = _process_dataset(
        ds_tas_canesm, lwa_can, events_can, lag_coord, half_window, n_day_min=0
    )
    era_tas_comp_all, era_lwa_comp_all = _process_dataset(
        ds_tas_era5, lwa_era, events_era, lag_coord, half_window, n_day_min=0
    )

    if event_duration_threshold > 0:
        canesm_tas_comp_3d, canesm_lwa_comp_3d = _process_dataset(
            ds_tas_canesm, lwa_can, events_can, lag_coord, half_window, n_day_min=event_duration_threshold
        )
        era_tas_comp_3d, era_lwa_comp_3d = _process_dataset(
            ds_tas_era5, lwa_era, events_era, lag_coord, half_window, n_day_min=event_duration_threshold
        )
    else:
        canesm_tas_comp_3d = canesm_lwa_comp_3d = None
        era_tas_comp_3d = era_lwa_comp_3d = None

    if (
        canesm_tas_comp_all is None and era_tas_comp_all is None
        and canesm_lwa_comp_all is None and era_lwa_comp_all is None
    ):
        print("plot_composite_TAS_during_LWA_events: no valid events to composite.")
        return None

    fig, (ax_0, ax_1) = plt.subplots(
        figsize=(10, 12),
        nrows=2,
        sharex=True,
        tight_layout=True,
    )

    _plot_panel(
        ax_0,
        canesm_tas_comp_all,
        era_tas_comp_all,
        canesm_tas_comp_3d,
        era_tas_comp_3d,
        y_label or "TAS (K)",
        title or "Composite TAS around LWA events",
        lag_coord,
        event_duration_threshold=event_duration_threshold if event_duration_threshold > 0 else None,
    )

    ax_0.tick_params(axis="x", labelbottom=False)

    lwa_ylabel = f"{LWA_var} (m)"
    _plot_panel(
        ax_1,
        canesm_lwa_comp_all,
        era_lwa_comp_all,
        canesm_lwa_comp_3d,
        era_lwa_comp_3d,
        lwa_ylabel,
        f"Composite {LWA_var} around LWA events",
        lag_coord,
        event_duration_threshold=event_duration_threshold if event_duration_threshold > 0 else None,
    )

    ax_1.set_xlabel(x_label or "Days relative to LWA peak")
    ax_1.set_xticks(lag_coord)

    fig_name = f"{output_path}/{LWA_var}_Q{Q_LWA}_tas_composite_{REGION}_{SEASON}_mindur{event_duration_threshold}d.png"
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")

    return None






def plot_composite_TAS_during_all_LWA_events(
    ds_tas_era5: xr.DataArray,    # dim("time") regional average TAS
    lwa_era: xr.DataArray,        # dim("time") regional average LWA
    events_era: xr.DataArray,     # dim("time") event IDs
    half_window: int,
    event_duration_threshold: int,
    title: str = "",
    x_label: str = "Days relative to LWA peak",
    y_label: str = "TAS (K)",
    output_path: str = './',
) -> None: 
    """
    Plot composites of TAS and LWA centred on the peak LWA day for each event.

    For every contiguous event (from `events_*`), find the day with the largest
    LWA, extract ±half_window days for TAS and LWA, average per event, then:
      • For CanESM: average windows per member, then summarize ensemble mean and spread.
      • For ERA5: average across events once (single blue line).
    """

    lag_coord = np.arange(-half_window, half_window + 1)

    era_tas_comp_all, era_lwa_comp_all = _process_dataset(
        ds_tas_era5, lwa_era, events_era, lag_coord, half_window, n_day_min=0
    )

    if event_duration_threshold > 0:
        era_tas_comp_3d, era_lwa_comp_3d = _process_dataset(
            ds_tas_era5, lwa_era, events_era, lag_coord, half_window, n_day_min=event_duration_threshold
        )
    else:
        era_tas_comp_3d = era_lwa_comp_3d = None

    fig, (ax_0, ax_1) = plt.subplots(
        figsize=(10, 12),
        nrows=2,
        sharex=True,
        tight_layout=True,
    )

    _plot_panel_all(
        ax_0,
        era_tas_comp_all,
        era_tas_comp_3d,
        y_label or r"$\Delta$T (K)",
        title or "Composite TAS around LWA events",
        lag_coord,
        event_duration_threshold=event_duration_threshold if event_duration_threshold > 0 else None,
    )

    ax_0.tick_params(axis="x", labelbottom=False)

    lwa_ylabel = f"{LWA_var} (m)"
    _plot_panel_all(
        ax_1,
        era_lwa_comp_all,
        era_lwa_comp_3d,
        lwa_ylabel,
        f"Composite {LWA_var} around LWA events",
        lag_coord,
        event_duration_threshold=event_duration_threshold if event_duration_threshold > 0 else None,
    )

    ax_1.set_xlabel(x_label or "Days relative to LWA peak")
    ax_1.set_xticks(lag_coord)

    fig_name = f"{output_path}/{LWA_var}_Q{Q_LWA}_tas_composite_{REGION}_{SEASON}_mindur{event_duration_threshold}d.png"
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")

    return None


# ------------------------------ Main ----------------------------------

def main(REGION, LWA_var, SEASON, ZG_COORD, Q_LWA, EVENT_DURATION_THRESHOLD: int):

    #load in data
    # ds_thresh_canesm        = open_thresh_data_canesm(VAR, ENSEMBLE_LIST, Q_TEMP, REGION)
    # ds_thresh_canesm_smooth = ds_thresh_canesm.rolling(dayofyear=7, center=True).mean().isel(time=0)
    # # ds_thresh_canesm_smooth = ds_thresh_canesm_smooth.sel(MEMBER=MEMBER)

    ds_lwa_thresh_canesm        = open_lwa_thresh_data_canesm(ENSEMBLE_LIST, Q_LWA, REGION, ZG_COORD)
    ds_lwa_thresh_canesm_smooth = ds_lwa_thresh_canesm.rolling(dayofyear=7, center=True).mean()
    # # ds_lwa_thresh_canesm_smooth = ds_lwa_thresh_canesm_smooth.sel(MEMBER=MEMBER)

    # ds_thresh_era5        = open_thresh_data_era5(VAR, Q_TEMP, REGION)
    # ds_thresh_era5_smooth = ds_thresh_era5.rolling(dayofyear=7, center=True).mean()

    ds_lwa_thresh_era5        = open_lwa_thresh_data_era5(Q_LWA, REGION, ZG_COORD)
    ds_lwa_thresh_era5_smooth = ds_lwa_thresh_era5.rolling(dayofyear=7, center=True).mean()

    ds_tas_canesm = open_temp_data_canesm(VAR, ENSEMBLE_LIST)
    ds_tas_canesm = ds_tas_canesm.chunk({"time": 365})
    ds_tas_canesm = compute_region_mean(ds_tas_canesm, REGION).compute()
    
    ds_tas_canesm_clim = compute_doy_climatology(ds_tas_canesm)
    ds_tas_canesm_anom = ds_tas_canesm.groupby("time.dayofyear") - ds_tas_canesm_clim

    # print(ds_tas_canesm)
    # print(ds_tas_canesm_clim)
    # print(ds_tas_canesm_anom)

    ds_tas_era5 = open_temp_data_era5(VAR)
    ds_tas_era5 = ds_tas_era5.chunk({"time": 365})
    ds_tas_era5 = compute_region_mean(ds_tas_era5, REGION).compute()
    
    ds_tas_era5_clim = compute_doy_climatology(ds_tas_era5)

    ds_tas_era5_anom = ds_tas_era5.groupby("time.dayofyear") - ds_tas_era5_clim

    # doy_all_canesm = ds_tas_canesm.time.dt.dayofyear
    # doy_all_era5   = ds_tas_era5.time.dt.dayofyear

    # ds_tas_yearly_canesm = ds_tas_canesm.groupby("time.year")
    # ds_tas_yearly_era5   = ds_tas_era5.groupby("time.year")

    ## loading LWA data
    # 1) Open CanESM and ERA5 for all VARiables
    ds_canesm_lwas: Dict[str, xr.DataArray] = open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas: Dict[str, xr.DataArray]   = open_era5_lwa(ZG_COORD)

    ds_canesm_lwa = ds_canesm_lwas[LWA_var]#.sel(MEMBER=MEMBER)
    ds_era5_lwa   = ds_era5_lwas[LWA_var]

    ds_canesm_lwa_reg = compute_region_mean(ds_canesm_lwa, REGION).chunk({"time": 365}).compute()
    ds_era5_lwa_reg   = compute_region_mean(ds_era5_lwa, REGION).chunk({"time": 365}).compute()

    # ds_canesm_lwa_reg_yearly = ds_canesm_lwa_reg.groupby("time.year")
    # ds_era5_lwa_reg_yearly   = ds_era5_lwa_reg.groupby("time.year")

    # plot HW thresholds
    # plot_temp_thresh_canesm_era5(ds_thresh_canesm_smooth, ds_thresh_era5_smooth, ds_tas_era5, OUTPUT_PLOTS_PATH)
    # plot_lwa_thresh_canesm_era5(ds_lwa_thresh_canesm_smooth[LWA_var], ds_lwa_thresh_era5_smooth[LWA_var], ds_era5_lwa_reg, OUTPUT_PLOTS_PATH)

    mask_era = (ds_era5_lwa_reg.groupby("time.dayofyear") >= ds_lwa_thresh_era5_smooth[LWA_var]).compute()
    mask_can = (ds_canesm_lwa_reg.groupby("time.dayofyear") >= ds_lwa_thresh_canesm_smooth[LWA_var].isel(time=0)).compute()

    # turn mask into events
    events_era = mask_to_events(mask_era) #dim("time")
    events_can = mask_to_events(mask_can) #dim("member", "time")

    # keep only events fully contained within the target season
    events_era_filtered = filter_events_by_season(events_era, SEASON)
    events_can_filtered = filter_events_by_season(events_can, SEASON)

    print(events_can_filtered)
    print(events_era_filtered)

    ds_tas_era5 = ds_tas_era5.assign_coords(time=ds_tas_era5.time.dt.floor("D"))

    # masked_lwa_era = ds_era5_lwa_reg.where(mask_era,drop=True)
    # masked_lwa_can = ds_canesm_lwa_reg.where(mask_can,drop=True)

    # masked_tas_era = ds_tas_era5.where(mask_era,drop=True)
    # masked_tas_can = ds_tas_canesm.where(mask_can,drop=True)

    # print(masked_lwa_era)
    # print(masked_tas_era)

    # plot_joint_kde_with_marginals(
    #     masked_lwa_era,
    #     masked_tas_era,
    #     title=f"ERA5 {LWA_var} vs TAS, {REGION}, {SEASON}",
    #     sim="ERA5",
    #     x_label=f"{LWA_var}",
    #     output_path=OUTPUT_PLOTS_PATH
    # )

    plot_histogram_of_duration(
        lwa_era=events_era_filtered, # ERA5 list of event IDs
        lwa_can=events_can_filtered, # CanESM list of event IDs
        x_label=f"{LWA_var}>q{Q_LWA} Event Duration [days]",
        y_label="Frequency",
        title=f"{LWA_var} Event Duration Histogram {REGION} {SEASON}",
        output_path=OUTPUT_PLOTS_PATH,
    )

    plot_composite_TAS_during_LWA_events(
        ds_tas_canesm=ds_tas_canesm_anom,  # dim("member", "time")
        ds_tas_era5=ds_tas_era5_anom,      # dim("time")
        lwa_can=ds_canesm_lwa_reg,    # dim("member", "time")
        lwa_era=ds_era5_lwa_reg,      # dim("time")
        events_can=events_can_filtered,        # dim("member", "time")
        events_era=events_era_filtered,        # dim("time")
        half_window=7,
        event_duration_threshold=EVENT_DURATION_THRESHOLD,
        title=f"TAS Composite during {LWA_var}>q{Q_LWA} Events {REGION} {SEASON}",
        x_label="Days relative to LWA peak",
        y_label=r"$\Delta$T (K)",
        output_path=OUTPUT_PLOTS_PATH,
    )


if __name__ == "__main__":
    args = arg_parser()
    REGION   = args.region
    LWA_var  = args.lwa_var
    Q_LWA    = args.lwa_quantile
    SEASON   = args.season
    ZG_LEVEL = args.zg

    EVENT_DURATION_THRESHOLD = args.event_duration_threshold

    main(REGION, LWA_var, SEASON, ZG_LEVEL, Q_LWA, EVENT_DURATION_THRESHOLD)
