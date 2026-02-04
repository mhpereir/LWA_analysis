from typing import Tuple
import numpy as np
import xarray as xr

from . import config


def floor_daily_time(da: xr.DataArray) -> xr.DataArray:
    """Ensure time coordinates are day-based."""
    return da.assign_coords(time=da.time.dt.floor("D"))

def drop_leap_day(da: xr.DataArray) -> xr.DataArray:
    return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))


def compute_region_mean(ds: xr.DataArray, region: str) -> xr.DataArray:
    """Compute area-weighted mean over region."""
    lat_slice, lon_slice = config.REGIONS[region]
    ds_region = ds.sel(lat=lat_slice, lon=lon_slice)

    weights = np.cos(np.deg2rad(ds_region.lat))
    return ds_region.weighted(weights).mean(dim=["lat", "lon"])


def dayofyear_anomaly(da: xr.DataArray) -> xr.DataArray:
    """Return anomalies by removing day-of-year climatology."""
    clim, _ = dayofyear_clim_and_std(da)
    return da.groupby("time.dayofyear") - clim
    

def dayofyear_clim_and_std(da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    da = floor_daily_time(da)
    clim = da.groupby("time.dayofyear").mean("time")
    std = da.groupby("time.dayofyear").std("time")
    return clim, std

def dayofyear_clim_and_std_noleap(da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    da = floor_daily_time(da)

    t = da["time"]
    doy = t.dt.dayofyear
    shift = xr.where(t.dt.is_leap_year & (t.dt.month > 2), 1, 0)  # shift after Feb in leap years
    doy_noleap = (doy - shift).astype(int)

    da2 = da.assign_coords(doy_noleap=("time", doy_noleap.data))
    clim = da2.groupby("doy_noleap").mean("time").rename({"doy_noleap": "dayofyear"})
    std  = da2.groupby("doy_noleap").std("time").rename({"doy_noleap": "dayofyear"})
    return clim, std

def canesm_dayofyear_stats(da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Return (ensemble_mean_clim, ensemble_spread, daily_variability).
    daily_variability is the mean of interannual std across members.
    """
    clim_m, std_m = dayofyear_clim_and_std(da)
    ens_mean = clim_m.mean("member")
    ens_spread = clim_m.std("member")
    daily_var = std_m.mean("member")
    return ens_mean, ens_spread, daily_var