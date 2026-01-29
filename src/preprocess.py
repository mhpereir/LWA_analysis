import numpy as np
import xarray as xr

from . import config


def compute_region_mean(ds: xr.DataArray, region: str) -> xr.DataArray:
    """Compute area-weighted mean over region."""
    lat_slice, lon_slice = config.REGIONS[region]
    ds_region = ds.sel(lat=lat_slice, lon=lon_slice)

    weights = np.cos(np.deg2rad(ds_region.lat))
    return ds_region.weighted(weights).mean(dim=["lat", "lon"])


def dayofyear_anomaly(da: xr.DataArray) -> xr.DataArray:
    """Return anomalies by removing day-of-year climatology."""
    clim = da.groupby("time.dayofyear").mean("time")
    return da.groupby("time.dayofyear") - clim


def floor_daily_time(da: xr.DataArray) -> xr.DataArray:
    """Ensure time coordinates are day-based."""
    return da.assign_coords(time=da.time.dt.floor("D"))
