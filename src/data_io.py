import glob
from typing import Dict, List, Tuple

import xarray as xr

from . import config


def open_canesm_temperature(var: str, ensemble_list: List[str]) -> xr.DataArray:
    path_to_data = config.CANESM_TAS_ROOT

    files_by_member = []
    for m in ensemble_list:
        filepat = f"{path_to_data}/{var}_daily_CanESM5_historical_{m}_1850_2014_2x2_bil.nc"
        files = sorted(glob.glob(filepat))
        if not files:
            raise FileNotFoundError(f"No files matched for {m}; pattern: {filepat}")
        files_by_member.append(files)

    ds_canesm = xr.open_mfdataset(
        files_by_member,
        combine="nested",
        concat_dim=["member", "time"],
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365},
    )

    ds_canesm = ds_canesm.sel(time=config.TIME_SLICE)
    ds_canesm = ds_canesm.assign_coords(member=("member", ensemble_list))
    return ds_canesm[var]


def open_era5_temperature(var: str) -> xr.DataArray:
    path_to_data = f"{config.ERA5_TAS_ROOT}"

    filepat = f"{path_to_data}/{var}_daily_ERA_*_2x2_bil.nc"
    files = sorted(glob.glob(filepat))
    if not files:
        raise FileNotFoundError(f"No files matched for ERA5 temperature {var}; pattern: {filepat}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365},
    )

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    ds = ds.sel(time=config.TIME_SLICE)
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    ds = ds.sel(lat=config.LAT_SLICE)
    return ds[var]


def open_canesm_mrsos(var: str, ensemble_list: List[str]) -> xr.DataArray:
    path_to_data = config.CANESM_MRSOS_ROOT

    files_by_member = []
    for m in ensemble_list:
        filepat = f"{path_to_data}/{var}_daily_CanESM5_historical_{m}_1850_2014_2x2_bil.nc"
        files = sorted(glob.glob(filepat))
        if not files:
            raise FileNotFoundError(f"No files matched for {m}; pattern: {filepat}")
        files_by_member.append(files)

    ds_canesm = xr.open_mfdataset(
        files_by_member,
        combine="nested",
        concat_dim=["member", "time"],
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365},
    )

    ds_canesm = ds_canesm.sel(time=config.TIME_SLICE)
    ds_canesm = ds_canesm.assign_coords(member=("member", ensemble_list))
    return ds_canesm[var]


def open_era5_mrsos(var: str) -> xr.DataArray:
    path_to_data = f"{config.ERA5_MRSOS_ROOT}"
 
    filepat = f"{path_to_data}/soil_moisture_daily_ERA5_*_2x2_bil.nc" #era5 files are called "soil_moisture", non-standard name
    files = sorted(glob.glob(filepat))
    if not files:
        raise FileNotFoundError(f"No files matched for ERA5 soil moisture {var}; pattern: {filepat}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        engine="h5netcdf",
        chunks={"time": 365},
    )

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    ds = ds.sel(time=config.TIME_SLICE)
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    ds = ds.sel(lat=config.LAT_SLICE)

    #var name in ERA5 is svwl1; rename to "mrsos" and convert units from m3/m3 to mm
    ds = ds.rename({var: "mrsos"})
    ds = ds["mrsos"] * 0.1 * 1000.0
    return ds


def open_canesm_lwa(ensemble_list: List[str], zg_coord: int) -> Dict[str, xr.DataArray]:
    path_to_data = config.CANESM_LWA_ROOT

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
        parallel=True,
        chunks={"time": 365, "member": 10, "lat": 35, "lon": 180},
        engine="h5netcdf",
    )
    ds = ds.sel(time=config.TIME_SLICE)
    ds = ds.assign_coords(member=("member", ensemble_list)).sel(lat=config.LAT_SLICE)

    ds_lwa = ds["LWA"]
    ds_lwaa = ds["LWA_a"]
    ds_lwac = ds["LWA_c"]
    return {"LWA": ds_lwa, "LWA_a": ds_lwaa, "LWA_c": ds_lwac}


def open_era5_lwa(zg_coord: int) -> Dict[str, xr.DataArray]:
    filepat = f"{config.ERA5_LWA_ROOT}/z{zg_coord}/LWA_day_ERA5_2deg.{zg_coord}.nc"
    files = sorted(glob.glob(filepat))
    if not files:
        raise FileNotFoundError(f"No ERA5 files matched: {filepat}")

    ds = xr.open_mfdataset(
        files,
        parallel=True,
        cache=False,
        chunks={"time": 3650, "lat": 35, "lon": 180},
        engine="h5netcdf",
    )

    ds = ds.sel(time=config.TIME_SLICE)
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    ds = ds.sel(lat=config.LAT_SLICE)

    ds_lwa = ds["LWA"]
    ds_lwaa = ds["LWA_a"]
    ds_lwac = ds["LWA_c"]
    return {"LWA": ds_lwa, "LWA_a": ds_lwaa, "LWA_c": ds_lwac}

def open_canesm_lwa_thresh(ensemble_list: List[str], q: int, region: str, zg_coord: int) -> xr.Dataset:
    path_to_data = config.LWA_THRESH_ROOT
    
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


def open_era5_lwa_thresh(q: int, region: str, zg_coord: int) -> xr.Dataset:
    path_to_data = config.LWA_THRESH_ROOT

    filepat = f"{path_to_data}/ERA5_LWAthresh_block_1970_2014_q{q}_{region}.{zg_coord}.nc"

    # 2) Open all lazily and stack along new 'MEMBER' then 'time'
    ds_ERA5 = xr.open_dataset(
        filepat,
        engine="h5netcdf",
        chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
    )

    # 3) Give the MEMBER axis meaningful labels
    return ds_ERA5
