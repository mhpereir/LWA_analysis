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
from matplotlib import ticker as mticker

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
VAR_NAMES: List[str] = ["LWA", "LWA_a"] # "LWA_a",

# Period and latitude band (to match existing practice)
TIME_SLICE = slice("1970-01-01", "2014-12-31")
LAT_SLICE  = slice(20, 85)  # 20–90°N

# REGIONs (consistent with your threshold files)
# REGIONS: dict[str, Tuple[slice, slice]] = {
#     "canada":       (slice(40, 70), slice(-140,    -60)),
#     "canada_north": (slice(55, 70), slice(-140, -60)),
#     "canada_south": (slice(40, 55), slice(-140, -60)),
#     "west":       (slice(40, 70), slice(-140,    -113.33)),
#     "west_north": (slice(55, 70), slice(-140, -113.33)),
#     "west_south": (slice(40, 55), slice(-140, -113.33)),
#     "central":       (slice(40, 70), slice(-113.33, -88.66)),
#     "central_north": (slice(55, 70), slice(-113.33, -88.66)),
#     "central_south": (slice(40, 55), slice(-113.33, -88.66)),
#     "east":       (slice(40, 70), slice(-88.66,  -60)),
#     "east_north": (slice(55, 70), slice(-88.66, -60)),
#     "east_south": (slice(40, 55), slice(-88.66, -60))
# }

LAT_BAND: dict[str, Tuple[float, float]] = {
    "north": (55, 70),
    "south": (40, 55),
    "all":   (40, 70)
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
# LWA_THRESH_ROOT = "/space/hall5/sitestore/eccc/crd/ccrn/users/mpw000/development/HW_analysis/lwa_thresholds"
# ERA5_TAS_ROOT   = "/home/mhpereir/data-mhpereir/standard_grid_daily/REANALYSIS/ERA5/2mT"
# CANESM_TAS_ROOT = "/home/mhpereir/data-mhpereir/standard_grid_daily/CMIP6/CanESM5/tas/historical"

OUTPUT_PLOTS_PATH = "/home/mhpereir/LWA_analysis/plots"
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

# Plot options
HATCH_K: float = 2.0
PROJ = ccrs.EqualEarth() 

def arg_parser():
    parser = argparse.ArgumentParser(
        description="LWA vs deltaT correlation analysis and plotting."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=list(LAT_BAND.keys()),
        default="all",
        help="Latitude range to analyze.",
    )
    parser.add_argument(
        "--season",
        type=str,
        choices=SEASON_NAMES,
        default="all",
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


# def open_lwa_thresh_data_canesm(ensemble_list, q, zg, region):
#     path_to_data = LWA_THRESH_ROOT
    
#     files_by_MEMBER = []
#     for m in ensemble_list:
#         filepat = f"{path_to_data}/CanESM5_LWAthresh_block_1970_2014_q{q}_{region}_{m}.{zg}.nc"
#         files = sorted(glob.glob(filepat))
#         if not files:
#             raise FileNotFoundError(f"No files matched for {m}")
#         files_by_MEMBER.append(files)

#     # 2) Open all lazily and stack along new 'MEMBER' then 'time'
#     ds_CanESM = xr.open_mfdataset(
#         files_by_MEMBER,
#         combine="nested",
#         concat_dim=["member", "time"],   # first stacks MEMBERs, then concatenates their time files
#         parallel=True,
#         engine="h5netcdf",
#         chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
#     )

#     # 3) Give the MEMBER axis meaningful labels
#     ds_CanESM = ds_CanESM.assign_coords(member=("member", ensemble_list))
#     #VAR_name = f"{VAR}_thresh_p{Q}_win31"
#     return ds_CanESM#[VAR_name]


# def open_lwa_thresh_data_era5(q, zg, region):
#     path_to_data = LWA_THRESH_ROOT


#     filepat = f"{path_to_data}/ERA5_LWAthresh_block_1970_2014_q{q}_{region}.{zg}.nc"

#     # 2) Open all lazily and stack along new 'MEMBER' then 'time'
#     ds_ERA5 = xr.open_dataset(
#         filepat,
#         engine="h5netcdf",
#         chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
#     )

#     # 3) Give the MEMBER axis meaningful labels
#     return ds_ERA5



# def open_temp_data_canesm(var, ensemble_list):
#     path_to_data = CANESM_TAS_ROOT
#     # {BASE_DIR}/${MEMBER}/day/${VAR}/gn/v20190429/${VAR}_day_CanESM5_historical_${MEMBER}_gn_18500101-20141231_2x2_bil.nc
#     files_by_member = []
#     for m in ensemble_list:
#         filepat = f"{path_to_data}/tas_daily_CanESM5_historical_{m}_1850_2014_2x2_bil.nc"
#         files = sorted(glob.glob(filepat))
#         if not files:
#             raise FileNotFoundError(f"No files matched for {m}")
#         files_by_member.append(files)

#     # 2) Open all lazily and stack along new 'MEMBER' then 'time'
#     ds_CanESM = xr.open_mfdataset(
#         files_by_member,
#         combine="nested",
#         concat_dim=["member", "time"],   # first stacks MEMBERs, then concatenates their time files
#         parallel=True,
#         engine="h5netcdf",
#         chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
#     )

#     ds_CanESM = ds_CanESM.sel(time=TIME_SLICE)
#     # 3) Give the MEMBER axis meaningful labels
#     ds_CanESM = ds_CanESM.assign_coords(member=("member", ensemble_list))
#     var_name = f"{var}"
#     return ds_CanESM[var_name]


# def open_temp_data_era5(var):
#     path_to_data = f"{ERA5_TAS_ROOT}"
#     # {BASE_DIR}/${MEMBER}/day/${VAR}/gn/v20190429/${VAR}_day_CanESM5_historical_${MEMBER}_gn_18500101-20141231_2x2_bil.nc

#     filepat = f"{path_to_data}/ERA5_CMOR6_v1_day_{var}_*_2x2_bil.nc"
#     files = sorted(glob.glob(filepat))
#     if not files:
#         raise FileNotFoundError(f"No files matched for ERA5 temperature {var}")


#     # 2) Open all lazily and stack along new 'MEMBER' then 'time'
#     ds = xr.open_mfdataset(
#         files,
#         combine="by_coords",
#         parallel=True,
#         engine="h5netcdf",
#         chunks={"time": 365}             # tune to your chunking (e.g., ~1 year per chunk)
#     )
#     ds = ds.sel(time=TIME_SLICE)
#     #remove leap days
#     ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
#     # crop to lat slice
#     ds = ds.sel(lat=LAT_SLICE)
#     # 3) Give the MEMBER axis meaningful labels
#     var_name = f"{var}"
#     return ds[var_name]


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

# def compute_region_mean(ds: xr.DataArray, region: str) -> xr.DataArray:
#     """Compute area-weighted mean over region."""
#     lat_slice, lon_slice = REGIONS[region]
#     ds_region = ds.sel(lat=lat_slice, lon=lon_slice)

#     weights = np.cos(np.deg2rad(ds_region.lat))
#     return ds_region.weighted(weights).mean(dim=["lat", "lon"])

def compute_lat_band_mean(ds: xr.DataArray, lat_band: str) -> xr.DataArray:
    """Compute area-weighted mean over latitude band."""
    lat_min, lat_max = LAT_BAND[lat_band]
    ds_band = ds.sel(lat=slice(lat_min, lat_max))

    weights = np.cos(np.deg2rad(ds_band.lat))
    return ds_band.weighted(weights).mean(dim=["lat"]) # dimension 'lon' remains


# ------------------------------ Helper Functions ----------------------------------




# ------------------------------ Plotting ----------------------------------


def plot_LWA_phase(
    lwa_era5_spatial: xr.DataArray,
    dict_canesm: Dict[str, xr.DataArray],
    dict_era5: Dict[str, xr.DataArray],
    title: str = "",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    
    # Create figure and axes

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(nrows=4, ncols=1,
                          height_ratios=[2.3, 0.25, 1.4, 1.4], hspace=0.35,
    )
    ax_3 = fig.add_subplot(gs[0], projection=ccrs.EqualEarth())
    ax_4 = fig.add_subplot(gs[1])
    ax_0 = fig.add_subplot(gs[2])
    ax_1 = fig.add_subplot(gs[3], sharex=ax_0)

    linestyle=['-','--',':']

    # plot geospatial data
    im = ax_3.contourf(
        lwa_era5_spatial.lon,
        lwa_era5_spatial.lat,
        lwa_era5_spatial,
        cmap="Reds",
        transform=ccrs.PlateCarree(),
    )
    
    gl = ax_3.gridlines( #type: ignore
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linestyle="--",
    linewidth=0.8,
    color="gray"
)
    gl.xlocator = mticker.FixedLocator([-150, -100, -50, 0, 50, 100, 150])

    # lat_min, lat_max = LAT_BAND[REGION]
    # for lat in (lat_min, lat_max):
    #     ax_3.plot(
    #         np.linspace(-180, 180, 360),
    #         np.full(360, lat),
    #         transform=ccrs.PlateCarree(),
    #         color="blue",
    #         linewidth=2
    # )
        
    # ax_3.vlines(x=[-140, -113], ymin=20, ymax=85, colors='gray', linestyles='-', linewidth=1)

    ax_3.set_title('LWA spatial distribution', fontsize=14)
    ax_3.set_extent([-180, 180, 20, 85], crs=ccrs.PlateCarree()) #type: ignore 
    ax_3.coastlines() #type: ignore
    # ax_3.set_position(gs[0].get_position(fig))

    # fig.colorbar(im, ax=ax_3, orientation="horizontal", label="LWA [m]", pad=0.02)
    
    
    fig.colorbar(
        im,
        cax=ax_4,
        orientation="horizontal",
        label="LWA [m]"
    )
    ax_4.xaxis.set_label_position("top")  # optional
    ax_4.tick_params(axis="x", labeltop=True, labelbottom=False)
    
    # ax_4.axis('off')

    # for i, LWA_var in enumerate(VAR_NAMES):
    canesm_data = dict_canesm['LWA']  #(member, lon)
    era5_data   = dict_era5['LWA']    #(lon)

    # plot CanESM5
    canesm_mean = canesm_data.mean(dim="member")
    canesm_5th  = canesm_data.quantile(dim="member", q=0.05)
    canesm_95th = canesm_data.quantile(dim="member", q=0.95)

    ax_0.plot(
        canesm_mean.lon,
        canesm_mean,
        linewidth=2,
        linestyle='-',
        color='blue',
        label='CanESM5'
    )

    # ax_0.fill_between(
    #     canesm_mean.lon,
    #     canesm_5th,
    #     canesm_95th,
    #     alpha=0.3,
    #     color='C0',
    # )

    # if LWA_var == "LWA_c":
    #     ax_0.fill_between(
    #         canesm_mean.lon,
    #         np.max(np.asarray([canesm_95th, era5_data]), axis=0),
    #         np.max(np.asarray([dict_canesm["LWA"].quantile(dim="member", q=0.05), dict_era5["LWA"]]), axis=0),
    #         color='red', alpha=0.3, label='LWA_a'
    #     )
    #     ax_0.fill_between(
    #         canesm_mean.lon,
    #         np.min(np.asarray([canesm_5th, era5_data]), axis=0),
    #         np.zeros_like(canesm_mean.lon),
    #         color='blue', alpha=0.3, label='LWA_c'
    #     )

    # plot ERA5
    ax_0.plot(
        era5_data.lon,
        era5_data,
        linewidth=2,
        linestyle='-',
        color='black',
        label='ERA5'
    )

    ax_0.vlines(x=[-140, -113], ymin=0, ymax=1.2*np.max([canesm_mean, era5_data]), colors='gray', linestyles='-', linewidth=1)

    ax_0.set_ylim([0, 1.2*np.max([canesm_mean, era5_data])]) #type: ignore
    ax_0.set_xlabel("")
    ax_0.tick_params(axis="x", which="both", labelbottom=False)
    ax_0.set_ylabel("LWA [m]")
    ax_0.legend(loc='lower right')


    # plot normalized panel

    canesm_data = dict_canesm["LWA_a"]/dict_canesm["LWA"]  #(member, lon)
    era5_data   = dict_era5["LWA_a"]/dict_era5["LWA"]      #(lon)

    # plot CanESM5
    canesm_mean = canesm_data.mean(dim="member")
    canesm_5th  = canesm_data.quantile(dim="member", q=0.05)
    canesm_95th = canesm_data.quantile(dim="member", q=0.95)

    ax_1.plot(
        canesm_mean.lon,
        canesm_mean,
        linewidth=2,
        linestyle='--',
        color='blue',
    )

    ax_1.fill_between(
        canesm_mean.lon,
        canesm_5th,
        canesm_95th,
        alpha=0.3,
        color='C0',
    )

    ax_1.fill_between(
        canesm_mean.lon,
        np.max(np.asarray([canesm_95th, era5_data]), axis=0),
        np.max(np.asarray([dict_canesm["LWA"].quantile(dim="member", q=0.05), dict_era5["LWA"]]), axis=0),
        color='blue', alpha=0.3, label='LWA_c'
    )
    ax_1.fill_between(
        canesm_mean.lon,
        np.min(np.asarray([canesm_5th, era5_data]), axis=0),
        np.zeros_like(canesm_mean.lon),
        color='red', alpha=0.3, label='LWA_a'
    )

    # plot ERA5
    ax_1.plot(
        era5_data.lon,
        era5_data,
        linewidth=2,
        linestyle='--',
        color='black',
    )

    ax_1.vlines(x=[-140, -113], ymin=0, ymax=1, colors='gray', linestyles='-', linewidth=1)

    ax_1.axhline(0.5, color='gray', linestyle=':', linewidth=1)

    ax_1.set_xlabel("Longitude [°]")
    ax_1.set_ylabel("LWA fraction")
    ax_1.legend()

    ax_1.set_xlim([-180, 180]) #type: ignore
    ax_1.set_ylim([0, 1])      #type: ignore


    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    # # tighten layout a bit (but keep our manual colorbar placement)
    # plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    fig_name = f"{output_path}/LWA_phase_plot_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None


# ------------------------------ Main ----------------------------------

def main(REGION, SEASON, ZG_COORD):

    #load in data
    # ds_thresh_canesm        = open_thresh_data_canesm(VAR, ENSEMBLE_LIST, Q_TEMP, REGION)
    # ds_thresh_canesm_smooth = ds_thresh_canesm.rolling(dayofyear=7, center=True).mean().isel(time=0)
    # # ds_thresh_canesm_smooth = ds_thresh_canesm_smooth.sel(MEMBER=MEMBER)

    # ds_lwa_thresh_canesm        = open_lwa_thresh_data_canesm(ENSEMBLE_LIST, Q_LWA, ZG_COORD, REGION)
    # ds_lwa_thresh_canesm_smooth = ds_lwa_thresh_canesm.rolling(dayofyear=7, center=True).mean()
    # # ds_lwa_thresh_canesm_smooth = ds_lwa_thresh_canesm_smooth.sel(MEMBER=MEMBER)

    # ds_thresh_era5        = open_thresh_data_era5(VAR, Q_TEMP, REGION)
    # ds_thresh_era5_smooth = ds_thresh_era5.rolling(dayofyear=7, center=True).mean()

    # ds_lwa_thresh_era5        = open_lwa_thresh_data_era5(Q_LWA, ZG_COORD, REGION)
    # ds_lwa_thresh_era5_smooth = ds_lwa_thresh_era5.rolling(dayofyear=7, center=True).mean()

    # ds_tas_canesm = open_temp_data_canesm(VAR, ENSEMBLE_LIST)
    # ds_tas_canesm = ds_tas_canesm.chunk({"time": 365})
    # ds_tas_canesm = ds_tas_canesm.sel(lat=REGIONS[REGION][0], lon=REGIONS[REGION][1]).mean(dim=["lat", "lon"]).compute()

    # ds_tas_era5 = open_temp_data_era5(VAR)
    # ds_tas_era5 = ds_tas_era5.chunk({"time": 365})
    # ds_tas_era5 = ds_tas_era5.sel(lat=REGIONS[REGION][0], lon=REGIONS[REGION][1]).mean(dim=["lat", "lon"]).compute()

    # doy_all_canesm = ds_tas_canesm.time.dt.dayofyear
    # doy_all_era5   = ds_tas_era5.time.dt.dayofyear

    # ds_tas_yearly_canesm = ds_tas_canesm.groupby("time.year")
    # ds_tas_yearly_era5   = ds_tas_era5.groupby("time.year")

    ## loading LWA data
    # 1) Open CanESM and ERA5 for all VARiables
    ds_canesm_lwas: Dict[str, xr.DataArray] = open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas:   Dict[str, xr.DataArray] = open_era5_lwa(ZG_COORD)

    

    canesm_lwas_lon: Dict[str, xr.DataArray] = {}
    era5_lwas_lon: Dict[str, xr.DataArray]   = {}
    
    
    
    for LWA_var in VAR_NAMES:
        print(f"Processing LWA variable: {LWA_var}")

        ds_canesm_lwa = ds_canesm_lwas[LWA_var]#.sel(MEMBER=MEMBER) #(time, member, lat, lon)
        ds_era5_lwa   = ds_era5_lwas[LWA_var] #(time, lat, lon)

        if SEASON == 'all':
            ds_canesm_lwa = ds_canesm_lwa.mean(dim="time")
            ds_era5_lwa   = ds_era5_lwa.mean(dim="time")

        else:
            ds_canesm_lwa = ds_canesm_lwa.sel(time=ds_canesm_lwa.time.dt.season == SEASON).mean(dim="time") #(member, lat, lon)
            ds_era5_lwa   = ds_era5_lwa.sel(time=ds_era5_lwa.time.dt.season == SEASON).mean(dim="time") # (lat, lon)

        if LWA_var == 'LWA':
            lwa_era5_spatial = ds_era5_lwa.compute()  #(lat, lon)

        canesm_lwas_lon[LWA_var] = compute_lat_band_mean(ds_canesm_lwa, REGION).compute() #(member, lon)
        era5_lwas_lon[LWA_var]   = compute_lat_band_mean(ds_era5_lwa, REGION).compute()   #(lon)

    plot_LWA_phase(
        lwa_era5_spatial=lwa_era5_spatial, #type: ignore
        dict_canesm=canesm_lwas_lon,
        dict_era5=era5_lwas_lon,
        title=f"LWA Phase Plot - {REGION}, {SEASON}",
        output_path=OUTPUT_PLOTS_PATH
    )


    # ds_canesm_lwa_reg_yearly = ds_canesm_lwa_reg.groupby("time.year")
    # ds_era5_lwa_reg_yearly   = ds_era5_lwa_reg.groupby("time.year")

    # plot HW thresholds
    # plot_temp_thresh_canesm_era5(ds_thresh_canesm_smooth, ds_thresh_era5_smooth, ds_tas_era5, OUTPUT_PLOTS_PATH)
    # plot_lwa_thresh_canesm_era5(ds_lwa_thresh_canesm_smooth[LWA_var], ds_lwa_thresh_era5_smooth[LWA_var], ds_era5_lwa_reg, OUTPUT_PLOTS_PATH)

    # mask_era = (ds_era5_lwa_reg.groupby("time.dayofyear") >= ds_lwa_thresh_era5_smooth[LWA_var]).compute()
    # mask_can = (ds_canesm_lwa_reg.groupby("time.dayofyear") >= ds_lwa_thresh_canesm_smooth[LWA_var].isel(time=0)).compute()

    

    # ds_tas_era5 = ds_tas_era5.assign_coords(time=ds_tas_era5.time.dt.floor("D"))

    # masked_lwa_era = ds_era5_lwa_reg.where(mask_era,drop=True)
    # masked_lwa_can = ds_canesm_lwa_reg.where(mask_can,drop=True)

    # masked_tas_era = ds_tas_era5.where(mask_era,drop=True)
    # masked_tas_can = ds_tas_canesm.where(mask_can,drop=True)

    # print(masked_lwa_era)
    # print(masked_tas_era)

    



if __name__ == "__main__":
    args = arg_parser()
    REGION   = args.region
    SEASON   = args.season
    ZG_LEVEL = args.zg

    main(REGION, SEASON, ZG_LEVEL)
