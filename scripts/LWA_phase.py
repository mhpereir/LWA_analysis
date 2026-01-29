import os
import sys

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, preprocess, data_io
from typing import Dict, List, Tuple, Any

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib import ticker as mticker


import argparse


# Define script specific constants
LAT_BANDs: dict[str, Tuple[float, float]] = {
    "north": (55, 70),
    "south": (40, 55),
    "all":   (40, 70)
}

PLOTS_OUTPUT_PATH = os.path.join(config.OUTPUT_PATH, "plots/LWA_phase")

# Define argument parser
def arg_parser():
    parser = argparse.ArgumentParser(
        description="LWA vs deltaT correlation analysis and plotting."
    )
    parser.add_argument(
        "--lat_band",
        type=str,
        choices=list(LAT_BANDs.keys()),
        default="all",
        help="Latitude range to analyze.",
    )
    parser.add_argument(
        "--season",
        type=str,
        choices=config.SEASON_NAMES,
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


# ----------------------------- Helpers -----------------------------

def compute_lat_band_mean(ds: xr.DataArray, lat_band: str) -> xr.DataArray:
    """Compute area-weighted mean over latitude band."""
    lat_min, lat_max = LAT_BANDs[lat_band]
    ds_band = ds.sel(lat=slice(lat_min, lat_max))

    weights = np.cos(np.deg2rad(ds_band.lat))
    return ds_band.weighted(weights).mean(dim=["lat"]) # dimension 'lon' remains


# ----------------------------- Plotting helpers -----------------------------

def plot_LWA_phase(
    lwa_era5_spatial: xr.DataArray,
    dict_canesm: Dict[str, xr.DataArray],
    dict_era5: Dict[str, xr.DataArray],
    lat_band_name: str,
    title: str = "",
    output_path: str = PLOTS_OUTPUT_PATH,
    fig_name: str = ""
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

    lat_min, lat_max = LAT_BANDs[lat_band_name]
    for lat in (lat_min, lat_max):
        ax_3.plot(
            np.linspace(-180, 180, 360),
            np.full(360, lat),
            transform=ccrs.PlateCarree(),
            color="blue",
            linewidth=2
    )
        
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

    file_name = os.path.join(output_path, fig_name)
    fig.savefig(file_name, dpi=300, bbox_inches='tight')

    return None

# ----------------------------- Main analysis function -----------------------------

def run_analysis(lat_band_name: str, season: str, zg_coord: int) -> None:
    ## loading LWA data
    # 1) Open CanESM and ERA5 for all VARiables
    ds_canesm_lwas: Dict[str, xr.DataArray] = data_io.open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas:   Dict[str, xr.DataArray] = data_io.open_era5_lwa(ZG_COORD)

    canesm_lwas_lon: Dict[str, xr.DataArray] = {}
    era5_lwas_lon: Dict[str, xr.DataArray]   = {}
    
    lwa_era5_spatial: xr.DataArray = None # type: ignore
    for LWA_var in LWA_VARS:
        print(f"Processing LWA variable: {LWA_var}")

        ds_canesm_lwa = ds_canesm_lwas[LWA_var]#.sel(MEMBER=MEMBER) #(time, member, lat, lon)
        ds_era5_lwa   = ds_era5_lwas[LWA_var] #(time, lat, lon)

        if SEASON == 'all' or SEASON == 'ALL':
            ds_canesm_lwa = ds_canesm_lwa.mean(dim="time")
            ds_era5_lwa   = ds_era5_lwa.mean(dim="time")

        else:
            ds_canesm_lwa = ds_canesm_lwa.sel(time=ds_canesm_lwa.time.dt.season == SEASON).mean(dim="time") #(member, lat, lon)
            ds_era5_lwa   = ds_era5_lwa.sel(time=ds_era5_lwa.time.dt.season == SEASON).mean(dim="time") # (lat, lon)

        if LWA_var == 'LWA':
            lwa_era5_spatial = ds_era5_lwa.compute()  #(lat, lon)

        canesm_lwas_lon[LWA_var] = compute_lat_band_mean(ds_canesm_lwa, lat_band_name).compute() #(member, lon)
        era5_lwas_lon[LWA_var]   = compute_lat_band_mean(ds_era5_lwa, lat_band_name).compute()   #(lon)

    if lwa_era5_spatial is None:
        raise ValueError("LWA_VARS must contain 'LWA' for spatial plotting")

    plot_LWA_phase(
        lwa_era5_spatial=lwa_era5_spatial,
        dict_canesm=canesm_lwas_lon,
        dict_era5=era5_lwas_lon,
        lat_band_name=lat_band_name,
        title=f"ERA5 LWA Phase Analysis - {lat_band_name}, {season}, z={zg_coord}hPa",
        fig_name=f"LWA_phase_plot_{lat_band_name}_{season}_z{zg_coord}.png",
        output_path=PLOTS_OUTPUT_PATH
    )

    return None



if __name__ == "__main__":
    args = arg_parser()
    LAT_BAND = args.lat_band
    SEASON   = args.season
    ZG_COORD = args.zg

    ENSEMBLE_LIST = config.ENSEMBLE_LIST
    LWA_VARS = ["LWA", "LWA_a", "LWA_c"]

    os.makedirs(PLOTS_OUTPUT_PATH, exist_ok=True)

    run_analysis(
        lat_band_name=LAT_BAND,
        season=SEASON,
        zg_coord=ZG_COORD
    )
