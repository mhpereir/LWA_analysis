import os
import sys
import argparse
from typing import Tuple

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from dask.distributed import Client, wait

from src import config, data_io, preprocess


OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/SM_baseline")
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot day-of-year soil moisture baseline for ERA5 and CanESM5."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="pnw_bartusek",
        choices=list(config.REGIONS.keys()),
        help="Region to analyze.",
    )
    
    return parser.parse_args()



def plot_sm_baseline(
    era5_clim: xr.DataArray,
    era5_std: xr.DataArray,
    canesm_mean: xr.DataArray,
    canesm_spread: xr.DataArray,
    canesm_daily_var: xr.DataArray,
    region: str,
    output_path: str,
) -> None:
    
    doy = canesm_mean["dayofyear"].values

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(
        doy,
        era5_clim.values - era5_std.values,
        era5_clim.values + era5_std.values,
        color="tab:blue",
        alpha=0.2,
        label="ERA5 daily variability",
    )
    ax.plot(doy, era5_clim.values, color="tab:blue", lw=2, label="ERA5 mean")

    # ax.fill_between(
    #     doy,
    #     canesm_mean.values - canesm_spread.values,
    #     canesm_mean.values + canesm_spread.values,
    #     color="tab:red",
    #     alpha=0.15,
    #     label="CanESM5 ensemble spread",
    # )
    ax.fill_between(
        doy,
        canesm_mean.values - canesm_daily_var.values,
        canesm_mean.values + canesm_daily_var.values,
        color="tab:red",
        alpha=0.3,
        label="CanESM5 daily variability",
    )
    ax.plot(doy, canesm_mean.values, color="tab:red", lw=2, label="CanESM5 ensemble mean")

    ax.set_title(f"Soil moisture day-of-year climatology ({region})")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Soil moisture (kg/m²)")
    ax.set_xlim(doy.min(), doy.max())
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(region: str) -> None:
    client = Client(processes=True, n_workers=8, threads_per_worker=1, memory_limit="2.5GB")
    ensemble_list = config.ENSEMBLE_LIST

    ds_canesm = data_io.open_canesm_mrsos(var="mrsos", ensemble_list=ensemble_list)
    ds_canesm = preprocess.compute_region_mean(ds_canesm, region).compute()

    ds_era5 = data_io.open_era5_mrsos(var="swvl1")
    ds_era5 = preprocess.compute_region_mean(ds_era5, region).compute()

    ds_era5 = preprocess.drop_leap_day(ds_era5)

    era5_clim, era5_std = preprocess.dayofyear_clim_and_std_noleap(ds_era5)
    canesm_mean, canesm_spread, canesm_daily_var = preprocess.canesm_dayofyear_stats(ds_canesm)

    fig_name = f"SM_baseline_{region}.png"
    output_path = os.path.join(OUTPUT_PLOTS_PATH, fig_name)
    plot_sm_baseline(
        era5_clim=era5_clim,
        era5_std=era5_std,
        canesm_mean=canesm_mean,
        canesm_spread=canesm_spread,
        canesm_daily_var=canesm_daily_var,
        region=region,
        output_path=output_path,
    )
    print(f"Saved: {output_path}")

    client.close()


    return None


if __name__ == "__main__":
    args = arg_parser()
    main(args.region)
    