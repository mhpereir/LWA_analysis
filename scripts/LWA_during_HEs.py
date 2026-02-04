from __future__ import annotations

import os
import sys
import glob
import argparse
from typing import Dict, List, Optional

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize

import cartopy.crs as ccrs

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, data_io, preprocess


plt.rcParams.update({"font.size": 16})

# ------------------------------- Configuration -------------------------------

ENSEMBLE_LIST: List[str] = config.ENSEMBLE_LIST
VAR_NAMES: List[str] = config.LWA_VARS
PROJ = config.PROJ
HATCH_K: float = 2.0

OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/LWA_during_HEs")
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)

# ----------------------------- Argument parsing -----------------------------

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LWA composites during hot/cold temperature extremes (CanESM5 vs ERA5)."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=list(config.REGIONS.keys()),
        default="central",
        help="Region to analyze.",
    )
    parser.add_argument(
        "--season",
        type=str,
        choices=sorted(config.SEASON_NAMES),
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
        "--q-hot",
        type=int,
        default=95,
        help="Hot-day quantile for temperature thresholds.",
    )
    parser.add_argument(
        "--q-cold",
        type=int,
        default=5,
        help="Cold-day quantile for temperature thresholds.",
    )
    parser.add_argument(
        "--which",
        type=str,
        choices=["hot", "cold", "both"],
        default="hot",
        help="Which composite(s) to compute.",
    )
    parser.add_argument(
        "--hw-thresh-root",
        type=str,
        default=config.HW_THRESH_ROOT,
        help="Root folder for HW threshold files.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Custom output prefix (optional).",
    )
    return parser.parse_args()


# ------------------------------- I/O utilities -------------------------------



# ------------------------------ Computation utils ----------------------------

def smooth_doy_threshold(da_doy: xr.DataArray, win: int = 7) -> xr.DataArray:
    return da_doy.rolling(dayofyear=win, center=True).mean()


def select_season_time(da: xr.DataArray, season: Optional[str]) -> xr.DataArray:
    if season is None or season == "ALL":
        return da
    return da.sel(time=da.time.dt.season == season)


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

# --------------------------------- Plotting ----------------------------------

def plot_base_lwa_panel(
    composites_canesm: Dict[str, xr.DataArray],
    composites_era5: Dict[str, xr.DataArray],
    out_png: str,
    region: str,
    title_suffix: str,
) -> None:
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, hspace=0.15, wspace=0.04)

    cmap = mpl.colormaps.get_cmap("magma")

    for i, v in enumerate(VAR_NAMES):
        era = composites_era5[v]
        can = composites_canesm[v].mean("member")
        vmax = float(np.nanpercentile([era.compute().values, can.compute().values], 99))
        vmin = 0.0 if np.isfinite(vmax) else 0.0
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0

        ax0 = fig.add_subplot(gs[i, 0], projection=PROJ)
        era.plot.pcolormesh(
            ax=ax0,
            x="lon",
            y="lat",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            add_labels=False,
            shading="nearest",
            rasterized=True,
        )
        ax0.coastlines(linewidth=0.5, color="0.35")  # type: ignore
        ax0.set_title(f"ERA5 {v}")

        lat_slice, lon_slice = config.REGIONS[region]
        rect_lats = [lat_slice.start, lat_slice.stop, lat_slice.stop, lat_slice.start, lat_slice.start]
        rect_lons = [lon_slice.start, lon_slice.start, lon_slice.stop, lon_slice.stop, lon_slice.start]
        ax0.plot(rect_lons, rect_lats, color="red", linewidth=2, transform=ccrs.PlateCarree())

        ax1 = fig.add_subplot(gs[i, 1], projection=PROJ)
        can.plot.pcolormesh(
            ax=ax1,
            x="lon",
            y="lat",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            add_labels=False,
            shading="nearest",
            rasterized=True,
        )
        ax1.coastlines(linewidth=0.5, color="0.35")  # type: ignore
        ax1.set_title(f"CanESM mean {v}")
        ax1.plot(rect_lons, rect_lats, color="red", linewidth=2, transform=ccrs.PlateCarree())

        cax = fig.add_axes([0.92, 0.72 - i * 0.31, 0.015, 0.22])  # type: ignore
        sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(f"{v} (units)")

    fig.suptitle(f"LWA composites during {title_suffix}", y=0.995, fontsize=18, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_panel(
    composites_canesm: Dict[str, xr.DataArray],
    composites_era5: Dict[str, xr.DataArray],
    out_png: str,
    hatch_k: float,
    title_suffix: str,
) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, hspace=0.08)

    cmap = mpl.colormaps.get_cmap("RdBu_r")

    max_abs = 0.0
    anoms: Dict[str, xr.DataArray] = {}
    stds: Dict[str, xr.DataArray] = {}

    for v in VAR_NAMES:
        ens_mean = composites_canesm[v].mean("member")
        ens_std = composites_canesm[v].std("member")
        anom = (ens_mean - composites_era5[v]) / composites_era5[v]
        anoms[v] = anom
        stds[v] = ens_std
        m = float(np.nanmax(np.abs(anom.values))) if anom.size > 0 else 0.0
        if np.isfinite(m):
            max_abs = np.nanpercentile(np.asarray([max_abs, m]), 99)

    if max_abs == 0.0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    for i, v in enumerate(VAR_NAMES):
        ax = fig.add_subplot(gs[i, 0], projection=PROJ)
        a = anoms[v]
        a.plot.pcolormesh(
            ax=ax,
            x="lon",
            y="lat",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            add_labels=False,
            shading="nearest",
            rasterized=True,
        )
        ax.coastlines(linewidth=0.5, color="0.35")  # type: ignore
        ax.set_title(f"{v}: CanESM − ERA5", fontsize=16)

        ax.contour(a["lon"], a["lat"], a, levels=[0], colors="k", linewidths=0.6, transform=ccrs.PlateCarree())

    cax = fig.add_axes([0.12, 0.05, 0.76, 0.025])  # type: ignore
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("CanESM − ERA5 / CanESM")

    fig.suptitle(f"LWA composite anomaly during {title_suffix}", y=0.995, fontsize=18, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_panel_seasonal(
    composites_canesm: Dict[str, xr.DataArray],
    ds_canesm: Dict[str, xr.DataArray],
    composites_era5: Dict[str, xr.DataArray],
    ds_era5: Dict[str, xr.DataArray],
    region: str,
    season: str,
    out_png: str,
    hatch_k: float,
    title_suffix: str,
) -> None:
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, hspace=0.15, wspace=0.04)

    cmap = mpl.colormaps.get_cmap("RdBu_r")

    for i, v in enumerate(VAR_NAMES):
        era = composites_era5[v]
        can = composites_canesm[v].mean("member")

        era_seasonal = select_season_time(ds_era5[v], season).mean("time")
        can_seasonal = select_season_time(ds_canesm[v], season).mean("time").mean("member")

        era = era - era_seasonal
        can = can - can_seasonal

        vmax = float(np.nanpercentile([np.abs(era.compute().values), np.abs(can.compute().values)], 99))
        vmin = -vmax

        ax0 = fig.add_subplot(gs[i, 0], projection=PROJ)
        era.plot.pcolormesh(
            ax=ax0,
            x="lon",
            y="lat",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            add_labels=False,
            shading="nearest",
            rasterized=True,
        )
        ax0.coastlines(linewidth=0.5, color="0.35")  # type: ignore
        ax0.set_title(f"ERA5 {v}")

        lat_slice, lon_slice = config.REGIONS[region]
        rect_lats = [lat_slice.start, lat_slice.stop, lat_slice.stop, lat_slice.start, lat_slice.start]
        rect_lons = [lon_slice.start, lon_slice.start, lon_slice.stop, lon_slice.stop, lon_slice.start]
        ax0.plot(rect_lons, rect_lats, color="red", linewidth=2, transform=ccrs.PlateCarree())

        ax1 = fig.add_subplot(gs[i, 1], projection=PROJ)
        can.plot.pcolormesh(
            ax=ax1,
            x="lon",
            y="lat",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            add_labels=False,
            shading="nearest",
            rasterized=True,
        )
        ax1.coastlines(linewidth=0.5, color="0.35")  # type: ignore
        ax1.set_title(f"CanESM mean {v}")
        ax1.plot(rect_lons, rect_lats, color="red", linewidth=2, transform=ccrs.PlateCarree())

        cax = fig.add_axes([0.92, 0.72 - i * 0.31, 0.015, 0.22])  # type: ignore
        sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(f"{v} (units)")

    fig.suptitle(f"LWA climatological anomaly {title_suffix}", y=0.995, fontsize=18, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------- Pipeline ----------------------------------

def run_analysis(
    region: str,
    q_hot: int,
    q_cold: int,
    which: str,
    season: str,
    zg_level: int,
    out_prefix: Optional[str],
) -> None:
    if out_prefix is None:
        out_prefix = os.path.join(
            OUTPUT_PLOTS_PATH,
            f"LWA_extremes_CanESM_ERA5.zg{zg_level}.{region}.{season}",
        )

    tas_canesm = data_io.open_canesm_temperature(config.TEMP_VAR, ENSEMBLE_LIST)
    tas_era5   = data_io.open_era5_temperature(config.TEMP_VAR)

    tas_canesm = preprocess.drop_leap_day(preprocess.floor_daily_time(tas_canesm))
    tas_era5   = preprocess.drop_leap_day(preprocess.floor_daily_time(tas_era5))

    tmean_canesm = preprocess.compute_region_mean(tas_canesm, region)
    tmean_era5   = preprocess.compute_region_mean(tas_era5, region)

    lwa_can = data_io.open_canesm_lwa(ENSEMBLE_LIST, zg_level)
    lwa_era = data_io.open_era5_lwa(zg_level)

    for v in VAR_NAMES:
        lwa_can[v] = preprocess.drop_leap_day(preprocess.floor_daily_time(lwa_can[v]))
        lwa_era[v] = preprocess.drop_leap_day(preprocess.floor_daily_time(lwa_era[v]))
        lwa_can[v] = lwa_can[v].reindex(time=tmean_canesm.time)
        lwa_era[v] = lwa_era[v].reindex(time=tmean_era5.time)

        lwa_can[v] = preprocess.dayofyear_anomaly(lwa_can[v])
        lwa_era[v] = preprocess.dayofyear_anomaly(lwa_era[v])

    if tmean_canesm.sizes["time"] != tmean_era5.sizes["time"]:
        raise ValueError("CanESM and ERA5 time lengths do not match after removing leap days.")

    thr_hot_can = smooth_doy_threshold(
        data_io.open_canesm_hw_thresh(config.TEMP_VAR, q_hot, region, ENSEMBLE_LIST)
    )
    thr_hot_era = smooth_doy_threshold(data_io.open_era5_hw_thresh(config.TEMP_VAR, q_hot, region))

    thr_hot_can = prep_doy_threshold(thr_hot_can, tmean_canesm)
    thr_hot_era = prep_doy_threshold(thr_hot_era, tmean_era5)  # no-op, but consistent

    

    # thr_cold_can = smooth_doy_threshold(
    #     data_io.open_canesm_hw_thresh(config.TEMP_VAR, q_cold, region, ENSEMBLE_LIST)
    # )
    # thr_cold_era = smooth_doy_threshold(data_io.open_era5_hw_thresh(config.TEMP_VAR, q_cold, region))

    tmean_canesm = select_season_time(tmean_canesm, season)
    tmean_era5   = select_season_time(tmean_era5, season)

    mask_hot_can = build_extreme_mask(tmean_canesm, thr_hot_can, is_hot=True)
    mask_hot_era = build_extreme_mask(tmean_era5, thr_hot_era, is_hot=True)

    print(np.sum(mask_hot_can.compute().values), len(tmean_canesm.time), "hot days in CanESM")
    print(np.sum(mask_hot_era.compute().values), len(tmean_era5.time), "hot days in ERA5")

    # mask_cold_can = build_extreme_mask(tmean_canesm, thr_cold_can, is_hot=False)
    # mask_cold_era = build_extreme_mask(tmean_era5, thr_cold_era, is_hot=False)

    def composites(mask_can: xr.DataArray, mask_era: xr.DataArray, label_suffix: str):
        comps_can: Dict[str, xr.DataArray] = {}
        comps_era: Dict[str, xr.DataArray] = {}
        for v in VAR_NAMES:
            comp_can = composite_on_mask(lwa_can[v], mask_can).compute()
            comp_era = composite_on_mask(lwa_era[v], mask_era).compute()

            if np.all(~np.isfinite(comp_can.values)):
                print(f"[warn] CanESM composite for {v} during {label_suffix} is NaN everywhere (empty mask?)")
            if np.all(~np.isfinite(comp_era.values)):
                print(f"[warn] ERA5 composite for {v} during {label_suffix} is NaN everywhere (empty mask?)")

            comps_can[v] = comp_can
            comps_era[v] = comp_era
        return comps_can, comps_era

    if which in ("hot", "both"):
        comps_can_hot, comps_era_hot = composites(mask_hot_can, mask_hot_era, "hot")
        plot_anomaly_panel(
            comps_can_hot,
            comps_era_hot,
            out_png=f"{out_prefix}.hot.png",
            hatch_k=HATCH_K,
            title_suffix=f"regional hot days (q={q_hot}; {season})",
        )
        plot_base_lwa_panel(
            comps_can_hot,
            comps_era_hot,
            out_png=f"{out_prefix}.hot.base.png",
            region=region,
            title_suffix=f"regional hot days (q={q_hot}; {season})",
        )
        plot_anomaly_panel_seasonal(
            composites_canesm=comps_can_hot,
            ds_canesm=lwa_can,
            composites_era5=comps_era_hot,
            ds_era5=lwa_era,
            region=region,
            season=season,
            out_png=f"{out_prefix}.hot.clim.png",
            hatch_k=HATCH_K,
            title_suffix=f" (hot; {season})",
        )

    if which in ("cold", "both"):
        sys.exit("Cold composites not implemented yet.")
        # comps_can_cold, comps_era_cold = composites(mask_cold_can, mask_cold_era, "cold")
        # plot_anomaly_panel(
        #     comps_can_cold,
        #     comps_era_cold,
        #     out_png=f"{out_prefix}.cold.png",
        #     hatch_k=HATCH_K,
        #     title_suffix=f"regional cold days (q={q_cold}; {season})",
        # )

        # plot_base_lwa_panel(
        #     comps_can_cold,
        #     comps_era_cold,
        #     out_png=f"{out_prefix}.cold.base.png",
        #     region=region,
        #     title_suffix=f"base LWA composites (cold; {season})",
        # )

        # plot_anomaly_panel_seasonal(
        #     composites_canesm=comps_can_cold,
        #     ds_canesm=lwa_can,
        #     composites_era5=comps_era_cold,
        #     ds_era5=lwa_era,
        #     region=region,
        #     season=season,
        #     out_png=f"{out_prefix}.cold.clim.png",
        #     hatch_k=HATCH_K,
        #     title_suffix=f" (cold; {season})",
        # )


if __name__ == "__main__":
    args = arg_parser()
    run_analysis(
        region=args.region,
        q_hot=args.q_hot,
        q_cold=args.q_cold,
        which=args.which,
        season=args.season,
        zg_level=args.zg,
        out_prefix=args.out_prefix,
    )
