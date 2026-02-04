#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite temperature anomalies during LWA extreme ("wavy") days in
CanESM5 vs ERA5, using modular src utilities.

Workflow
--------
1) Load LWA day-of-year thresholds for a region (q).
2) Compute regional-mean LWA time series for each LWA component.
3) Build masks for days exceeding LWA thresholds (wavy days).
4) Composite temperature anomalies on those days:
   - ERA5: mean over extreme days
   - CanESM5: per-member mean → ensemble mean + spread
5) Plot base composites and CanESM−ERA5 anomalies.

Date: 2026-02-02
"""
from __future__ import annotations

import os
import sys
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


# ------------------------------- Configuration -------------------------------

ENSEMBLE_LIST: List[str] = config.ENSEMBLE_LIST
VAR_NAMES: List[str] = config.LWA_VARS
PROJ = config.PROJ
HATCH_K: float = 2.0

OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/T_during_LWAs")
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)


# ----------------------------- Argument parsing -----------------------------

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temperature composites during extreme LWA (wavy) days (CanESM5 vs ERA5)."
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
        default="ALL",
        help="Season to analyze.",
    )
    parser.add_argument(
        "--zg",
        type=int,
        choices=[250, 500],
        default=500,
        help="Geopotential height level for LWA thresholds.",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=90,
        help="Quantile for LWA thresholds (e.g., 90 for wavy days).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Custom output prefix (optional).",
    )
    return parser.parse_args()


# ------------------------------ Computation utils ----------------------------

def smooth_doy_threshold(da_doy: xr.DataArray, win: int = 7) -> xr.DataArray:
    """Centered rolling mean over dayofyear with wrap-around."""
    if win % 2 == 0:
        raise ValueError("Window length must be odd for a centered mean.")

    half = win // 2
    n = da_doy.sizes["dayofyear"]

    padded = xr.concat(
        [da_doy.isel(dayofyear=slice(-half, None)), da_doy, da_doy.isel(dayofyear=slice(0, half))],
        dim="dayofyear",
    )

    rolled = padded.rolling(dayofyear=win, center=True).mean()
    result = rolled.isel(dayofyear=slice(half, n + half))
    result = result.assign_coords(dayofyear=da_doy.dayofyear)

    if "time" in result.dims:
        result = result.isel(time=0)

    return result


def select_season_time(da: xr.DataArray, season: Optional[str]) -> xr.DataArray:
    if season is None or season == "ALL":
        return da
    return da.sel(time=da.time.dt.season == season)


def build_extreme_mask(series: xr.DataArray, thresh_doy: xr.DataArray) -> xr.DataArray:
    return series.groupby("time.dayofyear") > thresh_doy


def align_mask_to_da(mask: xr.DataArray, da: xr.DataArray) -> xr.DataArray:
    mask = preprocess.floor_daily_time(mask)
    da = preprocess.floor_daily_time(da)
    mask = mask.reindex(time=da["time"]).fillna(False).astype(bool)
    return mask


def composite_on_mask(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    mask = align_mask_to_da(mask, da)
    ntrue = int(mask.sum().compute() if hasattr(mask.data, "compute") else mask.sum())
    if ntrue == 0:
        print("[warn] composite_on_mask: mask has zero True after time alignment")
    return da.where(mask).mean("time")


def prep_doy_threshold(thr: xr.DataArray, series: xr.DataArray) -> xr.DataArray:
    if "time" in thr.dims:
        if thr.sizes["time"] != 1:
            raise ValueError(f"Expected threshold time dim to be size 1, got {thr.sizes['time']}")
        thr = thr.squeeze("time", drop=True)

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
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, hspace=0.15, wspace=0.04)

    cmap = mpl.colormaps.get_cmap("magma")

    for i, v in enumerate(VAR_NAMES):
        era = composites_era5[v]
        can = composites_canesm[v].mean("member")
        both = xr.concat([era, can], dim="source")
        vmax = float(both.max(skipna=True).compute())
        vmin = float(both.min(skipna=True).compute())
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
        cbar.set_label("TAS [K]")

    fig.suptitle(f"TAS composites during {title_suffix}", y=0.995, fontsize=18, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_panel(
    composites_canesm: Dict[str, xr.DataArray],
    composites_era5: Dict[str, xr.DataArray],
    region: str,
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
        era = composites_era5[v]
        can = composites_canesm[v]

        anom_per_mem = can - era
        anom_ens_mean = anom_per_mem.mean("member")
        anom_ens_std = anom_per_mem.std("member")

        anoms[v] = anom_ens_mean
        stds[v] = anom_ens_std

        m = float(np.nanmax(np.abs(anom_ens_mean.values))) if anom_ens_mean.size > 0 else 0.0
        if np.isfinite(m):
            max_abs = max(max_abs, m)

    if max_abs == 0.0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    for i, v in enumerate(VAR_NAMES):
        ax = fig.add_subplot(gs[i, 0], projection=PROJ)
        a = anoms[v]
        s = stds[v]

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

        lat_slice, lon_slice = config.REGIONS[region]
        rect_lats = [lat_slice.start, lat_slice.stop, lat_slice.stop, lat_slice.start, lat_slice.start]
        rect_lons = [lon_slice.start, lon_slice.start, lon_slice.stop, lon_slice.stop, lon_slice.start]
        ax.plot(rect_lons, rect_lats, color="red", linewidth=2, transform=ccrs.PlateCarree())

        ax.contour(a["lon"], a["lat"], a, levels=[0], colors="k", linewidths=0.6, transform=ccrs.PlateCarree())

        mask = (np.abs(a) <= hatch_k * s).where(~np.isnan(s), False) #type: ignore
        ax.contourf(
            mask["lon"],
            mask["lat"],
            mask.astype(int),
            levels=[-0.5, 0.5, 1.5],
            hatches=["", "////"],
            colors="none",
            transform=ccrs.PlateCarree(),
        )

    cax = fig.add_axes([0.12, 0.05, 0.76, 0.025])  # type: ignore
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("CanESM − ERA5")

    fig.suptitle(f"TAS composite anomaly during {title_suffix}", y=0.995, fontsize=18, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_panel_seasonal(
    composites_canesm: Dict[str, xr.DataArray],
    composites_era5: Dict[str, xr.DataArray],
    region: str,
    out_png: str,
    title_suffix: str,
) -> None:
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, hspace=0.15, wspace=0.04)

    cmap = mpl.colormaps.get_cmap("RdBu_r")

    for i, v in enumerate(VAR_NAMES):
        era = composites_era5[v]
        can = composites_canesm[v].mean("member")

        both = xr.concat([era.abs(), can.abs()], dim="source")
        vmax = float(both.max(skipna=True).compute()) #type: ignore
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
        cbar.set_label("TAS [K]")

    fig.suptitle(f"TAS anomaly during wavy days {title_suffix}", y=0.995, fontsize=18, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------- Pipeline ----------------------------------

def run_analysis(
    region: str,
    q: int,
    season: str,
    zg_level: int,
    out_prefix: Optional[str],
) -> None:
    if out_prefix is None:
        out_prefix = os.path.join(
            OUTPUT_PLOTS_PATH,
            f"TAS_extremes_CanESM_minus_ERA5.zg{zg_level}.{region}.{season}",
        )

    tas_canesm = data_io.open_canesm_temperature(config.TEMP_VAR, ENSEMBLE_LIST)
    tas_era5 = data_io.open_era5_temperature(config.TEMP_VAR)

    tas_canesm = preprocess.drop_leap_day(preprocess.floor_daily_time(tas_canesm))
    tas_era5 = preprocess.drop_leap_day(preprocess.floor_daily_time(tas_era5))

    if tas_canesm.sizes["time"] != tas_era5.sizes["time"]:
        raise ValueError("CanESM and ERA5 time lengths do not match after removing leap days.")

    tas_canesm_anom = preprocess.dayofyear_anomaly(tas_canesm)
    tas_era5_anom = preprocess.dayofyear_anomaly(tas_era5)

    tas_canesm_anom = select_season_time(tas_canesm_anom, season)
    tas_era5_anom = select_season_time(tas_era5_anom, season)

    lwa_can = data_io.open_canesm_lwa(ENSEMBLE_LIST, zg_level)
    lwa_era = data_io.open_era5_lwa(zg_level)

    for v in VAR_NAMES:
        lwa_can[v] = preprocess.drop_leap_day(preprocess.floor_daily_time(lwa_can[v]))
        lwa_era[v] = preprocess.drop_leap_day(preprocess.floor_daily_time(lwa_era[v]))
        lwa_can[v] = lwa_can[v].reindex(time=tas_canesm.time)
        lwa_era[v] = lwa_era[v].reindex(time=tas_era5.time)

    lwa_thresh_can = data_io.open_canesm_lwa_thresh(ENSEMBLE_LIST, q, region, zg_level)
    lwa_thresh_era = data_io.open_era5_lwa_thresh(q, region, zg_level)

    comps_can: Dict[str, xr.DataArray] = {}
    comps_era: Dict[str, xr.DataArray] = {}

    for lwa_var in VAR_NAMES:
        lwa_mean_canesm = preprocess.compute_region_mean(lwa_can[lwa_var], region)
        lwa_mean_era5 = preprocess.compute_region_mean(lwa_era[lwa_var], region)

        lwa_mean_canesm = select_season_time(lwa_mean_canesm, season)
        lwa_mean_era5 = select_season_time(lwa_mean_era5, season)

        thr_can = prep_doy_threshold(
            smooth_doy_threshold(lwa_thresh_can[lwa_var]),
            lwa_mean_canesm,
        )
        thr_era = prep_doy_threshold(
            smooth_doy_threshold(lwa_thresh_era[lwa_var]),
            lwa_mean_era5,
        )

        mask_can = build_extreme_mask(lwa_mean_canesm, thr_can)
        mask_era = build_extreme_mask(lwa_mean_era5, thr_era)

        print(
            f"number of extreme LWA days for {lwa_var} (CanESM, ERA5):",
            int(mask_can.sum().values),
            int(mask_era.sum().values),
        )

        comp_can = composite_on_mask(tas_canesm_anom, mask_can).compute()
        comp_era = composite_on_mask(tas_era5_anom, mask_era).compute()

        if np.all(~np.isfinite(comp_can.values)):
            print(f"[warn] CanESM composite for {lwa_var} is NaN everywhere (empty mask?)")
        if np.all(~np.isfinite(comp_era.values)):
            print(f"[warn] ERA5 composite for {lwa_var} is NaN everywhere (empty mask?)")

        comps_can[lwa_var] = comp_can
        comps_era[lwa_var] = comp_era

    plot_base_lwa_panel(
        comps_can,
        comps_era,
        out_png=f"{out_prefix}.base.png",
        region=region,
        title_suffix=f"regional wavy days (q={q}; {season})",
    )

    plot_anomaly_panel_seasonal(
        composites_canesm=comps_can,
        composites_era5=comps_era,
        region=region,
        out_png=f"{out_prefix}.clim.png",
        title_suffix=f" (q={q}; {season})",
    )

    plot_anomaly_panel(
        composites_canesm=comps_can,
        composites_era5=comps_era,
        region=region,
        out_png=f"{out_prefix}.png",
        hatch_k=HATCH_K,
        title_suffix=f"regional wavy days (q={q}; {season})",
    )


if __name__ == "__main__":
    args = arg_parser()
    run_analysis(
        region=args.region,
        q=args.q,
        season=args.season,
        zg_level=args.zg,
        out_prefix=args.out_prefix,
    )
