from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List


from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import colors
from matplotlib.patches import Patch

import cartopy.crs as ccrs

from dask.distributed import Client, wait

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, data_io, preprocess


plt.rcParams.update({"font.size": 16})


# ------------------------------- Configuration -------------------------------

# Model ensemble members and variables from config
ENSEMBLE_LIST: List[str] = config.ENSEMBLE_LIST
VAR_NAMES: List[str] = config.LWA_VARS
SEASONS: List[str] = ["DJF", "MAM", "JJA", "SON"]
# DEFAULT_Q_LIST: List[float] = [0.01, 0.05, 0.95, 0.99]
DEFAULT_HATCH_K = 2.0

OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/LWA_baseline")
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seasonal LWA baseline anomalies (CanESM5 minus ERA5)."
    )
    parser.add_argument(
        "--zg",
        type=int,
        choices=[250, 500],
        default=500,
        help="Geopotential height level for LWA.",
    )
    parser.add_argument(
        "--hatch-k",
        type=float,
        default=DEFAULT_HATCH_K,
        help="Hatching threshold multiplier (|anom| <= k × std).",
    )
    # parser.add_argument(
    #     "--quantiles",
    #     type=str,
    #     default=",".join(str(q) for q in DEFAULT_Q_LIST),
    #     help="Comma-separated quantiles (e.g., '0.01,0.05,0.95,0.99').",
    # )
    # parser.add_argument(
    #     "--no-quantiles",
    #     action="store_true",
    #     help="Skip seasonal quantile plots.",
    # )
    return parser.parse_args()


# ------------------------------ Computation utils -----------------------------

def seasonal_clim_and_spread_canesm(da: xr.DataArray) -> xr.Dataset:
    """
    Compute seasonal climatology mean and ensemble std for CanESM.

    Workflow:
    - Subset by season
    - Average by year within season (seasonal mean for each year)
    - Climatological mean across years (per member)
    - Aggregate across members: mean and std

    Parameters
    ----------
    da : xr.DataArray
        CanESM variable with dims (member, time, lat, lon).

    Returns
    -------
    xr.Dataset
        'mean' and 'std' with dims (season, lat, lon).
    """
    seasons = SEASONS
    out_mean = []
    out_std = []

    for s in seasons:
        ds_season = da.sel(time=da.time.dt.season == s)
        # Yearly seasonal means, then climatology per member
        ds_yearly = ds_season.groupby("time.year").mean("time")
        ds_clim_members = ds_yearly.mean("year")

        # Across-member aggregation
        out_mean.append(ds_clim_members.mean("member"))
        out_std.append(ds_clim_members.std("member"))

    return xr.Dataset({
            "mean": xr.concat(out_mean, dim="season"),
            "std": xr.concat(out_std, dim="season"),
        },
        coords={"season": seasons},
    )


def seasonal_clim_era5(da: xr.DataArray) -> xr.Dataset:
    """
    Compute seasonal climatology for ERA5.

    Parameters
    ----------
    da : xr.DataArray
        ERA5 variable with dims (time, lat, lon).

    Returns
    -------
    xr.Dataset
        'mean' with dims (season, lat, lon).
    """
    seasons = SEASONS
    out_mean = []

    for s in seasons:
        ds_season = da.sel(time=da.time.dt.season == s)
        ds_yearly = ds_season.groupby("time.year").mean("time")
        out_mean.append(ds_yearly.mean("year"))

    return xr.Dataset({"mean": xr.concat(out_mean, dim="season")}, coords={"season": seasons})

def seasonal_daily_quantiles(da: xr.DataArray, q_list: List[float], is_canesm: bool) -> xr.Dataset:
    """
    Compute seasonal daily quantiles and aggregate across members (if present).

    Returns
    -------
    xr.Dataset
        Variables:
          - 'mean': ensemble mean of per-member quantiles (or the quantile itself for ERA5)
          - 'std' : ensemble std of per-member quantiles (NaN for ERA5)
        Dims: (quantile, season, lat, lon)
    """
    means = []
    stds  = []

    for s in SEASONS:
        sel = da.where(da.time.dt.season == s, drop=True)
        # Quantiles per member (if 'member' dim exists) or per field otherwise
        q = sel.quantile(q_list, dim="time")  # dims: (quantile, [member], lat, lon)

        if is_canesm:
            # Aggregate across ensemble members
            mean_q = q.mean("member")
            std_q  = q.std("member")
        else:
            mean_q = q
            # std undefined for non-ensemble: fill with NaN
            std_q  = xr.full_like(q, np.nan)

        means.append(mean_q)
        stds.append(std_q)

    mean = xr.concat(means, dim="season")
    std  = xr.concat(stds,  dim="season")

    # Ensure coords are set explicitly
    mean = mean.assign_coords(season=SEASONS, quantile=q_list)
    std  = std.assign_coords(season=SEASONS, quantile=q_list)

    ds = xr.Dataset({"mean": mean, "std": std})
    ds["mean"].attrs.update(description="Ensemble mean of per-member time-quantiles (or single-field quantile)")
    ds["std"].attrs.update(description="Ensemble std of per-member time-quantiles (NaN for non-ensemble)")
    return ds


# --------------------------------- Plotting ----------------------------------

def plot_baseline_grid(
    sim_stats: Dict[str, xr.Dataset],
    sim_name: str,
    out_png: str,
    hatch_k: float = DEFAULT_HATCH_K
) -> None:
    
    # Figure + grid
    fig = plt.figure(figsize=(14, 4.5))
    nrows, ncols = len(VAR_NAMES)+1, len(SEASONS)
    gs = gridspec.GridSpec(
            nrows=nrows, ncols=ncols, figure=fig,
            wspace=0.02, hspace=0,
            height_ratios=[1]*len(VAR_NAMES) + [0.08]  # <- allocate real height to cbar
)
    # Style
    cmap = mpl.colormaps.get_cmap("RdBu_r")
    mpl.rcParams["hatch.linewidth"] = 0.6

    # Global color scale across all vars/seasons
    all_means = [sim_stats[var]["mean"] for var in VAR_NAMES]
    vmin = 0
    # vmax = float(xr.concat(all_means, dim="stack").max().compute())

    

    for i, var in enumerate(VAR_NAMES):
        variable_mean = sim_stats[var]["mean"]
        for j, season in enumerate(SEASONS):
            vmax = np.max(sim_stats['LWA']["mean"].sel(season=season).values)
            anom_season = variable_mean.sel(season=season)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            ax = fig.add_subplot(gs[i, j], projection=config.PROJ)

            # Main field (rasterized keeps PDF size small; text/lines stay vector)
            anom_season.plot.pcolormesh(
                ax=ax, x="lon", y="lat",
                transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm,
                add_colorbar=False, add_labels=False,
                shading="nearest", rasterized=True,
            )

            # Coastlines + zero contour
            ax.coastlines(linewidth=0.5, color="0.35") # type: ignore
            ax.contour(
                anom_season["lon"], anom_season["lat"], anom_season,
                levels=[0], colors="k", linewidths=0.6, alpha=0.7,
                transform=ccrs.PlateCarree(),
            )

            # Titles / row labels
            if i == 0:
                ax.set_title(season, fontsize=16)
            if j == 0:
                ax.text(-0.08, 0.5, var, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=16) #type: ignore

            # # Hatching where |anom| ≤ k × σ
            # mask = (np.abs(anom_season) <= (hatch_k * s)).where(~np.isnan(s), False) #type: ignore
            # ax.contourf(
            #     mask["lon"], mask["lat"], mask.astype(int),
            #     levels=[-0.5, 0.5, 1.5],
            #     hatches=["", "////"], colors="none",
            #     transform=ccrs.PlateCarree(),
            # )

    for j, season in enumerate(SEASONS):
        host = fig.add_subplot(gs[nrows-1, j])
        host.set_axis_off()  # don't show the host axes frame/ticks

        # inset box: [left, bottom, width, height] in host-axes coordinates
        cax = host.inset_axes([0.08, 0.25, 0.84, 0.55])  #type: ignore # <- controls length + thickness

        # Shared colorbar
        # cax = fig.add_axes([0.12, 0.06, 0.76, 0.03]) # type: ignore
        vmax = np.max(sim_stats['LWA']["mean"].sel(season=season).values)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) #type: ignore
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((0, 0))
        # formatter.format = "%.1f"
        # cbar.ax.xaxis.set_major_formatter(formatter)
        # cbar.update_ticks()

    fig.text(
        0.5,        # centered horizontally
        -0.03,       # vertical position 
        "Seasonal Mean",
        ha="center",
        va="center",
        fontsize=16,
    )

    fig.suptitle(f"{sim_name} Seasonal Climatology", fontsize=16, fontweight="bold", y=1.02)

    fig.savefig(out_png + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_anomaly_grid(
    canesm_stats: Dict[str, xr.Dataset],
    era5_stats: Dict[str, xr.Dataset],
    out_png: str,
    hatch_k: float = DEFAULT_HATCH_K
) -> None:
    
    # Figure + grid
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(nrows=4, ncols=4, figure=fig, wspace=0.02, hspace=0.,
    height_ratios=[1]*len(VAR_NAMES) + [0.08])  # <- allocate real height to cbar)

    # Style
    cmap = mpl.colormaps.get_cmap("RdBu_r")
    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    mpl.rcParams["hatch.linewidth"] = 0.6

    lwa_anom = canesm_stats["LWA"]["mean"] - era5_stats["LWA"]["mean"]

    for i, var in enumerate(VAR_NAMES):
        anomaly = canesm_stats[var]["mean"] - era5_stats[var]["mean"]
        for j, season in enumerate(SEASONS):
            
            anom_season = anomaly.sel(season=season)
            std_season  = (canesm_stats[var]["std"]).sel(season=season)

            norm_factor = float(np.nanpercentile( np.abs(lwa_anom.sel(season=season)), 99)) or 1.0
            a = anom_season / norm_factor
            s = std_season / norm_factor

            ax = fig.add_subplot(gs[i, j], projection=config.PROJ)

            # Main field (rasterized keeps PDF size small; text/lines stay vector)
            a.plot.pcolormesh(
                ax=ax, x="lon", y="lat",
                transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm,
                add_colorbar=False, add_labels=False,
                shading="nearest", rasterized=True,
            )

            # Coastlines + zero contour
            ax.coastlines(linewidth=0.5, color="0.35") # type: ignore
            ax.contour(
                a["lon"], a["lat"], a,
                levels=[0], colors="k", linewidths=0.6, alpha=0.7,
                transform=ccrs.PlateCarree(),
            )

            # Titles / row labels
            if i == 0:
                ax.set_title(season, fontsize=20, pad=8, fontweight="bold")
            if j == 0:
                ax.text(-0.08, 0.5, var, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=16) #type: ignore

            # Hatching where |anom| ≤ k × σ
            mask = (np.abs(a) <= (hatch_k * s)).where(~np.isnan(s), False) #type: ignore
            ax.contourf(
                mask["lon"], mask["lat"], mask.astype(int),
                levels=[-0.5, 0.5, 1.5],
                hatches=["", "////"], colors="none",
                transform=ccrs.PlateCarree(),
            )

    # Shared colorbar
    cax = fig.add_subplot(gs[-1, :])

    #fig.add_axes([0.12, 0.06, 0.76, 0.03]) # type: ignore
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) #type: ignore
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1], shrink=0.7)
    cbar.set_label("Normalized CanESM − ERA5 anomaly (unitless)")

    # Hatch legend
    hatch_proxy = Patch(facecolor="none", edgecolor="0.25", hatch="////",
                        label=fr"|anom| ≤ {hatch_k}×σ (ensemble spread)")
    fig.legend(handles=[hatch_proxy], bbox_to_anchor=(0.90, -0), frameon=False)

    fig.savefig(out_png + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)





# --------------------------------- Pipeline ----------------------------------

def main(zg_coord: int, hatch_k: float) -> List[str]:
    """Run the full read → compute → plot pipeline."""
    client = Client(processes=True, n_workers=8, threads_per_worker=1, memory_limit="2.5GB")
    # 1) Open CanESM and ERA5 for all variables
    ds_canesm: Dict[str, xr.DataArray] = data_io.open_canesm_lwa(ENSEMBLE_LIST, zg_coord)
    ds_era5: Dict[str, xr.DataArray] = data_io.open_era5_lwa(zg_coord)

    # Normalize time coordinates to daily boundaries
    ds_canesm = {k: preprocess.floor_daily_time(v) for k, v in ds_canesm.items()}
    ds_era5 = {k: preprocess.floor_daily_time(v) for k, v in ds_era5.items()}

    # 2) Build seasonal climatology datasets
    canesm_stats_lazy: Dict[str, xr.Dataset] = {v: seasonal_clim_and_spread_canesm(ds_canesm[v]) for v in VAR_NAMES}
    era5_stats_lazy: Dict[str, xr.Dataset]   = {v: seasonal_clim_era5(ds_era5[v]) for v in VAR_NAMES}

    # 2.1) Persist to avoid oversized graphs later
    canesm_stats = {v: canesm_stats_lazy[v].persist() for v in VAR_NAMES}
    era5_stats   = {v: era5_stats_lazy[v].persist() for v in VAR_NAMES}

    wait([da.data for ds in canesm_stats.values() for da in ds.data_vars.values()] +
         [da.data for ds in era5_stats.values()   for da in ds.data_vars.values()])

    # 3) (Optional) daily quantiles — kept for parity with your script
    # if make_quantiles:
    #     _canesm_q = {v: seasonal_daily_quantiles(ds_canesm[v], q_list, is_canesm=True) for v in VAR_NAMES}
    #     _era5_q   = {v: seasonal_daily_quantiles(ds_era5[v],   q_list, is_canesm=False) for v in VAR_NAMES}
    #     _canesm_q = {v: _canesm_q[v].persist() for v in VAR_NAMES}
    #     _era5_q   = {v: _era5_q[v].persist()   for v in VAR_NAMES}

    # 4) Plot baseline grids
    out_png = os.path.join(
        OUTPUT_PLOTS_PATH,
        f"CanESM5_seasonal_clim_LWA_a_c.zg{zg_coord}",
    )
    plot_baseline_grid(canesm_stats, "CanESM5", out_png)

    out_png = os.path.join(
        OUTPUT_PLOTS_PATH,
        f"ERA5_seasonal_clim_LWA_a_c.zg{zg_coord}",
    )
    plot_baseline_grid(era5_stats, "ERA5", out_png)

    # 5) Plot anomalies grid
    outpaths: List[str] = []
    out_png = os.path.join(
        OUTPUT_PLOTS_PATH,
        f"CanESM5_minus_ERA5_seasonal_clim_anom_hatched_LWA_a_c.zg{zg_coord}",
    )
    plot_anomaly_grid(canesm_stats, era5_stats, out_png, hatch_k=hatch_k)
    outpaths.append(out_png + ".png")

    # if make_quantiles:
    #     out_png = os.path.join(
    #         OUTPUT_PLOTS_PATH,
    #         f"CanESM5_minus_ERA5_seasonal_quantile_anom_hatched_LWA_a_c.zg{zg_coord}",
    #     )
    #     plot_anomaly_quantiles_grid(_canesm_q, _era5_q, out_png, q_list=q_list, hatch_k=hatch_k)
    #     for q in q_list:
    #         outpaths.append(out_png + f".q{q:0.2f}.png")

    # Close lazy datasets
    for ds in list(ds_canesm.values()) + list(ds_era5.values()):
        ds.close()

    # if make_quantiles:
    #     wait([da.data for ds in _canesm_q.values() for da in ds.data_vars.values()] +
    #          [da.data for ds in _era5_q.values()   for da in ds.data_vars.values()])
    client.close()
    return outpaths





if __name__ == "__main__":
    args = arg_parser()
    outpaths = main(
        zg_coord=args.zg,
        hatch_k=args.hatch_k
    )
    if outpaths:
        print("Saved:")
        for p in outpaths:
            print(p)



# --------------------------------- End of file --------------------------------
