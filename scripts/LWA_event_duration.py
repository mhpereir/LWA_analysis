import os
import sys
import argparse

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

# Define script specific constants

OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/LWA_event_duration")
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)


# Define argument parser

def arg_parser():
    parser = argparse.ArgumentParser(
        description="LWA vs deltaT correlation analysis and plotting."
    )
    parser.add_argument(
        "--lwa_var",
        type=str,
        choices=config.LWA_VARS,
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
        choices=list(config.REGIONS.keys()),
        default="west_south",
        help="Region to analyze.",
    )
    parser.add_argument(
        "--season",
        type=str,
        choices=config.SEASON_NAMES,
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


# ----------------------------- Helpers -----------------------------

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



# ----------------------------- Plotting Helpers -----------------------------


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


# ----------------------------- Plotting -----------------------------


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

# ----------------------------- Main -----------------------------


def run_analysis(REGION, LWA_var, SEASON, ZG_COORD, Q_LWA, EVENT_DURATION_THRESHOLD: int):

    #load in data
 
    ds_lwa_thresh_canesm        = data_io.open_canesm_lwa_thresh(ENSEMBLE_LIST, Q_LWA, REGION, ZG_COORD)
    ds_lwa_thresh_canesm_smooth = ds_lwa_thresh_canesm.rolling(dayofyear=7, center=True).mean()
    
    ds_lwa_thresh_era5        = data_io.open_era5_lwa_thresh(Q_LWA, REGION, ZG_COORD)
    ds_lwa_thresh_era5_smooth = ds_lwa_thresh_era5.rolling(dayofyear=7, center=True).mean()

    ds_tas_canesm = data_io.open_canesm_temperature(TEMP_VAR, ENSEMBLE_LIST)
    ds_tas_canesm = ds_tas_canesm.chunk({"time": 365})
    ds_tas_canesm = preprocess.compute_region_mean(ds_tas_canesm, REGION).compute()
    
    ds_tas_canesm_anom = preprocess.dayofyear_anomaly(ds_tas_canesm)


    ds_tas_era5 = data_io.open_era5_temperature(TEMP_VAR)
    ds_tas_era5 = ds_tas_era5.chunk({"time": 365})
    ds_tas_era5 = preprocess.compute_region_mean(ds_tas_era5, REGION).compute()
    
    ds_tas_era5_anom = preprocess.dayofyear_anomaly(ds_tas_era5)


    ## loading LWA data
    # 1) Open CanESM and ERA5 for all LWA VARIABLES
    ds_canesm_lwas: Dict[str, xr.DataArray] = data_io.open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas: Dict[str, xr.DataArray]   = data_io.open_era5_lwa(ZG_COORD)

    ds_canesm_lwa = ds_canesm_lwas[LWA_var]#.sel(MEMBER=MEMBER)
    ds_era5_lwa   = ds_era5_lwas[LWA_var]

    ds_canesm_lwa_reg = preprocess.compute_region_mean(ds_canesm_lwa, REGION).chunk({"time": 365}).compute()
    ds_era5_lwa_reg   = preprocess.compute_region_mean(ds_era5_lwa, REGION).chunk({"time": 365}).compute()

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

    ENSEMBLE_LIST = config.ENSEMBLE_LIST
    TEMP_VAR      = config.TEMP_VAR

    EVENT_DURATION_THRESHOLD = args.event_duration_threshold

    run_analysis(REGION, LWA_var, SEASON, ZG_LEVEL, Q_LWA, EVENT_DURATION_THRESHOLD)