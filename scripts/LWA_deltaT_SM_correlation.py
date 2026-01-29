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
import statsmodels.api as sm #type:ignore
import scipy.stats as stats

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib import ticker as mticker

# Define script specific constants

OUTPUT_PLOTS_PATH = os.path.join(config.OUTPUT_PATH, "plots/LWA_deltaT_correlation")
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
    return parser.parse_args()



# ------------------------------ Helper Functions ----------------------------------

def quantile_fit(x, y, quantiles: List[float]) -> Dict[float, Any]:
    X = sm.add_constant(x)
    fits = {}
    for q in quantiles:
        model = sm.QuantReg(y, X)
        res = model.fit(q=q)
        fits[q] = res
    return fits


def _stack_clean(a: xr.DataArray, b: xr.DataArray, has_member: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper: flatten (member,time) or (time,) into 1D arrays and drop NaNs.
    """
    if has_member:
        a = a.stack(points=("member", "time"))
        b = b.stack(points=("member", "time"))
    else:
        pass

    xa = np.asarray(a).ravel()
    ya = np.asarray(b).ravel()

    good = np.isfinite(xa) & np.isfinite(ya)
    return xa[good], ya[good]


def _stack_clean_3(a: xr.DataArray, b: xr.DataArray, c: xr.DataArray, has_member: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper: flatten (member,time) or (time,) into 1D arrays and drop NaNs.
    """
    if has_member:
        a = a.stack(points=("member", "time"))
        b = b.stack(points=("member", "time"))
        c = c.stack(points=("member", "time"))
    else:
        pass

    xa = np.asarray(a).ravel()
    ya = np.asarray(b).ravel()
    ca = np.asarray(c).ravel()

    good = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(ca)
    return xa[good], ya[good], ca[good]

def _normalize(pdf, dx):
        area = np.trapezoid(pdf, dx=dx)
        return pdf / area if area > 0 else pdf, area


def _safe_gaussian_kde(x, y, bw='scott'):
    """
    Return a callable KDE(xgrid,ygrid) that is robust to singular covariance.
    On failure, returns a zero-density KDE.
    """
    pts = np.vstack([x, y])

    kde = stats.gaussian_kde(pts, bw_method=bw)

    bw  = kde.factor

    def eval_on(xx, yy):
        return kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    return eval_on, bw

def _relative_error_pct(value: float, error: float) -> float:
    """Return |error/value| in percent, guarding against division by zero."""
    if not np.isfinite(value) or value == 0 or not np.isfinite(error):
        return np.nan
    return abs(error / value) * 100.0


def _format_quantile_label(q: float, res: Any, prefix: str = "") -> str:
    """Format legend text with slope/intercept and relative errors."""
    intercept = float(res.params[0])
    slope = float(res.params[1])
    intercept_rel = _relative_error_pct(intercept, float(res.bse[0]))
    slope_rel = _relative_error_pct(slope, float(res.bse[1]))

    def _fmt(val: float) -> str:
        return f"{val:.1f}" if np.isfinite(val) else "nan"

    prefix_txt = f"{prefix} " if prefix else ""
    return (
        f"{prefix_txt}Q{int(q*100)} slope={slope:.2e} +/-{_fmt(slope_rel)}%, "
        f"b0={intercept:.2f} +/-{_fmt(intercept_rel)}%"
    )


# ------------------------------ Plotting ----------------------------------




def plot_joint_kde_with_marginals(
    x_in: xr.DataArray,
    y_in: xr.DataArray,
    
    title: str = "",
    sim: str = "",
    x_label: str = f"LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    point_alpha: float = 0.4,
    max_scatter: int = 5000,
    gridsize: int = 200,
    cmap: str = "afmhot_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) with marginal PDFs and a best-fit regression line.

    Layout:
        - main panel: 2D KDE background (+ scatter sample, + linear fit)
        - top panel:  1D PDF of x
        - right panel:1D PDF of y
    Returns slope, intercept, r-value for logging.

    Parameters
    ----------
    x_in, y_in : np.ndarray
        1D arrays of equal length. These should already be anomalies
        or whatever you want to compare.
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    if "member" in x_in.dims:
        x, y = _stack_clean(x_in, y_in, has_member=True)
    else:
        x, y = _stack_clean(x_in, y_in, has_member=False)

    if x.size < 3 or y.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    # ---------------------------------------------------------------------
    # 2. Fit linear regression using SciPy
    # ---------------------------------------------------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
 
    # x-range for plotting the regression line
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = slope * x_line + intercept

    # ---------------------------------------------------------------------
    # 2.1 Fit quantile regression lines (50,90,95%)
    lw = [2., 2. ,2.]
    if   SEASON == "JJA":
        quantile_list = [0.5, 0.95]
        q_color_list  = ["cyan", "magenta", "red"]
    elif SEASON == "DJF":
        quantile_list = [0.05, 0.5]
        q_color_list  = ["blue", "cyan"]
    else:
        quantile_list = [0.05, 0.5, 0.95]
        q_color_list  = ["blue", "cyan", "magenta"]
    quantile_fits = quantile_fit(x, y, quantiles=quantile_list)

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d = stats.gaussian_kde(np.vstack([x, y]))
    # grid
    xi = np.linspace(0, x_limits[1]*1.25, gridsize)
    yi = np.linspace(y_limits[0]*1.25, y_limits[1]*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # 1D KDEs for marginals ( PDFs )
    def kde_1d(arr: np.ndarray, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        k = stats.gaussian_kde(arr)
        # xs = np.linspace(np.min(arr), np.max(arr), npts)
        pdf = k(xs)

        # normalize pdf so area ~ 1
        dx = xs[1] - xs[0]
        area = np.trapezoid(pdf, dx=dx)
        if area > 0:
            pdf = pdf / area
        return xs, pdf

    xs_pdf, x_pdf = kde_1d(x, xs=np.linspace(x_limits[0], x_limits[1], gridsize))
    ys_pdf, y_pdf = kde_1d(y, xs=np.linspace(y_limits[0], y_limits[1], gridsize))

    # ---------------------------------------------------------------------
    # 4. Layout: Joint plot + marginals
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[6, 1, 0.25],
        height_ratios=[1, 6],
        wspace=0.0,
        hspace=0.0,
        figure=fig,
    )

    ax_top    = fig.add_subplot(gs[0, 0])
    ax_main   = fig.add_subplot(gs[1, 0])
    ax_right  = fig.add_subplot(gs[1, 1])
    ax_cbar   = fig.add_subplot(gs[1, 2])

    # --------------------
    # main panel (joint)
    # --------------------
    # KDE as background
    pcm = ax_main.pcolormesh(
        Xg, Yg, Z,
        shading="auto",
        cmap=cmap,
    )

    # light scatter sample: subselect if huge
    if x.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
    else:
        xs_sc = x
        ys_sc = y

    ax_main.scatter(
        xs_sc,
        ys_sc,
        s=6,
        c="gray",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    # # regression line
    # ax_main.plot(
    #     x_line,
    #     y_line,
    #     color="cyan",
    #     linewidth=2,
    #     label=f"fit: y = {slope:.2e} x + {intercept:.2f}\n" #type:ignore
    #           f"r = {r_value:.2f}",
    # )

    # quantile regression lines
    xs = np.linspace(x_limits[0], x_limits[1], 200)
    X_eval = sm.add_constant(xs)
    
    for ii,q in enumerate(quantile_list):
        yq_line = quantile_fits[q].predict(X_eval)
        ax_main.plot(
            xs,
            yq_line,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle="--",
            label=_format_quantile_label(q, quantile_fits[q]),
            zorder=30
        )

    # ax_main.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_main.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_main.set_ylabel(y_label + " (K)")
    ax_main.set_xlim(x_limits if x_limits else (np.min(Xg), np.max(Xg)))
    ax_main.set_ylim(y_limits if y_limits else (np.min(Yg), np.max(Yg)))
    ax_main.legend(loc="lower right", frameon=True, fontsize=9)

    # colorbar for KDE
    #cax = fig.add_axes([0.12, 0.04, 0.65, 0.015])  # [left, bottom, width, height] in fig coords #type: ignore
    cb = fig.colorbar(pcm, cax=ax_cbar, orientation="vertical")
    cb.set_label("Joint density (KDE)")

    # ax_cbar.xaxis.set_ticks_position("bottom")
    ax_cbar.xaxis.set_ticks([])

    # --------------------
    # top marginal (x PDF)
    # --------------------
    ax_top.plot(xs_pdf, x_pdf, color="black", lw=1.5)
    ax_top.fill_between(xs_pdf, 0, x_pdf, color="0.7", alpha=0.4)
    ax_top.set_ylabel("log(P(LWA))")
    #set y-axis to log scale
    ax_top.set_yscale("log")
    ax_top.set_ylim((1e-6, 1e-3))
    ax_top.set_xticks([])
    ax_top.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
        pad=0,
    )
    # sync x-lims with main panel's x
    ax_top.set_xlim(ax_main.get_xlim())
    # make top spine pretty
    ax_top.spines["right"].set_visible(True)
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["left"].set_visible(True)
    ax_top.spines["bottom"].set_visible(False)

    # --------------------
    # right marginal (y PDF)
    # --------------------
    ax_right.plot(y_pdf, ys_pdf, color="black", lw=1.5)
    ax_right.fill_betweenx(ys_pdf, 0, y_pdf, color="0.7", alpha=0.4)
    ax_right.set_xlabel("log(P(T))")
    #set x-axis to log scale
    ax_right.set_xscale("log")
    ax_right.set_yticks([])
    ax_right.set_xlim((1e-4, 1))
    ax_right.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        pad=0,
    )
    # sync y-lims with main panel's y
    ax_right.set_ylim(ax_main.get_ylim())
    # cosmetic spines
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(True)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(True)

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    # tighten layout a bit (but keep our manual colorbar placement)
    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    fig_name = f"{output_path}/{sim}_{LWA_var}_deltaT_kde_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None







def plot_joint_kde_with_marginals_mrsos(
    x_in: xr.DataArray,
    y_in: xr.DataArray,
    z_in: xr.DataArray,
    title: str = "",
    sim: str = "",
    x_label: str = f"LWA",
    y_label: str = r"$\Delta T$",
    z_label: str = r"$\Delta$MRSOS (kg/m2)",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    z_limits: Tuple[float, float] = None, #type:ignore
    point_alpha: float = 0.8,
    max_scatter: int = 5000,
    gridsize: int = 200,
    cmap: str = "afmhot_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) with marginal PDFs and a best-fit regression line.

    Layout:
        - main panel: 2D KDE background (+ scatter sample, + linear fit)
        - top panel:  1D PDF of x
        - right panel:1D PDF of y
    Returns slope, intercept, r-value for logging.

    Parameters
    ----------
    x_in, y_in : np.ndarray
        1D arrays of equal length. These should already be anomalies
        or whatever you want to compare.
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    if "member" in x_in.dims:
        x, y, z = _stack_clean_3(x_in, y_in, z_in, has_member=True)
    else:
        x, y, z = _stack_clean_3(x_in, y_in, z_in, has_member=False)

    if x.size < 3 or y.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    # ---------------------------------------------------------------------
    # 2. Fit linear regression using SciPy
    # ---------------------------------------------------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
 
    # x-range for plotting the regression line
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = slope * x_line + intercept

    # ---------------------------------------------------------------------
    # 2.1 Fit quantile regression lines (50,90,95%)
    lw = [2., 2. ,2.]
    if   SEASON == "JJA":
        quantile_list = [0.5, 0.95]
        q_color_list  = ["cyan", "magenta", "red"]
    elif SEASON == "DJF":
        quantile_list = [0.05, 0.5]
        q_color_list  = ["blue", "cyan"]
    else:
        quantile_list = [0.05, 0.5, 0.95]
        q_color_list  = ["blue", "cyan", "magenta"]
    quantile_fits = quantile_fit(x, y, quantiles=quantile_list)

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d = stats.gaussian_kde(np.vstack([x, y]))
    # grid
    xi = np.linspace(0, x_limits[1]*1.25, gridsize)
    yi = np.linspace(y_limits[0]*1.25, y_limits[1]*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # 1D KDEs for marginals ( PDFs )
    def kde_1d(arr: np.ndarray, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        k = stats.gaussian_kde(arr)
        # xs = np.linspace(np.min(arr), np.max(arr), npts)
        pdf = k(xs)

        # normalize pdf so area ~ 1
        dx = xs[1] - xs[0]
        area = np.trapezoid(pdf, dx=dx)
        if area > 0:
            pdf = pdf / area
        return xs, pdf

    xs_pdf, x_pdf = kde_1d(x, xs=np.linspace(x_limits[0], x_limits[1], gridsize))
    ys_pdf, y_pdf = kde_1d(y, xs=np.linspace(y_limits[0], y_limits[1], gridsize))

    # ---------------------------------------------------------------------
    # 4. Layout: Joint plot + marginals
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[6, 1, 0.25],
        height_ratios=[1, 6],
        wspace=0.0,
        hspace=0.0,
        figure=fig,
    )

    ax_top    = fig.add_subplot(gs[0, 0])
    ax_main   = fig.add_subplot(gs[1, 0])
    ax_right  = fig.add_subplot(gs[1, 1])
    ax_cbar   = fig.add_subplot(gs[1, 2])

    # --------------------
    # main panel (joint)
    # --------------------
    # KDE as background
    pcm = ax_main.pcolormesh(
        Xg, Yg, Z,
        shading="auto",
        cmap=cmap,
    )

    # light scatter sample: subselect if huge
    if x.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
        cs_sc = z[idx]

        scatter = ax_main.scatter(
            xs_sc,
            ys_sc,
            c=cs_sc,
            s=10,
            cmap="RdBu",
            edgecolors="k",
            linewidths=0.1,
            alpha=point_alpha,
            vmin=z_limits[0],
            vmax=z_limits[1],
        )

    else:
        xs_sc = x
        ys_sc = y
        cs_sc = z

        scatter = ax_main.scatter(
            xs_sc,
            ys_sc,
            c=cs_sc,
            s=10,
            cmap="RdBu",
            edgecolors="k",
            linewidths=0.1,
            alpha=point_alpha,
            vmin=z_limits[0],
            vmax=z_limits[1],
        )

    
    cbar_scatter = fig.colorbar(scatter, ax=ax_main, pad=0.02)
    cbar_scatter.set_label(f"{z_in.name if hasattr(z_in, 'name') else 'MRSOS'} (kg/m²)")

    # # regression line
    # ax_main.plot(
    #     x_line,
    #     y_line,
    #     color="cyan",
    #     linewidth=2,
    #     label=f"fit: y = {slope:.2e} x + {intercept:.2f}\n" #type:ignore
    #           f"r = {r_value:.2f}",
    # )

    # quantile regression lines
    xs = np.linspace(x_limits[0], x_limits[1], 200)
    X_eval = sm.add_constant(xs)
    
    for ii,q in enumerate(quantile_list):
        yq_line = quantile_fits[q].predict(X_eval)
        ax_main.plot(
            xs,
            yq_line,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle="--",
            label=_format_quantile_label(q, quantile_fits[q]),
            zorder=30
        )

    # ax_main.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_main.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_main.set_ylabel(y_label + " (K)")
    ax_main.set_xlim(x_limits if x_limits else (np.min(Xg), np.max(Xg)))
    ax_main.set_ylim(y_limits if y_limits else (np.min(Yg), np.max(Yg)))
    ax_main.legend(loc="lower right", frameon=True, fontsize=9)

    # colorbar for KDE
    #cax = fig.add_axes([0.12, 0.04, 0.65, 0.015])  # [left, bottom, width, height] in fig coords #type: ignore
    cb = fig.colorbar(pcm, cax=ax_cbar, orientation="vertical")
    cb.set_label("Joint density (KDE)")

    # ax_cbar.xaxis.set_ticks_position("bottom")
    ax_cbar.xaxis.set_ticks([])

    # --------------------
    # top marginal (x PDF)
    # --------------------
    ax_top.plot(xs_pdf, x_pdf, color="black", lw=1.5)
    ax_top.fill_between(xs_pdf, 0, x_pdf, color="0.7", alpha=0.4)
    ax_top.set_ylabel("log(P(LWA))")
    #set y-axis to log scale
    ax_top.set_yscale("log")
    ax_top.set_ylim((1e-6, 1e-3))
    ax_top.set_xticks([])
    ax_top.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
        pad=0,
    )
    # sync x-lims with main panel's x
    ax_top.set_xlim(ax_main.get_xlim())
    # make top spine pretty
    ax_top.spines["right"].set_visible(True)
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["left"].set_visible(True)
    ax_top.spines["bottom"].set_visible(False)

    # --------------------
    # right marginal (y PDF)
    # --------------------
    ax_right.plot(y_pdf, ys_pdf, color="black", lw=1.5)
    ax_right.fill_betweenx(ys_pdf, 0, y_pdf, color="0.7", alpha=0.4)
    ax_right.set_xlabel("log(P(T))")
    #set x-axis to log scale
    ax_right.set_xscale("log")
    ax_right.set_yticks([])
    ax_right.set_xlim((1e-4, 1))
    ax_right.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        pad=0,
    )
    # sync y-lims with main panel's y
    ax_right.set_ylim(ax_main.get_ylim())
    # cosmetic spines
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(True)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(True)

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    # tighten layout a bit (but keep our manual colorbar placement)
    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    fig_name = f"{output_path}/{sim}_{LWA_var}_deltaT_kde_MRSOS_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None





def plot_kde_comparison_model_vs_ref(
    x_mod,  # xr.DataArray, dims ('member','time') or ('time',) #model, CanESM
    y_mod,  # xr.DataArray, dims ('member','time') or ('time',)
    x_ref,  # xr.DataArray, dims ('time',) #reference, ERA5
    y_ref,  # xr.DataArray, dims ('time',)
    title: str = "",
    x_label: str = "LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    member_dim: str = "member",
    gridsize: int = 200,
    max_scatter: int = 4000,
    cmap: str = "RdBu",
    hatch_alpha: float = 0.15,
    output_path: str = "./",
) -> None:
    """
    Compare model joint density vs reference joint density in (x,y) space.

    Main panel:
      - Filled colormap: model *ensemble-mean* KDE
      - Contour lines:   reference (e.g. ERA5) KDE
      - Scatter:         reference points (optional thinning)
      - Regression lines: model-mean fit and ref fit
      - Hatching: where ref KDE lies inside the (p5,p95) band of
                  per-member KDE densities from the model.

    Inputs
    ------
    x_mod, y_mod : xr.DataArray
        Model quantities. Must share dims. If they have `member_dim`, we
        treat them as ensemble; otherwise treated as single-field model.
    x_ref, y_ref : xr.DataArray
        Reference quantities (e.g. ERA5). No member dim.
    member_dim : str
        Name of ensemble dimension in x_mod/y_mod. Ignored if absent.
    gridsize : int
        Resolution of KDE grid in both x and y directions.
    hatch_alpha : float
        Alpha for the hatched overlay mask.

    Output
    ------
    Saves a figure to `output_path`. Returns None.
    """

    if x_label == 'LWA_a' or x_label == 'LWA':
        leg_loc="upper left"
    elif x_label == 'LWA_c':
        leg_loc="lower left"
    else:
        Warning("x_label not recognized, setting legend location to 'best'")
        leg_loc="best"


    # ------------------------------------------------------------------
    # 1. Flatten data
    # ------------------------------------------------------------------
    has_member = (member_dim in x_mod.dims) and (member_dim in y_mod.dims)

    # model ALL points (for xrange/yrange)
    xm_all, ym_all = _stack_clean(x_mod, y_mod, has_member=has_member)
    # reference ALL points
    xr_all, yr_all = _stack_clean(x_ref, y_ref, has_member=False)

    if xm_all.size < 5 or xr_all.size < 5:
        raise ValueError("Not enough points to build KDEs.")

    # ------------------------------------------------------------------
    # 2. Build grid in (x,y) space
    #    Use combined 1st-99th percentiles of BOTH model+ref so both fit
    # ------------------------------------------------------------------
    x_lo = np.min(np.concatenate([xm_all, xr_all]))
    x_hi = np.max(np.concatenate([xm_all, xr_all]))
    y_lo = np.min(np.concatenate([ym_all, yr_all]))
    y_hi = np.max(np.concatenate([ym_all, yr_all]))

    xi = np.linspace(x_lo, x_hi, gridsize)
    yi = np.linspace(y_lo, y_hi, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)

    # ---------------------------------------------------------------------
    # 2.1 Fit quantile regression lines (50,90,95%)
    lw = [2., 2. ,2.]
    if   SEASON == "JJA":
        quantile_list = [0.5, 0.95]
        q_color_list  = ["cyan", "magenta", "red"]
    elif SEASON == "DJF":
        quantile_list = [0.05, 0.5]
        q_color_list  = ["blue", "cyan"]
    else:
        quantile_list = [0.05, 0.5, 0.95]
        q_color_list  = ["blue", "cyan", "magenta"]

    quantile_fits_ref = quantile_fit(xr_all, yr_all, quantiles=quantile_list)
    quantile_fits_mod = quantile_fit(xm_all, ym_all, quantiles=quantile_list)

    # ------------------------------------------------------------------
    # 3. KDEs:
    #    - ref KDE (single)
    #    - model member KDEs, then mean, then p05/p95
    # ------------------------------------------------------------------

    # reference KDE
    ref_eval, bw_ref = _safe_gaussian_kde(xr_all, yr_all)
    Z_ref = ref_eval(Xg, Yg)  # (gridsize, gridsize)

    Z_mean = None
    Z_p05  = None
    Z_p95  = None

    # model KDEs
    if has_member:
        Z_members = []
        for m in x_mod[member_dim].values:
            xm_m, ym_m = _stack_clean(x_mod.sel({member_dim: m}),
                                      y_mod.sel({member_dim: m}),
                                      has_member=False)
            if xm_m.size < 5:
                continue
            eval_m, _ = _safe_gaussian_kde(xm_m, ym_m, bw=bw_ref) #type:ignore
            Z_m = eval_m(Xg, Yg)
            Z_members.append(Z_m)

        if len(Z_members) == 0:
            raise ValueError("No valid member KDEs could be estimated.")

        Z_members = np.stack(Z_members, axis=0)  # (nmem, ny, nx)

        # ensemble mean density
        Z_mean = np.nanmean(Z_members, axis=0)

        # ensemble spread band (5th, 95th)
        Z_p05 = np.nanpercentile(Z_members, 5, axis=0)
        Z_p95 = np.nanpercentile(Z_members, 95, axis=0)
    else:
        SystemExit(f"Model data has no member dimension of name {member_dim}; ensemble spread hatching not possible.")
        # # no member dim: treat "model" as single field
        # mod_eval = _safe_gaussian_kde(xm_all, ym_all)
        # Z_mean = mod_eval(Xg, Yg)
        # Z_p05 = Z_mean.copy()
        # Z_p95 = Z_mean.copy()

    # ------------------------------------------------------------------
    # 4. Regression fits (ref vs model-mean)
    # ------------------------------------------------------------------
    def linfit(xarr, yarr):
        slope, intercept, r, p, se = stats.linregress(xarr, yarr)
        xline = np.linspace(x_lo, x_hi, 200)
        yline = slope * xline + intercept #type:ignore
        return slope, intercept, r, xline, yline

    slope_ref, intercept_ref, r_ref, xline_ref, yline_ref = linfit(xr_all, yr_all)
    slope_mod, intercept_mod, r_mod, xline_mod, yline_mod = linfit(xm_all, ym_all)

    # ------------------------------------------------------------------
    # 5. Hatching mask:
    #    hatch where ref KDE is within [p05, p95] of model members
    # ------------------------------------------------------------------
    within_band = (Z_ref >= Z_p05) & (Z_ref <= Z_p95)

    # ------------------------------------------------------------------
    # 6. Plot layout
    #    We'll imitate your current setup: main joint panel,
    #    plus top/right marginals using the reference data only
    #    (since that's your observed distribution anchor).
    #    Colorbar sits next to main panel.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[6, 1, 0.25],
        height_ratios=[1, 6],
        wspace=0.0,
        hspace=0.0,
        figure=fig,
    )

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_cbar  = fig.add_subplot(gs[1, 2])

    # -------------------------
    # 6a. MAIN PANEL
    # -------------------------

    # If you really want anomaly instead of model density:
    Z_plot = Z_mean - Z_ref
    vmax = np.nanmax(np.abs(Z_plot))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin,
                        vcenter=0.0,
                        vmax=vmax)
    pcm = ax_main.pcolormesh(Xg, Yg, Z_plot,
                             shading="auto",
                             cmap="RdBu_r",
                             norm=norm)

    # Otherwise: shaded model-mean density
    # pcm = ax_main.pcolormesh(
    #     Xg, Yg, Z_mean, #type: ignore
    #     shading="auto",
    #     cmap=cmap,
    # )

    # Hatching: where ERA5 KDE is within [p05,p95] of ensemble KDEs
    # We'll overlay a transparent contourf hatch layer.
    ax_main.contourf(
        Xg, Yg, within_band.astype(int),
        levels=[-0.5, 0.5, 1.5],
        hatches=["", ".."],
        colors="none",
        linewidth=0.1,
        alpha=hatch_alpha,
        linewidths=0.0,
    )

    # Ref KDE contours
    # Use a few log-spaced-ish levels so we see structure
    # (add small epsilon to avoid log(0)).
    eps = np.nanmax(Z_ref) * 1e-3
    levels = np.linspace(np.nanmax(Z_ref)*0.1, np.nanmax(Z_ref), 5)
    cs = ax_main.contour(
        Xg, Yg, Z_ref,
        levels=levels,
        colors="white",
        linewidths=1,
        alpha=0.9,
    )
    ax_main.clabel(cs, inline=True, fmt="%.1e", fontsize=8)

    # Scatter reference points (thin if huge)
    if xr_all.size > max_scatter:
        rng = np.random.default_rng(123)
        idx = rng.choice(xr_all.size, size=max_scatter, replace=False)
        xs_sc = xr_all[idx]
        ys_sc = yr_all[idx]
    else:
        xs_sc = xr_all
        ys_sc = yr_all

    ax_main.scatter(
        xs_sc, ys_sc,
        s=6,
        c="gray",
        edgecolors="none",
        linewidths=0.2,
        alpha=0.4,
        zorder=10,
    )

    # Regression lines
    # ax_main.plot(
    #     xline_ref, yline_ref,
    #     color="cyan",
    #     lw=2,
    #     label=f"ERA5: y={slope_ref:.2e}x+{intercept_ref:.2f}, r={r_ref:.2f}",
    #     zorder=11,
    # )
    # ax_main.plot(
    #     xline_mod, yline_mod,
    #     color="yellow",
    #     lw=2,
    #     label=f"CanESM: y={slope_mod:.2e}x+{intercept_mod:.2f}, r={r_mod:.2f}",
    #     zorder=11,
    # )

    # quantile regression lines
    xs = np.linspace(x_lo, x_hi, 200)
    X_eval = sm.add_constant(xs)
    
    for ii,q in enumerate(quantile_list):
        yq_line_ref = quantile_fits_ref[q].predict(X_eval)
        yq_line_mod = quantile_fits_mod[q].predict(X_eval)
        ax_main.plot(
            xs,
            yq_line_ref,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle="--",
            label=_format_quantile_label(q, quantile_fits_ref[q], prefix="ERA5"),
            zorder=30
        )

        ax_main.plot(
            xs,
            yq_line_mod,
            color=q_color_list[ii],
            linewidth=lw[ii],
            linestyle=":",
            label=_format_quantile_label(q, quantile_fits_mod[q], prefix="CanESM"),
            zorder=30
        )


    # axes labels / limits
    # ax_main.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_main.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_main.set_ylabel(y_label + ' [K]')
    ax_main.set_xlim(x_limits if x_limits else (x_lo, x_hi))
    ax_main.set_ylim(y_limits if y_limits else (y_lo, y_hi))

    ax_main.legend(
        loc=leg_loc,
        frameon=True,
        fontsize=8,
    )

    # -------------------------
    # 6b. TOP marginal (PDF of x_ref and x_mod-mean, optional)
    # -------------------------
    # We'll show reference in black/gray fill, and model-mean in a line.

    # --- reference KDE ---
    kx_ref = stats.gaussian_kde(xr_all, bw_method='scott')
    xs = np.linspace(x_lo, x_hi, 200)
    pdf_x_ref = kx_ref(xs)

    kx_ref_factor = kx_ref.factor

    # --- model: per-member KDEs ---
    pdfs_members = []
    for m in x_mod["member"].values:              # assumes x_mod has member dim
        xm = np.asarray(x_mod.sel(member=m)).ravel()
        xm = xm[np.isfinite(xm)]
        
        kde_m = stats.gaussian_kde(xm, bw_method=kx_ref_factor)
        pdf_m = kde_m(xs)
        pdfs_members.append(pdf_m)

    pdfs_members = np.stack(pdfs_members, axis=0)   # shape (nmem, nx)

    # ensemble mean + 5–95% range
    pdf_x_mod_mean = np.nanmean(pdfs_members, axis=0)
    pdf_x_mod_p05  = np.nanpercentile(pdfs_members, 5, axis=0)
    pdf_x_mod_p95  = np.nanpercentile(pdfs_members, 95, axis=0)

    # --- normalize all PDFs to integrate ≈1 ---
    dx = xs[1] - xs[0]

    pdf_x_ref, area_x_ref           = _normalize(pdf_x_ref, dx)
    pdf_x_mod_mean, area_x_mod_mean = _normalize(pdf_x_mod_mean, dx)
    pdf_x_mod_p05                   = pdf_x_mod_p05 / area_x_mod_mean
    pdf_x_mod_p95                   = pdf_x_mod_p95 / area_x_mod_mean

    # --- plotting ---
    ax_top.fill_between(xs, 0, pdf_x_ref, color="0.7", alpha=0.4)
    ax_top.plot(xs, pdf_x_ref, color="black", lw=1.5, label="ERA5")

    # ensemble-spread shading
    ax_top.fill_between(
        xs, pdf_x_mod_p05, pdf_x_mod_p95,
        color="gold", alpha=0.25, label="CanESM 5–95%"
    )

    # mean line
    ax_top.plot(xs, pdf_x_mod_mean, color="gold", lw=1.8, label="CanESM mean")

    # cosmetics
    ax_top.set_xlim(x_lo, x_hi)
    ax_top.set_xticks([])
    ax_top.set_ylabel("log(P(LWA))")
    #make ax_top y-axis log scale
    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-6, 1e-3)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False, pad=0)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.legend(frameon=False, fontsize=8)


    # -------------------------
    # 6c. RIGHT marginal (PDF of y_ref vs y_mod)
    # -------------------------
    # --- reference KDE (1D, y) ---
    ky_ref = stats.gaussian_kde(yr_all, bw_method='scott')
    ys = np.linspace(y_lo, y_hi, 200)
    pdf_y_ref = ky_ref(ys)
    ky_ref_factor = ky_ref.factor

    # --- model: per-member KDEs for y ---
    pdfs_members_y = []
    for m in y_mod["member"].values:                 # assumes y_mod has a 'member' dim
        ym = np.asarray(y_mod.sel(member=m)).ravel()
        ym = ym[np.isfinite(ym)]
        
        kde_m = stats.gaussian_kde(ym, bw_method=ky_ref_factor)
        pdf_m = kde_m(ys)

        pdfs_members_y.append(pdf_m)

    if len(pdfs_members_y) == 0:
        # Fallback: no valid members → use zeros so plotting still works
        pdfs_members_y = [np.zeros_like(ys)]

    pdfs_members_y = np.stack(pdfs_members_y, axis=0)  # (nmem, ny)

    # ensemble mean + 5–95% range
    pdf_y_mod_mean = np.nanmean(pdfs_members_y, axis=0)
    pdf_y_mod_p05  = np.nanpercentile(pdfs_members_y, 5, axis=0)
    pdf_y_mod_p95  = np.nanpercentile(pdfs_members_y, 95, axis=0)

    # --- normalize all PDFs to integrate ≈1 ---
    dy = ys[1] - ys[0]

    pdf_y_ref, area_y_ref           = _normalize(pdf_y_ref, dy)
    pdf_y_mod_mean, area_y_mod_mean = _normalize(pdf_y_mod_mean, dy)
    pdf_y_mod_p05                   = pdf_y_mod_p05 / area_y_mod_mean
    pdf_y_mod_p95                   = pdf_y_mod_p95 / area_y_mod_mean

    # --- plotting (RIGHT marginal) ---
    # REF as filled baseline
    ax_right.fill_betweenx(ys, 0, pdf_y_ref, color="0.7", alpha=0.4)
    ax_right.plot(pdf_y_ref, ys, color="black", lw=1.5, label="ERA5")

    # ensemble-spread shading (model 5–95%)
    ax_right.fill_betweenx(
        ys, pdf_y_mod_p05, pdf_y_mod_p95,
        color="gold", alpha=0.25, label="CanESM 5–95%"
    )

    # model mean line
    ax_right.plot(pdf_y_mod_mean, ys, color="gold", lw=1.8, label="CanESM mean")

    # cosmetics
    ax_right.set_ylim(y_lo, y_hi)
    ax_right.set_yticks([])
    ax_right.set_xlabel("log(P(T))")
    ax_right.set_xscale("log")
    ax_right.set_xlim(1e-4, 1)
    ax_right.tick_params(axis="y", left=False, labelleft=False, pad=0)
    ax_right.spines["left"].set_visible(False)
    # (optional) show legend here, or keep it in ax_top/ax_main
    # ax_right.legend(frameon=False, fontsize=8, loc="lower right")

    # -------------------------
    # 6d. Colorbar
    # -------------------------
    cb = fig.colorbar(pcm, cax=ax_cbar, orientation="vertical")
    cb.set_label("CanESM - ERA5 KDE Anomaly")
    ax_cbar.xaxis.set_ticks([])

    # hatch legend proxy
    # hatch_proxy = Patch(facecolor="none", edgecolor="0.25",
    #                     hatch="////",
    #                     label="ERA5 KDE within CanESM 5–95% band")
    # leg_2 = ax_main.legend(handles=[hatch_proxy], loc="upper left", frameon=True, fontsize=8)
    # ax_main.add_artist(leg_1)


    handles1, labels1 = ax_main.get_legend_handles_labels() #grabs existing legend handles

    # build proxies for contour and hatch
    contour_proxy = Line2D([], [], color="white", lw=1.5, label="ERA5 KDE contours")
    hatch_proxy = Patch(facecolor="none", edgecolor="0.25", hatch="////",
                        label="ERA5 KDE within CanESM 5–95% band")

    # combine all
    handles = handles1 + [contour_proxy, hatch_proxy]
    ax_main.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)

    # -------------------------
    # 6e. Title & layout
    # -------------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    fig_name = f"{output_path}/KDE_{LWA_var}_modelvref_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    plt.close(fig)


def plot_conditional_kdes(
    x_in: xr.DataArray,
    y_in: xr.DataArray,
    title: str = "",
    sim: str = "",
    x_label: str = "LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    point_alpha: float = 0.4,
    max_scatter: int = 5000,
    gridsize: int = 400,
    cmap: str = "afmhot_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) with marginal PDFs and a best-fit regression line.

    Layout:
        - main panel: 2D KDE background (+ scatter sample, + linear fit)
        - top panel:  1D PDF of x
        - right panel:1D PDF of y
    Returns slope, intercept, r-value for logging.

    Parameters
    ----------
    x_in, y_in : np.ndarray
        1D arrays of equal length. These should already be anomalies
        or whatever you want to compare.
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    if "member" in x_in.dims:
        x, y = _stack_clean(x_in, y_in, has_member=True)
    else:
        x, y = _stack_clean(x_in, y_in, has_member=False)

    if x.size < 3 or y.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    if x_label == 'LWA_a' or x_label == 'LWA':
        leg_loc="upper left"
    elif x_label == 'LWA_c':
        leg_loc="lower left"
    else:
        Warning("x_label not recognized, setting legend location to 'best'")
        leg_loc="best"

    if x_limits is None:
        xmin, xmax = np.min([x_era.min(), x_can.min()]), np.max([x_era.max(), x_can.max()])
    else:
        xmin, xmax = x_limits

    if y_limits is None:
        ymin, ymax = np.min([y_era.min(), y_can.min()]), np.max([y_era.max(), y_can.max()])
    else:
        ymin, ymax = y_limits

    # ---------------------------------------------------------------------
    # 2. Fit linear regression using SciPy
    # ---------------------------------------------------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
 
    # x-range for plotting the regression line
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), gridsize)
    y_line = slope * x_line + intercept

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d = stats.gaussian_kde(np.vstack([x, y]))
    # grid
    xi = np.linspace(0, xmax*1.25, gridsize+5)
    yi = np.linspace(ymin*1.25, ymax*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    # 1D KDEs for marginals ( PDFs )
    def kde_1d(arr: np.ndarray, npts: int = gridsize) -> Tuple[np.ndarray, np.ndarray]:

        k = stats.gaussian_kde(arr)
        xs = np.linspace(np.min(arr), np.max(arr), npts)
        pdf = k(xs)

        # normalize pdf so area ~ 1
        dx = xs[1] - xs[0]
        area = np.trapezoid(pdf, dx=dx)
        if area > 0:
            pdf = pdf / area
        return xs, pdf

    xs_pdf, x_pdf = kde_1d(x) #PDF of LWA
    ys_pdf, y_pdf = kde_1d(y) #PDF of T

    # ---------------------------------------------------------------------
    # 4. Layout: two main conditional panels + cbar below
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig, axs = plt.subplots(ncols=2, figsize=(13, 8), tight_layout=True)
    ax_1     = axs[0]
    ax_2     = axs[1]

    x_marg = np.trapezoid(Z, x=yi, axis=0)   # p(LWA=x), shape (nx,)
    y_marg = np.trapezoid(Z, x=xi, axis=1)   # p(T=y),   shape (ny,)

    print(np.trapezoid(x_marg, x=xi))  # should be ~1
    print(np.trapezoid(y_marg, x=yi))  # should be ~1

    eps = np.finfo(float).eps
    Z_cond_LWA = Z / np.maximum(x_marg, eps)[None, :]   # p(T | LWA)  (normalize columns)
    Z_cond_T   = Z / np.maximum(y_marg, eps)[:, None]   # p(LWA | T)  (normalize rows)

    ## testing that the conditional densities integrate to 1
    arr_1 = np.trapezoid(Z_cond_LWA, x=xi, axis=1) # "Conditional density P(T | LWA) does not integrate to 1"
    arr_2 = np.trapezoid(Z_cond_T,   x=yi, axis=0)   # "Conditional density P(LWA | T) does not integrate to 1"

    print("range of conditional prob Z_cond_LWA", arr_1.min(), arr_1.max())
    print("range of conditional prob Z_cond_T", arr_2.min(), arr_2.max())

    # --------------------
    # Left main panel (x | y)
    # --------------------
    # KDE as background
    pcm = ax_1.pcolormesh(
        Xg, Yg, Z_cond_LWA,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=0.45
    )

    #conditional mean line
    bins = np.linspace(x.min(), x.max(), 30)
    digitized = np.digitize(x, bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cond_mean = np.array([y[digitized == i].mean() for i in range(1, len(bins))])
    
    cond_lowb = []
    cond_upb = []
    for i in range(1, len(bins)):
        vals = y[digitized == i]
        if vals.size == 0:
            cond_lowb.append(cond_mean[i-1])
            cond_upb.append(cond_mean[i-1])
        else:
            cond_lowb.append(np.percentile(vals, 25))
            cond_upb.append(np.percentile(vals, 75))
    cond_lowb = np.array(cond_lowb)
    cond_upb = np.array(cond_upb)

    ax_1.plot(bin_centers, cond_mean, color='cyan', lw=2, label='conditional mean')
    ax_1.fill_between(bin_centers, cond_lowb, cond_upb, color='cyan', alpha=0.3, label='IQR')

    # light scatter sample: subselect if huge
    if x.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
    else:
        xs_sc = x
        ys_sc = y

    ax_1.scatter(
        xs_sc,
        ys_sc,
        s=8,
        c="white",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    ax_1.set_title("P(T | LWA)")
    # ax_1.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_1.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_1.set_ylabel(y_label + " (K)")
    ax_1.set_xlim(x_limits if x_limits else (np.min(x), np.max(x)))
    ax_1.set_ylim(y_limits if y_limits else (np.min(y), np.max(y)))
    ax_1.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm, ax=ax_1, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # Right main panel (y | x)
    # --------------------
    pcm2 = ax_2.pcolormesh(
        Xg, Yg, Z_cond_T,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=4e-4
    )

    #conditional mean line
    bins = np.linspace(y.min(), y.max(), 30)
    digitized = np.digitize(y, bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    cond_mean = np.array([x[digitized == i].mean() for i in range(1, len(bins))])
    
    cond_lowb = []
    cond_upb = []
    for i in range(1, len(bins)):
        vals = x[digitized == i]
        if vals.size == 0:
            cond_lowb.append(cond_mean[i-1])
            cond_upb.append(cond_mean[i-1])
        else:
            cond_lowb.append(np.percentile(vals, 25))
            cond_upb.append(np.percentile(vals, 75))
    cond_lowb = np.array(cond_lowb)
    cond_upb = np.array(cond_upb)

    #conditional mean line
    ax_2.plot(cond_mean, bin_centers, color='cyan', lw=2, label='conditional mean')
    ax_2.fill_betweenx(bin_centers, cond_lowb, cond_upb, color='cyan', alpha=0.3, label='IQR')

    # light scatter sample: subselect if huge
    if y.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(y.size, size=max_scatter, replace=False)
        xs_sc = x[idx]
        ys_sc = y[idx]
    else:
        xs_sc = x
        ys_sc = y

    ax_2.scatter(
        xs_sc,
        ys_sc,
        s=8,
        c="white",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    ax_2.set_title("P(LWA | T)")
    # ax_2.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_2.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_2.set_ylabel(y_label + " (K)")
    ax_2.set_xlim(x_limits if x_limits else (np.min(x), np.max(x)))
    ax_2.set_ylim(y_limits if y_limits else (np.min(y), np.max(y)))
    ax_2.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm2, ax=ax_2, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    fig_name = f"{output_path}/{sim}_{LWA_var}_deltaT_conditional_kde_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None


def plot_conditional_kdes_model_vs_ref(
    lwa_can: xr.DataArray,
    deltaT_can: xr.DataArray,
    lwa_era: xr.DataArray,
    deltaT_era: xr.DataArray,
    title: str = "",
    x_label: str = "LWA",
    y_label: str = r"$\Delta T$",
    x_limits: Tuple[float, float] = None, #type:ignore
    y_limits: Tuple[float, float] = None, #type:ignore
    gridsize: int = 400,
    cmap: str = "RdBu_r",
    output_path: str = './',
) -> None: # Tuple[float, float, float]:
    """
    Plot a 2D KDE (x vs y) of conditional probability anomaly, between CanESM ensemble and ERA5.

    Layout:
        - LHS main panel: 2D KDE of P(T | LWA) for CanESM - ERA5
        - RHS main panel: 2D KDE of P(LWA | T) for CanESM - ERA5

    Parameters
    ----------
    lwa_can, deltaT_can : np.ndarray 
        dimensions (member, time, lat, lon).
    lwa_era, deltaT_era : np.ndarray
        dimensions (time, lat, lon).
    title : str
        Title for the whole figure.
    x_label, y_label : str
        Axis labels for main panel.
    point_alpha : float
        Transparency for scatter sample.
    max_scatter : int
        Max number of points to scatter-plot (for readability).
        All points are still used for KDE and regression.
    gridsize : int
        Resolution of KDE grid in each dimension.
    cmap : str
        Matplotlib colormap name for the KDE.

    Returns
    -------
    None
    """

    # ---------------------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------------------

    x_era, y_era = _stack_clean(lwa_era, deltaT_era, has_member=False)

    if "member" in lwa_can.dims:
        x_can, y_can = _stack_clean(lwa_can, deltaT_can, has_member=True)
    else:
        x_can, y_can = _stack_clean(lwa_can, deltaT_can, has_member=False)

    if x_limits is None:
        xmin, xmax = np.min([x_era.min(), x_can.min()]), np.max([x_era.max(), x_can.max()])
    else:
        xmin, xmax = x_limits

    if y_limits is None:
        ymin, ymax = np.min([y_era.min(), y_can.min()]), np.max([y_era.max(), y_can.max()])
    else:
        ymin, ymax = y_limits

    if x_can.size < 3 or y_can.size < 3:
        raise ValueError("Not enough valid points after NaN filtering for KDE/regression.")

    if x_label == 'LWA_a' or x_label == 'LWA':
        leg_loc="upper left"
    elif x_label == 'LWA_c':
        leg_loc="lower left"
    else:
        Warning("x_label not recognized, setting legend location to 'best'")
        leg_loc="best"

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d_era = stats.gaussian_kde(np.vstack([x_era, y_era]))
    kde2d_can = stats.gaussian_kde(np.vstack([x_can, y_can]))
    # grid
    xi = np.linspace(0, xmax*1.25, gridsize+5)
    yi = np.linspace(ymin*1.25, ymax*1.25, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    
    Z_era = kde2d_era(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    Z_can = kde2d_can(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    #1D Marginal KDEs for CanESM ( PDFs )

    x_marg_era = np.trapezoid(Z_era, x=yi, axis=0)   # p(LWA=x), shape (nx,)
    y_marg_era = np.trapezoid(Z_era, x=xi, axis=1)   # p(T=y),   shape (ny,)

    x_marg_can = np.trapezoid(Z_can, x=yi, axis=0)   # p(LWA=x), shape (nx,)
    y_marg_can = np.trapezoid(Z_can, x=xi, axis=1)   # p(T=y),   shape (ny,)

    eps = np.finfo(float).eps
    Z_cond_LWA_can = Z_can / np.maximum(x_marg_can, eps)[None, :]   # p(T | LWA)  (normalize columns)
    Z_cond_T_can   = Z_can / np.maximum(y_marg_can, eps)[:, None]   # p(LWA | T)  (normalize rows)

    Z_cond_LWA_era = Z_era / np.maximum(x_marg_era, eps)[None, :]   # p(T | LWA)  (normalize columns)
    Z_cond_T_era   = Z_era / np.maximum(y_marg_era, eps)[:, None]   # p(LWA | T)  (normalize rows)

    Z_cond_LWA_anom = Z_cond_LWA_can - Z_cond_LWA_era
    Z_cond_T_anom   = Z_cond_T_can - Z_cond_T_era

    # ---------------------------------------------------------------------
    # 4. Layout: two main conditional panels + cbar below
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig, axs = plt.subplots(ncols=2, figsize=(13, 8), tight_layout=True)
    ax_1     = axs[0]
    ax_2     = axs[1]

    # --------------------
    # Left main panel (x | y)
    # --------------------
    # KDE as background
    pcm = ax_1.pcolormesh(
        Xg, Yg, Z_cond_LWA_anom,
        shading="auto",
        cmap=cmap,
        vmin=-0.3,
        vmax=0.3
    )

    #conditional mean line
    bins = np.linspace(xmin, xmax, 30)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    digitized_can = np.digitize(x_can, bins)
    cond_mean_can = np.array([y_can[digitized_can == i].mean() for i in range(1, len(bins))])

    digitized_era = np.digitize(x_era, bins)
    cond_mean_era = np.array([y_era[digitized_era == i].mean() for i in range(1, len(bins))])
    
    cond_lowb_can = []
    cond_upb_can = []
    for i in range(1, len(bins)):
        vals = y_can[digitized_can == i]
        if vals.size == 0:
            cond_lowb_can.append(cond_mean_can[i-1])
            cond_upb_can.append(cond_mean_can[i-1])
        else:
            cond_lowb_can.append(np.percentile(vals, 25))
            cond_upb_can.append(np.percentile(vals, 75))
    cond_lowb_can = np.array(cond_lowb_can)
    cond_upb_can = np.array(cond_upb_can)

    cond_lowb_era = []
    cond_upb_era = []
    for i in range(1, len(bins)):
        vals = y_era[digitized_era == i]
        if vals.size == 0:
            cond_lowb_era.append(cond_mean_era[i-1])
            cond_upb_era.append(cond_mean_era[i-1])
        else:
            cond_lowb_era.append(np.percentile(vals, 25))
            cond_upb_era.append(np.percentile(vals, 75))
    cond_lowb_era = np.array(cond_lowb_era)
    cond_upb_era = np.array(cond_upb_era)

    ax_1.plot(bin_centers, cond_mean_can, color='yellow', lw=2)
    ax_1.fill_between(bin_centers, cond_lowb_can, cond_upb_can, color='yellow', alpha=0.3)
    
    ax_1.plot(bin_centers, cond_mean_era, color='cyan', lw=2, label='conditional mean (ERA5)')
    ax_1.fill_between(bin_centers, cond_lowb_era, cond_upb_era, color='cyan', alpha=0.3, label='IQR (ERA5)')

    ax_1.set_title("P(T | LWA)")
    # ax_1.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_1.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_1.set_ylabel(y_label + " (K)")
    ax_1.set_xlim(x_limits if x_limits else (xmin, xmax))
    ax_1.set_ylim(y_limits if y_limits else (ymin, ymax))
    ax_1.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm, ax=ax_1, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # Right main panel (y | x)
    # --------------------
    pcm2 = ax_2.pcolormesh(
        Xg, Yg, Z_cond_T_anom,
        shading="auto",
        cmap=cmap,
        vmin=-3e-4,
        vmax=3e-4
    )

    #conditional mean line
    bins = np.linspace(ymin, ymax, 30)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    digitized_can = np.digitize(y_can, bins)
    cond_mean_can = np.array([x_can[digitized_can == i].mean() for i in range(1, len(bins))])

    digitized_era = np.digitize(y_era, bins)
    cond_mean_era = np.array([x_era[digitized_era == i].mean() for i in range(1, len(bins))])
    
    cond_lowb_can = []
    cond_upb_can = []
    for i in range(1, len(bins)):
        vals = x_can[digitized_can == i]
        if vals.size == 0:
            cond_lowb_can.append(cond_mean_can[i-1])
            cond_upb_can.append(cond_mean_can[i-1])
        else:
            cond_lowb_can.append(np.percentile(vals, 25))
            cond_upb_can.append(np.percentile(vals, 75))
    cond_lowb_can = np.array(cond_lowb_can)
    cond_upb_can = np.array(cond_upb_can)

    cond_lowb_era = []
    cond_upb_era = []
    for i in range(1, len(bins)):
        vals = x_era[digitized_era == i]
        if vals.size == 0:
            cond_lowb_era.append(cond_mean_era[i-1])
            cond_upb_era.append(cond_mean_era[i-1])
        else:
            cond_lowb_era.append(np.percentile(vals, 25))
            cond_upb_era.append(np.percentile(vals, 75))
    cond_lowb_era = np.array(cond_lowb_era)
    cond_upb_era = np.array(cond_upb_era)

    ax_2.plot(cond_mean_can, bin_centers, color='yellow', lw=2)
    ax_2.fill_betweenx(bin_centers, cond_lowb_can, cond_upb_can, color='yellow', alpha=0.3)
    
    ax_2.plot(cond_mean_era, bin_centers, color='cyan', lw=2, label='conditional mean (ERA5)')
    ax_2.fill_betweenx(bin_centers, cond_lowb_era, cond_upb_era, color='cyan', alpha=0.3, label='IQR (ERA5)')

    ax_2.set_title("P(LWA | T)")
    # ax_2.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_2.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_2.set_ylabel(y_label + " (K)")
    ax_2.set_xlim(x_limits if x_limits else (xmin, xmax))
    ax_2.set_ylim(y_limits if y_limits else (ymin, ymax))
    ax_2.legend(loc=leg_loc, frameon=True, fontsize=9)

    # colorbar for KDE
    cb = fig.colorbar(pcm2, ax=ax_2, orientation="horizontal")
    cb.set_label("Conditional density (KDE)")

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    fig_name = f"{output_path}/KDE_{LWA_var}_conditional_modelvref_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None


# -----------------------------------------------------------------------------

def main(REGION, LWA_var, SEASON, ZG_COORD):

    #load in data

    ds_tas_canesm = data_io.open_canesm_temperature(TEMP_VAR, ENSEMBLE_LIST)
    ds_tas_canesm = ds_tas_canesm.chunk({"time": 365})
    ds_tas_canesm = preprocess.compute_region_mean(ds_tas_canesm, REGION).compute()
    
    ds_tas_canesm_clim = ds_tas_canesm.groupby("time.dayofyear").mean("time")
    ds_tas_canesm_anom = ds_tas_canesm.groupby("time.dayofyear") - ds_tas_canesm_clim

    ds_tas_era5 = data_io.open_era5_temperature(TEMP_VAR)
    ds_tas_era5 = ds_tas_era5.chunk({"time": 365})
    ds_tas_era5 = preprocess.compute_region_mean(ds_tas_era5, REGION).compute()
    
    ds_tas_era5_clim = ds_tas_era5.groupby("time.dayofyear").mean("time")
    ds_tas_era5_anom = ds_tas_era5.groupby("time.dayofyear") - ds_tas_era5_clim

    ds_tas_canesm_anom = ds_tas_canesm_anom.assign_coords(time=ds_tas_canesm_anom.time.dt.floor("D"))
    ds_tas_era5_anom = ds_tas_era5_anom.assign_coords(time=ds_tas_era5_anom.time.dt.floor("D")) #ensure time coords match
    
    ds_mrsos_canesm = data_io.open_canesm_mrsos(var='mrsos', ensemble_list=ENSEMBLE_LIST)
    ds_mrsos_canesm = preprocess.compute_region_mean(ds_mrsos_canesm, REGION).compute()
    ds_mrsos_canesm_clim = ds_mrsos_canesm.groupby("time.dayofyear").mean("time")
    ds_mrsos_canesm_anom = ds_mrsos_canesm.groupby("time.dayofyear") - ds_mrsos_canesm_clim

    ds_mrsos_era5 = data_io.open_era5_mrsos(var='swvl1')
    ds_mrsos_era5 = preprocess.compute_region_mean(ds_mrsos_era5, REGION).compute()

    ds_mrsos_era5_clim = ds_mrsos_era5.groupby("time.dayofyear").mean("time")
    ds_mrsos_era5_anom = ds_mrsos_era5.groupby("time.dayofyear") - ds_mrsos_era5_clim


    ## loading LWA data
    # 1) Open CanESM and ERA5 for all VARiables
    ds_canesm_lwas: Dict[str, xr.DataArray] = data_io.open_canesm_lwa(ENSEMBLE_LIST, ZG_COORD)
    ds_era5_lwas: Dict[str, xr.DataArray]   = data_io.open_era5_lwa(ZG_COORD)

    ds_canesm_lwa = ds_canesm_lwas[LWA_var]#.sel(MEMBER=MEMBER)
    ds_era5_lwa   = ds_era5_lwas[LWA_var]

    ds_canesm_lwa_reg = preprocess.compute_region_mean(ds_canesm_lwa, REGION).chunk({"time": 365}).compute()
    ds_era5_lwa_reg   = preprocess.compute_region_mean(ds_era5_lwa, REGION).chunk({"time": 365}).compute()

    ds_canesm_lwa_reg = xr.apply_ufunc(np.sqrt, ds_canesm_lwa_reg)
    ds_era5_lwa_reg   = xr.apply_ufunc(np.sqrt, ds_era5_lwa_reg)

    ds_canesm_lwa_reg = ds_canesm_lwa_reg.assign_coords(time=ds_canesm_lwa_reg.time.dt.floor("D"))
    ds_era5_lwa_reg = ds_era5_lwa_reg.assign_coords(time=ds_era5_lwa_reg.time.dt.floor("D"))

    ds_mrsos_canesm_anom = ds_mrsos_canesm_anom.assign_coords(time=ds_mrsos_canesm_anom.time.dt.floor("D"))
    ds_mrsos_era5_anom   = ds_mrsos_era5_anom.assign_coords(time=ds_mrsos_era5_anom.time.dt.floor("D"))


    season_mask_era = ds_era5_lwa_reg.time.dt.season == SEASON
    season_mask_can = ds_canesm_lwa_reg.time.dt.season == SEASON

    masked_lwa_era = ds_era5_lwa_reg.where(season_mask_era,drop=True)
    masked_lwa_can = ds_canesm_lwa_reg.where(season_mask_can,drop=True)

    masked_tas_era = ds_tas_era5_anom.where(season_mask_era,drop=True)
    masked_tas_can = ds_tas_canesm_anom.where(season_mask_can,drop=True)

    masked_mrsos_era = ds_mrsos_era5_anom.where(season_mask_era,drop=True)
    masked_mrsos_can = ds_mrsos_canesm_anom.where(season_mask_can,drop=True)

    print(masked_mrsos_can, masked_lwa_can)
    print(masked_mrsos_era, masked_lwa_era)


    fig,ax = plt.subplots(figsize=(8,6))
    ax.scatter(masked_lwa_era, masked_mrsos_era, color='blue', alpha=0.5, label='ERA5')
    ax.scatter(masked_lwa_can, masked_mrsos_can, color='red', alpha=0.5, label='CanESM')
    ax.set_xlabel(f"{LWA_var} (sqrt hPa m)")
    ax.set_ylabel("Soil Moisture Anomaly (kg/m2)")
    ax.set_title(f"{LWA_var} vs Soil Moisture Anomaly, {REGION}, {SEASON}")
    ax.legend()
    fig_name = f"{OUTPUT_PLOTS_PATH}/{LWA_var}_vs_SoilMoisture_{REGION}_{SEASON}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')


    print(masked_lwa_era)
    print(masked_tas_era)

    x_lo = np.min([masked_lwa_era.min().item(), masked_lwa_can.min().item()])
    x_hi = np.max([masked_lwa_era.max().item(), masked_lwa_can.max().item()])
    y_lo = np.min([masked_tas_era.min().item(), masked_tas_can.min().item()])
    y_hi = np.max([masked_tas_era.max().item(), masked_tas_can.max().item()])
    z_lo = np.min([masked_mrsos_era.min().item(), masked_mrsos_can.min().item()])
    z_hi = np.max([masked_mrsos_era.max().item(), masked_mrsos_can.max().item()])

    z_max = np.max([np.abs(z_lo), np.abs(z_hi)])

    plot_joint_kde_with_marginals_mrsos(
        masked_lwa_era,
        masked_tas_era,
        masked_mrsos_era,
        title=f"ERA5 {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="ERA5",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        z_limits=(-z_max, z_max),
        output_path=OUTPUT_PLOTS_PATH
    )



    plot_joint_kde_with_marginals_mrsos(
        masked_lwa_can,
        masked_tas_can,
        masked_mrsos_can,
        title=f"CanESM {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="CanESM",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        z_limits=(-z_max, z_max),
        output_path=OUTPUT_PLOTS_PATH
    )


    plot_joint_kde_with_marginals(
        masked_lwa_era,
        masked_tas_era,
        title=f"ERA5 {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="ERA5",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    plot_conditional_kdes(
        x_in=masked_lwa_era,
        y_in=masked_tas_era,
        title=f"ERA5 {LWA_var} vs TAS (conditional), {REGION}, {SEASON}",
        sim="ERA5",
        x_label=f"{LWA_var}", #just lwa_var
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )
    
    plot_joint_kde_with_marginals(
        masked_lwa_can,
        masked_tas_can,
        title=f"CanESM {LWA_var} vs TAS, {REGION}, {SEASON}",
        sim="CanESM",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    plot_conditional_kdes(
        x_in=masked_lwa_can,
        y_in=masked_tas_can,
        title=f"CanESM {LWA_var} vs TAS (conditional), {REGION}, {SEASON}",
        sim="CanESM",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    plot_kde_comparison_model_vs_ref(
        masked_lwa_can,  # xr.DataArray, dims ('member','time') or ('time',) #model, CanESM
        masked_tas_can,  # xr.DataArray, dims ('member','time') or ('time',)
        masked_lwa_era,  # xr.DataArray, dims ('time',) #reference, ERA5
        masked_tas_era,  # xr.DataArray, dims ('time',)
        title = "CanESM vs ERA5 KDE Comparison",
        x_label = f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )


    plot_conditional_kdes_model_vs_ref(
        lwa_can=masked_lwa_can,
        deltaT_can=masked_tas_can,
        lwa_era=masked_lwa_era,
        deltaT_era=masked_tas_era,
        title=f"CanESM vs ERA5 {LWA_var} vs TAS (conditional), {REGION}, {SEASON}",
        x_label=f"{LWA_var}",
        x_limits=(x_lo, x_hi),
        y_limits=(y_lo, y_hi),
        output_path=OUTPUT_PLOTS_PATH
    )

    return None



if __name__ == "__main__":
    args = arg_parser()
    REGION   = args.region
    LWA_var  = args.lwa_var
    SEASON   = args.season
    ZG_LEVEL = args.zg

    TEMP_VAR      = config.TEMP_VAR
    ENSEMBLE_LIST = config.ENSEMBLE_LIST

    main(REGION, LWA_var, SEASON, ZG_LEVEL)
