import os
import sys
import argparse
from typing import Tuple, Any, List, Dict

# Ensure project root is on sys.path when running directly from scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src import config, data_io, preprocess

import statsmodels.api as sm #type:ignore
import scipy.stats as stats


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
    parser.add_argument(
        "--lwa_var",
        type=str,
        default="LWA",
        choices=config.LWA_VARS,
        help="LWA variable to analyze.",
    )
    parser.add_argument(
        "--zg",
        type=int,
        default=500,
        choices=[250, 500],
        help="Geopotential height level for LWA.",
    )
    
    return parser.parse_args()


# ---------------------------- HELPER FUNCTIONS ---------------------------- #

def load_lwa_data(region: str, lwa_var: str, zg_coord: int, ensemble_list: list) -> Tuple[xr.DataArray, xr.DataArray]:

    ds_canesm_lwas = data_io.open_canesm_lwa(ensemble_list, zg_coord)
    ds_era5_lwas = data_io.open_era5_lwa(zg_coord)

    ds_canesm_lwa = ds_canesm_lwas[lwa_var]
    ds_era5_lwa = ds_era5_lwas[lwa_var]

    ds_canesm_lwa_reg = preprocess.compute_region_mean(ds_canesm_lwa, region).chunk({"time": 365}).compute()
    ds_era5_lwa_reg = preprocess.compute_region_mean(ds_era5_lwa, region).chunk({"time": 365}).compute()

    ds_canesm_lwa_reg = xr.apply_ufunc(np.sqrt, ds_canesm_lwa_reg)
    ds_era5_lwa_reg = xr.apply_ufunc(np.sqrt, ds_era5_lwa_reg)

    ds_canesm_lwa_reg = preprocess.floor_daily_time(ds_canesm_lwa_reg)
    ds_era5_lwa_reg = preprocess.floor_daily_time(ds_era5_lwa_reg)

    return ds_canesm_lwa_reg, ds_era5_lwa_reg

def load_temp_data(region: str, temp_var: str, ensemble_list: list) -> Tuple[xr.DataArray, xr.DataArray]:

    ds_canesm_tas = data_io.open_canesm_temperature(temp_var,ensemble_list)
    ds_era5_tas = data_io.open_era5_temperature(temp_var,)

    ds_canesm_tas_reg = preprocess.compute_region_mean(ds_canesm_tas, region).chunk({"time": 365}).compute()
    ds_era5_tas_reg = preprocess.compute_region_mean(ds_era5_tas, region).chunk({"time": 365}).compute()

    ds_canesm_tas_reg = preprocess.floor_daily_time(ds_canesm_tas_reg)
    ds_era5_tas_reg = preprocess.floor_daily_time(ds_era5_tas_reg)

    return ds_canesm_tas_reg, ds_era5_tas_reg



# ---------------------------- KDE HELPERS --------------------------- #


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


# ---------------------------- PLOTTING FUNCTION --------------------------- #

def plot_sm_baseline(
    era5_clim: xr.DataArray,
    era5_std: xr.DataArray,
    canesm_mean: xr.DataArray,
    canesm_spread: xr.DataArray,
    canesm_daily_var: xr.DataArray,
    region: str,
    output_path: str,
) -> None:
    doy = era5_clim["dayofyear"].values

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

    ax.fill_between(
        doy,
        canesm_mean.values - canesm_spread.values,
        canesm_mean.values + canesm_spread.values,
        color="tab:red",
        alpha=0.15,
        label="CanESM5 ensemble spread",
    )
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
    ax.legend(ncol=2, frameon=False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)





def plot_joint_kde_panels(
    x_in_era: xr.DataArray,
    y_in_era: xr.DataArray,
    x_in_canesm: xr.DataArray,
    y_in_canesm: xr.DataArray,
    title: str = "",
    x_label: str = f"LWA [m hPa]",
    y_label: str = r"$\Delta$SM [kg/m$^2$]",
    point_alpha: float = 0.4,
    max_scatter: int = 5000,
    gridsize: int = 200,
    cmap: str = "afmhot_r",
    output_path: str = './',
    lwa_var: str = "LWA",
    season: str = "JJA",
    region: str = "pnw_bartusek",
) -> None:
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

    x_era, y_era = _stack_clean(x_in_era, y_in_era, has_member=False)
    x_canesm, y_canesm = _stack_clean(x_in_canesm, y_in_canesm, has_member=True)

    
    # ---------------------------------------------------------------------
    # 2. Fit linear regression using SciPy
    # ---------------------------------------------------------------------
    res_era    = stats.linregress(x_era, y_era)
    res_canesm = stats.linregress(x_canesm, y_canesm)

    m_era = float(res_era.slope)           #type: ignore
    b_era = float(res_era.intercept)       #type: ignore

    m_canesm = float(res_canesm.slope)     #type: ignore
    b_canesm = float(res_canesm.intercept) #type: ignore

    rval_era    = float(res_era.rvalue)      #type: ignore
    rval_canesm = float(res_canesm.rvalue)   #type: ignore

    x_limits = (0, float(np.percentile(np.concatenate([x_era, x_canesm]), 99.5)))
    y_max    = np.max(np.abs(np.concatenate([y_era, y_canesm])))
    y_limits = (-y_max, y_max)

    # x-range for plotting the regression line
    x_line = np.linspace(x_limits[0], x_limits[1], 200)
    y_line_era    = m_era * x_line + b_era
    y_line_canesm = m_canesm * x_line + b_canesm

    # ---------------------------------------------------------------------
    # 2.1 Fit quantile regression lines (50,90,95%)
    lw = [2., 2. ,2.]
    if   season == "JJA":
        quantile_list = [0.5, 0.95]
        q_color_list  = ["cyan", "magenta", "red"]
    elif season == "DJF":
        quantile_list = [0.05, 0.5]
        q_color_list  = ["blue", "cyan"]
    else:
        quantile_list = [0.05, 0.5, 0.95]
        q_color_list  = ["blue", "cyan", "magenta"]
    quantile_fits_era    = quantile_fit(x_era, y_era, quantiles=quantile_list)
    quantile_fits_canesm = quantile_fit(x_canesm, y_canesm, quantiles=quantile_list)

    # ---------------------------------------------------------------------
    # 3. Build KDE
    # ---------------------------------------------------------------------
    # 2D KDE for joint density
    kde2d_era   , bw_era    = _safe_gaussian_kde(x_era, y_era)
    kde2d_canesm, bw_canesm = _safe_gaussian_kde(x_canesm, y_canesm)

    # grid
    xi = np.linspace(0, x_limits[1]*1.25, gridsize)
    yi = np.linspace(y_limits[0]*1.25, y_limits[1]*1.25, gridsize)

    Xg, Yg   = np.meshgrid(xi, yi)
    Z_era    = kde2d_era(Xg.ravel(), Yg.ravel()).reshape(Xg.shape)
    Z_canesm = kde2d_canesm(Xg.ravel(), Yg.ravel()).reshape(Xg.shape)

    # # 1D KDEs for marginals ( PDFs )
    # def kde_1d(arr: np.ndarray, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    #     k = stats.gaussian_kde(arr)
    #     # xs = np.linspace(np.min(arr), np.max(arr), npts)
    #     pdf = k(xs)

    #     # normalize pdf so area ~ 1
    #     dx = xs[1] - xs[0]
    #     area = np.trapezoid(pdf, dx=dx)
    #     if area > 0:
    #         pdf = pdf / area
    #     return xs, pdf

    # xs_pdf, x_pdf = kde_1d(x, xs=np.linspace(x_limits[0], x_limits[1], gridsize))
    # ys_pdf, y_pdf = kde_1d(y, xs=np.linspace(y_limits[0], y_limits[1], gridsize))

    # ---------------------------------------------------------------------
    # 4. Layout: Joint plot + marginals
    # ---------------------------------------------------------------------
    # We'll manually place axes with gridspec to control spacing
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        ncols=2, nrows=1,
        width_ratios=[1,1],
        wspace=0.1,
        hspace=0.0,
        figure=fig,
    )

    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # --------------------
    # LHS panel
    # --------------------
    # KDE as background
    pcm = ax_left.pcolormesh(
        Xg, Yg, Z_era,
        shading="auto",
        cmap=cmap,
    )

    # light scatter sample: subselect if huge
    if x_era.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x_era.size, size=max_scatter, replace=False)
        xs_sc = x_era[idx]
        ys_sc = y_era[idx]
    else:
        xs_sc = x_era
        ys_sc = y_era

    ax_left.scatter(
        xs_sc,
        ys_sc,
        s=6,
        c="gray",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    # # regression line
    ax_left.plot(
        x_line,
        y_line_era,
        color="k",
        linewidth=2,
        label=f"fit: y = {m_era:.2e} x + {b_era:.2f}\n" #type:ignore
              f"r = {rval_era:.2f}",
    )

    # quantile regression lines
    # xs = np.linspace(x_limits[0], x_limits[1], 200)
    # X_eval = sm.add_constant(xs)
    
    # for ii,q in enumerate(quantile_list):
    #     yq_line = quantile_fits_era[q].predict(X_eval)
    #     ax_left.plot(
    #         xs,
    #         yq_line,
    #         color=q_color_list[ii],
    #         linewidth=lw[ii],
    #         linestyle="--",
    #         label=_format_quantile_label(q, quantile_fits_era[q]),
    #         zorder=30
    #     )

    # ax_left.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_left.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    ax_left.set_ylabel(y_label + " (K)")
    ax_left.set_xlim(x_limits)
    ax_left.set_ylim(y_limits)
    ax_left.legend(loc="lower right", frameon=True, fontsize=9)
    # colorbar for KDE
    # colorbar for KDE
    cb = fig.colorbar(pcm, ax=ax_left, orientation="horizontal")
    cb.set_label("ERA5 KDE")


   
    # --------------------
    # RHS plot
    # --------------------

    # KDE as background
    pcm = ax_right.pcolormesh(
        Xg, Yg, Z_canesm,
        shading="auto",
        cmap=cmap,
    )

    # light scatter sample: subselect if huge
    if x_canesm.size > max_scatter:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(x_canesm.size, size=max_scatter, replace=False)
        xs_sc = x_canesm[idx]
        ys_sc = y_canesm[idx]
    else:
        xs_sc = x_canesm
        ys_sc = y_canesm

    ax_right.scatter(
        xs_sc,
        ys_sc,
        s=6,
        c="gray",
        edgecolors="k",
        linewidths=0.2,
        alpha=point_alpha,
    )

    # # regression line
    ax_right.plot(
        x_line,
        y_line_canesm,
        color="k",
        linewidth=2,
        label=f"fit: y = {m_canesm:.2e} x + {b_canesm:.2f}\n" #type:ignore
              f"r = {rval_canesm:.2f}",
    )

    # quantile regression lines
    # xs = np.linspace(x_limits[0], x_limits[1], 200)
    # X_eval = sm.add_constant(xs)
    
    # for ii,q in enumerate(quantile_list):
    #     yq_line = quantile_fits_canesm[q].predict(X_eval)
    #     ax_right.plot(
    #         xs,
    #         yq_line,
    #         color=q_color_list[ii],
    #         linewidth=lw[ii],
    #         linestyle="--",
    #         label=_format_quantile_label(q, quantile_fits_canesm[q]),
    #         zorder=30
    #     )

    # ax_left.set_xlabel(f"sqrt[{x_label} (hPa m)]")
    ax_right.set_xlabel(rf'$\sqrt{{{x_label}\ \mathrm{{(hPa\,m)}}}}$')
    # ax_right.set_ylabel(y_label + " (K)")
    ax_right.set_xlim(x_limits)
    ax_right.set_ylim(y_limits)
    ax_right.legend(loc="lower right", frameon=True, fontsize=9)
    # colorbar for KDE
    # colorbar for KDE
    cb = fig.colorbar(pcm, ax=ax_right, orientation="horizontal")
    cb.set_label("CanESM KDE")

    # --------------------
    # overall title
    # --------------------
    if title:
        fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    fig_name = f"{output_path}/SM_{lwa_var}_KDE_{region}_{season}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    return None




# ---------------------------- MAIN FUNCTION --------------------------- #

def main(region: str, lwa_var: str, zg_coord: int) -> None:
    ensemble_list = config.ENSEMBLE_LIST

    ds_canesm_mrsos = data_io.open_canesm_mrsos(var="mrsos", ensemble_list=ensemble_list)
    ds_canesm_mrsos = preprocess.compute_region_mean(ds_canesm_mrsos, region).compute()

    ds_era5_mrsos = data_io.open_era5_mrsos(var="swvl1")
    ds_era5_mrsos = preprocess.compute_region_mean(ds_era5_mrsos, region).compute()

    ds_era5_mrsos = preprocess.drop_leap_day(ds_era5_mrsos)

    # era5_mrsos_clim, era5_mrsos_std                                = preprocess.dayofyear_clim_and_std(ds_era5_mrsos)
    # canesm_mrsos_mean, canesm_mrsos_spread, canesm_mrsos_daily_var = preprocess.canesm_dayofyear_stats(ds_canesm_mrsos)

    era5_mrsos_anom = preprocess.dayofyear_anomaly(ds_era5_mrsos)
    canesm_mrsos_anom = preprocess.dayofyear_anomaly(ds_canesm_mrsos)

    ds_canesm_lwa, ds_era5_lwa = load_lwa_data(region, lwa_var, zg_coord, ensemble_list)

    ds_canesm_temp, ds_era5_temp = load_temp_data(region, config.TEMP_VAR, ensemble_list)
    
    ds_canesm_temp_anom = preprocess.dayofyear_anomaly(ds_canesm_temp)
    ds_era5_temp_anom   = preprocess.dayofyear_anomaly(ds_era5_temp)

    fig_name = f"SM_LWA_relation_{region}_{lwa_var}_{zg_coord}.png"
    output_path = os.path.join(OUTPUT_PLOTS_PATH, fig_name)

    plot_joint_kde_panels(
        x_in_era=ds_era5_lwa,
        y_in_era=era5_mrsos_anom,
        x_in_canesm=ds_canesm_lwa,
        y_in_canesm=canesm_mrsos_anom,
        title=f"SM vs LWA relation ({region}, {lwa_var} at {zg_coord} hPa)",
        x_label=f"{lwa_var} [m hPa]",
        y_label=r"$\Delta$SM [kg/m$^2$]",
        point_alpha=0.3,
        max_scatter=5000,
        gridsize=200,
        cmap="afmhot_r",
        output_path=OUTPUT_PLOTS_PATH,
        lwa_var=lwa_var,
        season="JJA",
        region=region,
    )

    print(f"Saved: {output_path}")

    # add anomaly plot

    return None


if __name__ == "__main__":
    args = arg_parser()
    
    main(args.region, args.lwa_var, args.zg)
    
