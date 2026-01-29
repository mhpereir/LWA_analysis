from typing import Any, Dict, List, Tuple

import numpy as np
import statsmodels.api as sm  # type: ignore
from scipy import stats


def quantile_fit(x, y, quantiles: List[float]) -> Dict[float, Any]:
    X = sm.add_constant(x)
    fits = {}
    for q in quantiles:
        model = sm.QuantReg(y, X)
        res = model.fit(q=q)
        fits[q] = res
    return fits


def _stack_clean(a, b, has_member: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten (member,time) or (time,) into 1D arrays and drop NaNs."""
    if has_member:
        a = a.stack(points=("member", "time"))
        b = b.stack(points=("member", "time"))

    xa = np.asarray(a).ravel()
    ya = np.asarray(b).ravel()

    good = np.isfinite(xa) & np.isfinite(ya)
    return xa[good], ya[good]


def _normalize(pdf, dx):
    area = np.trapezoid(pdf, dx=dx)
    return pdf / area if area > 0 else pdf, area


def _safe_gaussian_kde(x, y, bw="scott"):
    """Return a callable KDE(xgrid,ygrid) robust to singular covariance."""
    pts = np.vstack([x, y])

    kde = stats.gaussian_kde(pts, bw_method=bw)
    bw_factor = kde.factor

    def eval_on(xx, yy):
        return kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    return eval_on, bw_factor


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
