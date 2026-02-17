from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, Dict, Any, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
import arviz as az


@dataclass(frozen=True)
class Variation:
    name: str
    include_lwa: bool = True
    include_sm: bool = True
    include_interaction: bool = False


@dataclass(frozen=True)
class Term:
    """One additive component of mu: coef * f(data_vars)."""
    coef_name: str
    uses: Sequence[str]  # names of pm.Data variables required
    active: Callable[[Variation], bool]
    expr: Callable[[Dict[str, Any]], Any]  # maps data dict -> pytensor expr


@dataclass(frozen=True)
class ModelSpec:
    key: str
    ar1: bool
    data_names: Sequence[str]
    intercept_name: str
    terms: Sequence[Term]
    likelihood: str = "normal"  # "studentt" or "normal"

    # names used by build_pymc_model
    rho_name: str = "rho"
    sigma_name: str = "sigma"
    nu_name: str = "nu_minus_2"

    def __post_init__(self) -> None:
        if self.likelihood not in ("normal", "studentt"):
            raise ValueError(f"invalid likelihood name: {self.likelihood}")
        # For normal likelihood, nu_name is unused (fine); for no-AR, rho_name unused (fine).

    def posterior_param_names(self, variation: Optional[Variation] = None) -> list[str]:
        """
        Return posterior parameter names expected for this spec.
        If variation is provided, include only coefficients active for that variation.
        """
        names: list[str] = [self.intercept_name]
        seen = {self.intercept_name}

        for term in self.terms:
            if variation is not None and not term.active(variation):
                continue
            if term.coef_name not in seen:
                names.append(term.coef_name)
                seen.add(term.coef_name)

        if self.ar1 and self.rho_name not in seen:
            names.append(self.rho_name)
            seen.add(self.rho_name)
        if self.sigma_name not in seen:
            names.append(self.sigma_name)
            seen.add(self.sigma_name)
        if self.likelihood == "studentt" and self.nu_name not in seen:
            names.append(self.nu_name)

        return names


def single_lwa_spec(ar1: bool = True) -> ModelSpec:
    return ModelSpec(
        key="single_lwa_ar1" if ar1 else "single_lwa_noar",
        ar1=ar1,
        data_names=("x_lwa", "x_sm", "y", "same_year"),  # x1=LWA, x2=SM
        intercept_name="b0",
        terms=(
            Term(
                coef_name="b_lwa",
                uses=("x_lwa",),
                active=lambda v: v.include_lwa,
                expr=lambda d: d["x_lwa"],
            ),
            Term(
                coef_name="b_sm",
                uses=("x_sm",),
                active=lambda v: v.include_sm,
                expr=lambda d: d["x_sm"],
            ),
            Term(
                coef_name="b_int",
                uses=("x_lwa", "x_sm"),
                active=lambda v: v.include_interaction,
                expr=lambda d: d["x_lwa"] * d["x_sm"],
            ),
        ),
    )


def both_lwa_spec(ar1: bool = True) -> ModelSpec:
    # Here x1=LWA_a, x2=LWA_c, x3=SM (to match how you’ve been naming)
    return ModelSpec(
        key="both_lwa_ar1" if ar1 else "both_lwa_noar",
        ar1=ar1,
        data_names=("x_lwa_a", "x_lwa_c", "x_sm", "y", "same_year"),
        intercept_name="b0",
        terms=(
            Term(
                coef_name="b_lwa_a",
                uses=("x_lwa_a",),
                active=lambda v: True,          # always include LWA_a for this spec
                expr=lambda d: d["x_lwa_a"],
            ),
            Term(
                coef_name="b_lwa_c",
                uses=("x_lwa_c",),
                active=lambda v: True,          # always include LWA_c for this spec
                expr=lambda d: d["x_lwa_c"],
            ),
            Term(
                coef_name="b_sm",
                uses=("x_sm",),
                active=lambda v: v.include_sm,  # allow SM on/off
                expr=lambda d: d["x_sm"],
            ),
            # If you want an “interaction” for both-LWA, define it explicitly here.
            # Don’t reuse include_interaction unless you implement what it means.
        ),
    )


def _nc_safe(v):
    # NetCDF attrs: keep to str/int/float, arrays of those, etc.
    if isinstance(v, (bool, np.bool_)):
        return int(v)  # 0/1
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

def save_idata(idata: az.InferenceData, out_nc: str, *, spec: ModelSpec, variation: Variation, context: dict):
    idata.attrs["spec_key"] = spec.key
    idata.attrs["ar1"] = _nc_safe(spec.ar1)
    idata.attrs["likelihood"] = spec.likelihood
    idata.attrs["variation"] = variation.name
    idata.attrs["include_lwa"] = _nc_safe(variation.include_lwa)
    idata.attrs["include_sm"] = _nc_safe(variation.include_sm)
    idata.attrs["include_interaction"] = _nc_safe(variation.include_interaction)
    for k, v in context.items():
        idata.attrs[k] = _nc_safe(v)
    az.to_netcdf(idata, out_nc)



def build_pymc_model(
    spec: ModelSpec,
    variation: Variation,
    *,
    time: xr.DataArray,                 # for same_year
    data: Dict[str, np.ndarray],        # x1/x2[/x3], y
) -> pm.Model:
    """
    data must contain at least the names in spec.data_names except 'same_year' which is computed.
    Arrays are expected already aligned, cleaned, and standardized.
    """

    # compute same_year from time
    years = time.dt.year.values.astype(np.int32)
    same_year = np.concatenate([[0], (years[1:] == years[:-1]).astype(np.int8)])

    # sanity checks
    if "y" not in data:
        raise ValueError("data must include 'y'")
    n = data["y"].shape[0]
    for k, v in data.items():
        if v.shape[0] != n:
            raise ValueError(f"Length mismatch: {k} has {v.shape[0]} but y has {n}")
    if same_year.shape[0] != n:
        raise ValueError(f"same_year has {same_year.shape[0]} but y has {n}")

    with pm.Model() as model:
        # Create pm.Data nodes
        d: Dict[str, Any] = {}
        for name in spec.data_names:
            if name == "same_year":
                d[name] = pm.Data("same_year", same_year)
            else:
                if name not in data:
                    raise ValueError(f"Missing data['{name}'] required by spec {spec.key}")
                d[name] = pm.Data(name, data[name])

        # Priors: intercept always exists
        b0 = pm.Normal(spec.intercept_name, 0.0, 1.0)

        # Priors for all term coefficients (only those used by this spec)
        coefs: Dict[str, Any] = {}
        for term in spec.terms:
            if term.coef_name not in coefs:
                coefs[term.coef_name] = pm.Normal(term.coef_name, 0.0, 1.0)

        # Noise / AR params
        sigma = pm.HalfNormal(spec.sigma_name, 1.0)
        rho = None
        if spec.ar1:
            rho = pm.Uniform(spec.rho_name, lower=-0.99, upper=0.99)
        
        nu = None
        if spec.likelihood == "studentt":
            nu = pm.Exponential(spec.nu_name, 1/10) + 2.0
        
        # Build mu from active terms
        mu = b0
        for term in spec.terms:
            if term.active(variation):
                mu = mu + coefs[term.coef_name] * term.expr(d)

        # Likelihood
        y = d["y"]

        if spec.ar1:
            syd = d["same_year"]
            y_prev  = pt.concatenate([y[:1], y[:-1]])
            mu_prev = pt.concatenate([mu[:1], mu[:-1]])
            cond_mu = mu + rho * syd * (y_prev - mu_prev) # type: ignore

            if spec.likelihood == "studentt":
                pm.StudentT("y_like", nu=nu, mu=cond_mu, sigma=sigma, observed=y)
            elif spec.likelihood == "normal":
                pm.Normal("y_like", mu=cond_mu, sigma=sigma, observed=y)
            else:
                raise ValueError(spec.likelihood)

        else:

            if spec.likelihood == "studentt":
                pm.StudentT("y_like", nu=nu, mu=mu, sigma=sigma, observed=y)
            elif spec.likelihood == "normal":
                pm.Normal("y_like", mu=mu, sigma=sigma, observed=y)
            else:
                raise ValueError(spec.likelihood)

    return model
