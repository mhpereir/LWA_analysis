import os

from .pymc_models import ModelSpec, Term

# Model variations
_full     = {'include_lwa': True, 'include_sm': True, 'include_interaction': True}
_no_int   = {'include_lwa': True, 'include_sm': True, 'include_interaction': False}
_sm_only  = {'include_lwa': False, 'include_sm': True, 'include_interaction': False}
_lwa_only = {'include_lwa': True, 'include_sm': False, 'include_interaction': False}

model_variations = {
    "full":     _full,
    # "no_int":   _no_int,
    "sm_only":  _sm_only,
    "lwa_only": _lwa_only,
}

# PyMC sampling parameters
PYMC_NDRAW         = 2000  # number of posterior draws to use for R2 calculation
PYMC_NTUNE         = 2000  # number of tuning steps
PYMC_CHAINS        = 4
PYMC_TARGET_ACCEPT = 0.9


BOTH_LWA_NOAR_studt_SPEC = ModelSpec(
    key="both_lwa_noar_studentt",
    ar1=False,
    likelihood="studentt",
    data_names=("x_lwa_a", "x_lwa_c", "x_sm", "y", "same_year"),
    intercept_name="b0",
    terms=(
        Term(
            coef_name="b1",
            uses=("x_lwa_a",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_a"],
        ),
        Term(
            coef_name="b2",
            uses=("x_lwa_c",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_c"],
        ),
        Term(
            coef_name="b3",
            uses=("x_sm",),
            active=lambda v: v.include_sm,
            expr=lambda d: d["x_sm"],
        ),
    ),
)


BOTH_LWA_NOAR_normal_SPEC = ModelSpec(
    key="both_lwa_noar_normal",
    ar1=False,
    likelihood="normal",
    data_names=("x_lwa_a", "x_lwa_c", "x_sm", "y", "same_year"),
    intercept_name="b0",
    terms=(
        Term(
            coef_name="b1",
            uses=("x_lwa_a",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_a"],
        ),
        Term(
            coef_name="b2",
            uses=("x_lwa_c",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_c"],
        ),
        Term(
            coef_name="b3",
            uses=("x_sm",),
            active=lambda v: v.include_sm,
            expr=lambda d: d["x_sm"],
        ),
    ),
)



BOTH_LWA_AR1_studt_SPEC = ModelSpec(
    key="both_lwa_ar1_studentt",
    ar1=True,
    likelihood="studentt",
    data_names=("x_lwa_a", "x_lwa_c", "x_sm", "y", "same_year"),
    intercept_name="b0",
    terms=(
        Term(
            coef_name="b1",
            uses=("x_lwa_a",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_a"],
        ),
        Term(
            coef_name="b2",
            uses=("x_lwa_c",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_c"],
        ),
        Term(
            coef_name="b3",
            uses=("x_sm",),
            active=lambda v: v.include_sm,
            expr=lambda d: d["x_sm"],
        ),
    ),
)

BOTH_LWA_AR1_normal_SPEC = ModelSpec(
    key="both_lwa_ar1_normal",
    ar1=True,
    likelihood="normal",
    data_names=("x_lwa_a", "x_lwa_c", "x_sm", "y", "same_year"),
    intercept_name="b0",
    terms=(
        Term(
            coef_name="b1",
            uses=("x_lwa_a",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_a"],
        ),
        Term(
            coef_name="b2",
            uses=("x_lwa_c",),
            active=lambda v: v.include_lwa,
            expr=lambda d: d["x_lwa_c"],
        ),
        Term(
            coef_name="b3",
            uses=("x_sm",),
            active=lambda v: v.include_sm,
            expr=lambda d: d["x_sm"],
        ),
    ),
)


bayesian_model_specs = {
    "both_lwa_noar_studentt": BOTH_LWA_NOAR_studt_SPEC,
    "both_lwa_noar_normal": BOTH_LWA_NOAR_normal_SPEC,
    "both_lwa_ar1_studentt": BOTH_LWA_AR1_studt_SPEC,
    "both_lwa_ar1_normal": BOTH_LWA_AR1_normal_SPEC,
}
