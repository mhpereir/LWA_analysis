import os

# Model variations
_full     = {'include_lwa': True, 'include_sm': True, 'include_interaction': True}
_no_int   = {'include_lwa': True, 'include_sm': True, 'include_interaction': False}
_sm_only  = {'include_lwa': False, 'include_sm': True, 'include_interaction': False}
_lwa_only = {'include_lwa': True, 'include_sm': False, 'include_interaction': False}

model_variations = {
    "full":     _full,
    "no_int":   _no_int,
    "sm_only":  _sm_only,
    "lwa_only": _lwa_only,
}

# PyMC sampling parameters
PYMC_NDRAW         = 2000  # number of posterior draws to use for R2 calculation
PYMC_NTUNE         = 2000  # number of tuning steps
PYMC_CHAINS        = 4
PYMC_TARGET_ACCEPT = 0.9