import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ------------------------------ Configuration --------------------------------

FAST_IO = False  # Set to True to speed up testing with fewer ensemble members

if FAST_IO:
    # Ensemble members
    ENSEMBLE_LIST: list[str] = ["r1i1p1f1", "r2i1p1f1" ]

    # Time slice
    TIME_SLICE: slice = slice("1970-01-01", "1999-12-31")

else:
    # Ensemble members
    ENSEMBLE_LIST: list[str] = [
        "r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1", "r5i1p1f1",
        "r6i1p1f1", "r7i1p1f1", "r8i1p1f1", "r9i1p1f1", "r10i1p1f1",
        "r1i1p2f1", "r2i1p2f1", "r3i1p2f1", "r4i1p2f1", "r5i1p2f1",
        "r6i1p2f1", "r7i1p2f1", "r8i1p2f1", "r9i1p2f1", "r10i1p2f1"
    ]

    # Time slice
    TIME_SLICE: slice = slice("1970-01-01", "2014-12-31")

# LWA variables
LWA_VARS: list[str] = ["LWA", "LWA_a", "LWA_c"]

# TEMPERATURE VARIABLE = "tas"
TEMP_VAR: str = "tas"

# Spatial slices
LAT_SLICE: slice = slice(20, 85)  # 20–90°N

# Regions (consistent with threshold files)
REGIONS: dict[str, tuple[slice, slice]] = {
    "canada": (slice(40, 70), slice(-140, -60)),
    "canada_north": (slice(55, 70), slice(-140, -60)),
    "canada_south": (slice(40, 55), slice(-140, -60)),
    "west": (slice(40, 70), slice(-140, -113.33)),
    "west_north": (slice(55, 70), slice(-140, -113.33)),
    "west_south": (slice(40, 55), slice(-140, -113.33)),
    "central": (slice(40, 70), slice(-113.33, -88.66)),
    "central_north": (slice(55, 70), slice(-113.33, -88.66)),
    "central_south": (slice(40, 55), slice(-113.33, -88.66)),
    "east": (slice(40, 70), slice(-88.66, -60)),
    "east_north": (slice(55, 70), slice(-88.66, -60)),
    "east_south": (slice(40, 55), slice(-88.66, -60)),
    "pnw_bartusek": (slice(40, 60), slice(-130.0, -110.0)),
}

SEASON_NAMES: set[str] = {"ALL", "DJF", "MAM", "JJA", "SON"}

# Root paths (adapt if needed)
CANESM_LWA_ROOT: str = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/CanESM5/historical"
ERA5_LWA_ROOT: str = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/ERA5"

ERA5_TAS_ROOT: str = "/home/mhpereir/data-mhpereir/standard_grid_daily/REANALYSIS/ERA5/tas"
CANESM_TAS_ROOT: str = "/home/mhpereir/data-mhpereir/standard_grid_daily/CMIP6/CanESM5/tas/historical"

ERA5_MRSOS_ROOT: str = "/home/mhpereir/data-mhpereir/standard_grid_daily/REANALYSIS/ERA5/soil_moisture"
CANESM_MRSOS_ROOT: str = "/home/mhpereir/data-mhpereir/standard_grid_daily/CMIP6/CanESM5/mrsos/historical"

LWA_THRESH_ROOT: str = "/home/mhpereir/data-mhpereir/LWA_thresholds/outputs"

OUTPUT_PATH: str = "/home/mhpereir/LWA_analysis/results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Plot options
PROJ = ccrs.EqualEarth()
plt.rcParams.update({"font.size": 16})