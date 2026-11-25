#!/usr/bin/env python3
"""
Regrid global forecasts onto Baltic mask grid and apply ocean masks.

Steps:
1. Interpolate (depth + lat/lon) using linear + extrapolation
2. Rename depths
3. Apply:
   - 2D surface mask for zos
   - 3D thetao mask for ocean variables
4. Save output
"""

from pathlib import Path
import xarray as xr
import numpy as np

import dask
from dask.diagnostics import ProgressBar


# -----------------------------
# CONFIG
# -----------------------------
FORECAST_PATH = Path("data/glo12_baltic_forecasts.zarr")
MASK_PATH     = Path("data/baltic_mask.zarr")
OUT_PATH      = Path("data/glo12_baltic_forecasts_interp_extra.zarr")

TARGET_DEPTHS = [0.5, 9.45, 28.15, 46.91, 91.31]
RENAMED_DEPTHS = [1, 9, 28, 47, 91]


# -----------------------------
# LOAD DATA
# -----------------------------
print("Opening datasets...")

ds = xr.open_dataset(FORECAST_PATH, chunks={
    "init_time": 1,
    "lead_time": 1
})

m = xr.open_dataset(MASK_PATH)

print(ds)
print(m)


# -----------------------------
# BUILD TARGET GRID
# -----------------------------
print("Preparing target grid...")

target_depths = xr.DataArray(
    TARGET_DEPTHS,
    dims="depth",
    name="depth"
)


# -----------------------------
# INTERPOLATION
# -----------------------------
print("Interpolating to Baltic grid...")

ds_interp = ds.interp(
    depth=target_depths,
    method="linear",
)

ds_interp = ds_interp.interp(
    latitude=m.latitude,
    longitude=m.longitude,
    method="linear",
)


# -----------------------------
# RENAME DEPTHS
# -----------------------------
print("Renaming depth levels...")

ds_interp = ds_interp.assign_coords(
    depth=("depth", RENAMED_DEPTHS)
)


# -----------------------------
# BUILD MASKS
# -----------------------------
print("Building masks...")

# Surface mask (2D)
surface_mask = (m.sla == 1)

# 3D ocean mask
mask_3d = (m.thetao == 1)


# -----------------------------
# APPLY MASKS
# -----------------------------
print("Applying masks...")

ds_out = xr.Dataset()

for var in ds_interp.data_vars:
    da = ds_interp[var]

    if var == "zos":
        # 2D mask
        masked = da.where(surface_mask)
    else:
        # 3D mask (auto-broadcast over init_time, lead_time)
        masked = da.where(mask_3d)

    # preserve attributes
    masked.attrs = da.attrs

    ds_out[var] = masked


# -----------------------------
# ASSIGN FINAL COORDS
# -----------------------------
print("Assigning final coordinates...")

ds_out = ds_out.assign_coords({
    "depth": ("depth", RENAMED_DEPTHS),
    "latitude": m.latitude,
    "longitude": m.longitude,
})


# -----------------------------
# SAVE OUTPUT
# -----------------------------
print(f"Saving to {OUT_PATH} ...")

with dask.config.set(scheduler="threads", num_workers=10):
    with ProgressBar():
        ds_out.to_zarr(OUT_PATH, mode="w")

print("Done.")