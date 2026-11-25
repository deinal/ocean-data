#!/usr/bin/env python3
import xarray as xr
import numpy as np
from pathlib import Path


import dask
from dask.diagnostics import ProgressBar

# -----------------------------
# PATHS
# -----------------------------
IN_PATH = "data/ifs2024_daily_means.zarr"
# OUT_PATH = "data/ifs_forecasts.zarr"
# MASK_PATH = "data/global_mask_1_4.zarr"

OUT_PATH = "data/ifs_analysis_forecasts.zarr"
MASK_PATH = "data/global_analysis_mask_1_4.zarr"

# -----------------------------
# VARIABLE MAP
# -----------------------------
LEXICON = {
    "2m_temperature": "sotemair",
    "10m_u_component_of_wind": "sowinu10",
    "10m_v_component_of_wind": "sowinv10",
    "ssrd": "sosudosw",
    "strd": "sosudolw",
    "tprate": "sowaprec",
    "2d": "sod2m",
    "mean_sea_level_pressure": "somslpre",
}

# -----------------------------
# LOAD DATA
# -----------------------------
print("Opening dataset...", flush=True)
ds = xr.open_zarr(IN_PATH)

# -----------------------------
# FIX LONGITUDE (0–360 → -180–180)
# -----------------------------
print("Fixing longitude...", flush=True)

lon_new = np.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
ds = ds.assign_coords(longitude=lon_new)

# sort for monotonic coords
ds = ds.sortby("latitude").sortby("longitude")

# roll so -180 is first
ds = ds.roll(longitude=-1, roll_coords=True)
lon_vals = ds.longitude.values.astype(np.float64)
lon_vals[0] = -180.0
ds = ds.assign_coords(longitude=lon_vals)

# -----------------------------
# LOAD MASK
# -----------------------------
print("Loading mask...", flush=True)
mask = xr.open_zarr(MASK_PATH)

mask_2d = mask.thetao.isel(depth=0).astype(bool)

# -----------------------------
# INTERPOLATE
# -----------------------------
print("Interpolating to mask grid...", flush=True)

ds_interp = ds.interp(
    latitude=mask.latitude,
    longitude=mask.longitude,
)

# -----------------------------
# APPLY MASK
# -----------------------------
print("Applying mask...", flush=True)

for var in ds_interp.data_vars:
    ds_interp[var] = ds_interp[var].where(mask_2d)

# -----------------------------
# RENAME VARIABLES
# -----------------------------
print("Renaming variables...", flush=True)

rename_dict = {k: v for k, v in LEXICON.items() if k in ds_interp.data_vars}
ds_interp = ds_interp.rename(rename_dict)

# -----------------------------
# CHUNKING (important for Zarr)
# -----------------------------
print("Chunking...", flush=True)

ds_interp = ds_interp.chunk({
    "init_time": 1,
    "lead_time": 1,
    "latitude": -1,
    "longitude": -1,
})

# -----------------------------
# WRITE OUTPUT
# -----------------------------
print(f"Writing to {OUT_PATH}...", flush=True)

with dask.config.set(scheduler="threads", num_workers=20):
    with ProgressBar():
        ds_interp.to_zarr(OUT_PATH, mode="w")

print("Done.", flush=True)