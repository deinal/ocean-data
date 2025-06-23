#!/usr/bin/env python3
"""
Convert CMEMS daily data to weekly means starting on Thursdays,
and generate a 20-year weekly climatology (1993–2012) repeated backward to 1940,
including all ISO weeks (1–53).
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

input_zarr  = "cmems_phy_combined.zarr"
output_zarr = "cmems_phy_weekly.zarr"

# ─────────────────────────────────────────────────────────────
# 0) Load daily data
# ─────────────────────────────────────────────────────────────
ds_daily = xr.open_zarr(input_zarr)

# check for nans
# da0 = ds_daily.thetao.isel(depth=0)
# nan_mask = da0.isnull().all(dim=("latitude", "longitude")).compute()
# bad_dates = da0.time[nan_mask].values
# for d in bad_dates:
#     print(np.datetime_as_string(d, unit="D"), flush=True)


# ─────────────────────────────────────────────────────────────
# 1) Weekly means starting Thursday→Wednesday, labeled on Thursday
# ─────────────────────────────────────────────────────────────
# Find the first Thursday on or after the start date:
first_thu = pd.date_range(ds_daily.time.values[0], periods=1, freq="W-THU")[0] # 1993-01-07
print(ds_daily, ds_daily.thetao.isel(depth=0).sel(time="1993-01-07").compute(), flush=True)

# Resample in 7-day bins, anchored at that Thursday, closed & labeled on the left edge
ds_weekly = ds_daily.resample(
    time="7D",
    origin=first_thu,
    closed="left",
    label="left"
).mean()
# Note first bin [1992-12-31, 1993-01-07) starts in 1992
print(ds_weekly, ds_weekly.thetao.isel(depth=0).isel(time=1).compute(), flush=True)

# ─────────────────────────────────────────────────────────────
# 2) Build 20-year weekly climatology (1993–2012), all ISO weeks 1–53
# ─────────────────────────────────────────────────────────────
# Select base period
ds_clim_base = ds_weekly.sel(time=slice("1993-01-01", "2012-12-31"))

# Extract ISO week numbers (1–53) for each timestamp
iso = ds_clim_base.time.dt.isocalendar()
ds_clim_base = ds_clim_base.assign_coords(week=iso.week)

# Group by week index and average
clim = ds_clim_base.groupby("week").mean("time")
print(clim, clim.thetao.isel(depth=0).isel(week=1).compute(), flush=True)
# clim now has dimension 'week' with labels 1 ... up to 53 (where present)

# ─────────────────────────────────────────────────────────────
# 3) Repeat climatology back to 1940
# ─────────────────────────────────────────────────────────────
# Build all Thursdays from 1940-01-01 through 1992-12-30 (Thu 31 Dec. 1992 already in ds)
ref_weeks = pd.date_range("1940-01-01", "1992-12-30", freq="W-THU")

n_weeks = clim.sizes["week"]  # 52 or 53

# Repeat the pattern: pick week‐indices in order, wrapping around
clim_repeated = clim.isel(week=(np.arange(len(ref_weeks)) % n_weeks))

# Rename 'week' → 'time' and assign the actual dates
clim_repeated = clim_repeated.rename({"week": "time"})
clim_repeated = clim_repeated.assign_coords(time=ref_weeks)

# ─────────────────────────────────────────────────────────────
# 4) Concatenate climatology + actual weekly data
# ─────────────────────────────────────────────────────────────
ds_combined = xr.concat([clim_repeated, ds_weekly], dim="time")
ds_combined = ds_combined.sortby("time")

# ─────────────────────────────────────────────────────────────
# 5) Rechunk for Zarr (10 weeks per chunk)
# ─────────────────────────────────────────────────────────────
chunk_map = {
    "time":      10,
    "depth":     ds_combined.sizes["depth"],
    "latitude":  ds_combined.sizes["latitude"],
    "longitude": ds_combined.sizes["longitude"],
}
ds_combined = ds_combined.chunk(chunk_map)

# ─────────────────────────────────────────────────────────────
# 6) Write out
# ─────────────────────────────────────────────────────────────
if os.path.exists(output_zarr):
    os.system(f"rm -rf {output_zarr}")

ds_combined.to_zarr(output_zarr, mode="w")
print(f"Saved weekly climatology+data to {output_zarr}")
