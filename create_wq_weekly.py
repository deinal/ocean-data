#!/usr/bin/env python3
"""
Convert CMEMS daily data to weekly means starting on a user-specified weekday,
and generate a 20-year weekly climatology (1993–2012) repeated backward to 1940,
making sure the repeated climatology does not overlap the actual data.
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr


parser = argparse.ArgumentParser(
    description="Make weekly CMEMS means starting on chosen weekday"
)
parser.add_argument(
    "--weekday", "-w",
    type=int, choices=range(1,8), required=True,
    help="Weekday to anchor weeks on: 1=Mon, …, 7=Sun"
)
parser.add_argument(
    "--input", "-i",
    default="cmems_phy_combined.zarr",
    help="Input daily Zarr store"
)
args = parser.parse_args()
wd = args.weekday

# map 1–7 → pandas weekly freq code
wd_map = {
    1: "MON", 2: "TUE", 3: "WED", 4: "THU",
    5: "FRI", 6: "SAT", 7: "SUN"
}
wd_code = wd_map[wd]

input_zarr  = args.input
output_zarr = f"cmems_phy_{wd}.zarr"

# -------------------------------------------------------------------
# 1) Load daily data
# -------------------------------------------------------------------
ds_daily = xr.open_zarr(input_zarr)

# -------------------------------------------------------------------
# 2) Find the first desired weekday on or after the first time point
# -------------------------------------------------------------------
start = pd.to_datetime(ds_daily.time.values[0])
first_anchor = pd.date_range(start, periods=1, freq=f"W-{wd_code}")[0]

# -------------------------------------------------------------------
# 3) Resample to weekly means, closed & labeled on the left (anchor day)
# -------------------------------------------------------------------
ds_weekly = ds_daily.resample(
    time="7D",
    origin=first_anchor,
    closed="left",
    label="left"
).mean()

# -------------------------------------------------------------------
# 4) Build 20-year climatology (1993–2012) by ISO-week group
# -------------------------------------------------------------------
ds_clim_base = ds_weekly.sel(time=slice("1993-01-01", "2012-12-31"))
iso = ds_clim_base.time.dt.isocalendar()
ds_clim_base = ds_clim_base.assign_coords(week=iso.week)
clim = ds_clim_base.groupby("week").mean("time")
n_weeks = clim.sizes["week"]

# -------------------------------------------------------------------
# 5) Repeat climatology back to 1940, up to just before real data
# -------------------------------------------------------------------
# end of repeated segment = first real-week date minus one day
end_repeat = pd.to_datetime(ds_weekly.time.values[0]) - pd.Timedelta(days=1)

ref_weeks = pd.date_range("1940-01-01", end_repeat, freq=f"W-{wd_code}")
# tile the 1…n_weeks pattern across ref_weeks
idx = np.arange(len(ref_weeks)) % n_weeks
clim_repeat = clim.isel(week=idx)\
                  .rename({"week": "time"})\
                  .assign_coords(time=ref_weeks)

# -------------------------------------------------------------------
# 6) Concatenate repeated climatology + actual weekly data
# -------------------------------------------------------------------
ds_combined = xr.concat([clim_repeat, ds_weekly], dim="time").sortby("time")

ds_combined = ds_combined.fillna(0.0)
for vn in ds_combined.data_vars:
    ds_combined[vn] = ds_combined[vn].astype("float32")
print(ds_combined, flush=True)

# -------------------------------------------------------------------
# 7) Rechunk and write out
# -------------------------------------------------------------------
chunk_map = {
    "time":      10,
    "depth":     ds_combined.sizes["depth"],
    "latitude":  ds_combined.sizes["latitude"],
    "longitude": ds_combined.sizes["longitude"],
}
ds_combined = ds_combined.chunk(chunk_map)

if os.path.exists(output_zarr):
    os.system(f"rm -rf {output_zarr}")

ds_combined.to_zarr(output_zarr, mode="w")
print(f"Saved weekly climatology + data to {output_zarr}")
