#!/usr/bin/env python3
import xarray as xr
import pandas as pd
import os

# Paths to input stores
parts = [
    "cmems_phy_19930101_19991231.zarr",
    "cmems_phy_20000101_20081231.zarr",
    "cmems_phy_20090101_20170906.zarr",
    "cmems_phy_20170907_20250501.zarr",
]

# Open each and select appropriate time slice
ds1 = xr.open_zarr(parts[0])
ds2 = xr.open_zarr(parts[1])
ds3 = xr.open_zarr(parts[2])
ds4 = xr.open_zarr(parts[3])

# Concatenate along time
ds_combined = xr.concat([ds1, ds2, ds3, ds4], dim="time")
ds_combined = ds_combined.sortby("time")

# Load static mask (surface layer)
static = xr.open_zarr("cmems_phy_static.zarr")
surface_mask = static["mask"].isel(depth=0)  # dims: latitude, longitude

# For sithick and siconc: wherever mask==1 and value is NaN, set to 0
for var in ("sithick", "siconc"):
    ds_combined[var] = ds_combined[var].where(~((surface_mask == 1) & ds_combined[var].isnull()), 0)

# Write out
out_store = "cmems_phy_combined.zarr"

chunk_map = {
    "time":      10,
    "depth":     ds_combined.sizes["depth"],
    "latitude":  ds_combined.sizes["latitude"],
    "longitude": ds_combined.sizes["longitude"],
}
ds_combined = ds_combined.chunk(chunk_map)

ds_combined.to_zarr(out_store, mode="w")
print(f"Wrote combined dataset to {out_store}")
