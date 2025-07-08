#!/usr/bin/env python3
import xarray as xr
import numpy as np


zarr_paths = [
    "cmems_phy_1.zarr",
    "cmems_phy_2.zarr",
    "cmems_phy_3.zarr",
    "cmems_phy_4.zarr",
    "cmems_phy_5.zarr",
    "cmems_phy_6.zarr",
    "cmems_phy_7.zarr",
]

# Open and concatenate along time
datasets = [xr.open_zarr(path, chunks={}) for path in zarr_paths]
ds_combined = xr.concat(datasets, dim="time")

# Sort by time
ds_combined = ds_combined.sortby("time")

# Rechunk
ds_combined = ds_combined.chunk({
    "time": 10,
    "depth": ds_combined.sizes["depth"],
    "latitude": ds_combined.sizes["latitude"],
    "longitude": ds_combined.sizes["longitude"],
})

# Save output
ds_combined.to_zarr("weekly_cmems_phy.zarr", mode="w")
print("Merged and saved to weekly_cmems_phy.zarr")
