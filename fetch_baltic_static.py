#!/usr/bin/env python3
import os
import numpy as np
import copernicusmarine as cm

# Output zarr store
out_store = "data/baltic/static/bathymetry.zarr"
os.makedirs(os.path.dirname(out_store), exist_ok=True)

# Load the bathy & mask part
ds = cm.open_dataset(
    dataset_id    = "cmems_mod_bal_phy_anfc_static",
    dataset_part  = "bathy",
    variables     = ["deptho", "mask"],
    maximum_depth = 204,
).isel(depth=slice(None, None, 3))

# Discard extra river
keep = (ds.longitude > 10) | (ds.latitude > 54)
ds["mask"] = ds["mask"].where(keep, other=0)
ds["deptho"] = ds["deptho"].where(keep, other=np.nan)

# Crop to southern edge
ds = ds.sel(latitude=slice(53.6, None))

# Create 2D lat/lon grids
lat = ds.latitude.values
lon = ds.longitude.values
lon2d, lat2d = np.meshgrid(lon, lat)  # shape: (lat, lon)

# Add to dataset
ds["lat2d"] = (("latitude", "longitude"), lat2d)
ds["lon2d"] = (("latitude", "longitude"), lon2d)

# Re‐chunk
chunk_map = {dim: size for dim, size in ds.sizes.items()}
ds = ds.chunk(chunk_map)

# Write to zarr
ds.to_zarr(out_store, mode="w", consolidated=True)

print(f"Static bathymetry store (masked) written to {out_store}")
