#!/usr/bin/env python3
import os
import xarray as xr
import copernicusmarine as cm

# Output Zarr store
out_store = "data/baltic/static/bathymetry.zarr"
os.makedirs(os.path.dirname(out_store), exist_ok=True)

# Load mean dynamic topography (MDT)
# ds_mdt = cm.open_dataset(
#     dataset_id="cmems_mod_bal_phy_my_static",
#     dataset_part="mdt",
#     variables=["mdt"],
# )

# # Load bathymetry & mask
# ds_bathy = cm.open_dataset(
#     dataset_id="cmems_mod_bal_phy_my_static",
#     dataset_part="bathy",
#     variables=["deptho", "mask"],
# )

# Merge into one Dataset
# ds = xr.merge([ds_mdt, ds_bathy])# 

# 1) Load the bathy & mask part
ds = cm.open_dataset(
    dataset_id    = "cmems_mod_bal_phy_my_static",
    dataset_part = "bathy",
    variables     = ["deptho", "mask"],
)

# 2) Build a boolean “keep” mask: True where we want to keep data
#    (i.e. lon ≥ 10 OR lat ≥ 54), broadcasted over the grid.
keep = (ds.longitude > 10) | (ds.latitude > 54)

# 3) Apply it: everywhere keep==False becomes NaN
ds = ds.where(keep)

# 4) (Optional) Re‐chunk so each variable is in a single chunk
chunk_map = {dim: size for dim, size in ds.sizes.items()}
ds = ds.chunk(chunk_map)

# 5) Write out a consolidated Zarr
ds.to_zarr(out_store, mode="w", consolidated=True)

print(f"Static bathymetry store (masked) written to {out_store}")