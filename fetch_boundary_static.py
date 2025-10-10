#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import copernicusmarine as cm


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

# Load the bathy & mask part
ds_bathy = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
    dataset_part="bathy",
    variables=["deptho"],
    minimum_longitude=5,
    maximum_longitude=10,
    minimum_latitude=55,
    maximum_latitude=60,
)

ds_mdt = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
    dataset_part="mdt",
    variables=["mdt"],
    minimum_longitude=5,
    maximum_longitude=10,
    minimum_latitude=55,
    maximum_latitude=60,
)
ds_mdt = ds_mdt.assign_coords(
    longitude=ds_bathy.longitude,
    latitude=ds_bathy.latitude,
)

ds = xr.merge([ds_bathy, ds_mdt])

# discard interior data
ds = ds.where(~((ds.longitude > 9) & (ds.latitude < 56)))

# re‐chunk
chunk_map = {dim: size for dim, size in ds.sizes.items()}
ds = ds.chunk(chunk_map)

# Write to zarr
ds.to_zarr(os.path.join(args.base_dir, "boundary_static.zarr"), mode="w", consolidated=True)
print(f"Saved static boundary data")

# Mean Dynamic Topography
os.makedirs("plots", exist_ok=True)
ds.mdt.plot()
plt.savefig(os.path.join("plots", "boundary_mdt.png"))
plt.close()

# Bathymetry
ds.deptho.plot()
plt.savefig(os.path.join("plots", "boundary_deptho.png"))
plt.close()

# Load the mask
ds_mask = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
    dataset_part="bathy",
    variables=["mask"],
    minimum_longitude=5,
    maximum_longitude=10,
    minimum_latitude=55,
    maximum_latitude=60,
    maximum_depth=110,
)
glorys_depths = ds_mask.depth.values

# Load baltic mask to get target depth levels
baltic_grid = xr.open_zarr(os.path.join(args.base_dir, "baltic_mask.zarr"))
baltic_depths = baltic_grid.depth.values

# Find nearest GLORYS depth for each Baltic depth
closest_idxs = np.abs(glorys_depths[:, None] - baltic_depths[None, :]).argmin(axis=0)
closest_depths = glorys_depths[closest_idxs]

# Round to nearest integer and get unique sorted levels
unique_depths = np.unique(np.round(closest_depths).astype(int))
print("Selected depth levels:", unique_depths)

# Subset the depth dimension to the closest matching GLORYS depths
ds_mask = ds_mask.sel(depth=glorys_depths[closest_idxs])

# Replace the depth coordinate values with the rounded integers
ds_mask = ds_mask.assign_coords(depth=unique_depths)

# Mask out redundant baltic data in the boundary
ds_mask = ds_mask.where(~((ds_mask.longitude > 9) & (ds_mask.latitude < 56)), other=0)

# Plot surface mask
ds_mask.mask.isel(depth=0).plot()
plt.savefig(os.path.join("plots", "boundary_mask.png"))
plt.close()

# Create feature mask
variable_depth = {
    "mlotst": False,
    "siconc": False,
    "sithick": False,
    "zos": False,
    "so": True,
    "thetao": True,
    "uo": True,
    "vo": True,
}

# Selected depth levels
mask_selection = ds_mask["mask"]

# Surface mask
surface_mask = ds_mask["mask"].isel(depth=0)

# Build output dataset
feature_mask_dict = {}

for var, has_depth in variable_depth.items():
    if has_depth:
        feature_mask_dict[var] = mask_selection.copy(deep=False)
    else:
        feature_mask_dict[var] = surface_mask.copy(deep=False)

# Combine as Dataset
feature_mask_ds = xr.Dataset(data_vars=feature_mask_dict)

# Save to Zarr
chunk_map = {dim: size for dim, size in feature_mask_ds.sizes.items()}
feature_mask_ds = feature_mask_ds.chunk(chunk_map)
feature_mask_ds.to_zarr(os.path.join(args.base_dir, "boundary_mask.zarr"), mode="w")
print("Saved boundary mask")
