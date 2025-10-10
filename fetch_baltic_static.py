#!/usr/bin/env python3
import os
import argparse
import numpy as np
import xarray as xr
import copernicusmarine as cm
import matplotlib.pyplot as plt


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

# Load static data
ds_bathy = cm.open_dataset(
    dataset_id="cmems_mod_bal_phy_anfc_static",
    dataset_part="bathy",
    variables=["mask"],
    maximum_depth=104,
).isel(depth=slice(None, None, 3))

rounded_depths = np.round(ds_bathy.depth.values).astype(int)
ds_bathy = ds_bathy.assign_coords(depth=rounded_depths)
print("Selected depth levels:", rounded_depths)

ds_deptho = cm.open_dataset(
    dataset_id="cmems_mod_bal_phy_anfc_static",
    dataset_part="bathy",
    variables=["deptho"],
)

ds_mdt = cm.open_dataset(
    dataset_id="cmems_mod_bal_phy_anfc_static",
    dataset_part="mdt",
    variables=["mdt"],
)

ds_mdt = ds_mdt.assign_coords(
    longitude=ds_bathy.longitude,
    latitude=ds_bathy.latitude,
)

ds = xr.merge([ds_bathy, ds_deptho, ds_mdt])

# Discard extra river
keep = (ds.longitude > 10) | (ds.latitude > 54)
ds["mask"] = ds["mask"].where(keep, other=0)
ds["deptho"] = ds["deptho"].where(keep, other=np.nan)
ds["mdt"] = ds["mdt"].where(keep, other=np.nan)

# Crop to southern edge
ds = ds.sel(latitude=slice(53.6, None))

# Plot static data
os.makedirs("plots", exist_ok=True)

ds.mask.isel(depth=0).plot()
plt.savefig(os.path.join("plots", "baltic_mask.png"))
plt.close()

ds.deptho.plot()
plt.savefig(os.path.join("plots", "baltic_deptho.png"))
plt.close()

ds.mdt.plot()
plt.savefig(os.path.join("plots", "baltic_mdt.png"))
plt.close()

# Create feature mask
variable_depth = {
    "mlotst": False,
    "siconc": False,
    "sithick": False,
    "sla": False,
    "mwp": False,
    "swh": False,
    "uw": False,
    "vw": False,
    "zsd": False,
    "so": True,
    "thetao": True,
    "uo": True,
    "vo": True,
    "chl": True,
}

# Selected depth levels
mask_selection = ds["mask"]

# Surface mask
surface_mask = ds["mask"].isel(depth=0)

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
feature_mask_ds = feature_mask_ds.chunk({"depth": -1, "latitude": -1, "longitude": -1})
feature_mask_ds.to_zarr(os.path.join(args.base_dir, "baltic_mask.zarr"), mode="w")
print("Saved baltic mask")

# Store static variables
static_ds = ds[["deptho", "mdt"]]
static_ds = static_ds.chunk({"latitude": -1, "longitude": -1})
static_ds.to_zarr(os.path.join(args.base_dir, "baltic_static.zarr"), mode="w")
print("Saved baltic static")