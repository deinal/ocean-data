#!/usr/bin/env python3
import os
import argparse
import numpy as np
import xarray as xr
import copernicusmarine as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial import cKDTree

# Sphere radius (m) for Lambert
R_EARTH = 6371229.0

LAMBERT = ccrs.LambertConformal(
    central_longitude=20.0,
    central_latitude=60.0,
    standard_parallels=(60.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=R_EARTH, semiminor_axis=R_EARTH),
)
SRC_CRS = ccrs.PlateCarree()


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="data",
    help="Base output directory (default: %(default)s)"
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

TARGET_DEPTHS = [1, 9, 28, 47, 91]  # m

# Load static data
ds_bathy = cm.open_dataset(
    dataset_id="cmems_mod_bal_phy_my_static",
    dataset_part="bathy",
    variables=["mask"],
    maximum_depth=100,
)
depths = ds_bathy.depth.values
depth_indices = sorted(
    set(np.argmin(np.abs(depths - z)) for z in TARGET_DEPTHS)
)
ds_bathy = ds_bathy.isel(depth=depth_indices)
ds_bathy = ds_bathy.assign_coords(depth=np.round(ds_bathy.depth.values).astype(int))
print("Selected depth levels:", list(ds_bathy.depth.values))

ds_deptho = cm.open_dataset(
    dataset_id="cmems_mod_bal_phy_my_static",
    dataset_part="bathy",
    variables=["deptho"],
)
ds_deptho = ds_deptho.assign_coords(
    longitude=ds_bathy.longitude,
    latitude=ds_bathy.latitude,
)

ds_mdt = cm.open_dataset(
    dataset_id="cmems_mod_bal_phy_my_static",
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

# Lambert projection: x, y as positional static features (m)
lat = ds.latitude.values
lon = ds.longitude.values
lon2d, lat2d = np.meshgrid(lon, lat)
pts = LAMBERT.transform_points(SRC_CRS, lon2d, lat2d)
grid_x = pts[:, :, 0]
grid_y = pts[:, :, 1]
ds["x_coord"] = (("latitude", "longitude"), grid_x.astype(np.float32))
ds["y_coord"] = (("latitude", "longitude"), grid_y.astype(np.float32))

# Distance to coast (m) using Lambert projection
surface_mask = ds["mask"].isel(depth=0).values
land = ~surface_mask.astype(bool)
land_xy = np.column_stack([grid_x[land].ravel(), grid_y[land].ravel()])
ocean_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
tree = cKDTree(land_xy)
dist, _ = tree.query(ocean_xy, k=1)
coast_dist_2d = dist.reshape(grid_x.shape).astype(np.float32)
# Land points: 0 distance to self
coast_dist_2d[land] = 0.0
ds["coast_dist"] = (("latitude", "longitude"), coast_dist_2d)

# Apply surface mask so created variables are NaN over land like the rest
surface_mask = ds["mask"].isel(depth=0)
ds["x_coord"] = ds["x_coord"].where(surface_mask)
ds["y_coord"] = ds["y_coord"].where(surface_mask)
ds["coast_dist"] = ds["coast_dist"].where(surface_mask)

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

ds.coast_dist.plot()
plt.savefig(os.path.join("plots", "baltic_coast_dist.png"))
plt.close()

ds.x_coord.plot()
plt.savefig(os.path.join("plots", "baltic_x_coord.png"))
plt.close()

ds.y_coord.plot()
plt.savefig(os.path.join("plots", "baltic_y_coord.png"))
plt.close()

# Create feature mask
variable_depth = {
    "thetao": True,
    "so": True,
    "uo": True,
    "vo": True,
    "sla": False,
    "siconc": False,
    "sithick": False,
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
static_ds = ds[["deptho", "mdt", "x_coord", "y_coord", "coast_dist"]]
static_ds = static_ds.chunk({"latitude": -1, "longitude": -1})
static_ds.to_zarr(os.path.join(args.base_dir, "baltic_static.zarr"), mode="w")
print("Saved baltic static")