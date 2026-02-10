#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import copernicusmarine as cm
import cartopy.crs as ccrs


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
parser.add_argument(
    "--dist-threshold-km",
    type=float,
    default=150.0,
    help="Distance (km) from Baltic Sea surface points to include boundary points."
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

# Define Lambert Conformal projection
lambert_proj = ccrs.LambertConformal(
    central_longitude=20.0,
    central_latitude=60.0,
    standard_parallels=(60.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6371229.0, semiminor_axis=6371229.0),
)
src_crs = ccrs.PlateCarree()

# Load Baltic mask for distance-to-sea source points
baltic_grid = xr.open_zarr(os.path.join(args.base_dir, "baltic_mask.zarr"))
baltic_surface_mask = baltic_grid.thetao.isel(depth=0).astype(bool).values  # (nlat, nlon)
baltic_lats = baltic_grid.latitude.values
baltic_lons = baltic_grid.longitude.values

# Baltic 2D lon/lat
b_lat2d, b_lon2d = np.meshgrid(baltic_lats, baltic_lons, indexing="ij")

# Find the westernmost longitude among True surface pixels
true_lon_vals = b_lon2d[baltic_surface_mask]
westmost_lon = true_lon_vals.min()

# Map that longitude to the nearest longitude index in the 1D lon coordinate
lon_idx = int(np.argmin(np.abs(baltic_lons - westmost_lon)))

# Take the whole longitude "column" but only where the Baltic surface mask is True
col_mask = baltic_surface_mask[:, lon_idx]  # (nlat,)

mask_lat_points = baltic_lats[col_mask]
mask_lon_points = np.full(mask_lat_points.shape, baltic_lons[lon_idx], dtype=float)
print("Using westmost Baltic surface lon:", float(baltic_lons[lon_idx]), "with", mask_lat_points.size, "points")

# Project Baltic sea points to Lambert x/y (meters)
baltic_pts = lambert_proj.transform_points(src_crs, mask_lon_points, mask_lat_points)
baltic_x = baltic_pts[:, 0]
baltic_y = baltic_pts[:, 1]

# Load the bathy & mdt part (boundary static)
ds_bathy = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
    dataset_part="bathy",
    variables=["deptho"],
    minimum_longitude=5,
    maximum_longitude=9,
    minimum_latitude=55,
    maximum_latitude=60,
)

ds_mdt = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
    dataset_part="mdt",
    variables=["mdt"],
    minimum_longitude=5,
    maximum_longitude=9,
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
print("Saved static boundary data")

# Plots
os.makedirs("plots", exist_ok=True)
ds.mdt.plot()
plt.savefig(os.path.join("plots", "boundary_mdt.png"))
plt.close()

ds.deptho.plot()
plt.savefig(os.path.join("plots", "boundary_deptho.png"))
plt.close()

# Load the mask
ds_mask = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
    dataset_part="bathy",
    variables=["mask"],
    minimum_longitude=5,
    maximum_longitude=9,
    minimum_latitude=55,
    maximum_latitude=60,
    maximum_depth=110,
)
glorys_depths = ds_mask.depth.values

# Find nearest GLORYS depth for each Baltic depth
baltic_depths = baltic_grid.depth.values
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

# Compute distance-based boundary mask
bnd_lats = ds_mask.latitude.values
bnd_lons = ds_mask.longitude.values

# 2D boundary lon/lat
lon2d, lat2d = np.meshgrid(bnd_lons, bnd_lats)  # shape (nlat, nlon)

# Project boundary grid to Lambert x/y
bnd_pts = lambert_proj.transform_points(src_crs, lon2d, lat2d)
bnd_x = bnd_pts[:, :, 0]
bnd_y = bnd_pts[:, :, 1]

# Distance-based mask (union-of-discs)
mask_bnd = np.zeros_like(bnd_x, dtype=np.uint8)
r2 = (args.dist_threshold_km * 1000.0) ** 2

for xi, yi in zip(baltic_x, baltic_y):
    dist2 = (bnd_x - xi) ** 2 + (bnd_y - yi) ** 2
    mask_bnd[dist2 <= r2] = 1

mask_dist_da = xr.DataArray(
    mask_bnd,
    dims=("latitude", "longitude"),
    coords={"latitude": bnd_lats, "longitude": bnd_lons},
    name="mask_dist",
)

glorys_mask_bool = ds_mask.mask.astype(bool)
dist_mask_bool = mask_dist_da.astype(bool)
boundary_dist_mask = (glorys_mask_bool & dist_mask_bool).astype(int)

# Plot distance mask
boundary_dist_mask.isel(depth=0).plot()
plt.title("Boundary distance mask (surface)")
plt.savefig(os.path.join("plots", "boundary_dist_mask_surface.png"))
plt.close()

boundary_dist_mask.isel(depth=-1).plot()
plt.title("Boundary distance mask (deepest)")
plt.savefig(os.path.join("plots", "boundary_dist_mask_deep.png"))
plt.close()


# Create feature mask
variable_depth = {
    "zos": False,
    "thetao": True,
    "so": True,
    "uo": True,
    "vo": True,
}

# Build output dataset
feature_mask_dict = {}

for var, has_depth in variable_depth.items():
    if has_depth:
        feature_mask_dict[var] = boundary_dist_mask.copy(deep=False)
    else:
        feature_mask_dict[var] = boundary_dist_mask.isel(depth=0).copy(deep=False)

# Combine as Dataset
feature_mask_ds = xr.Dataset(data_vars=feature_mask_dict)

# Save to Zarr
chunk_map = {dim: size for dim, size in feature_mask_ds.sizes.items()}
feature_mask_ds = feature_mask_ds.chunk(chunk_map)
feature_mask_ds.to_zarr(os.path.join(args.base_dir, "boundary_mask.zarr"), mode="w")
print("Saved boundary mask")
