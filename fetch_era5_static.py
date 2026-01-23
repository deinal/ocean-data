import os
import argparse
import numpy as np
import xarray as xr
import earthkit.data as ekd
import xarray_regrid
import matplotlib.pyplot as plt
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
    default=25.0,
    help="Distance (km) from Baltic Sea surface points to include atmospheric points."
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

# Load input datasets
era5_ds = xr.open_dataset(os.path.join(args.base_dir, "era5_data.zarr"))
baltic_grid = xr.open_zarr(os.path.join(args.base_dir, "baltic_mask.zarr"))
target_lat = baltic_grid.latitude
target_lon = baltic_grid.longitude
surface_mask = baltic_grid.thetao.isel(depth=0).astype(bool)

# Define Lambert Conformal projection
lambert_proj = ccrs.LambertConformal(
    central_longitude=20.0,
    central_latitude=60.0,
    standard_parallels=(60.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6371229.0, semiminor_axis=6371229.0),
)
src_crs = ccrs.PlateCarree()

# Fetch static variables
ds_geo = ekd.from_source(
    "cds",
    "reanalysis-era5-land",
    variable="geopotential",
    area=[70, 5, 50, 35],
    data_format="netcdf",
    download_format="unarchived",
).to_xarray()
ds_geo = ds_geo.isel(time=0, drop=True)
ds_geo["geopotential_height"] = ds_geo.z / 9.80665
ds_geo["geopotential_height"].attrs["units"] = "m"
ds_geo = ds_geo.drop_vars("z")

# Regrid to ERA5 grid
ds_re = ds_geo.regrid.conservative(
    era5_ds,
    latitude_coord="latitude",
    skipna=True,
)

# Make 2D lat/lon grids from Baltic dataset
lat2d, lon2d = np.meshgrid(target_lat.values, target_lon.values, indexing="ij")

# Select points where mask == True
mask_indices = surface_mask.values  # shape (nlat, nlon)
mask_lat_points = lat2d[mask_indices]
mask_lon_points = lon2d[mask_indices]

# Project to Lambert x/y
baltic_pts = lambert_proj.transform_points(
    src_crs, mask_lon_points, mask_lat_points
)
baltic_x = baltic_pts[:, 0]
baltic_y = baltic_pts[:, 1]

# Project ERA5 grid to x/y
era5_lats = ds_re.latitude.values
era5_lons = ds_re.longitude.values
lon2d, lat2d = np.meshgrid(era5_lons, era5_lats)

era5_pts = lambert_proj.transform_points(src_crs, lon2d, lat2d)
era5_x = era5_pts[:, :, 0]
era5_y = era5_pts[:, :, 1]

# Compute distance-based mask in projected space
mask_era5 = np.zeros_like(era5_x, dtype=np.uint8)

# Compute min distance from each ERA5 point to any Baltic point
for xi, yi in zip(baltic_x, baltic_y):
    dist2 = (era5_x - xi) ** 2 + (era5_y - yi) ** 2
    mask_era5[dist2 <= (args.dist_threshold_km * 1000.0) ** 2] = 1

mask_da = xr.DataArray(
    mask_era5,
    dims=("latitude", "longitude"),
    coords={"latitude": era5_lats, "longitude": era5_lons},
    name="mask",
)
mask_da.attrs["description"] = "Binary mask"

# Add to regridded dataset
ds_re["mask"] = mask_da

# sort
ds_re = ds_re.sortby("latitude")
ds_re = ds_re.sortby("longitude")

# Save to Zarr
ds_re.to_zarr(os.path.join(args.base_dir, "era5_static.zarr"), mode="w")
print(f"Saved ERA5 static data")

# Plot mask and geopotential height
os.makedirs("plots", exist_ok=True)

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6))
mask_da.plot(ax=ax)
ax.coastlines(resolution="10m", linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
gl.xlines = False
gl.ylines = False
plt.savefig(os.path.join("plots", "era5_mask.png"))
plt.close()

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6))
ds_re.geopotential_height.plot(ax=ax)
ax.coastlines(resolution="10m", linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
gl.xlines = False
gl.ylines = False
plt.savefig(os.path.join("plots", "era5_geopotential_height.png"))
plt.close()
