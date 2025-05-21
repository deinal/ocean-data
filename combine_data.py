import os
import xarray as xr
import numpy as np
from glob import glob

years = [2000, 2001, 2002]
era5_path = "data/baltic/raw/era5"
reanalysis_path = "data/baltic/raw/reanalysis"
static_grid_path = "data/baltic/static/bathymetry.zarr"
output_path = "data/baltic/preprocessed"

target_grid = xr.open_zarr(static_grid_path)
target_lat = target_grid.latitude
target_lon = target_grid.longitude
surface_mask = target_grid.mask.isel(depth=0).astype(bool)

def interpolate_to_grid(ds, target_lat, target_lon):
    return ds.interp(latitude=target_lat, longitude=target_lon, method="linear")

def get_depth_mask(ds, max_depth=150, step=3):
    if "depth" in ds.coords:
        depth_vals = ds.depth.values
        depth_mask = (depth_vals <= max_depth)
        return depth_vals[depth_mask][::step]
    return None

def process_3d_subset(ds):
    selected_depths = get_depth_mask(ds)
    if selected_depths is not None:
        ds = ds.sel(depth=selected_depths)
    return ds

def process_wave(ds):
    # Subset wave dataset to match the surface_mask grid (has 1 extra pixel in lat and lon)
    ds = ds.sel(
        latitude=ds.latitude[:-1], # drop max latitude
        longitude=ds.longitude[1:] # drop min longitude
    )
    
    ds = ds.assign_coords(
        latitude=surface_mask.latitude,
        longitude=surface_mask.longitude
    )

    # Replace NaNs with 0 where the grid is sea
    ds["VHM0"] = ds["VHM0"].where(~surface_mask | ~np.isnan(ds["VHM0"]), 0)
    ds["VMDR"] = ds["VMDR"].where(~surface_mask | ~np.isnan(ds["VMDR"]), 0)
    ds["VTM10"] = ds["VTM10"].where(~surface_mask | ~np.isnan(ds["VTM10"]), 0)

    ds = ds.rename({"VHM0": "swh"})  # Rename significant wave height to swh
    ds = ds.rename({"VTM10": "mwp"})  # Rename mean wave period to mwp

    # Convert wave direction from degrees to radians
    angle_rad = np.deg2rad(ds["VMDR"])

    # Convert wave direction to components
    ds["uw"] = np.sin(angle_rad)  # u component
    ds["vw"] = np.cos(angle_rad)  # v component
    ds = ds.drop_vars("VMDR")

    # Recover wave direction VMDR, or MWD, from u, v components
    # theta_from_rad = np.arctan2(ds["uw"], ds["vw"])  # radians
    # ds["mwd"] = np.rad2deg(theta_from_rad) % 360  # wrap to [0, 360)

    return ds

def merge_daily_zarr(input_dir, output_file, target_lat=None, target_lon=None, include_vars=None, years=None, custom_process=None):
    all_files = sorted(glob(os.path.join(input_dir, "*.zarr")))

    if years is not None:
        all_files = [f for f in all_files if any(str(y) in os.path.basename(f) for y in years)]

    first = True

    for f in all_files:
        ds = xr.open_zarr(f)

        # Optional filtering of variables
        if include_vars:
            ds = ds[[v for v in include_vars if v in ds.data_vars]]

        # Optional interpolation
        if target_lat is not None and target_lon is not None:
            ds = interpolate_to_grid(ds, target_lat, target_lon)

        # Optional custom processing
        if custom_process:
            ds = custom_process(ds)
        
        # Make sure time is a proper dimension, not just a scalar coordinate
        if "time" in ds.coords and "time" not in ds.dims:
            ds = ds.expand_dims("time")

        # Write one day at a time to output Zarr
        if first:
            ds.to_zarr(output_file, mode="w")
        else:
            ds.to_zarr(output_file, mode="a", append_dim="time")
        first = False


merge_daily_zarr(
    input_dir=era5_path,
    output_file=os.path.join(output_path, "era5.zarr"),
    target_lat=target_lat,
    target_lon=target_lon,
    years=years
)

merge_daily_zarr(
    input_dir=os.path.join(reanalysis_path, "bal_phy"),
    output_file=os.path.join(output_path, "bal_phy.zarr"),
    years=years,
    custom_process=process_3d_subset,
)

merge_daily_zarr(
    input_dir=os.path.join(reanalysis_path, "bal_bgc"),
    output_file=os.path.join(output_path, "bal_bgc.zarr"),
    include_vars=["chl", "o2", "zsd"],
    years=years,
    custom_process=process_3d_subset
)

merge_daily_zarr(
    input_dir=os.path.join(reanalysis_path, "bal_wav"),
    output_file=os.path.join(output_path, "bal_wav.zarr"),
    years=years,
    custom_process=process_wave
)
