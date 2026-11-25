"""Regrid ERA5 daily average Zarr to Baltic and global mask grids."""
import os
import argparse
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

LEXICON = {
    "2m_temperature": "sotemair",
    "10m_u_component_of_wind": "sowinu10",
    "10m_v_component_of_wind": "sowinv10",
    "mean_surface_downward_short_wave_radiation_flux": "sosudosw",
    "mean_surface_downward_long_wave_radiation_flux": "sosudolw",
    "mean_total_precipitation_rate": "sowaprec",
    "2m_dewpoint_temperature": "sod2m",
    "mean_sea_level_pressure": "somslpre",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="data",
    help="Base data directory",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=20,
    help="Number of workers for dask (default: %(default)s)",
)
args = parser.parse_args()

base_dir = args.base_dir
era5_path = os.path.join(base_dir, "arco_era5.zarr")

# (mask_zarr_name, output_zarr_name)
targets = [
    (os.path.join(base_dir, "baltic_mask.zarr"), os.path.join(base_dir, "baltic_era5.zarr")),
    (os.path.join(base_dir, "global_mask_1_4.zarr"), os.path.join(base_dir, "global_era5_1_4.zarr")),
    (os.path.join(base_dir, "global_mask_1.zarr"), os.path.join(base_dir, "global_era5_1.zarr")),
    (os.path.join(base_dir, "global_analysis_mask_1_4.zarr"), os.path.join(base_dir, "global_analysis_era5_1_4.zarr")),
]

# ERA5 daily average
era5_ds = xr.open_zarr(era5_path)
print(era5_ds.latitude, era5_ds.longitude, flush=True)

# Convert if ERA5 is 0–360 and target grids are -180–180
lon = era5_ds.longitude
longitude_new = np.where(
    era5_ds["longitude"] > 180, era5_ds["longitude"] - 360, era5_ds["longitude"]
)
era5_ds = era5_ds.assign_coords(longitude=longitude_new).sortby("latitude").sortby("longitude")

# Roll so the 180° column becomes the first column and assign it -180°,
# matching GLORYS grid.
era5_ds = era5_ds.roll(longitude=-1, roll_coords=True)
lon_vals = np.asarray(era5_ds.longitude.values, dtype=np.float64)
lon_vals[0] = -180.0
era5_ds = era5_ds.assign_coords(longitude=lon_vals)
print(era5_ds.latitude, era5_ds.longitude, flush=True)

for mask_path, out_path in targets:
    mask = xr.open_zarr(mask_path)
    print(mask, flush=True)
    mask_2d = mask.thetao.isel(depth=0).astype(bool)

    print(f"Interpolating to {os.path.basename(mask_path)} ...", flush=True)
    ds_r = era5_ds.interp(latitude=mask.latitude, longitude=mask.longitude)

    for var in ds_r.data_vars:
        ds_r[var] = ds_r[var].where(mask_2d)

    ds_r = ds_r.rename(LEXICON)
    ds_r = ds_r.chunk({"time": 1, "latitude": -1, "longitude": -1})

    # Check for NaNs inside mask after regrid (print time and variable)
    for var in ds_r.data_vars:
        nan_inside = (ds_r[var].isnull() & mask_2d).any(dim=["latitude", "longitude"])
        if nan_inside.any():
            times_with_nan = ds_r.time.values[nan_inside.compute().values]
            for t in np.atleast_1d(times_with_nan):
                print(f"  NaN inside mask: var={var} time={t}", flush=True)

    with dask.config.set(scheduler="threads", num_workers=args.num_workers):
        with ProgressBar():
            ds_r.to_zarr(out_path, mode="w")
    print(f"  -> {out_path}", flush=True)
