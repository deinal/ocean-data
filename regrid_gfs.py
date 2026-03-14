"""Regrid GFS forecasts to Baltic and global mask grids."""
import os
import argparse
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--base-dir",
    type=str,
    default="data",
    help="Base data directory",
)
parser.add_argument(
    "--gfs",
    type=str,
    default="gfs_forecasts.nc",
    help="GFS forecast file name under base-dir (default: gfs_forecasts.nc)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=20,
    help="Number of workers for dask (default: %(default)s)",
)
args = parser.parse_args()

base_dir = args.base_dir
gfs_path = os.path.join(base_dir, args.gfs)

# (mask_zarr_name, output_zarr_name)
targets = [
    (os.path.join(base_dir, "baltic_mask.zarr"), os.path.join(base_dir, "baltic_gfs.zarr")),
    (os.path.join(base_dir, "global_mask_1_4.zarr"), os.path.join(base_dir, "global_gfs_1_4.zarr")),
    (os.path.join(base_dir, "global_mask_1.zarr"), os.path.join(base_dir, "global_gfs_1.zarr")),
]

# Open GFS
gfs_ds = xr.open_dataset(gfs_path)

print(gfs_ds.latitude, gfs_ds.longitude, flush=True)

for mask_path, out_path in targets:
    mask = xr.open_zarr(mask_path)
    print(mask, flush=True)
    mask_2d = mask.thetao.isel(depth=0).astype(bool)

    print(f"Interpolating GFS to {os.path.basename(mask_path)} ...", flush=True)
    ds_r = gfs_ds.interp(latitude=mask.latitude, longitude=mask.longitude)

    for var in ds_r.data_vars:
        ds_r[var] = ds_r[var].where(mask_2d)

    # Chunk by forecast dims + space (no time)
    ds_r = ds_r.chunk(
        {"first_day_datetime": 1, "lead_day_index": 1, "latitude": -1, "longitude": -1}
    )

    # NaN check inside mask (over forecast indices)
    for var in ds_r.data_vars:
        nan_inside = (ds_r[var].isnull() & mask_2d).any(dim=["latitude", "longitude"])
        if nan_inside.any():
            bad = nan_inside.compute()
            for i, j in zip(*np.where(bad.values), strict=True):
                fd = ds_r.first_day_datetime.values[i]
                ld = ds_r.lead_day_index.values[j]
                print(f"  NaN inside mask: var={var} first_day={fd} lead_day_index={ld}", flush=True)

    with dask.config.set(scheduler="threads", num_workers=args.num_workers):
        with ProgressBar():
            ds_r.to_zarr(out_path, mode="w")
    print(f"  -> {out_path}", flush=True)
