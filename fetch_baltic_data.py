#!/usr/bin/env python3
import os
import argparse
import calendar
import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
parser.add_argument(
    "--start",
    "-s",
    type=str,
    default="2020-01-01",
    help="Start date YYYY-MM-DD (default: %(default)s)"
)
parser.add_argument(
    "--end",
    "-e",
    type=str,
    default="2020-03-31",
    help="End date YYYY-MM-DD (default: %(default)s)"
)
args = parser.parse_args()

start_dt = pd.to_datetime(args.start)
end_dt = pd.to_datetime(args.end)
month_starts = pd.date_range(start=start_dt.replace(day=1), end=end_dt, freq="MS")


TARGET_DEPTHS = [1, 9, 28, 47, 91]  # m

ds_mask = xr.open_zarr(os.path.join(args.base_dir, "baltic_mask.zarr"))

def to_zarr_append(ds, store):
    """Write first chunk, then append each subsequent chunk along time."""
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")

for month_start in month_starts:
    year, month = month_start.year, month_start.month
    last_day = calendar.monthrange(year, month)[1]
    month_end = month_start + pd.Timedelta(days=last_day - 1, unit="D")
    fetch_start = max(month_start, start_dt)
    fetch_end = min(month_end, end_dt)
    start_time = fetch_start.strftime("%Y-%m-%dT00:00:00")
    end_time = fetch_end.strftime("%Y-%m-%dT23:59:59")

    print(f"[{year}-{month:02d}] fetching …")
    ds = cm.open_dataset(
        dataset_id     = "cmems_mod_bal_phy_my_P1D-m",
        variables      = ["thetao", "so", "uo", "vo", "sla", "siconc", "sithick"],
        start_datetime = start_time,
        end_datetime   = end_time,
        maximum_depth  = 100,
    )

    depths = ds.depth.values
    depth_indices = sorted(
        set(np.argmin(np.abs(depths - z)) for z in TARGET_DEPTHS)
    )
    ds = ds.isel(depth=depth_indices)
    ds = ds.assign_coords(depth=np.round(ds.depth.values).astype(int))
    print("Selected depth levels:", list(ds.depth.values))

    # Discard extra river
    ds = ds.where((ds.longitude > 10) | (ds.latitude > 54))

    # Crop to southern edge
    ds = ds.sel(latitude=slice(53.6, None))

    # Assign coordinates to match mask
    ds = ds.assign_coords(
        longitude=ds_mask.longitude,
        latitude=ds_mask.latitude,
    )

    # Chunk for writing
    if "depth" in ds.dims:
        ds = ds.chunk({"time": ds.sizes["time"], "depth": ds.sizes["depth"]})
    else:
        ds = ds.chunk({"time": ds.sizes["time"]})

    to_zarr_append(ds, os.path.join(args.base_dir, "baltic_data.zarr"))
    print(f"→ Appended {ds.sizes['time']} steps for {year}-{month:02d}\n", flush=True)