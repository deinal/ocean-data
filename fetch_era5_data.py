#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import earthkit.data as ekd
import xarray as xr
from time import sleep


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
parser.add_argument(
    "--start-year",
    type=int,
    default=2022,
    help="Start year (default: %(default)s)"
)
parser.add_argument(
    "--end-year",
    type=int,
    default=2022,
    help="End year (default: %(default)s)"
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

# Paths and date range
zarr_store = os.path.join(args.base_dir, "era5_data.zarr")
start_date = pd.Timestamp(f"{args.start_year}-01-01")
end_date   = pd.Timestamp(f"{args.end_year}-12-31")
all_months = pd.date_range(start=start_date, end=end_date, freq="MS")

# Determine resume point
if os.path.exists(zarr_store):
    existing = xr.open_zarr(zarr_store)
    last_time = pd.to_datetime(existing.time.values).max()
    resume_date = pd.Timestamp(last_time.year, last_time.month, 1) + pd.DateOffset(months=1)
    months_to_download = [d for d in all_months if d >= resume_date]
else:
    months_to_download = all_months

# Convert to 6-hourly daily variables
def convert_to_6h_variables(ds: xr.Dataset) -> xr.Dataset:
    """Convert 6-hourly ERA5 data to daily variables with *_00, *_06, *_12, *_18 suffixes."""
    hours = ds["time"].dt.hour
    dates = ds["time"].dt.floor("D")

    new_vars = {}
    for var in ds.data_vars:
        for hr in [0, 6, 12, 18]:
            sel = ds[var].where(hours == hr, drop=True)
            sel = sel.assign_coords(time=dates.where(hours == hr, drop=True))
            sel = sel.rename(var + f"_{hr:02d}")
            new_vars[sel.name] = sel

    out_ds = xr.Dataset(new_vars)
    out_ds = out_ds.groupby("time").first()
    return out_ds

# Download function 
def safe_download_and_decode(year: int, month_str: str) -> xr.Dataset:
    for attempt in range(10):
        try:
            ds = ekd.from_source(
                "cds",
                "reanalysis-era5-single-levels",
                variable=[
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                ],
                product_type="reanalysis",
                area=[70, 5, 50, 35],  # N,W,S,E
                year=str(year),
                month=month_str,
                day=[f"{day:02d}" for day in range(1, 32)],
                time=["00:00", "06:00", "12:00", "18:00"],
                format="netcdf"
            ).to_xarray()
            return ds
        except Exception as e:
            print(f"Download or decode failed (attempt {attempt+1}): {e}")
            sleep(5)
    raise RuntimeError(f"Failed to download {year}-{month_str} after multiple attempts.")

# Append to Zarr
def to_zarr_append(ds: xr.Dataset, store: str):
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")

# Download loop
for date in months_to_download:
    year = date.year
    month = date.month
    month_str = f"{month:02d}"
    print(f"Downloading {year}-{month_str}…", flush=True)

    # download 6-hourly ERA5 data for this month
    ds = safe_download_and_decode(year, month_str)

    # cleanup
    ds = ds.rename({"valid_time": "time"})
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds])

    # convert to daily variables with *_00, *_06, *_12, *_18
    ds_daily = convert_to_6h_variables(ds)

    # append to zarr store
    to_zarr_append(ds_daily, zarr_store)

    print(f"• appended {ds_daily.time.size} daily steps to {zarr_store}", flush=True)

    ds.close()
    ds_daily.close()
