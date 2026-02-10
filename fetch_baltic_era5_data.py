#!/usr/bin/env python3
"""Fetch ERA5 6-hourly data and store daily averages (mean over 00, 06, 12, 18)."""
import os
import argparse
import pandas as pd
import earthkit.data as ekd
import xarray as xr
from time import sleep


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
    default=2010,
    help="Start year (default: %(default)s)"
)
parser.add_argument(
    "--end-year",
    type=int,
    default=2013,
    help="End year (default: %(default)s)"
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

zarr_store = os.path.join(args.base_dir, "era5_data_daily_avg.zarr")
start_date = pd.Timestamp(f"{args.start_year}-01-01")
end_date   = pd.Timestamp(f"{args.end_year}-12-31")
all_months = pd.date_range(start=start_date, end=end_date, freq="MS")

if os.path.exists(zarr_store):
    existing = xr.open_zarr(zarr_store)
    last_time = pd.to_datetime(existing.time.values).max()
    resume_date = pd.Timestamp(last_time.year, last_time.month, 1) + pd.DateOffset(months=1)
    months_to_download = [d for d in all_months if d >= resume_date]
else:
    months_to_download = all_months


def to_daily_avg(ds: xr.Dataset) -> xr.Dataset:
    """Average 6-hourly ERA5 over each day (00, 06, 12, 18)."""
    return ds.resample(time="1D").mean()


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
                area=[70, 5, 50, 35],
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


def to_zarr_append(ds: xr.Dataset, store: str):
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")


max_retries = 5
for date in months_to_download:
    year = date.year
    month = date.month
    month_str = f"{month:02d}"
    print(f"Downloading {year}-{month_str}…", flush=True)

    for attempt in range(max_retries):
        try:
            ds = safe_download_and_decode(year, month_str)

            ds = ds.rename({"valid_time": "time"})
            ds = ds.drop_vars([v for v in ("number", "expver") if v in ds])
            ds = ds.sortby("latitude")
            ds = ds.sortby("longitude")

            ds_daily = to_daily_avg(ds)
            # Load into memory so zarr write does not read from NetCDF after it may be closed
            ds_daily = ds_daily.load()
            ds.close()

            to_zarr_append(ds_daily, zarr_store)

            print(f"• appended {ds_daily.time.size} daily avg steps to {zarr_store}", flush=True)
            break
        except RuntimeError as e:
            if "NetCDF" in str(e) or "HDF" in str(e):
                if attempt < max_retries - 1:
                    print(f"• NetCDF/HDF error (attempt {attempt + 1}/{max_retries}), retrying…", flush=True)
                    sleep(10)
                else:
                    raise
            else:
                raise
