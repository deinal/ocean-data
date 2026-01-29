import argparse
import os
from time import sleep
import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid


def convert_to_6h_variables(ds: xr.Dataset) -> xr.Dataset:
    """Convert 6-hourly ERA5 data to daily variables with *_00, *_06, *_12, *_18 suffixes."""
    hours = ds["time"].dt.hour
    dates = ds["time"].dt.floor("D")

    new_vars: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        for hr in (0, 6, 12, 18):
            sel = ds[var].where(hours == hr, drop=True)
            sel = sel.assign_coords(time=dates.where(hours == hr, drop=True))
            sel = sel.rename(var + f"_{hr:02d}")
            new_vars[sel.name] = sel

    out_ds = xr.Dataset(new_vars)
    out_ds = out_ds.groupby("time").first()
    return out_ds


def safe_download_and_decode(
    year: int,
    month_str: str,
    variables: list[str],
) -> xr.Dataset:
    for attempt in range(10):
        try:
            ds = (
                ekd.from_source(
                    "cds",
                    "reanalysis-era5-single-levels",
                    variable=variables,
                    product_type="reanalysis",
                    year=str(year),
                    month=month_str,
                    day=[f"{day:02d}" for day in range(1, 32)],
                    time=["00:00", "06:00", "12:00", "18:00"],
                    format="netcdf",
                )
                .to_xarray()
            )
            return ds
        except Exception as e:
            print(f"Download or decode failed (attempt {attempt+1}): {e}", flush=True)
            sleep(5)
    raise RuntimeError(f"Failed to download {year}-{month_str} after multiple attempts.")


def to_zarr_append(ds: xr.Dataset, store: str):
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
        help="Base output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format (inclusive)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format (inclusive)",
    )
    args = parser.parse_args()

    fpath = os.path.join(args.base_dir, "global_era5_data.zarr")

    start_date = pd.to_datetime(args.start).normalize()
    end_date = pd.to_datetime(args.end).normalize()

    # Target grid
    lat_target = np.arange(90.0, -91.0, -1.5)
    lon_target = np.arange(0.0, 360.0, 1.5)
    ds_target_grid = xr.Dataset(
        coords={
            "latitude": ("latitude", lat_target),
            "longitude": ("longitude", lon_target),
        }
    )

    # Variables
    variables = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
    ]

    # Download per month
    all_months = pd.date_range(start=start_date, end=end_date, freq="MS")

    # Determine resume point if Zarr exists
    if os.path.exists(fpath):
        existing = xr.open_zarr(fpath)
        last_time = pd.to_datetime(existing.time.values).max()
        resume_date = pd.Timestamp(last_time.year, last_time.month, 1) + pd.DateOffset(
            months=1
        )
        months_to_download = [d for d in all_months if d >= resume_date]
    else:
        months_to_download = list(all_months)

    for date in months_to_download:
        year = date.year
        month_str = f"{date.month:02d}"
        print(f"Downloading {year}-{month_str}…", flush=True)

        ds = safe_download_and_decode(year, month_str, variables)

        if "valid_time" in ds:
            ds = ds.rename({"valid_time": "time"})
        ds = ds.drop_vars([v for v in ("number", "expver") if v in ds], errors="ignore")

        # sort for regrid
        ds = ds.sortby("latitude")
        ds = ds.sortby("longitude")

        ds_daily = convert_to_6h_variables(ds)

        # trim to requested interval (month requests may include extra days)
        ds_daily = ds_daily.sel(time=slice(start_date, end_date))
        if ds_daily.time.size == 0:
            print(f"• no days in range for {year}-{month_str}; skipping", flush=True)
            ds.close()
            ds_daily.close()
            continue

        # conservative regrid to target
        ds_re = ds_daily.regrid.conservative(
            ds_target_grid,
            latitude_coord="latitude",
            skipna=True,
        )

        chunk_map = {dim: size for dim, size in ds_re.sizes.items()}
        ds_re = ds_re.chunk(chunk_map)
        ds_re = ds_re.transpose("time", "latitude", "longitude")

        to_zarr_append(ds_re, fpath)
        print(f"• appended {ds_re.time.size} daily steps to {fpath}", flush=True)

        ds.close()
        ds_daily.close()
        ds_re.close()


if __name__ == "__main__":
    main()

