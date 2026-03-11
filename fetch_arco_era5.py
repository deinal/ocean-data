"""
Fetch ERA5 single-level data from ARCO-ERA5 (GCP public dataset).
https://github.com/google-research/arco-era5
"""
import os
import argparse
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar


ARCO_ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_total_precipitation_rate",
]

parser = argparse.ArgumentParser(
    description="Fetch global ERA5 from ARCO-ERA5 (GCP)."
)
parser.add_argument("--base-dir", type=str, default="data")
parser.add_argument("--start-date", type=str, default="1993-01-01")
parser.add_argument("--end-date", type=str, default="2024-12-31")
parser.add_argument("--source", type=str, default=ARCO_ERA5_ZARR)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Remove existing output zarr before writing.",
)
args = parser.parse_args()

os.makedirs(args.base_dir, exist_ok=True)
storage_options = {"token": "anon"} if args.source.startswith("gs://") else None


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start_month = start.replace(day=1)
    end_month = end.replace(day=1)
    return pd.date_range(start_month, end_month, freq="MS")


def main():
    zarr_store = os.path.join(args.base_dir, "arco_era5.zarr")

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)

    print(f"Period: {start.date()} to {end.date()}", flush=True)
    print(f"Output: {zarr_store}", flush=True)

    if args.overwrite and os.path.exists(zarr_store):
        print(f"Removing existing store: {zarr_store}", flush=True)
        import shutil
        shutil.rmtree(zarr_store)

    print("Opening ARCO-ERA5 zarr...", flush=True)
    ds = xr.open_zarr(args.source, storage_options=storage_options)
    print(ds, flush=True)

    vars_present = [v for v in VARIABLES if v in ds.data_vars]
    missing = [v for v in VARIABLES if v not in ds.data_vars]

    print(f"Variables found: {vars_present}", flush=True)
    if missing:
        print(f"Variables missing: {missing}", flush=True)

    ds = ds[vars_present]

    months = month_starts(start, end)
    first_write = not os.path.exists(zarr_store)

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=args.num_workers):
            for month_start in months:
                month_end = month_start + pd.offsets.MonthBegin(1)

                sel_start = max(start, month_start)
                sel_end = min(end, month_end - pd.Timedelta(hours=1))

                if sel_start > sel_end:
                    continue

                print(
                    f"\nProcessing month: {month_start.strftime('%Y-%m')} "
                    f"({sel_start} -> {sel_end})",
                    flush=True,
                )

                ds_month = ds.sel(time=slice(sel_start, sel_end))

                if ds_month.sizes.get("time", 0) == 0:
                    print("  No timesteps in selection, skipping.", flush=True)
                    continue

                ds_daily = ds_month.resample(time="1D").mean()

                ds_daily = ds_daily.chunk({
                    "time": 1,
                    "latitude": -1,
                    "longitude": -1,
                })

                print(f"  Daily timesteps: {ds_daily.sizes['time']}", flush=True)

                if first_write:
                    print("  Writing first month...", flush=True)
                    ds_daily.to_zarr(zarr_store, mode="w")
                    first_write = False
                else:
                    print("  Appending month...", flush=True)
                    ds_daily.to_zarr(zarr_store, append_dim="time")


if __name__ == "__main__":
    main()