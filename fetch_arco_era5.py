"""
Fetch ERA5 single-level data from ARCO-ERA5 (GCP public dataset).
https://github.com/google-research/arco-era5
"""
import os
import argparse
import pandas as pd
import xarray as xr
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
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
)
parser.add_argument("--start-date", type=str, default="2020-01-01")
parser.add_argument("--end-date", type=str, default="2020-03-31")
parser.add_argument("--source", type=str, default=ARCO_ERA5_ZARR)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

zarr_store = os.path.join(args.base_dir, "arco_era5.zarr")
storage_options = {"token": "anon"} if args.source.startswith("gs://") else None


def main():
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    print(f"Period: {start.date()} to {end.date()}", flush=True)

    print("Opening ARCO-ERA5 zarr…", flush=True)
    ds = xr.open_zarr(args.source, storage_options=storage_options)
    print(ds, flush=True)

    vars_present = [v for v in VARIABLES if v in ds.data_vars]

    ds = ds[vars_present].sel(time=slice(start, end))
    print(f"Daily mean…", flush=True)
    ds_daily = ds.resample(time="1D").mean()

    print(f"Writing {zarr_store}…", flush=True)
    with ProgressBar():
        ds_daily.to_zarr(zarr_store, mode="w")

    print(f"Done. {zarr_store}", flush=True)


if __name__ == "__main__":
    main()