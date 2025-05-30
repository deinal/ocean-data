import os
import pandas as pd
import earthkit.data as ekd
import xarray as xr
from time import sleep


# setup
output_dir = "data/baltic/raw"
os.makedirs(output_dir, exist_ok=True)
zarr_store = os.path.join(output_dir, "era5.zarr")

start_date = pd.Timestamp("1993-01-01")
end_date = pd.Timestamp("2025-05-01")

# generate list of (year, month) pairs between start and end
all_months = pd.date_range(start=start_date, end=end_date, freq="MS")

# check if zarr store already has data
if os.path.exists(zarr_store):
    existing = xr.open_zarr(zarr_store)
    last_time = pd.to_datetime(existing.time.values).max()
    resume_date = pd.Timestamp(last_time.year, last_time.month, 1) + pd.DateOffset(months=1)
    months_to_download = [d for d in all_months if d >= resume_date]
else:
    months_to_download = all_months


def to_zarr_append(ds, store):
    """Write first chunk with mode='w', then append along time."""
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")


def safe_download_and_decode():
    for attempt in range(10):
        try:
            ds = ekd.from_source("cds",
                "reanalysis-era5-single-levels",
                variable=[
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "total_precipitation",
                    "surface_solar_radiation_downwards",
                    "surface_thermal_radiation_downwards",
                    "snowfall"
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
    raise RuntimeError("Failed to download.")


# download
for date in months_to_download:
    year = date.year
    month = date.month
    month_str = f"{month:02d}"

    print(f"Downloading {year}-{month_str}…", flush=True)

    ds = safe_download_and_decode()

    ds = ds.rename({"valid_time": "time"})
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds])

    to_zarr_append(ds, zarr_store)
    print(f"  ⇒ Appended {ds.time.size} steps to {zarr_store}", flush=True)
    ds.close()
