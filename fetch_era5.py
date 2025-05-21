import cdsapi
import numpy as np
import xarray as xr
import os
from datetime import datetime

# Set base output folder
output_dir = "data/baltic/raw/era5"
os.makedirs(output_dir, exist_ok=True)

# Define fixed parameters
dataset = "reanalysis-era5-single-levels"
variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure"
]
years = [2002]
area = [70, 5, 50, 35]  # [north, west, south, east]
client = cdsapi.Client()

for year in years:
    for month in range(1, 13):
        month_str = f"{month:02d}"
        print(f"\nProcessing {year}-{month_str}...")

        # Temporary download filename
        netcdf_path = os.path.join(output_dir, f"{year}{month_str}.nc")

        # CDS request
        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": str(year),
            "month": month_str,
            "day": [f"{day:02d}" for day in range(1, 32)],
            "area": area,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        # Download the data
        client.retrieve(dataset, request).download(netcdf_path)

        # Load dataset
        ds = xr.open_dataset(netcdf_path)

        if "valid_time" in ds:
            ds = ds.rename({"valid_time": "time"})
        
        ds = ds.drop_vars([v for v in ["number", "expver"] if v in ds.variables])

        # Group by date and compute daily mean
        daily_ds = ds.resample(time="1D").mean()

        # Save each day separately
        for date in daily_ds.time.values:
            dt = np.datetime64(date).astype("datetime64[D]").astype(datetime)
            date_str = dt.strftime("%Y%m%d")

            ds_day = daily_ds.sel(time=date)
            ds_day = ds_day.expand_dims("time")  # Maintain time dim

            zarr_path = os.path.join(output_dir, f"{date_str}.zarr")
            ds_day.to_zarr(zarr_path, mode="w", consolidated=True)
            print(f"Saved: {zarr_path}")

        # Clean up
        ds.close()
