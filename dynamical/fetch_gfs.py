"""
Download NOAA GFS forecast from dynamical.org for all Wednesdays of 2024 at 00Z,
compute daily mean over lead_time per lead day, add dewpoint (Magnus),
rename to OceanBench-style names, and store as one zarr.
"""
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from dask.diagnostics import ProgressBar


# GFS forecasts
GFS_ZARR_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"

# OceanBench-style names
LEXICON = {
    "temperature_2m": "sotemair",
    "wind_u_10m": "sowinu10",
    "wind_v_10m": "sowinv10",
    "pressure_reduced_to_mean_sea_level": "somslpre",
    "precipitation_surface": "sowaprec",
    "downward_short_wave_radiation_flux_surface": "sosudosw",
    "downward_long_wave_radiation_flux_surface": "sosudolw",
}

VARS_KEEP = [
    "wind_u_10m",
    "wind_v_10m",
    "temperature_2m",
    "pressure_reduced_to_mean_sea_level",
    "precipitation_surface",
    "downward_short_wave_radiation_flux_surface",
    "downward_long_wave_radiation_flux_surface",
]
RH_VAR = "relative_humidity_2m"


def _wednesdays_2024() -> pd.DatetimeIndex:
    return pd.date_range(start="2024-01-01", end="2024-12-31", freq="W-WED")


def _magnus_dewpoint(T: xr.DataArray, RH: xr.DataArray) -> xr.DataArray:
    """Dewpoint (°C) from T (°C) and RH (%) using Magnus formula."""
    a, b = 17.625, 243.04
    gamma = np.log(RH / 100.0) + (a * T) / (b + T)
    return (b * gamma) / (a - gamma)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir",
        type=str,
        default="data",
        help="Output zarr path",
    )
    p.add_argument(
        "--url",
        type=str,
        default=GFS_ZARR_URL,
        help="GFS forecast zarr URL",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of Dask workers",
    )
    args = p.parse_args()

    # Set up dask client with n workers
    client = Client(n_workers=args.n_workers)
    print(f"Dask client: {client}", flush=True)

    wednesdays = _wednesdays_2024()
    print(f"Opening {args.url} ...", flush=True)
    ds = xr.open_zarr(args.url)
    print(ds)

    # Build list of init_time values that are Wed 2024 00:00
    inits = pd.to_datetime(ds.init_time.values)
    wed_set = set(wednesdays.normalize())
    mask = np.array(
        [pd.Timestamp(t).normalize() in wed_set and inits[i].hour == 0 for i, t in enumerate(inits)]
    )
    init_times_wed = ds.init_time.values[mask]
    print(f"Found {len(init_times_wed)} Wednesday 00Z init times", flush=True)

    # Select init_time
    ds_sub = ds.sel(init_time=init_times_wed)

    # Variables: keep only those present + RH for dewpoint
    vars_load = [v for v in VARS_KEEP if v in ds_sub.data_vars]
    missing = set(VARS_KEEP) - set(vars_load)
    if missing:
        print(f"Warning: variables not in dataset (skipped): {missing}", flush=True)
    if RH_VAR in ds_sub.data_vars:
        vars_load.append(RH_VAR)
    else:
        print(f"Warning: {RH_VAR} not in dataset; sod2m will be omitted", flush=True)
    ds_sub = ds_sub[vars_load]

    # Daily mean: group by lead_day_index = floor(lead_time in hours / 24)
    # Convert timedelta64[ns] to hours: divide by nanoseconds per hour (1e9 * 3600)
    lead_hours = (ds_sub.lead_time.values / np.timedelta64(1, 'h')).astype(np.int64)
    lead_day_index = lead_hours // 24
    ds_sub = ds_sub.assign_coords(lead_day_index=("lead_time", lead_day_index))
    print("Computing daily means...", flush=True)
    with ProgressBar():
        ds_daily = ds_sub.groupby("lead_day_index").mean(dim="lead_time")

    print(ds_daily.lead_day_index, flush=True)

    # Rename init_time -> first_day_datetime, use date (00:00:00)
    ds_daily = ds_daily.rename(init_time="first_day_datetime")
    first_day = ds_daily.first_day_datetime.values.astype("datetime64[D]")
    ds_daily = ds_daily.assign_coords(first_day_datetime=first_day)

    # Dewpoint from T and RH (daily means)
    if RH_VAR in ds_daily.data_vars:
        T = ds_daily["temperature_2m"]
        RH = ds_daily[RH_VAR]
        ds_daily["sod2m"] = _magnus_dewpoint(T, RH)
        ds_daily = ds_daily.drop_vars(RH_VAR)

    # Rename to OceanBench names
    to_rename = {k: v for k, v in LEXICON.items() if k in ds_daily.data_vars}
    ds_daily = ds_daily.rename(to_rename)

    # Select first 15 lead days
    ds_daily = ds_daily.isel(lead_day_index=slice(15))

    # Convert sod2m and sotemair from Celsius to Kelvin and add units
    ds_daily["sod2m"] = ds_daily["sod2m"] + 273.15
    ds_daily["sod2m"].attrs["units"] = "K"
    ds_daily["sotemair"] = ds_daily["sotemair"] + 273.15
    ds_daily["sotemair"].attrs["units"] = "K"

    # Reorder dimensions: first_day_datetime before lead_day_index
    dim_order = ["first_day_datetime", "lead_day_index", "latitude", "longitude"]
    # Only include dimensions that exist
    dim_order = [d for d in dim_order if d in ds_daily.dims]
    ds_daily = ds_daily.transpose(*dim_order, ...)

    # Drop unwanted coordinates
    coords_to_drop = ["expected_forecast_length", "ingested_forecast_length", "valid_time", "spatial_ref"]
    coords_to_drop = [c for c in coords_to_drop if c in ds_daily.coords]
    if coords_to_drop:
        ds_daily = ds_daily.drop_vars(coords_to_drop)

    # Chunk the dataset: 1 for first_day_datetime, 1 for lead_day_index, -1 (full size) for lat/lon
    chunks = {
        "first_day_datetime": 1,
        "lead_day_index": 1,
        "latitude": ds_daily.dims["latitude"],
        "longitude": ds_daily.dims["longitude"],
    }
    ds_daily = ds_daily.chunk(chunks)

    # Set up encoding for zarr writing
    encoding = {}
    for var in ds_daily.data_vars:
        var_chunks = tuple(chunks.get(dim, ds_daily.dims[dim]) for dim in ds_daily[var].dims)
        encoding[var] = {"chunks": var_chunks}
    # Also encode coordinates that are dimensions
    for coord in ds_daily.coords:
        if coord in ds_daily.dims:
            coord_chunks = tuple(chunks.get(dim, ds_daily.dims[dim]) for dim in ds_daily[coord].dims)
            encoding[coord] = {"chunks": coord_chunks}

    print(ds_daily, flush=True)
    print(f"Writing ...", flush=True)
    zarr_path = os.path.join(args.base_dir, "gfs_forecasts.zarr")
    with ProgressBar():
        ds_daily.to_zarr(zarr_path, encoding=encoding, mode="w")
    
    client.close()


if __name__ == "__main__":
    main()
