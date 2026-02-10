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

# global config
DATASETS = [
    {
        "dataset_id": "cmems_mod_bal_phy_my_P1D-m",
        "variables": ["mlotst", "siconc", "sithick", "sla", "so", "thetao", "uo", "vo"],
        "start_year": 1993,
        "end_year": 2022,
        "type": "daily",
    },
    {
        "dataset_id": "cmems_mod_bal_bgc_my_P1D-m",
        "variables": ["chl"],
        "start_year": 1993,
        "end_year": 2022,
        "type": "daily",
    },
    {
        "dataset_id": "cmems_mod_bal_wav_my_PT1H-i",
        "variables": ["VHM0", "VMDR", "VTM10"],
        "start_year": 1993,
        "end_year": 2022,
        "type": "wave",
    },
    {
        "dataset_id": "cmems_mod_bal_phy_anfc_P1D-m",
        "variables": ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo"],
        "start_year": 2023,
        "end_year": 2024,
        "type": "daily",
    },
    {
        "dataset_id": "cmems_mod_bal_phy_anfc_PT1H-i",
        "variables": ["sla"],
        "start_year": 2023,
        "end_year": 2024,
        "type": "hourly",
    },
    {
        "dataset_id": "cmems_mod_bal_bgc_anfc_P1D-m",
        "variables": ["chl"],
        "start_year": 2023,
        "end_year": 2024,
        "type": "daily",
    },
    {
        "dataset_id": "cmems_mod_bal_wav_anfc_PT1H-i",
        "variables": ["VHM0", "VMDR", "VTM10"],
        "start_year": 2023,
        "end_year": 2024,
        "type": "wave",
    },
]

for ds_cfg in DATASETS:
    os.makedirs(args.base_dir, exist_ok=True)

static_grid = xr.open_zarr(os.path.join(args.base_dir, "baltic_mask.zarr"))
target_lat = static_grid.latitude
target_lon = static_grid.longitude
surface_mask = static_grid.thetao.isel(depth=0).astype(bool)

def process_wave(ds):
    # Match wave hindcast to the forecast grid (has 1 extra pixel in lat and lon)
    ds = ds.sel(
        latitude=ds.latitude[:-1], # drop max latitude
        longitude=ds.longitude[1:] # drop min longitude
    )
    ds = ds.assign_coords(
        latitude=target_lat,
        longitude=target_lon,
    )

    for var in ["VHM0", "VMDR", "VTM10"]:
        ds[var] = ds[var].where(~surface_mask | ~np.isnan(ds[var]), 0)
    
    ds = ds.rename({"VHM0": "swh", "VTM10": "mwp"})
    
    ang = np.deg2rad(ds["VMDR"])
    ds["uw"] = np.sin(ang) # Sine component of sea surface wave from direction
    ds["vw"] = np.cos(ang) # Cosine component of sea surface wave from direction
    ds["uw"].attrs["units"] = "1"
    ds["vw"].attrs["units"] = "1"
    ds = ds.drop_vars("VMDR")

    ds = ds.resample(time="1D").mean()

    return ds

def process_hourly(ds):
    ds = ds.assign_coords(latitude=target_lat, longitude=target_lon)
    ds = ds.resample(time="1D").mean()
    return ds

def to_zarr_append(ds, store):
    """Write first chunk, then append each subsequent chunk along time."""
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")

for year in range(args.start_year, args.end_year + 1):
    for month in range(1, 13):
        monthly_datasets = []
        last_day   = calendar.monthrange(year, month)[1]
        start_time = f"{year:04d}-{month:02d}-01T00:00:00"
        end_time   = f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59"

        for ds_cfg in DATASETS:
            if not (ds_cfg["start_year"] <= year <= ds_cfg["end_year"]):
                continue

            dataset_id = ds_cfg["dataset_id"]
            variables  = ds_cfg["variables"]
            ds_type    = ds_cfg.get("type", "regular")

            print(f"[{year}-{month:02d}] fetching {dataset_id} ({ds_type})…")
            ds = cm.open_dataset(
                dataset_id     = dataset_id,
                variables      = variables,
                start_datetime = start_time,
                end_datetime   = end_time,
                maximum_depth  = 104,
            )

            if "depth" in ds.dims:
                ds = ds.isel(depth=slice(None, None, 3))
                rounded_depths = np.round(ds.depth.values).astype(int)
                ds = ds.assign_coords(depth=rounded_depths)

            ds = ds.where((ds.longitude > 10) | (ds.latitude > 54))
            ds = ds.sel(latitude=slice(53.6, None))

            if ds_type == "wave":
                ds = process_wave(ds)
            elif ds_type == "hourly":
                ds = process_hourly(ds)
            else:
                ds = ds.assign_coords(latitude=target_lat, longitude=target_lon)

            # Chunk for writing
            if "depth" in ds.dims:
                ds = ds.chunk({"time": ds.sizes["time"], "depth": ds.sizes["depth"]})
            else:
                ds = ds.chunk({"time": ds.sizes["time"]})

            monthly_datasets.append(ds)

        # Merge all vars for this month into one Dataset
        merged = xr.merge(monthly_datasets)
        to_zarr_append(merged, os.path.join(args.base_dir, "baltic_data.zarr"))
        print(f"→ Appended {merged.sizes['time']} steps for {year}-{month:02d}\n", flush=True)