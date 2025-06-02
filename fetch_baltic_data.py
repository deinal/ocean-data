#!/usr/bin/env python3
import os
import calendar
import pandas as pd
import copernicusmarine as cm

# === CONFIG ===
BASE_DIR   = "data/baltic/raw"
DATASETS   = [
    {
        "store": "bal_phy_rea",
        "dataset_id": "cmems_mod_bal_phy_my_P1D-m",
        "variables": ["mlotst", "siconc", "sithick", "sla", "so", "thetao", "uo", "vo"],
        "start_year": 1993,
        "end_year": 2022,
    },
    {
        "store": "bal_bgc_rea",
        "dataset_id": "cmems_mod_bal_bgc_my_P1D-m",
        "variables": ["chl", "o2", "zsd"],
        "start_year": 1993,
        "end_year": 2022,
    },
    {
        "store": "bal_wav_rea",
        "dataset_id": "cmems_mod_bal_wav_my_PT1H-i",
        "variables": ["VHM0", "VMDR", "VTM10"],
        "start_year": 1993,
        "end_year": 2022,
    },
    {
        "store": "bal_phy_ana",
        "dataset_id": "cmems_mod_bal_phy_anfc_P1D-m",
        "variables": ["mlotst", "siconc", "sithick", "so", "thetao", "uo", "vo"],
        "start_year": 2023,
        "end_year": 2024,
    },
    {
        "store": "bal_sla_ana",
        "dataset_id": "cmems_mod_bal_phy_anfc_PT1H-i",
        "variables": ["sla"],
        "start_year": 2023,
        "end_year": 2024,
    },
    {
        "store": "bal_bgc_ana",
        "dataset_id": "cmems_mod_bal_bgc_anfc_P1D-m",
        "variables": ["chl", "o2", "zsd"],
        "start_year": 2023,
        "end_year": 2024,
    },
    {
        "store": "bal_wav_ana",
        "dataset_id": "cmems_mod_bal_wav_anfc_PT1H-i",
        "variables": ["VHM0", "VMDR", "VTM10"],
        "start_year": 2023,
        "end_year": 2024,
    },
]

DEPTH_SLICE = slice(None, None, 3)

# ensure output dirs
for ds_cfg in DATASETS:
    os.makedirs(BASE_DIR, exist_ok=True)

def to_zarr_append(ds, store):
    """Write first chunk with mode='w', then append each subsequent chunk along 'time'."""
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")

# loop every month
for ds_cfg in DATASETS:
    store    = ds_cfg["store"]
    dataset_id= ds_cfg["dataset_id"]
    zarr_store= os.path.join(BASE_DIR, f"{store}.zarr")

    print(f"\n=== Building {zarr_store} ===")

    for year in range(ds_cfg["start_year"], ds_cfg["end_year"] + 1):
        for month in range(1, 13):
            last_day    = calendar.monthrange(year, month)[1]
            start_time  = f"{year:04d}-{month:02d}-01T00:00:00"
            end_time    = f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59"

            print(f"[{year}-{month:02d}] fetching {dataset_id}…")
            ds = cm.open_dataset(
                dataset_id     = dataset_id,
                variables      = ds_cfg["variables"],
                start_datetime = start_time,
                end_datetime   = end_time,
                maximum_depth  = 204,
            )

            if "depth" in ds.dims:
                # select depths
                ds = ds.isel(depth=DEPTH_SLICE)

                # chunk the entire month as one time‐block and all depths in one block
                ds = ds.chunk({
                    "time":  ds.sizes["time"],
                    "depth": ds.sizes["depth"],
                })
            else:
                 ds = ds.chunk({
                    "time":  ds.sizes["time"],
                })


            # mask out extra river and crop frame
            ds = ds.where((ds.longitude > 10) | (ds.latitude > 54))
            ds = ds.sel(latitude=slice(53.6, None))

            # append monthly data
            to_zarr_append(ds, zarr_store)
            print(f"   • appended {ds.sizes['time']} steps to {store}.zarr")

            ds.close()

    print(f"Done {store}.zarr\n")
