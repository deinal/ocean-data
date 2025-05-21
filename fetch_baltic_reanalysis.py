#!/usr/bin/env python3
import os
import calendar
import pandas as pd
import copernicusmarine as cm

# === CONFIG ===
BASE_DIR = "data/baltic/raw/reanalysis"
DATASETS = [
    {
        "subdir": "bal_phy",
        "dataset_id": "cmems_mod_bal_phy_my_P1D-m",
        "variables": ["mlotst", "siconc", "sithick", "sla", "so", "thetao", "uo", "vo"],
        "hourly": False,
    },
    {
        "subdir": "bal_bgc",
        "dataset_id": "cmems_mod_bal_bgc_my_P1D-m",
        "variables": ["chl", "nh4", "no3", "nppv", "o2", "ph", "po4", "spco2", "zsd"],
        "hourly": False,
    },
    {
        "subdir": "bal_wav",
        "dataset_id": "cmems_mod_bal_wav_my_PT1H-i",
        "variables": ["VHM0", "VMDR", "VTM10"],
        "hourly": True,
    },
]

START_YEAR = 2000
END_YEAR = 2002

# Ensure output dirs
for ds_cfg in DATASETS:
    os.makedirs(os.path.join(BASE_DIR, ds_cfg["subdir"]), exist_ok=True)

# Loop every month
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        # build month bounds
        last_day    = calendar.monthrange(year, month)[1]
        start_month = f"{year:04d}-{month:02d}-01T00:00:00"
        end_month   = f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59"

        for ds_cfg in DATASETS:
            subdir = ds_cfg["subdir"]
            outdir = os.path.join(BASE_DIR, subdir)
            print(f"[{year}-{month:02d}] fetching {ds_cfg['dataset_id']} → {subdir}/")

            # open full month
            ds = cm.open_dataset(
                dataset_id     = ds_cfg["dataset_id"],
                variables      = ds_cfg["variables"],
                start_datetime = start_month,
                end_datetime   = end_month,
            )

            # Make sure the full ds_daily is chunked along time=1 once up front:
            if ds_cfg["hourly"]:
                ds_daily = ds.resample(time="1D").mean(dim="time").chunk({"time": 1})
            else:
                # even for daily data, we can chunk time=1 so each store is exactly one time‐step
                ds_daily = ds.chunk({"time": 1})

            # now loop by integer index
            for i in range(ds_daily.sizes["time"]):
                t = ds_daily.time.values[i]
                ts = pd.to_datetime(str(t))
                day_str = ts.strftime("%Y%m%d")
                store = os.path.join(outdir, f"{day_str}.zarr")
                if os.path.exists(store):
                    continue

                # … after you select the daily slice …
                daily = ds_daily.isel(time=i)

                # build a chunk map that uses the full length of every dimension
                chunk_map = {dim: size for dim, size in daily.sizes.items()}

                # re-chunk so each dim is one chunk
                daily = daily.chunk(chunk_map)

                # ignore extra river
                daily = daily.where((daily.longitude > 10) | (daily.latitude > 54))

                # crop to southern edge
                daily = daily.sel(latitude=slice(53.6, None))

                # write it out
                daily.to_zarr(store, mode="w", consolidated=True)

                print(f"   • wrote {subdir}/{day_str}.zarr")

            ds.close()
            if ds_cfg["hourly"]:
                ds_daily.close()

print("✅ All done!")
