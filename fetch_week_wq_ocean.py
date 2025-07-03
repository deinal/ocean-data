#!/usr/bin/env python3
"""
Fetch the last 7 days of GLORYS daily data, regrid to target grid,
compute the weekly mean, and save as a one‐timestep Zarr store.
"""
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid
import copernicusmarine as cm


parser = argparse.ArgumentParser(
    description="Fetch & regrid last 7 days of GLORYS data, then compute weekly mean"
)
parser.add_argument(
    "--out", "-o",
    help="Output Zarr store path",
    default="cmems_phy_last_week_mean.zarr",
)
args = parser.parse_args()

# 7‑day window
end_date = pd.to_datetime(pd.Timestamp.today().strftime("%Y-%m-%d"))
start_date = end_date - pd.Timedelta(days=6)
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# Regrid settings
depth_slice = slice(None, None, 5)
lat_target = np.arange(90.0, -91.0, -1.5)
lon_target = np.arange(0.0, 360.0, 1.5)
ds_target_grid = xr.Dataset(coords={
    "latitude":  ("latitude",  lat_target),
    "longitude": ("longitude", lon_target),
})

def fetch_and_regrid(date: pd.Timestamp) -> xr.Dataset:
    """Download & conservative‐regrid one day, selecting every 5th depth."""
    t0 = date.strftime("%Y-%m-%d")

    ds1 = cm.open_dataset(
        "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        start_datetime=t0, end_datetime=t0,
        maximum_depth=200,
    ).isel(depth=depth_slice)
    ds2 = cm.open_dataset(
        "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        start_datetime=t0, end_datetime=t0,
        maximum_depth=200,
    ).isel(depth=depth_slice)
    ds3 = cm.open_dataset(
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["usi", "vsi", "zos", "siconc", "sithick"],
        start_datetime=t0, end_datetime=t0,
    )
    ds = xr.merge([ds1, ds2, ds3])
    
    # conservative regrid
    ds_re = ds.regrid.conservative(
        ds_target_grid,
        latitude_coord="latitude",
        skipna=True,
    )

    # one chunk per dim
    chunk_map = {dim: size for dim, size in ds_re.sizes.items()}
    ds_re = ds_re.chunk(chunk_map)

    # ensure dims order
    return ds_re.transpose("time", "depth", "latitude", "longitude")


def main():
    print(f"Fetching {len(dates)} days from {start_date.date()}→{end_date.date()}…")
    daily_ds = [fetch_and_regrid(d) for d in dates]
    
    # concatenate over time and average
    ds_week = xr.concat(daily_ds, dim="time")
    weekly_mean = ds_week.mean(dim="time", keep_attrs=True).fillna(0.0)
    
    # set time coordinate to the end_date
    weekly_mean = weekly_mean.assign_coords(time=[end_date])
    
    # write
    weekly_mean.to_zarr(args.out, mode="w")
    print("Saved weekly mean to", args.out)

if __name__ == "__main__":
    main()
