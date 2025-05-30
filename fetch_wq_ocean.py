#!/usr/bin/env python3
"""
Regrid CMEMS ocean data from 1/12° to 3/2° resolution
with resumable Zarr writing and depth-level chunking.
"""
import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm
import xarray_regrid
import os

zarr_store = "cmems_phy_19930101_20250501.zarr"

# Select every 5th depth level
depth_slice = slice(None, None, 5)

# Time range from July 1, 2021
time_index = pd.date_range("1993-01-01", "2025-05-01", freq="D")

# Target grid
lat_target = np.arange(90.0, -91.0, -1.5)
lon_target = np.arange(0.0, 360.0, 1.5)

# Target grid dataset
ds_target_grid = xr.Dataset(coords={
    "latitude":  ("latitude",  lat_target),
    "longitude": ("longitude", lon_target),
})


def fetch_and_regrid(date):
    t0 = date.strftime("%Y-%m-%d")

    # reanalysis
    if date <= pd.Timestamp("2021-06-30"):
        dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
        variables = ["thetao", "uo", "vo", "usi", "vsi", "zos", "siconc", "sithick"]
        ds = cm.open_dataset(
            dataset_id=dataset_id,
            variables=variables,
            start_datetime=t0,
            end_datetime=t0,
            maximum_depth=200,
        ).isel(depth=depth_slice)
    # analysis
    elif date >= pd.Timestamp("2022-06-01"):
        ds1 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            start_datetime=t0,
            end_datetime=t0,
            maximum_depth=200,
        ).isel(depth=depth_slice)
                
        ds2 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            start_datetime=t0,
            end_datetime=t0,
            maximum_depth=200,
        ).isel(depth=depth_slice)

        ds3 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["usi", "vsi", "zos", "siconc", "sithick"],
            start_datetime=t0,
            end_datetime=t0,
        )

        ds = xr.merge([ds1, ds2, ds3])
    # interim
    else:
        dataset_id = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
        variables = ["thetao", "uo", "vo", "usi", "vsi", "zos", "siconc", "sithick"]
        ds = cm.open_dataset(
            dataset_id=dataset_id,
            variables=variables,
            start_datetime=t0,
            end_datetime=t0,
            maximum_depth=200,
        ).isel(depth=depth_slice)

    ds_regridded = ds.regrid.conservative(
        ds_target_grid,
        latitude_coord="latitude",
        skipna=True,
    )

    # Build chunk map with full dimension lengths
    chunk_map = {dim: size for dim, size in ds_regridded.sizes.items()}

    # Re-chunk the dataset
    ds_regridded = ds_regridded.chunk(chunk_map)

    # Reorder dimensions
    return ds_regridded.transpose("time", "depth", "latitude", "longitude")


def truncate_last_time_step():
    """Remove the last time step from the Zarr store."""
    try:
        ds = xr.open_zarr(zarr_store)
        if ds.dims["time"] < 2:
            print("Only one time step, removing store.", flush=True)
            os.system(f"rm -r {zarr_store}")
            return None
        truncated = ds.isel(time=slice(None, -1))
        truncated.to_zarr(zarr_store, mode="w")
        print(f"Truncated last time step: {ds.time.values[-1]}", flush=True)
        return pd.to_datetime(truncated.time.values[-1])
    except Exception as e:
        print(f"Failed to truncate Zarr: {e}", flush=True)
        return None


def main():
    if not os.path.exists(zarr_store):
        start_idx = 0
        print("Initializing new Zarr store...", flush=True)
        ds_first = fetch_and_regrid(time_index[0])
        ds_first.to_zarr(zarr_store, mode="w")
        print(f"Initialized {zarr_store} with {time_index[0].date()}", flush=True)
        start_idx = 1
    else:
        last_written_time = truncate_last_time_step()
        if last_written_time is None:
            start_idx = 0
        else:
            start_idx = time_index.get_loc(last_written_time)

    for dt in time_index[start_idx:]:
        try:
            ds_day = fetch_and_regrid(dt)
            ds_day.to_zarr(zarr_store, mode="a", append_dim="time")
            print("Appended", dt.date(), flush=True)
        except Exception as e:
            print(f"Failed on {dt.date()}: {e}", flush=True)
            break

    print("Processing complete.", flush=True)


if __name__ == "__main__":
    main()
