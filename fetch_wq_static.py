#!/usr/bin/env python3
"""
Download and regrid CMEMS global static data to 3/2° resolution,
and store it in cmems_phy_static.zarr.
"""

import os
import numpy as np
import xarray as xr
import copernicusmarine as cm
import xarray_regrid

zarr_store = "cmems_phy_static.zarr"

# Select every 5th depth level
depth_slice = slice(None, None, 5)

# Target 1.5° global grid
lat_target = np.arange(90.0, -91.0, -1.5)
lon_target = np.arange(0.0, 360.0, 1.5)
ds_target_grid = xr.Dataset(coords={
    "latitude":  ("latitude",  lat_target),
    "longitude": ("longitude", lon_target),
})

def regrid_static(ds: xr.Dataset, skipna=True) -> xr.Dataset:
    """Regrid dataset to target grid. For 'mask', treat 0 as NaN before regridding, then restore."""
    if "mask" in ds:
        # Temporarily replace land (0) with NaN
        mask = ds["mask"].where(ds["mask"] != 0)
        ds = ds.copy()
        ds["mask"] = mask

    # Perform conservative regridding
    ds_regridded = ds.regrid.conservative(
        ds_target_grid,
        latitude_coord="latitude",
        skipna=skipna
    )

    # Replace NaNs in regridded mask with 0 to recover binary ocean/land
    if "mask" in ds_regridded:
        ds_regridded["mask"] = ds_regridded["mask"].fillna(0).astype(np.int8)

    # Rechunk
    chunk_map = {dim: size for dim, size in ds_regridded.sizes.items()}
    return ds_regridded.chunk(chunk_map)


def fetch_and_regrid_static():
    # MDT
    ds_mdt = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
        dataset_part="mdt",
        variables=["mdt"],
    )
    ds_mdt = regrid_static(ds_mdt)

    # Bathymetry + mask
    ds_bathy = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_static",
        dataset_part="bathy",
        variables=["deptho", "mask"],
        maximum_depth=200,
    ).isel(depth=depth_slice)
    ds_bathy = regrid_static(ds_bathy)

    # Merge and transpose consistently
    ds_merged = xr.merge([ds_mdt, ds_bathy])
    ds_merged = ds_merged.transpose(...)

    return ds_merged

def main():
    ds_static = fetch_and_regrid_static()

    ds_static = ds_static.fillna(0.0)
    for vn in ds_static.data_vars:
        ds_static[vn] = ds_static[vn].astype("float32")
    print(ds_static, flush=True)
    
    ds_static.to_zarr(zarr_store, mode="w")
    print(f"Static dataset written to {zarr_store}", flush=True)

if __name__ == "__main__":
    main()
