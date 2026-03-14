"""
Interpolate global data to 1/4° and 1° grids.
"""
import os
import argparse
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar


# 1/4° grid
LAT_1_4 = np.arange(-80.0, 90.0, 0.25) # [-80.0, -79.75, ..., 89.5 ,  89.75], shape=(680,)
LON_1_4 = np.arange(-180.0, 180.0, 0.25) # [-180.0, -179.75, ..., 179.5, 179.75], shape=(1440,)

# 1° grid
LAT_1 = np.arange(-79.5, 90.5, 1.0) # [-79.5, -78.5, ..., 88.5, 89.5], shape=(170,)
LON_1 = np.arange(-179.5, 180.5, 1.0) # [-179.5, -178.5, ..., 178.5, 179.5], shape=(360,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data",
        help="Base data directory",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of workers for dask (default: %(default)s)",
    )
    
    args = parser.parse_args()
    base = args.base_dir

    grids = [(LAT_1_4, LON_1_4, "1_4"), (LAT_1, LON_1, "1")]

    mask_path = os.path.join(base, "global_mask.zarr")
    ds_mask = xr.open_zarr(mask_path)
    ocean_surface = (ds_mask["zos"] == 1)
    ocean = (ds_mask["thetao"] == 1)
    ds_mask_prep = ds_mask.where(ds_mask != 0)

    mask_ds = xr.Dataset({
        "zos": ocean_surface,
        "siconc": ocean_surface,
        "sithick": ocean_surface,
        "uo": ocean,
        "vo": ocean,
        "thetao": ocean,
        "so": ocean,
    })

    # regrid mask to both grids
    for lat_arr, lon_arr, suffix in grids:
        print(f"Regridding global_mask -> {suffix}° ...")
        ds_r = ds_mask_prep.interp(latitude=lat_arr, longitude=lon_arr)
        ds_r = ds_r.notnull().astype(np.uint8)
        ds_r = ds_r.chunk({"depth": -1, "latitude": -1, "longitude": -1})
        with dask.config.set(scheduler="threads", num_workers=args.num_workers):
            with ProgressBar():
                ds_r.to_zarr(os.path.join(base, f"global_mask_{suffix}.zarr"), mode="w")
        print(f"  -> global_mask_{suffix}.zarr")

    # global_static.zarr -> global_static_1_4.zarr and global_static_1.zarr
    static_path = os.path.join(base, "global_static.zarr")
    print("Loading global_static.zarr and filling interior NaNs ...")
    ds_static = xr.open_zarr(static_path)
    ds_static = xr.where(ds_static.isnull() & ocean_surface, 0.0, ds_static)
    for lat_arr, lon_arr, suffix in grids:
        print(f"Regridding global_static -> {suffix}° ...")
        ds_r = ds_static.interp(latitude=lat_arr, longitude=lon_arr)
        chunk_map = {"latitude": -1, "longitude": -1}
        if "depth" in ds_r.dims:
            chunk_map["depth"] = -1
        ds_r = ds_r.chunk(chunk_map)
        with dask.config.set(scheduler="threads", num_workers=args.num_workers):
            with ProgressBar():
                ds_r.to_zarr(os.path.join(base, f"global_static_{suffix}.zarr"), mode="w")
        print(f"  -> global_static_{suffix}.zarr")

    # global_data.zarr -> global_data_1_4.zarr and global_data_1.zarr
    data_path = os.path.join(base, "global_data.zarr")
    print("Loading global_data.zarr and filling interior NaNs ...")
    ds_data = xr.open_zarr(data_path)
    ds_data = xr.where(ds_data.isnull() & mask_ds, 0.0, ds_data)
    for lat_arr, lon_arr, suffix in grids:
        print(f"Regridding global_data -> {suffix}° ...")
        ds_r = ds_data.interp(latitude=lat_arr, longitude=lon_arr)
        ds_r = ds_r.chunk({"time": 1, "depth": -1, "latitude": -1, "longitude": -1})
        with dask.config.set(scheduler="threads", num_workers=args.num_workers):
            with ProgressBar():
                ds_r.to_zarr(os.path.join(base, f"global_data_{suffix}.zarr"), mode="w")
        print(f"  -> global_data_{suffix}.zarr")

if __name__ == "__main__":
    main()
