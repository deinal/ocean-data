"""
Interpolate global data to 1/4° and 1° grids.
"""
import os
import argparse
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from scipy.ndimage import distance_transform_edt

# 1/4° grid
LAT_1_4 = np.arange(-80.0, 90.0, 0.25) # [-80.0, -79.75, ..., 89.5 ,  89.75], shape=(680,)
LON_1_4 = np.arange(-180.0, 180.0, 0.25) # [-180.0, -179.75, ..., 179.5, 179.75], shape=(1440,)


def _reduce_dims(da, keep=("depth",)):
    return [d for d in da.dims if d not in keep]

def _nearest_fill_2d(arr):
    """True 2-D nearest-neighbour fill via distance transform.
    ~one C pass per slice; orders of magnitude faster than chained interp1d."""
    invalid = np.isnan(arr)
    if not invalid.any():
        return arr
    if invalid.all():
        return np.zeros_like(arr, dtype=arr.dtype)
    ind = distance_transform_edt(
        invalid, return_distances=False, return_indices=True,
    )
    return arr[tuple(ind)].astype(arr.dtype, copy=False)

def _nearest_fill_lonlat(da):
    """Parallel, per-(time, depth) 2-D nearest fill over (latitude, longitude)."""
    chunks = {}
    if "longitude" in da.dims: chunks["longitude"] = -1
    if "latitude"  in da.dims: chunks["latitude"]  = -1
    if "time"      in da.dims: chunks["time"]      = 1
    if "depth"     in da.dims: chunks["depth"]     = 1
    da = da.chunk(chunks)
    return xr.apply_ufunc(
        _nearest_fill_2d, da,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["latitude", "longitude"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[da.dtype],
    )

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
        default=10,
        help="Number of workers for dask (default: %(default)s)",
    )
    
    args = parser.parse_args()
    base = args.base_dir

    grids = [(LAT_1_4, LON_1_4, "1_4")]

    mask_path = os.path.join(base, "global_mask.zarr")
    ds_mask = xr.open_zarr(mask_path)
    ocean_surface = (ds_mask["zos"] == 1)
    ocean = (ds_mask["thetao"] == 1)

    mask_ds = xr.Dataset({
        "zos": ocean_surface,
        "siconc": ocean_surface,
        "sithick": ocean_surface,
        "uo": ocean,
        "vo": ocean,
        "thetao": ocean,
        "so": ocean,
    })

    global_datasets = ["global_data_jan_2025", "glo12_nowcasts", "global_analysis_mondays_2024"]

    for name in global_datasets:
        data_path = os.path.join(base, f"{name}.zarr")

        print(f"Loading {name}.zarr and filling interior NaNs ...")
        ds_data = xr.open_zarr(data_path)
        ds_data = ds_data.assign_coords(
            longitude=ds_mask.longitude,
            latitude=ds_mask.latitude,
        )
        if name in ["glo12_nowcasts", "global_analysis_mondays_2024"]:
            ds_data["siconc"] = ds_data["siconc"].clip(min=0.0, max=1.0)
            ds_data["sithick"] = ds_data["sithick"].clip(min=0.0, max=4.0)

        if name == "glo12_nowcasts":
            ds_data["time"] = ds_data["time"].dt.floor("D").values

        ds_data = ds_data.assign(
            {
                n: (da.fillna(0.0) if n in {"siconc", "sithick"}
                    else _nearest_fill_lonlat(da))
                for n, da in ds_data.data_vars.items()
            }
        ).where(mask_ds)

        #ds_data = ds_data.fillna(0.0).where(mask_ds)
        for lat_arr, lon_arr, suffix in grids:
            print(f"Regridding {name} -> {suffix}° ...")
            ds_r = ds_data.interp(latitude=lat_arr, longitude=lon_arr)
            ds_r = ds_r.chunk({"time": 1, "depth": -1, "latitude": -1, "longitude": -1})
            with dask.config.set(scheduler="threads", num_workers=args.num_workers):
                with ProgressBar():
                    ds_r.to_zarr(os.path.join(base, f"{name}_{suffix}.zarr"), mode="w")
            print(f"  -> {name}_{suffix}.zarr")

if __name__ == "__main__":
    main()
