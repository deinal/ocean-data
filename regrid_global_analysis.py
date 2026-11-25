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

    mask_path = os.path.join(base, "global_analysis_mask.zarr")
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

    # # regrid mask to both grids
    # print(f"Regridding global_analysis_mask -> 1_4° ...")
    # ds_r = ds_mask_prep.interp(latitude=LAT_1_4, longitude=LON_1_4)
    # ds_r = ds_r.notnull().astype(np.uint8)
    # ds_r = ds_r.chunk({"depth": -1, "latitude": -1, "longitude": -1})
    # with dask.config.set(scheduler="threads", num_workers=args.num_workers):
    #     with ProgressBar():
    #         ds_r.to_zarr(os.path.join(base, "global_analysis_mask_1_4.zarr"), mode="w")
    # print(f"  -> global_analysis_mask_1_4.zarr")

    # # global_analysis_static.zarr -> global_analysis_static_1_4.zarr
    # static_path = os.path.join(base, "global_analysis_static.zarr")
    # print("Loading global_analysis_static.zarr and filling interior NaNs ...")
    # ds_static = xr.open_zarr(static_path)
    # ds_static = xr.where(ds_static.isnull() & ocean_surface, 0.0, ds_static)
    # print(f"Regridding global_analysis_static -> 1_4° ...")
    # ds_r = ds_static.interp(latitude=LAT_1_4, longitude=LON_1_4)
    # chunk_map = {"latitude": -1, "longitude": -1}
    # if "depth" in ds_r.dims:
    #     chunk_map["depth"] = -1
    # ds_r = ds_r.chunk(chunk_map)
    # with dask.config.set(scheduler="threads", num_workers=args.num_workers):
    #     with ProgressBar():
    #         ds_r.to_zarr(os.path.join(base, "global_analysis_static_1_4.zarr"), mode="w")
    # print(f"  -> global_analysis_static_1_4.zarr")

    # # global_analysis_data.zarr -> global_analysis_data_1_4.zarr
    # data_path = os.path.join(base, "global_analysis_data.zarr")
    # print("Loading global_analysis_data.zarr and filling interior NaNs ...")
    # ds_data = xr.open_zarr(data_path)
    # ds_data = ds_data.fillna(0.0).where(mask_ds)
    # print(f"Regridding global_analysis_data -> 1_4° ...")
    # ds_r = ds_data.interp(latitude=LAT_1_4, longitude=LON_1_4)
    # ds_r = ds_r.chunk({"time": 1, "depth": -1, "latitude": -1, "longitude": -1})
    # with dask.config.set(scheduler="threads", num_workers=args.num_workers):
    #     with ProgressBar():
    #         ds_r.to_zarr(os.path.join(base, "global_analysis_data_1_4.zarr"), mode="w")
    # print(f"  -> global_analysis_data_1_4.zarr")

    # glo12_nowcasts.zarr -> glo12_nowcasts_1_4.zarr
    data_path = os.path.join(base, "glo12_nowcasts.zarr")
    print("Loading glo12_nowcasts.zarr and filling interior NaNs ...")
    ds_data = xr.open_zarr(data_path)
    ds_data["time"] = ds_data["time"].dt.floor("D").values
    # nowcast has ice values outside of bounds
    ds_data["siconc"] = ds_data["siconc"].clip(min=0.0, max=1.0)
    ds_data["sithick"] = ds_data["sithick"].clip(min=0.0)
    ds_data = ds_data.fillna(0.0).where(mask_ds)
    print(f"Regridding glo12_nowcasts -> 1_4° ...")
    ds_r = ds_data.interp(latitude=LAT_1_4, longitude=LON_1_4)
    ds_r = ds_r.chunk({"time": 1, "depth": -1, "latitude": -1, "longitude": -1})
    with dask.config.set(scheduler="threads", num_workers=args.num_workers):
        with ProgressBar():
            ds_r.to_zarr(os.path.join(base, "glo12_nowcasts_1_4.zarr"), mode="w")
    print(f"  -> glo12_nowcasts_1_4.zarr")

if __name__ == "__main__":
    main()
