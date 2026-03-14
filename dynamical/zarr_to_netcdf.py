"""
Convert GFS forecasts from Zarr to NetCDF format.
"""
import argparse
import os
import xarray as xr
from dask.distributed import Client
from dask.diagnostics import ProgressBar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir",
        type=str,
        default="data",
        help="Directory containing the zarr file",
    )
    p.add_argument(
        "--zarr-name",
        type=str,
        default="gfs_forecasts.zarr",
        help="Name of the zarr file to read",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NetCDF file path",
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

    # Construct paths
    zarr_path = os.path.join(args.base_dir, args.zarr_name)
    if args.output:
        nc_path = args.output
    else:
        # Default: replace .zarr with .nc
        nc_path = os.path.join(args.base_dir, args.zarr_name.replace(".zarr", ".nc"))

    print(f"Reading zarr from: {zarr_path}", flush=True)
    ds = xr.open_zarr(zarr_path, chunks=None)
    print(ds, flush=True)

    # Remove all attributes
    ds.attrs = {}
    for coord in ds.coords:
        ds[coord].attrs = {}

    print(f"Writing NetCDF to: {nc_path}", flush=True)
    with ProgressBar():
        ds.to_netcdf(nc_path, mode="w")

    print(f"Cnverted to NetCDF: {nc_path}", flush=True)
    client.close()


if __name__ == "__main__":
    main()
