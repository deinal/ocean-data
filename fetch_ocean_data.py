import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm
import xarray_regrid
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
parser.add_argument(
    "--out",
    "-o",
    help="Output Zarr store path",
    default=None,
)
parser.add_argument(
    "--start", "-s",
    required=True,
    help="Start date in YYYY-MM-DD format"
)
parser.add_argument(
    "--end", "-e",
    required=True,
    help="End date in YYYY-MM-DD format"
)
args = parser.parse_args()

fpath = os.path.join(args.base_dir, "ocean_data.zarr")

start_date = pd.to_datetime(args.start).strftime("%Y-%m-%d")
end_date = pd.to_datetime(args.end).strftime("%Y-%m-%d")

# Select every 5th depth level
depth_slice = slice(None, None, 5)

# Time range from July 1, 2021
time_index = pd.date_range(start_date, end_date, freq="D")

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
        variables = ["zos", "thetao", "so", "uo", "vo"]
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
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["zos"],
            start_datetime=t0,
            end_datetime=t0,
        )

        ds2 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            start_datetime=t0,
            end_datetime=t0,
            maximum_depth=200,
        ).isel(depth=depth_slice)

        ds3 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=["so"],
            start_datetime=t0,
            end_datetime=t0,
            maximum_depth=200,
        ).isel(depth=depth_slice)
                
        ds4 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            start_datetime=t0,
            end_datetime=t0,
            maximum_depth=200,
        ).isel(depth=depth_slice)


        ds = xr.merge([ds1, ds2, ds3, ds4])

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


def main():
    start_idx = 0
    print("Initializing new Zarr store...", flush=True)
    ds_first = fetch_and_regrid(time_index[0])
    ds_first.to_zarr(fpath, mode="w")
    print(f"Initialized {fpath} with {time_index[0].date()}", flush=True)
    start_idx = 1

    for dt in time_index[start_idx:]:
        try:
            ds_day = fetch_and_regrid(dt)
            ds_day.to_zarr(fpath, mode="a", append_dim="time")
            print("Appended", dt.date(), flush=True)
        except Exception as e:
            print(f"Failed on {dt.date()}: {e}", flush=True)
            break

    print("Processing complete.", flush=True)


if __name__ == "__main__":
    main()
