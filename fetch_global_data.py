import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm
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
    default="2020-01-01",
    help="Start date in YYYY-MM-DD format"
)
parser.add_argument(
    "--end", "-e",
    default="2020-03-31",
    help="End date in YYYY-MM-DD format"
)
args = parser.parse_args()

fpath = os.path.join(args.base_dir, "global_data.zarr")

start_date = pd.to_datetime(args.start).strftime("%Y-%m-%d")
end_date = pd.to_datetime(args.end).strftime("%Y-%m-%d")

TARGET_DEPTHS = [0, 47, 92, 222, 318, 541]  # m

time_index = pd.date_range(start_date, end_date, freq="D")

def fetch_day(date):
    t0 = date.strftime("%Y-%m-%d")

    dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    variables = ["thetao", "so", "uo", "vo", "zos", "siconc", "sithick"]
    ds = cm.open_dataset(
        dataset_id=dataset_id,
        variables=variables,
        start_datetime=t0,
        end_datetime=t0,
        maximum_depth=550,
    )
    vars_with_depth = [v for v in ds.data_vars if "depth" in ds[v].dims]
    vars_2d = [v for v in ds.data_vars if "depth" not in ds[v].dims]
    depths = ds.depth.values
    depth_indices = sorted(
        set(np.argmin(np.abs(depths - z)) for z in TARGET_DEPTHS)
    )
    ds = xr.merge([
        ds[vars_with_depth].isel(depth=depth_indices),
        ds[vars_2d],
    ])
    ds = ds.assign_coords(depth=np.round(ds.depth.values).astype(int))

    chunk_map = {dim: size for dim, size in ds.sizes.items()}
    ds = ds.chunk(chunk_map)
    return ds.transpose("time", "depth", "latitude", "longitude")


def main():
    start_idx = 0
    print("Initializing new Zarr store...", flush=True)
    ds_first = fetch_day(time_index[0])
    ds_first.to_zarr(fpath, mode="w")
    print(f"Initialized {fpath} with {time_index[0].date()}", flush=True)
    start_idx = 1

    for dt in time_index[start_idx:]:
        try:
            ds_day = fetch_day(dt)
            ds_day.to_zarr(fpath, mode="a", append_dim="time")
            print("Appended", dt.date(), flush=True)
        except Exception as e:
            print(f"Failed on {dt.date()}: {e}", flush=True)
            break

    print("Processing complete.", flush=True)


if __name__ == "__main__":
    main()
