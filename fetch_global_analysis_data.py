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
    default="data",
    help="Base output directory (default: %(default)s)",
)
parser.add_argument(
    "--start",
    "-s",
    default="2023-01-01",
    help="Start date in YYYY-MM-DD format",
)
parser.add_argument(
    "--end",
    "-e",
    default="2025-01-31",
    help="End date in YYYY-MM-DD format",
)
args = parser.parse_args()


fpath = os.path.join(args.base_dir, "global_analysis_data.zarr")

start_date = pd.to_datetime(args.start)
end_date = pd.to_datetime(args.end)

TARGET_DEPTHS = [0, 47, 92, 222, 318, 541]  # m
VARS_3D = ["thetao", "so", "uo", "vo"]
VARS_2D = ["zos", "siconc", "sithick"]

# Analysis/forecast (anfc) datasets are split per variable group.
DATASETS_3D = {
    "curr":   ("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",    ["uo", "vo"]),
    "so":     ("cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",     ["so"]),
    "thetao": ("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ["thetao"]),
}
DATASET_ID_2D = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(
        start=start.normalize().replace(day=1),
        end=end.normalize().replace(day=1),
        freq="MS",
    )


def month_bounds(month_start: pd.Timestamp, global_start: pd.Timestamp, global_end: pd.Timestamp):
    month_end = month_start + pd.offsets.MonthEnd(0)
    t0 = max(month_start, global_start)
    t1 = min(month_end, global_end)
    return t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")


def fetch_2d_month(t0: str, t1: str) -> xr.Dataset:
    ds = cm.open_dataset(
        dataset_id=DATASET_ID_2D,
        variables=VARS_2D,
        start_datetime=t0,
        end_datetime=t1,
    )
    return ds


def fetch_3d_depth_month(
    dataset_id: str,
    variables: list,
    t0: str,
    t1: str,
    target_depth: int,
) -> xr.Dataset:
    zmin = max(0, target_depth - 1)
    zmax = target_depth + 1

    ds = cm.open_dataset(
        dataset_id=dataset_id,
        variables=variables,
        start_datetime=t0,
        end_datetime=t1,
        minimum_depth=zmin,
        maximum_depth=zmax,
    )

    if "depth" not in ds.dims:
        raise ValueError(
            f"No depth dimension returned for dataset {dataset_id} "
            f"at target depth {target_depth} m"
        )

    # We expect exactly one native level inside [z-1, z+1].
    if ds.sizes["depth"] == 0:
        raise ValueError(
            f"No depth returned within [{zmin}, {zmax}] for target "
            f"depth {target_depth} m in dataset {dataset_id}"
        )

    if ds.sizes["depth"] > 1:
        raise ValueError(
            f"Ambiguous depth selection for target {target_depth} m "
            f"in dataset {dataset_id}; native depths in window: "
            f"{ds['depth'].values.tolist()}"
        )

    # Force exact target depth coordinate in the output dataset
    ds = ds.assign_coords(depth=[int(target_depth)])
    return ds


def fetch_month(month_start: pd.Timestamp) -> xr.Dataset:
    t0, t1 = month_bounds(month_start, start_date, end_date)
    print(f"Fetching month {t0} to {t1}", flush=True)

    ds_2d = fetch_2d_month(t0, t1)

    per_group = []
    for group_name, (dataset_id, variables) in DATASETS_3D.items():
        depth_slices = []
        for z in TARGET_DEPTHS:
            print(
                f"  Fetching {group_name} {variables} at target depth {z} m",
                flush=True,
            )
            ds_z = fetch_3d_depth_month(dataset_id, variables, t0, t1, z)
            depth_slices.append(ds_z)

        ds_group = xr.concat(depth_slices, dim="depth")
        ds_group = ds_group.assign_coords(
            depth=np.array(TARGET_DEPTHS, dtype=int)
        )
        per_group.append(ds_group)

    ds_3d = xr.merge(per_group)
    ds = xr.merge([ds_3d, ds_2d])

    # Standardize dimension order where present
    ordered = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        if "depth" in dims:
            ordered[v] = ds[v].transpose("time", "depth", "latitude", "longitude")
        else:
            ordered[v] = ds[v].transpose("time", "latitude", "longitude")
    ds = xr.Dataset(ordered, coords=ds.coords, attrs=ds.attrs)

    # Chunking requested:
    # 3D vars -> time=1, depth=-1, latitude=-1, longitude=-1
    # 2D vars -> time=1, latitude=-1, longitude=-1
    chunked = {}
    for v in ds.data_vars:
        if "depth" in ds[v].dims:
            chunked[v] = ds[v].chunk({
                "time": 1,
                "depth": -1,
                "latitude": -1,
                "longitude": -1,
            })
        else:
            chunked[v] = ds[v].chunk({
                "time": 1,
                "latitude": -1,
                "longitude": -1,
            })

    ds = xr.Dataset(chunked, coords=ds.coords, attrs=ds.attrs)
    return ds


def get_encoding(ds: xr.Dataset) -> dict:
    nd = ds.sizes["depth"]
    ny = ds.sizes["latitude"]
    nx = ds.sizes["longitude"]
    encoding = {}

    for v in VARS_3D:
        encoding[v] = {"chunks": (1, nd, ny, nx)}
    for v in VARS_2D:
        encoding[v] = {"chunks": (1, ny, nx)}

    return encoding


def main():
    months = month_starts(start_date, end_date)

    print("Initializing new Zarr store...", flush=True)
    ds_first = fetch_month(months[0])
    encoding = get_encoding(ds_first)
    ds_first.to_zarr(fpath, mode="w", encoding=encoding)
    print(f"Initialized {fpath} with first month {months[0].strftime('%Y-%m')}", flush=True)

    for month in months[1:]:
        try:
            ds_month = fetch_month(month)
            ds_month.to_zarr(fpath, mode="a", append_dim="time")
            print(f"Appended {month.strftime('%Y-%m')}", flush=True)
        except Exception as e:
            print(f"Failed on {month.strftime('%Y-%m')}: {e}", flush=True)
            break

    print("Processing complete.", flush=True)


if __name__ == "__main__":
    main()
