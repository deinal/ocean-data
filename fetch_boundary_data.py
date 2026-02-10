import os
import argparse
import calendar
import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
parser.add_argument(
    "--start-year",
    type=int,
    default=2010,
    help="Start year (default: %(default)s)"
)
parser.add_argument(
    "--end-year",
    type=int,
    default=2013,
    help="End year (default: %(default)s)"
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

DATASETS = [
    {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["zos", "thetao", "so", "uo", "vo"],
        "start_year": 1993,
        "end_year": 2022,
        "type": "daily",
        "replace_nans": True,
    },
    {
        "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        "variables": ["zos"],
        "start_year": 2023,
        "end_year": 2024,
        "type": "daily",
    },
    {
        "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "variables": ["thetao"],
        "start_year": 2023,
        "end_year": 2025,
        "type": "daily",
    },
    {
        "dataset_id": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        "variables": ["so"],
        "start_year": 2023,
        "end_year": 2025,
        "type": "daily",
    },
    {
        "dataset_id": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "variables": ["uo", "vo"],
        "start_year": 2023,
        "end_year": 2024,
        "type": "daily",
    },
]

# Load boundary grid
ds_boundary = xr.open_zarr(os.path.join(args.base_dir, "boundary_mask.zarr"))
target_lat = ds_boundary.latitude
target_lon = ds_boundary.longitude

# Append to zarr store
out_store = os.path.join(args.base_dir, "boundary_data.zarr")

def to_zarr_append(ds, store):
    if not os.path.exists(store):
        ds.to_zarr(store, mode="w")
    else:
        ds.to_zarr(store, mode="a", append_dim="time")

# Preprocess dataset
def preprocess_ds(ds, ds_type):
    # Subsample depths
    if "depth" in ds.dims:
        # Select depth levels that match the boundary mask
        ds_depths = ds.depth.values
        boundary_depths = ds_boundary.depth.values
        closest_idxs = np.abs(ds_depths[:, None] - boundary_depths[None, :]).argmin(axis=0)
        ds = ds.sel(depth=ds_depths[closest_idxs])
        ds = ds.assign_coords(depth=boundary_depths)

    # Mask out extra river pixel (SW corner)
    ds = ds.where(~((ds.longitude > 9) & (ds.latitude < 56)))

    # Assign target grid coordinates
    ds = ds.assign_coords(
        latitude=("latitude", target_lat.values),
        longitude=("longitude", target_lon.values),
    )

    # Chunk for writing
    if "depth" in ds.dims:
        ds = ds.chunk({"time": ds.sizes["time"], "depth": ds.sizes["depth"]})
    else:
        ds = ds.chunk({"time": ds.sizes["time"]})

    return ds

# Main Loop
print(f"\n=== Building store {out_store} ===")
for year in range(args.start_year, args.end_year + 1):
    for month in range(1, 13):
        monthly_datasets = []
        last_day = calendar.monthrange(year, month)[1]
        start_time = f"{year:04d}-{month:02d}-01T00:00:00"
        end_time = f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59"

        vars_to_replace = set()
        for ds_cfg in DATASETS:
            # Skip if this year is outside the dataset's validity
            if year < ds_cfg["start_year"] or year > ds_cfg["end_year"]:
                continue

            dataset_id = ds_cfg["dataset_id"]
            ds_type = ds_cfg.get("type", "daily")
            
            # Track variables that need NaN replacement
            if ds_cfg.get("replace_nans", False):
                vars_to_replace.update(ds_cfg["variables"])

            print(f"[{year}-{month:02d}] fetching {dataset_id} ({ds_type})…", flush=True)

            ds = cm.open_dataset(
                dataset_id=dataset_id,
                variables=ds_cfg["variables"],
                start_datetime=start_time,
                end_datetime=end_time,
                maximum_depth=110,
                minimum_longitude=5,
                maximum_longitude=9,
                minimum_latitude=55,
                maximum_latitude=60,
            )

            ds = preprocess_ds(ds, ds_type)
            monthly_datasets.append(ds)

        merged = xr.merge(monthly_datasets)
        
        # Replace NaNs inside boundary mask with 0 (concerns reanalysis with slightly different grid)
        for var_name in vars_to_replace:
            if var_name in merged.data_vars:
                nan_in_mask = np.isnan(merged[var_name]) & (ds_boundary[var_name] == 1)
                merged[var_name] = merged[var_name].where(~nan_in_mask, 0.0)
        
        to_zarr_append(merged, out_store)
        print(f"→ Appended {merged.sizes['time']} steps for {year}-{month:02d}\n")
