import os
import argparse
import numpy as np
import xarray as xr
import copernicusmarine as cm
import xarray_regrid


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

# Select every 5th depth level
depth_slice = slice(None, None, 5)

# Target grid: 1° spacing, lon -180..180, lat -80..90
lat_target = np.arange(-80.0, 91.0, 1.0)
lon_target = np.arange(-180.0, 180.0, 1.0)
ds_target_grid = xr.Dataset(coords={
    "latitude": ("latitude", lat_target),
    "longitude": ("longitude", lon_target),
})

def regrid_static(ds: xr.Dataset, skipna=True) -> xr.Dataset:
    """Regrid dataset to 1° via bilinear. For 'mask', treat 0 as NaN before regridding, then binarize."""
    if "mask" in ds:
        # Temporarily replace land (0) with NaN
        mask = ds["mask"].where(ds["mask"] != 0)
        ds = ds.copy()
        ds["mask"] = mask

    ds_regridded = ds.regrid.linear(ds_target_grid)

    # Binarize regridded mask (bilinear gives fractional values)
    if "mask" in ds_regridded:
        ds_regridded["mask"] = (ds_regridded["mask"].fillna(0) > 0.5).astype(np.int8)

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

    # Merge
    ds_merged = xr.merge([ds_mdt, ds_bathy])
    
    # Transpose to consistent order (depth, latitude, longitude if depth exists)
    if "depth" in ds_merged.dims:
        ds_merged = ds_merged.transpose("depth", "latitude", "longitude")
    else:
        ds_merged = ds_merged.transpose("latitude", "longitude")

    return ds_merged

def main():
    ds_static = fetch_and_regrid_static()

    # Create feature mask for global ocean variables
    variable_depth = {
        "zos": False,
        "thetao": True,
        "so": True,
        "uo": True,
        "vo": True,
    }

    # Selected depth levels
    mask_selection = ds_static["mask"]

    # Surface mask
    surface_mask = ds_static["mask"].isel(depth=0)

    # Build output dataset
    feature_mask_dict = {}

    for var, has_depth in variable_depth.items():
        if has_depth:
            feature_mask_dict[var] = mask_selection.copy(deep=False)
        else:
            feature_mask_dict[var] = surface_mask.copy(deep=False)

    # Combine as Dataset
    feature_mask_ds = xr.Dataset(data_vars=feature_mask_dict)

    static_out = os.path.join(args.base_dir, "ocean_static.zarr")
    mask_out = os.path.join(args.base_dir, "ocean_mask.zarr")

    for vn in ds_static.data_vars:
        ds_static[vn] = ds_static[vn].astype("float32")
    
    print("Static dataset:", flush=True)
    print(ds_static, flush=True)
    
    # Save static variables (deptho, mdt)
    static_ds = ds_static[["deptho", "mdt"]]
    static_ds = static_ds.chunk({"latitude": -1, "longitude": -1})
    static_ds.to_zarr(static_out, mode="w")
    print(f"Saved ocean static to {static_out}", flush=True)

    # Save feature mask
    feature_mask_ds = feature_mask_ds.chunk({"depth": -1, "latitude": -1, "longitude": -1})
    feature_mask_ds.to_zarr(mask_out, mode="w")
    print(f"Saved ocean mask to {mask_out}", flush=True)

if __name__ == "__main__":
    main()
