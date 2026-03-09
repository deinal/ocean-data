import os
import argparse
import numpy as np
import xarray as xr
import copernicusmarine as cm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Earth radius (m) for great-circle distance
R_EARTH = 6371229.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=str,
    default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
    help="Base output directory (default: %(default)s)"
)
parser.add_argument(
    "--n-workers",
    type=int,
    default=20,
    help="Number of workers for parallel queries (default: %(default)s)"
)
args = parser.parse_args()
os.makedirs(args.base_dir, exist_ok=True)

TARGET_DEPTHS = [0, 47, 92, 222, 318, 541]  # m


def fetch_static():
    # MDT
    ds_mdt = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_static",
        dataset_part="mdt",
        variables=["mdt"],
    )

    # Bathymetry + mask
    ds_bathy = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_static",
        dataset_part="bathy",
        variables=["deptho", "mask"],
        maximum_depth=550,
    )
    depths = ds_bathy.depth.values
    depth_indices = sorted(
        set(np.argmin(np.abs(depths - z)) for z in TARGET_DEPTHS)
    )
    ds_bathy = ds_bathy.isel(depth=depth_indices)
    ds_bathy = ds_bathy.assign_coords(depth=np.round(ds_bathy.depth.values).astype(int))
    print("Selected depth levels:", list(ds_bathy.depth.values))

    # Merge
    ds_merged = xr.merge([ds_mdt, ds_bathy])

    if "depth" in ds_merged.dims:
        ds_merged = ds_merged.transpose("depth", "latitude", "longitude")
    else:
        ds_merged = ds_merged.transpose("latitude", "longitude")

    return ds_merged

def main():
    ds_static = fetch_static()

    # sin/cos lat/lon as static features
    lat_deg = ds_static.latitude.values
    lon_deg = ds_static.longitude.values
    nlat, nlon = lat_deg.size, lon_deg.size
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    sin_lat = (np.sin(lat_rad).reshape(-1, 1) * np.ones((nlat, nlon), dtype=np.float32)).astype(np.float32)
    cos_lat = (np.cos(lat_rad).reshape(-1, 1) * np.ones((nlat, nlon), dtype=np.float32)).astype(np.float32)
    sin_lon = (np.sin(lon_rad).reshape(1, -1) * np.ones((nlat, nlon), dtype=np.float32)).astype(np.float32)
    cos_lon = (np.cos(lon_rad).reshape(1, -1) * np.ones((nlat, nlon), dtype=np.float32)).astype(np.float32)
    ds_static["sin_lat"] = (("latitude", "longitude"), sin_lat)
    ds_static["cos_lat"] = (("latitude", "longitude"), cos_lat)
    ds_static["sin_lon"] = (("latitude", "longitude"), sin_lon)
    ds_static["cos_lon"] = (("latitude", "longitude"), cos_lon)

    # Distance to coast (m): great-circle via cKDTree on unit-sphere 3D (parallel query)
    mask_2d = ds_static["mask"].isel(depth=0).values
    lat_deg = ds_static.latitude.values
    lon_deg = ds_static.longitude.values
    nlat, nlon = lat_deg.size, lon_deg.size
    lon2d, lat2d = np.meshgrid(lon_deg, lat_deg)
    lat_rad = np.radians(lat2d.ravel())
    lon_rad = np.radians(lon2d.ravel())
    # Unit-sphere Cartesian (x,y,z) for great-circle distance
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    xyz = np.column_stack([x, y, z])
    land = (mask_2d.ravel() == 0)
    ocean = (mask_2d.ravel() == 1)
    land_xyz = xyz[land]
    ocean_xyz = xyz[ocean]
    tree = cKDTree(land_xyz)
    d_chord, _ = tree.query(ocean_xyz, k=1, workers=args.n_workers)
    # Chord length d = 2*sin(angle/2) => angle = 2*arcsin(d/2)
    dist_rad = 2.0 * np.arcsin(np.clip(d_chord.astype(np.float64) / 2.0, 0.0, 1.0))
    dist_m = (dist_rad * R_EARTH).astype(np.float32)
    coast_dist_2d = np.zeros((nlat, nlon), dtype=np.float32)
    coast_dist_2d.ravel()[ocean] = dist_m
    ds_static["coast_dist"] = (("latitude", "longitude"), coast_dist_2d)

    # Plot static data
    os.makedirs("plots", exist_ok=True)

    ds_static.mask.isel(depth=0).plot()
    plt.savefig(os.path.join("plots", "global_mask.png"))
    plt.close()

    ds_static.deptho.plot()
    plt.savefig(os.path.join("plots", "global_deptho.png"))
    plt.close()

    ds_static.mdt.plot()
    plt.savefig(os.path.join("plots", "global_mdt.png"))
    plt.close()

    # Create feature mask for global ocean variables
    variable_depth = {
        "thetao": True,
        "so": True,
        "uo": True,
        "vo": True,
        "zos": False,
        "siconc": False,
        "sithick": False,
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

    static_out = os.path.join(args.base_dir, "global_static.zarr")
    mask_out = os.path.join(args.base_dir, "global_mask.zarr")

    for vn in ds_static.data_vars:
        ds_static[vn] = ds_static[vn].astype("float32")
    
    print("Static dataset:", flush=True)
    print(ds_static, flush=True)
    
    # Save static variables (deptho, mdt, coast_dist, sin_lon, cos_lon, sin_lat, cos_lat)
    static_ds = ds_static[["deptho", "mdt", "sin_lon", "cos_lon", "sin_lat", "cos_lat", "coast_dist"]]
    static_ds = static_ds.chunk({"latitude": -1, "longitude": -1})
    static_ds.to_zarr(static_out, mode="w")
    print(f"Saved global static to {static_out}", flush=True)

    # Save feature mask
    feature_mask_ds = feature_mask_ds.chunk({"depth": -1, "latitude": -1, "longitude": -1})
    feature_mask_ds.to_zarr(mask_out, mode="w")
    print(f"Saved global mask to {mask_out}", flush=True)

if __name__ == "__main__":
    main()
