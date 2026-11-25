#!/usr/bin/env python3
"""
Create Baltic boundary data, static, and mask by cropping global_data.zarr and
global_static.zarr to the boundary box. Uses baltic_mask.zarr for distance-to-sea
and the same Lambert projection and threshold as fetch_boundary_static.
"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import dask
from dask.diagnostics import ProgressBar


MIN_LON, MAX_LON = 5.0, 9.0
MIN_LAT, MAX_LAT = 55.0, 60.0
MAX_DEPTH = 100.0
R_EARTH = 6371229.0


def discard_interior(ds):
    return ds.where(~((ds.longitude > 9) & (ds.latitude < 56)))


lambert_proj = None
src_crs = ccrs.PlateCarree()


def main():
    global lambert_proj
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="data", help="Base directory")
    parser.add_argument("--data", type=str, default="global_data.zarr", help="Global data zarr name")
    parser.add_argument("--static", type=str, default="global_static.zarr", help="Global static zarr name")
    parser.add_argument("--dist-threshold-km", type=float, default=150.0, help="Distance (km) for boundary mask")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of workers for dask (default: %(default)s)")
    args = parser.parse_args()
    base = args.base_dir
    os.makedirs(base, exist_ok=True)

    lambert_proj = ccrs.LambertConformal(
        central_longitude=20.0,
        central_latitude=60.0,
        standard_parallels=(60.0, 60.0),
        globe=ccrs.Globe(semimajor_axis=R_EARTH, semiminor_axis=R_EARTH),
    )

    data_path = os.path.join(base, args.data)
    static_path = os.path.join(base, args.static)
    baltic_mask_path = os.path.join(base, "baltic_mask.zarr")

    print("Loading global data and static...")
    ds_data = xr.open_zarr(data_path)
    ds_static = xr.open_zarr(static_path)

    ds_data = ds_data.sel(
        longitude=slice(MIN_LON, MAX_LON),
        latitude=slice(MIN_LAT, MAX_LAT),
    )
    ds_static = ds_static.sel(
        longitude=slice(MIN_LON, MAX_LON),
        latitude=slice(MIN_LAT, MAX_LAT),
    )

    if "depth" in ds_data.dims:
        depths = ds_data.depth.values
        depth_sel = depths[depths <= MAX_DEPTH]
        ds_data = ds_data.sel(depth=depth_sel)
        print("Depth levels (<= 100 m):", list(ds_data.depth.values))

    ds_data = discard_interior(ds_data)
    ds_static = discard_interior(ds_static)

    if "depth" in ds_static.dims:
        depths_s = ds_static.depth.values
        depth_sel_s = depths_s[depths_s <= MAX_DEPTH]
        ds_static = ds_static.sel(depth=depth_sel_s)

    static_out = ds_static[["deptho", "mdt"]].astype("float32")
    chunk_map = {dim: size for dim, size in static_out.sizes.items()}
    static_out = static_out.chunk(chunk_map)
    static_out.to_zarr(os.path.join(base, "baltic_boundary_static.zarr"), mode="w")
    print("Saved baltic_boundary_static.zarr")

    chunk_map = {dim: size for dim, size in ds_data.sizes.items()}
    ds_data = ds_data.chunk(chunk_map)
    with dask.config.set(scheduler="threads", num_workers=args.num_workers):
            with ProgressBar():
                ds_data.to_zarr(os.path.join(base, "baltic_boundary_data.zarr"), mode="w")
    print("Saved baltic_boundary_data.zarr")

    if "depth" in ds_data.dims:
        ocean_mask_3d = ds_data.thetao.isel(time=0).notnull().astype(np.uint8)
    else:
        ocean_mask_3d = ds_data.thetao.isel(time=0).notnull().astype(np.uint8).expand_dims(depth=[0])

    baltic_grid = xr.open_zarr(baltic_mask_path)
    baltic_surface_mask = baltic_grid.thetao.isel(depth=0).astype(bool).values
    baltic_lats = baltic_grid.latitude.values
    baltic_lons = baltic_grid.longitude.values

    b_lat2d, b_lon2d = np.meshgrid(baltic_lats, baltic_lons, indexing="ij")
    true_lon_vals = b_lon2d[baltic_surface_mask]
    westmost_lon = true_lon_vals.min()
    lon_idx = int(np.argmin(np.abs(baltic_lons - westmost_lon)))
    col_mask = baltic_surface_mask[:, lon_idx]
    mask_lat_points = baltic_lats[col_mask]
    mask_lon_points = np.full(mask_lat_points.shape, baltic_lons[lon_idx], dtype=float)
    print("Using westmost Baltic surface lon:", float(baltic_lons[lon_idx]), "with", mask_lat_points.size, "points")

    baltic_pts = lambert_proj.transform_points(src_crs, mask_lon_points, mask_lat_points)
    baltic_x = baltic_pts[:, 0]
    baltic_y = baltic_pts[:, 1]

    bnd_lats = ds_data.latitude.values
    bnd_lons = ds_data.longitude.values
    lon2d, lat2d = np.meshgrid(bnd_lons, bnd_lats)
    bnd_pts = lambert_proj.transform_points(src_crs, lon2d, lat2d)
    bnd_x = bnd_pts[:, :, 0]
    bnd_y = bnd_pts[:, :, 1]

    r2 = (args.dist_threshold_km * 1000.0) ** 2
    mask_bnd = np.zeros_like(bnd_x, dtype=np.uint8)
    for xi, yi in zip(baltic_x, baltic_y):
        dist2 = (bnd_x - xi) ** 2 + (bnd_y - yi) ** 2
        mask_bnd[dist2 <= r2] = 1

    mask_dist_da = xr.DataArray(
        mask_bnd,
        dims=("latitude", "longitude"),
        coords={"latitude": bnd_lats, "longitude": bnd_lons},
    )

    dist_mask_bool = mask_dist_da.astype(bool)
    ocean_bool = ocean_mask_3d.astype(bool)
    boundary_dist_mask = (ocean_bool & dist_mask_bool).astype(np.uint8)

    variable_depth = {
        "thetao": True,
        "so": True,
        "uo": True,
        "vo": True,
        "zos": False,
    }
    feature_mask_dict = {}
    for var, has_depth in variable_depth.items():
        if has_depth:
            feature_mask_dict[var] = boundary_dist_mask.copy(deep=False)
        else:
            feature_mask_dict[var] = boundary_dist_mask.isel(depth=0).copy(deep=False)

    feature_mask_ds = xr.Dataset(data_vars=feature_mask_dict)
    chunk_map = {dim: size for dim, size in feature_mask_ds.sizes.items()}
    feature_mask_ds = feature_mask_ds.chunk(chunk_map)
    feature_mask_ds.to_zarr(os.path.join(base, "baltic_boundary_mask.zarr"), mode="w")
    print("Saved baltic_boundary_mask.zarr")

    
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    static_plot = xr.open_zarr(os.path.join(base, "baltic_boundary_static.zarr"))
    static_plot.mdt.plot()
    plt.savefig(os.path.join(plot_dir, "baltic_boundary_mdt.png"))
    plt.close()

    static_plot.deptho.plot()
    plt.savefig(os.path.join(plot_dir, "baltic_boundary_deptho.png"))
    plt.close()

    bdm = boundary_dist_mask
    bdm.isel(depth=0).plot()
    plt.title("Boundary distance mask (surface)")
    plt.savefig(os.path.join(plot_dir, "baltic_boundary_dist_mask_surface.png"))
    plt.close()

    bdm.isel(depth=-1).plot()
    plt.title("Boundary distance mask (deepest)")
    plt.savefig(os.path.join(plot_dir, "baltic_boundary_dist_mask_deep.png"))
    plt.close()


    print("Done.")


if __name__ == "__main__":
    main()
