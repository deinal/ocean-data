import argparse
import os
import numpy as np
import xarray as xr
import earthkit.data as ekd
import xarray_regrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/monolith/global_data/ml_datasets/baltic_sea/data/prep",
        help="Base output directory (default: %(default)s)"
    )
    args = parser.parse_args()
    os.makedirs(args.base_dir, exist_ok=True)

    fpath = os.path.join(args.base_dir, "global_era5_static.zarr")

    # Target grid (match global ocean scripts)
    lat_target = np.arange(90.0, -91.0, -1.5)
    lon_target = np.arange(0.0, 360.0, 1.5)
    ds_target_grid = xr.Dataset(
        coords={
            "latitude": ("latitude", lat_target),
            "longitude": ("longitude", lon_target),
        }
    )

    ds_geo = ekd.from_source(
        "cds",
        "reanalysis-era5-land",
        variable="geopotential",
        data_format="netcdf",
        download_format="unarchived",
    ).to_xarray()

    # one timestep -> static
    if "valid_time" in ds_geo:
        ds_geo = ds_geo.rename({"valid_time": "time"})
    ds_geo = ds_geo.isel(time=0, drop=True)

    # geopotential height in meters
    ds_geo["geopotential_height"] = ds_geo.z / 9.80665
    ds_geo["geopotential_height"].attrs["units"] = "m"
    ds_geo = ds_geo.drop_vars("z")

    # sort for regrid
    ds_geo = ds_geo.sortby("latitude")
    ds_geo = ds_geo.sortby("longitude")

    ds_re = ds_geo.regrid.conservative(
        ds_target_grid,
        latitude_coord="latitude",
        skipna=True,
    )

    # Load ocean mask to get ocean points
    ocean_mask_path = os.path.join(args.base_dir, "ocean_mask.zarr")
    if os.path.exists(ocean_mask_path):
        ocean_mask = xr.open_zarr(ocean_mask_path)
        # Get surface mask
        if "zos" in ocean_mask.data_vars:
            surface_mask = ocean_mask["zos"].astype(bool)
        else:
            surface_mask = ocean_mask[list(ocean_mask.data_vars)[0]].isel(depth=0).astype(bool)
        
        # Get ocean point coordinates
        ocean_lat = surface_mask.latitude.values
        ocean_lon = surface_mask.longitude.values
        
        # Create 2D grids
        lat2d_ocean, lon2d_ocean = np.meshgrid(ocean_lat, ocean_lon, indexing="ij")
        
        # Select ocean points (where mask is True)
        ocean_mask_values = surface_mask.values
        ocean_lat_points = lat2d_ocean[ocean_mask_values]
        ocean_lon_points = lon2d_ocean[ocean_mask_values]
        
        print(f"Found {len(ocean_lat_points)} ocean points", flush=True)
        
        # Get ERA5 grid coordinates
        era5_lats = ds_re.latitude.values
        era5_lons = ds_re.longitude.values
        
        # Distance threshold
        dist_threshold_deg = 1.5
        
        print("Computing distance-based mask...", flush=True)
        
        # Initialize mask
        mask_era5 = np.zeros((len(era5_lats), len(era5_lons)), dtype=np.uint8)
        
        # Convert to radians for distance calculation
        era5_lats_rad = np.deg2rad(era5_lats)
        era5_lons_rad = np.deg2rad(era5_lons)
        ocean_lats_rad = np.deg2rad(ocean_lat_points)
        ocean_lons_rad = np.deg2rad(ocean_lon_points)
        
        # For each ocean point, mark nearby ERA5 grid points
        for ocean_lat_rad, ocean_lon_rad in zip(ocean_lats_rad, ocean_lons_rad):
            # Compute distances to all ERA5 grid points
            dlat = ocean_lat_rad - era5_lats_rad[:, np.newaxis]
            dlon = ocean_lon_rad - era5_lons_rad[np.newaxis, :]
            
            # Handle longitude wrapping (ensure dlon is in [-180, 180])
            dlon = np.mod(dlon + np.pi, 2 * np.pi) - np.pi
            
            # Haversine formula
            a = (np.sin(dlat/2)**2 + 
                 np.cos(era5_lats_rad[:, np.newaxis]) * np.cos(ocean_lat_rad) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical errors
            distances_deg = np.rad2deg(c)
            
            # Mark points within threshold
            mask_era5[distances_deg <= dist_threshold_deg] = 1
        
        mask_da = xr.DataArray(
            mask_era5,
            dims=("latitude", "longitude"),
            coords={"latitude": era5_lats, "longitude": era5_lons},
            name="mask",
        )
        mask_da.attrs["description"] = f"Binary mask: ERA5 points within {dist_threshold_deg} degrees of ocean"
        
        # Add to regridded dataset
        ds_re["mask"] = mask_da
        print(f"Created mask: {np.sum(mask_era5)}/{mask_era5.size} points marked", flush=True)
    else:
        print(f"Warning: {ocean_mask_path} not found, skipping mask creation", flush=True)

    chunk_map = {dim: size for dim, size in ds_re.sizes.items()}
    ds_re = ds_re.chunk(chunk_map)

    ds_re.to_zarr(fpath, mode="w")
    print("Saved ERA5 global static data to", fpath, flush=True)


if __name__ == "__main__":
    main()

