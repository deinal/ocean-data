#!/usr/bin/env python3
import os
import xarray as xr
import matplotlib.pyplot as plt

# Path to your static Zarr store
static_zarr = "data/baltic/static/bathymetry.zarr"

# Directory to save figures
fig_dir = "figures/static"
os.makedirs(fig_dir, exist_ok=True)

# Open the static dataset
ds_static = xr.open_zarr(static_zarr)

for var in ds_static.data_vars:
    da = ds_static[var]

    # Drop any singleton dims
    if "time" in da.dims:
        da = da.isel(time=0)
    if "depth" in da.dims:
        da = da.isel(depth=0)

    # Choose a sensible colormap for mask
    cmap = "gray_r" if var == "mask" else None

    # Plot and save
    fig = da.plot(figsize=(6, 5), cmap=cmap)
    outpath = os.path.join(fig_dir, f"{var}.png")
    fig.figure.savefig(outpath, dpi=150)
    plt.close(fig.figure)
    print(f"Saved static variable plot: {var}.png")

ds_static.close()
print("All static plots done.")


