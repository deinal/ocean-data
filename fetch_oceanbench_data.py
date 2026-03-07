"""
Fetch OceanBench input datasets (GLO12 nowcasts + IFS forcings) and save as zarr.
Source: https://oceanbench.readthedocs.io/en/latest/source/oceanbench.datasets.html
"""
import argparse
import numpy as np
import os
import xarray as xr

from dask.diagnostics import ProgressBar
from oceanbench.datasets.input import glo12_nowcasts, ifs_forcings

DEFAULT_OUT = "/monolith/global_data/ml_datasets/baltic_sea/data/prep"
TARGET_DEPTHS = [0, 50, 100, 200, 500]  # m
TARGET_VARS = ["thetao", "so", "uo", "vo", "zos", "siconc", "sithick"]
CHUNK_LAT, CHUNK_LON = 256, 540

parser = argparse.ArgumentParser(description="Fetch OceanBench input datasets.")
parser.add_argument(
    "--out-dir",
    type=str,
    default=DEFAULT_OUT,
    help="Output directory (default: %(default)s)",
)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

nowcast_path = os.path.join(args.out_dir, "glo12_nowcasts.zarr")
ds = glo12_nowcasts()

# Subset variables
vars_present = [v for v in TARGET_VARS if v in ds.data_vars]
ds = ds[vars_present]

# Subset depth only for variables that have a depth dimension
vars_with_depth = [v for v in ds.data_vars if "depth" in ds[v].dims]
vars_2d = [v for v in ds.data_vars if "depth" not in ds[v].dims]

depths = ds.depth.values
depth_indices = sorted(set(np.argmin(np.abs(depths - z)) for z in TARGET_DEPTHS))
ds_depth = ds[vars_with_depth].isel(depth=depth_indices)
n_depth = len(depth_indices)
ds_2d = ds[vars_2d]
ds_nowcasts = xr.merge([ds_depth, ds_2d])

# Subset tim to slice for testing
# ds_nowcasts = ds_nowcasts.isel(time=slice(0, 10))

# Rechunk
chunk_dict = {"time": 1, "depth": n_depth, "latitude": CHUNK_LAT, "longitude": CHUNK_LON}
ds_nowcasts = ds_nowcasts.chunk(chunk_dict)
encoding = {}
for v in ds_nowcasts.data_vars:
    dims = list(ds_nowcasts[v].dims)
    if "depth" in dims:
        encoding[v] = {"chunks": (1, n_depth, CHUNK_LAT, CHUNK_LON)}
    else:
        encoding[v] = {"chunks": (1, CHUNK_LAT, CHUNK_LON)}

print("GLO12 nowcasts (depths ~0, 50, 100, 200, 500 m):", flush=True)
print(ds_nowcasts, flush=True)
print(f"Writing {nowcast_path} ...", flush=True)
with ProgressBar():
    ds_nowcasts.to_zarr(nowcast_path, mode="w", encoding=encoding)
print(f"  -> {nowcast_path}\n")



# IFS forcings
forcings_path = os.path.join(args.out_dir, "ifs_forcings.zarr")
ds_forcings = ifs_forcings()
print(ds_forcings, flush=True)
# Drop conflicting fill/missing attrs so zarr can encode
for v in ds_forcings.data_vars:
    a = ds_forcings[v].attrs
    if "_FillValue" in a and "missing_value" in a:
        del a["missing_value"]  # keep _FillValue (nan) only
print(f"Writing {forcings_path} ...", flush=True)
with ProgressBar():
    ds_forcings.to_zarr(forcings_path, mode="w")
print(f"  -> {forcings_path}\n")
