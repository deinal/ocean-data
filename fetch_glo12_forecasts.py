import argparse
import numpy as np
import os
import pandas as pd
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from oceanbench.datasets.challenger import glo12

DEFAULT_OUT = "data"
TARGET_DEPTHS = [0, 47, 92]  # m
TARGET_VARS = ["thetao", "so", "uo", "vo", "zos"]

MIN_LON, MAX_LON = 5.0, 9.0
MIN_LAT, MAX_LAT = 55.0, 60.0

parser = argparse.ArgumentParser(description="Fetch GLO12 forecasts.")
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument(
    "--out-dir",
    type=str,
    default=DEFAULT_OUT,
    help="Output directory (default: %(default)s)",
)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

nowcast_path = os.path.join(args.out_dir, "glo12_forecasts.zarr")
ds = glo12()

ds = ds.sel(
    longitude=slice(MIN_LON, MAX_LON),
    latitude=slice(MIN_LAT, MAX_LAT),
)

# Subset variables
vars_present = [v for v in TARGET_VARS if v in ds.data_vars]
ds = ds[vars_present]

# Subset depth only for variables that have a depth dimension
vars_with_depth = [v for v in ds.data_vars if "depth" in ds[v].dims]
vars_2d = [v for v in ds.data_vars if "depth" not in ds[v].dims]

depths = ds.depth.values
print("Available depth levels:", depths, flush=True)
depth_indices = sorted(set(np.argmin(np.abs(depths - z)) for z in TARGET_DEPTHS))
ds_depth = ds[vars_with_depth].isel(depth=depth_indices)
print("Chosen depth levels:", ds_depth.depth.values, flush=True)
ds_depth = ds_depth.assign_coords(depth=np.round(ds_depth.depth.values).astype(int))
print("Rounded depth levels:", ds_depth.depth.values, flush=True)
n_depth = len(depth_indices)
ds_2d = ds[vars_2d]
ds_forecasts = xr.merge([ds_depth, ds_2d])


# first_day_datetime is init_time + 1 day -> shift back by 1 day and
# floor to 00:00:00 so it lines up with the other datasets, then rename.
fdd = pd.to_datetime(ds_forecasts["first_day_datetime"].values)
init_times = (fdd - pd.Timedelta(days=1)).floor("D")
ds_forecasts = ds_forecasts.rename({"first_day_datetime": "init_time"})
ds_forecasts = ds_forecasts.assign_coords(
    init_time=init_times.values.astype("datetime64[ns]")
)

# lead_day_index = 0, 1, ..., 9 -> lead_time = 1D, 2D, ..., 10D
ldi = np.asarray(ds_forecasts["lead_day_index"].values)
lead_times = pd.to_timedelta(ldi + 1, unit="D").to_numpy().astype("timedelta64[ns]")
ds_forecasts = ds_forecasts.rename({"lead_day_index": "lead_time"})
ds_forecasts = ds_forecasts.assign_coords(lead_time=lead_times)

print("init_time range:",
      str(pd.Timestamp(ds_forecasts.init_time.values.min()).date()),
      "..",
      str(pd.Timestamp(ds_forecasts.init_time.values.max()).date()),
      flush=True)
print("lead_time values:",
      [str(pd.Timedelta(lt)) for lt in ds_forecasts.lead_time.values],
      flush=True)

n_lat = ds_forecasts.sizes["latitude"]
n_lon = ds_forecasts.sizes["longitude"]

chunk_dict = {
    "init_time": 1,
    "lead_time": 1,
    "depth": n_depth,
    "latitude": n_lat,
    "longitude": n_lon,
}
ds_forecasts = ds_forecasts.chunk(chunk_dict)

encoding = {}
for v in ds_forecasts.data_vars:
    dims = list(ds_forecasts[v].dims)
    if "depth" in dims:
        encoding[v] = {"chunks": (1, 1, n_depth, n_lat, n_lon)}
    else:
        encoding[v] = {"chunks": (1, 1, n_lat, n_lon)}

print(f"GLO12 nowcasts (depths {ds_forecasts.depth.values.tolist()} m):", flush=True)
print(ds_forecasts, flush=True)
print(f"Writing {nowcast_path} ...", flush=True)
with ProgressBar():
    ds_forecasts.to_zarr(nowcast_path, mode="w", encoding=encoding)
print(f"  -> {nowcast_path}\n")