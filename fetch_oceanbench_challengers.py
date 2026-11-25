import argparse
import numpy as np
import os
import pandas as pd
import xarray as xr
import zarr
import dask
from dask.diagnostics import ProgressBar
from oceanbench.datasets.challenger import glo12, glonet, wenhai, xihe

DEFAULT_OUT = "data"
TARGET_VARS = ["thetao", "so", "uo", "vo", "zos"]
TARGET_DEPTHS = [0, 47, 92, 222, 318, 541]

# 1/4° grid
LAT_1_4 = np.arange(-80.0, 90.0, 0.25) # [-80.0, -79.75, ..., 89.5 ,  89.75], shape=(680,)
LON_1_4 = np.arange(-180.0, 180.0, 0.25) # [-180.0, -179.75, ..., 179.5, 179.75], shape=(1440,)


parser = argparse.ArgumentParser(description="Fetch oceanbench challengers.")
parser.add_argument("--num-workers", type=int, default=10)
parser.add_argument("--challenger", type=str)
parser.add_argument(
    "--out-dir",
    type=str,
    default=DEFAULT_OUT,
    help="Output directory (default: %(default)s)",
)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

data = {"glo12": glo12(), "glonet": glonet(), "wenhai": wenhai(), "xihe": xihe()}

ds = data[args.challenger]

output_path = os.path.join(args.out_dir, f"{args.challenger}_global_forecasts.zarr")

# Subset variables
vars_present = [v for v in TARGET_VARS if v in ds.data_vars]
ds = ds[vars_present]

# Subset depth only for variables that have a depth dimension
vars_with_depth = [v for v in ds.data_vars if "depth" in ds[v].dims]
vars_2d = [v for v in ds.data_vars if "depth" not in ds[v].dims]

depths = ds.depth.values
print("Available depth levels:", depths, flush=True)
ds_depth = ds[vars_with_depth]
if args.challenger in ["wenhai", "xihe"]:
    # 1. Select the exact targets + the two bounding depths for 47.37 (40 and 55)
    # This aggressively reduces the dataset size before any math happens
    ds_subset = ds_depth.sel(depth=[0, 40, 55, 92, 222, 318, 541], method="nearest")
    
    # 2. Isolate the bounding depths and interpolate to 47.37
    ds_bounds = ds_subset.sel(depth=[40, 55], method="nearest")
    ds_47 = ds_bounds.interp(depth=[47.37], method="linear")
    
    # 3. Isolate the other target depths
    ds_0 = ds_subset.sel(depth=[0], method="nearest")
    ds_deep = ds_subset.sel(depth=[92, 222, 318, 541], method="nearest")
    
    # 4. Concatenate back together in the correct order
    ds_depth = xr.concat([ds_0, ds_47, ds_deep], dim="depth")
else:
    ds_depth = ds.sel(depth=TARGET_DEPTHS, method="nearest")
print("Selected data:", ds_depth, flush=True)
ds_depth = ds_depth.assign_coords(depth=TARGET_DEPTHS)

print("Chosen depth levels:", ds_depth.depth.values, flush=True)
n_depth = len(ds_depth.depth)
ds_2d = ds[vars_2d]
ds_forecasts = xr.merge([ds_depth, ds_2d])

if args.challenger == "glonet":
    ds_forecasts = ds_forecasts.rename({"lat": "latitude", "lon": "longitude"})

ds_forecasts = ds_forecasts.interp(latitude=LAT_1_4, longitude=LON_1_4)

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

print(ds_forecasts, flush=True)
print(f"Writing {output_path} ...", flush=True)
with dask.config.set(scheduler="threads", num_workers=args.num_workers):
    with ProgressBar():
        ds_forecasts.to_zarr(output_path, mode="w", encoding=encoding)
print(f"  -> {output_path}\n")