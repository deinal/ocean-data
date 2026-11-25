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

# 1. Load Data
data = {"glo12": glo12(), "glonet": glonet(), "wenhai": wenhai(), "xihe": xihe()}
ds = data[args.challenger]

output_path = os.path.join(args.out_dir, f"{args.challenger}_raw_forecasts.zarr")

# 2. Subset to target variables
vars_present = [v for v in TARGET_VARS if v in ds.data_vars]
ds = ds[vars_present]

# 3. Subset Depth (< 600m)
if "depth" in ds.coords:
    print(f"Original depths: {ds.depth.values}", flush=True)
    ds = ds.isel(depth=ds.depth < 600)
    print(f"Filtered depths (<600m): {ds.depth.values}", flush=True)

# 4. Standardize spatial coordinates
if args.challenger == "glonet":
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

# 5. Format Timestamps
print("Formatting time coordinates...", flush=True)
fdd = pd.to_datetime(ds["first_day_datetime"].values)
init_times = (fdd - pd.Timedelta(days=1)).floor("D")
ds = ds.rename({"first_day_datetime": "init_time"})
ds = ds.assign_coords(init_time=init_times.values.astype("datetime64[ns]"))

ldi = np.asarray(ds["lead_day_index"].values)
lead_times = pd.to_timedelta(ldi + 1, unit="D").to_numpy().astype("timedelta64[ns]")
ds = ds.rename({"lead_day_index": "lead_time"})
ds = ds.assign_coords(lead_time=lead_times)

print("init_time range:",
    str(pd.Timestamp(ds.init_time.values.min()).date()), "..",
    str(pd.Timestamp(ds.init_time.values.max()).date()),
    flush=True)
print("lead_time values:",
    [str(pd.Timedelta(lt)) for lt in ds.lead_time.values],
    flush=True)

# 6. Chunking and Encoding
n_lat = ds.sizes["latitude"]
n_lon = ds.sizes["longitude"]
n_depth = ds.sizes.get("depth", 1)  # Default to 1 if no depth dimension remains

chunk_dict = {
    "init_time": 1,
    "lead_time": 1,
    "latitude": n_lat,
    "longitude": n_lon,
}
if "depth" in ds.dims:
    chunk_dict["depth"] = n_depth

ds = ds.chunk(chunk_dict)

encoding = {}
for v in ds.data_vars:
    dims = list(ds[v].dims)
    if "depth" in dims:
        encoding[v] = {"chunks": (1, 1, n_depth, n_lat, n_lon)}
    else:
        encoding[v] = {"chunks": (1, 1, n_lat, n_lon)}

# 7. Save to Zarr
print(f"Writing {output_path} ...", flush=True)
with dask.config.set(scheduler="threads", num_workers=args.num_workers):
    with ProgressBar():
        ds.to_zarr(output_path, mode="w", encoding=encoding)
print(f"  -> {output_path}\n")