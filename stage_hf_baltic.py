#!/usr/bin/env python3
"""
Stage the Baltic zarrs used by the Njord configs for upload to Hugging Face.

- Stores are hard-linked into the staging folder (no extra disk space).
- baltic_era5.zarr is rewritten with 31-day time chunks: one chunk per day
  gives ~11.7k files per variable, above HF's ~10k files-per-folder limit.
"""
import argparse
import os
import shutil

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

STORES = [
    "baltic_data.zarr",
    "baltic_mask.zarr",
    "baltic_static.zarr",
    "baltic_oper_data.zarr",
    "baltic_forecasts.zarr",
    "baltic_ifs_forecasts.zarr",
    "baltic_boundary_data.zarr",
    "baltic_boundary_mask.zarr",
    "baltic_boundary_static.zarr",
    "baltic_boundary_forecasts.zarr",
]
RECHUNK = {"baltic_era5.zarr": {"time": 31}}

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", default="data")
parser.add_argument("--out-dir", default="data/hf/njord-baltic")
parser.add_argument("--num-workers", type=int, default=10)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def hardlink_tree(src, dst):
    if os.path.exists(dst):
        print(f"skip {dst} (exists)", flush=True)
        return
    shutil.copytree(src, dst, copy_function=os.link)
    print(f"linked {src} -> {dst}", flush=True)


for name in STORES:
    hardlink_tree(os.path.join(args.base_dir, name), os.path.join(args.out_dir, name))

for name, chunks in RECHUNK.items():
    dst = os.path.join(args.out_dir, name)
    if os.path.exists(dst):
        print(f"skip {dst} (exists)", flush=True)
        continue
    ds = xr.open_zarr(os.path.join(args.base_dir, name))
    ds = ds.chunk({**{d: -1 for d in ds.dims}, **chunks})
    for v in ds.variables:
        ds[v].encoding.pop("chunks", None)
        ds[v].encoding.pop("preferred_chunks", None)
    print(f"rechunking {name} -> {dict(ds.chunks)}", flush=True)
    with dask.config.set(scheduler="threads", num_workers=args.num_workers):
        with ProgressBar():
            ds.to_zarr(dst, mode="w", consolidated=True)

n_files = sum(len(files) for _, _, files in os.walk(args.out_dir))
print(f"done: {n_files} files in {args.out_dir}", flush=True)
