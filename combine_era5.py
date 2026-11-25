#!/usr/bin/env python3
"""
Merge split ARCO ERA5 Zarr stores into one deduplicated time-concatenated Zarr.

Expected input layout:
    data/
      arco_era5_1.zarr
      arco_era5_2.zarr
      arco_era5_3.zarr

Behavior:
- Opens all matching `arco_era5_*.zarr` stores in numeric order.
- Keeps all times from the first store.
- For each later store, keeps data starting from its first timestamp onward,
  dropping any overlapping timestamps already present from earlier stores.
  This means later stores "win" from their first date onward.
- Rechunks output so that for each data variable:
    time      -> 1
    latitude  -> full length (-1)
    longitude -> full length (-1)
- Writes result to:
    data/arco_era5.zarr

Usage:
    python merge_arco_era5.py
    python merge_arco_era5.py --data-dir data
    python merge_arco_era5.py --workers 16
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar


def numeric_key(path: str) -> int:
    m = re.search(r"arco_era5_(\d+)\.zarr/?$", path)
    if not m:
        raise ValueError(f"Could not extract numeric suffix from {path}")
    return int(m.group(1))


def infer_spatial_dims(ds: xr.Dataset) -> tuple[str | None, str | None]:
    lat_name = next((d for d in ("latitude", "lat") if d in ds.dims), None)
    lon_name = next((d for d in ("longitude", "lon") if d in ds.dims), None)
    return lat_name, lon_name


def trim_for_nonoverlapping_concat(datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Later stores override earlier ones starting from the later store's first timestamp.
    Example:
      ds1: 2000-01-01 ... 2000-12-31
      ds2: 2000-12-01 ... 2001-12-31
    result keeps:
      ds1: 2000-01-01 ... 2000-11-30
      ds2: 2000-12-01 ... 2001-12-31
    """
    if not datasets:
        raise ValueError("No datasets provided")

    trimmed: list[xr.Dataset] = []

    for i, ds in enumerate(datasets):
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError(f"Dataset {i} has no 'time' coordinate")

    for i, ds in enumerate(datasets):
        ds = ds.sortby("time")

        if i < len(datasets) - 1:
            next_first_time = datasets[i + 1]["time"].values[0]
            ds = ds.sel(time=slice(None, next_first_time), drop=False)
            ds = ds.isel(time=ds["time"] < next_first_time)

        trimmed.append(ds)

    out = xr.concat(trimmed, dim="time")

    return out


def build_encoding(ds: xr.Dataset, lat_dim: str | None, lon_dim: str | None) -> dict:
    encoding = {}

    for var_name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif lat_dim is not None and dim == lat_dim:
                chunks.append(da.sizes[dim])
            elif lon_dim is not None and dim == lon_dim:
                chunks.append(da.sizes[dim])
            else:
                # Keep any other dimensions as full-size chunks unless you want something else.
                chunks.append(da.sizes[dim])

        encoding[var_name] = {
            "chunks": tuple(chunks),
        }

    return encoding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-name", type=str, default="arco_era5.zarr")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = data_dir / args.output_name

    input_paths = sorted(
        glob.glob(str(data_dir / "arco_era5_*.zarr")),
        key=numeric_key,
    )

    if not input_paths:
        raise FileNotFoundError(f"No input stores found in {data_dir}/arco_era5_*.zarr")

    # Avoid accidentally including an already-merged output if naming overlaps later.
    input_paths = [p for p in input_paths if Path(p).name != args.output_name]

    print("Input stores:")
    for p in input_paths:
        print(f"  - {p}")
    print(f"Output store:\n  - {output_path}")

    datasets = []
    for path in input_paths:
        print(f"Opening {path}")
        ds = xr.open_zarr(path, consolidated=False)
        ds = ds.sortby("time")
        datasets.append(ds)

    merged = trim_for_nonoverlapping_concat(datasets)

    lat_dim, lon_dim = infer_spatial_dims(merged)
    print(f"Inferred spatial dims: lat={lat_dim}, lon={lon_dim}")

    chunk_map = {"time": 1}
    if lat_dim is not None:
        chunk_map[lat_dim] = -1
    if lon_dim is not None:
        chunk_map[lon_dim] = -1

    merged = merged.chunk(chunk_map)
    encoding = build_encoding(merged, lat_dim, lon_dim)

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    print("\nMerged dataset:")
    print(merged)

    print("\nWriting with Dask threaded scheduler...")
    dask.config.set(scheduler="threads", num_workers=args.workers)

    with ProgressBar():
        merged.to_zarr(
            str(output_path),
            mode="w",
            consolidated=True,
            encoding=encoding,
        )

    print(f"\nDone: wrote {output_path}")

    print(merged)
    print(merged.data_vars)


if __name__ == "__main__":
    main()