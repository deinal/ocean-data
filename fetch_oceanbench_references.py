"""
Fetch GLORYS 1/4° and 1° reanalysis from Minio (OceanBench) and store as a local zarr.
"""
from datetime import datetime
import os
import argparse
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


def _glorys_1_4_path(first_day_datetime: np.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-glonet/public/glorys14_refull_2024/{first_day}.zarr"


def _glorys_1_degree_path(first_day_datetime: np.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glorys_1degree_2024_V2/{first_day}.zarr"


def _wednesdays_2024() -> np.ndarray:
    """Every Wednesday in 2024 as datetime64."""
    dr = pd.date_range(start="2024-01-01", end="2024-12-31", freq="W-WED")
    return dr.values.astype("datetime64[ns]")


def _preprocess(dataset):
    ntime = dataset.sizes["time"]
    return (
        dataset.rename({"time": "lead_day_index"})
        .assign_coords(lead_day_index=np.arange(ntime))
    )


def _fetch_and_write_glorys(path_fn, first_day_datetimes, out_path):
    """Fetch GLORYS from MinIO one first_day at a time, write intermediates, then concat and write final zarr."""
    n = len(first_day_datetimes)
    temp_dir = out_path.rstrip("/").replace(".zarr", "_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        print(f"Loading from Minio: {n} first days (Wednesdays 2024), one at a time...", flush=True)
        for i, fdd in enumerate(first_day_datetimes):
            url = path_fn(fdd)
            first_day_str = datetime.fromisoformat(str(fdd)).strftime("%Y%m%d")
            part_path = os.path.join(temp_dir, f"{first_day_str}.zarr")
            print(f"  [{i + 1}/{n}] {url} -> {part_path}", flush=True)
            ds = xr.open_dataset(url, engine="zarr")
            ds = _preprocess(ds)
            ds = ds.expand_dims(first_day_datetime=[np.datetime64(fdd, "ns")])
            with ProgressBar():
                ds.to_zarr(part_path, mode="w")
            del ds

        temp_paths = [
            os.path.join(temp_dir, datetime.fromisoformat(str(fdd)).strftime("%Y%m%d") + ".zarr")
            for fdd in first_day_datetimes
        ]
        print(f"Concatenating {len(temp_paths)} intermediates...", flush=True)
        ds = xr.open_mfdataset(
            temp_paths,
            engine="zarr",
            combine="nested",
            concat_dim="first_day_datetime",
        )
        print(ds, flush=True)
        print(f"Writing to {out_path}...", flush=True)
        with ProgressBar():
            ds.to_zarr(out_path, mode="w")
        print(f"Done: {out_path}", flush=True)
    finally:
        print(f"Removing temp dir {temp_dir}...", flush=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data",
        help="Output directory for the zarr stores (default: %(default)s)",
    )
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    first_day_datetimes = _wednesdays_2024()

    # 1/4 degree -> glorys_1_4.zarr
    out_1_4 = os.path.join(args.base_dir, "glorys_1_4.zarr")
    print("=== GLORYS 1/4° ===", flush=True)
    _fetch_and_write_glorys(_glorys_1_4_path, first_day_datetimes, out_1_4)

    # 1 degree -> glorys_1.zarr
    out_1 = os.path.join(args.base_dir, "glorys_1.zarr")
    print("=== GLORYS 1° ===", flush=True)
    _fetch_and_write_glorys(_glorys_1_degree_path, first_day_datetimes, out_1)


if __name__ == "__main__":
    main()
