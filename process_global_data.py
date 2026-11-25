#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr

import dask
from dask.diagnostics import ProgressBar

DATA_DIR = Path("data")

# -----------------------------
# INPUT FILES
# -----------------------------
ds_forecast = xr.open_zarr(DATA_DIR / "global_data_1_4.zarr")
ds_forecast_2025 = xr.open_zarr(DATA_DIR / "global_data_jan_2025_1_4.zarr")

ds_nowcast = xr.open_zarr(DATA_DIR / "glo12_nowcasts_1_4.zarr")
ds_analysis = xr.open_zarr(DATA_DIR / "global_analysis_mondays_2024_1_4.zarr")

# ds_data = xr.open_zarr(DATA_DIR / "global_data_1_4.zarr")
# ds_nowcast["sithick"] = ds_data["sithick"].sel(time=ds_nowcast["time"])
# ds_analysis["sithick"] = ds_data["sithick"].sel(time=ds_analysis["time"])
# ds_nowcast["siconc"] = ds_data["siconc"].sel(time=ds_nowcast["time"])
# ds_analysis["siconc"] = ds_data["siconc"].sel(time=ds_analysis["time"])
# ds_nowcast["so"] = ds_data["so"].sel(time=ds_nowcast["time"])
# ds_analysis["so"] = ds_data["so"].sel(time=ds_analysis["time"])

#ds_nowcast = xr.open_zarr(DATA_DIR / "global_data_1_4.zarr").sel(time=ds_nowcast["time"])
#ds_analysis = xr.open_zarr(DATA_DIR / "global_data_1_4.zarr").sel(time=ds_analysis["time"])

# -----------------------------
# INIT TIMES (from nowcast)
# -----------------------------
init_times = ds_nowcast["time"].values

# -----------------------------
# DEFINE LEAD TIMES
# -----------------------------
lead_days = np.arange(-1, 11)  # -1D to +10D
lead_time = lead_days.astype("timedelta64[D]")

# -----------------------------
# HELPER: select correct dataset
# -----------------------------
def get_forecast_source(valid_time):
    """Pick correct forecast dataset depending on year."""
    if np.datetime64(valid_time, "Y") >= np.datetime64("2025"):
        return ds_forecast_2025
    return ds_forecast

# -----------------------------
# BUILD DATASET
# -----------------------------
out_list = []

for init in init_times:
    print(f"Processing init_time {init} ...", flush=True)
    slices = []

    for ld in lead_days:
        valid_time = init + np.timedelta64(ld, "D")

        if ld == -1:
            ds_src = ds_analysis
        elif ld == 0:
            ds_src = ds_nowcast
        else:
            ds_src = get_forecast_source(valid_time)

        try:
            ds_sel = ds_src.sel(time=valid_time)
        except Exception:
            print(f"Skipping missing time {valid_time}")
            continue

        # expand with lead_time dim
        ds_sel = ds_sel.expand_dims(
            lead_time=[np.timedelta64(ld, "D")]
        )
            
        slices.append(ds_sel)

    # concat along lead_time for this init
    ds_init = xr.concat(slices, dim="lead_time")

    # add init_time dimension
    ds_init = ds_init.expand_dims(init_time=[init])

    out_list.append(ds_init)

# -----------------------------
# FINAL CONCAT
# -----------------------------
ds_out = xr.concat(out_list, dim="init_time")

# ensure ordering
ds_out = ds_out.sortby(["init_time", "lead_time"])

# -----------------------------
# WRITE ZARR
# -----------------------------
out_path = DATA_DIR / "global_forecasts_truth.zarr"

with dask.config.set(scheduler="threads", num_workers=10):
    with ProgressBar():
        ds_out.to_zarr(out_path, mode="w", consolidated=True)

print(f"Saved to {out_path}")