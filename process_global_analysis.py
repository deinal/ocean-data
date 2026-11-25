#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "global_analysis_forecasts.zarr"

ds_analysis = xr.open_zarr(DATA_DIR / "global_analysis_data_1_4.zarr")
ds_nowcast  = xr.open_zarr(DATA_DIR / "glo12_nowcasts_1_4.zarr")

init_times = ds_nowcast["time"].values
lead_time  = np.arange(-1, 11).astype("timedelta64[D]")

init_da = xr.DataArray(init_times, dims="init_time", coords={"init_time": init_times})
lead_da = xr.DataArray(lead_time,  dims="lead_time", coords={"lead_time": lead_time})
valid_time = init_da + lead_da

ds_ana_sel = ds_analysis.sel(time=valid_time).drop_vars("time")
ds_now     = ds_nowcast.sel(time=init_times).rename({"time": "init_time"})

is_lead_zero = lead_da == np.timedelta64(0, "D")
ds_out = xr.where(is_lead_zero, ds_now, ds_ana_sel)

ds_out = ds_out.assign_coords(time=valid_time)
ds_out = ds_out.sortby(["init_time", "lead_time"])

with dask.config.set(scheduler="threads", num_workers=10):
    with ProgressBar():
        ds_out.to_zarr(OUT_PATH, mode="w", consolidated=True)

print(f"Saved to {OUT_PATH}")