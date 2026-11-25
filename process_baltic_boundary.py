#!/usr/bin/env python3
"""
Build data/baltic_boundary_forecasts.zarr.

Sources
-------
data/glo12_forecasts.zarr               (init_time, lead_time 1D..10D, depth, lat, lon)
data/glo12_nowcasts.zarr                (time, depth, lat, lon)   -> lead_time  = 0D
data/global_analysis_mondays_2024.zarr  (time, depth, lat, lon)   -> lead_time = -1D
data/baltic_boundary_mask.zarr          (matches the subset box & depths exactly)

For every init_time present in glo12_forecasts.zarr:
  - lead_time = -1D  <-  global_analysis_mondays_2024 at time = init_time - 1 day
  - lead_time =  0D  <-  glo12_nowcasts              at time = init_time
  - lead_time  1D..10D  <- glo12_forecasts directly

Spatial/variable/depth subset:
  latitude  in [55, 60]
  longitude in [ 5,  9]
  depths    = [0, 47, 92]
  vars      = [thetao, so, uo, vo, zos]

Finally  ds.fillna(0.0).where(mask_ds)  sets interior NaNs to zero and
reinstates NaN outside the boundary ring.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar


DEFAULT_OUT = "data"
TARGET_DEPTHS = [0, 47, 92]  # m
TARGET_VARS = ["thetao", "so", "uo", "vo", "zos"]

MIN_LON, MAX_LON = 5.0, 9.0
MIN_LAT, MAX_LAT = 55.0, 60.0

FC_PATH   = os.path.join(DEFAULT_OUT, "glo12_forecasts.zarr")
NOW_PATH  = os.path.join(DEFAULT_OUT, "glo12_nowcasts.zarr")
ANA_PATH  = os.path.join(DEFAULT_OUT, "global_analysis_mondays_2024.zarr")
MASK_PATH = os.path.join(DEFAULT_OUT, "baltic_boundary_mask.zarr")
OUT_PATH  = os.path.join(DEFAULT_OUT, "baltic_boundary_forecasts.zarr")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _subset_box(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(
        latitude=slice(MIN_LAT, MAX_LAT),
        longitude=slice(MIN_LON, MAX_LON),
    )


def _subset_vars(ds: xr.Dataset) -> xr.Dataset:
    keep = [v for v in TARGET_VARS if v in ds.data_vars]
    missing = [v for v in TARGET_VARS if v not in ds.data_vars]
    if missing:
        print(f"  warning: missing vars {missing}", flush=True)
    return ds[keep]


def _subset_depths(ds: xr.Dataset) -> xr.Dataset:
    if "depth" not in ds.dims:
        return ds
    depths = np.asarray(ds.depth.values)
    # Pick the nearest source level to each target depth, then relabel exactly.
    idx = sorted({int(np.argmin(np.abs(depths - z))) for z in TARGET_DEPTHS})
    ds = ds.isel(depth=idx).assign_coords(
        depth=np.round(depths[idx]).astype(int)
    )
    # Now demand the target depths are present
    ds = ds.sel(depth=TARGET_DEPTHS)
    return ds


def _ensure_td64_ns(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind != "m":
        a = pd.to_timedelta(a).to_numpy()
    return a.astype("timedelta64[ns]")


def _per_init_from_time_series(
    ds_ts: xr.Dataset,
    init_times: pd.DatetimeIndex,
    offset: pd.Timedelta,
    lead_value: np.timedelta64,
    source_name: str,
) -> xr.Dataset:
    """For each init_time, pick ds_ts at time=init_time+offset; attach lead_time=lead_value."""
    t = pd.to_datetime(ds_ts.time.values).normalize()
    ds_ts = ds_ts.assign_coords(time=t.values)

    needed = pd.DatetimeIndex([it + offset for it in init_times])
    avail = set(pd.DatetimeIndex(ds_ts.time.values))
    missing = [x for x in needed if x not in avail]
    if missing:
        raise RuntimeError(
            f"{source_name}: missing {len(missing)} required valid_time(s), "
            f"e.g. {[str(x.date()) for x in missing[:5]]}"
        )

    sel = ds_ts.sel(time=needed.values)
    sel = sel.rename(time="init_time")
    sel = sel.assign_coords(
        init_time=np.asarray(init_times, dtype="datetime64[ns]")
    )
    sel = sel.expand_dims(lead_time=[lead_value])
    return sel


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_baltic_boundary_forecasts() -> None:
    print("Opening sources...", flush=True)
    ds_fc   = xr.open_zarr(FC_PATH,   decode_timedelta=True)
    ds_now  = xr.open_zarr(NOW_PATH)
    ds_ana  = xr.open_zarr(ANA_PATH)
    ds_mask = xr.open_zarr(MASK_PATH)

    # Spatial / var / depth subset on each source (fast, before heavy ops).
    ds_fc  = _subset_box(_subset_depths(_subset_vars(ds_fc)))
    ds_now = _subset_box(_subset_depths(_subset_vars(ds_now)))
    ds_ana = _subset_box(_subset_depths(_subset_vars(ds_ana)))

    # Normalize ds_fc lead_time dtype.
    ds_fc = ds_fc.assign_coords(lead_time=_ensure_td64_ns(ds_fc.lead_time.values))
    print(ds_mask)
    print(ds_fc)
    ds_fc = ds_fc.assign_coords(
        longitude=ds_mask.longitude,
        latitude=ds_mask.latitude,
    )

    init_times = pd.DatetimeIndex(ds_fc.init_time.values)
    print(
        f"init_time: n={len(init_times)}  "
        f"[{init_times.min().date()} .. {init_times.max().date()}]",
        flush=True,
    )
    print(
        f"lead_time from glo12_forecasts: {len(ds_fc.lead_time)} "
        f"[{pd.Timedelta(ds_fc.lead_time.values[0])} .. "
        f"{pd.Timedelta(ds_fc.lead_time.values[-1])}]",
        flush=True,
    )

    # --- lead_time =  0D  from glo12_nowcasts ---------------------------
    ds_lead0 = _per_init_from_time_series(
        ds_now, init_times,
        offset=pd.Timedelta(0),
        lead_value=np.timedelta64(0, "D").astype("timedelta64[ns]"),
        source_name="glo12_nowcasts",
    )
    print(ds_lead0)
    ds_lead0 = ds_lead0.assign_coords(
        longitude=ds_mask.longitude,
        latitude=ds_mask.latitude,
    )

    # --- lead_time = -1D  from global_analysis_mondays_2024 -------------
    ds_leadm1 = _per_init_from_time_series(
        ds_ana, init_times,
        offset=pd.Timedelta(days=-1),
        lead_value=np.timedelta64(-1, "D").astype("timedelta64[ns]"),
        source_name="global_analysis_mondays_2024",
    )
    print(ds_leadm1)
    ds_leadm1 = ds_leadm1.assign_coords(
        longitude=ds_mask.longitude,
        latitude=ds_mask.latitude,
    )

    # --- Concatenate along lead_time and sort ---------------------------
    ds_all = xr.concat(
        [ds_leadm1, ds_lead0, ds_fc],
        dim="lead_time",
    ).sortby("lead_time")

    # Fill NaNs where our reanalysis mask differ from analysis forecast.
    ds_out = ds_all.interpolate_na(
        dim="latitude", method="linear", fill_value="extrapolate"
    ).interpolate_na(
        dim="longitude", method="linear", fill_value="extrapolate"
    ).where(ds_mask == 1)

    # Valid-time coord if time forcing is needed (not used for boundary here)
    # ds_out = ds_out.assign_coords(time=ds_out.init_time + ds_out.lead_time)

    # --- Chunk and write -----------------------------------------------
    chunks = {"init_time": 1, "lead_time": -1, "latitude": -1, "longitude": -1}
    if "depth" in ds_out.dims:
        chunks["depth"] = -1
    ds_out = ds_out.chunk(chunks).unify_chunks()

    print(f"Output dims: {dict(ds_out.sizes)}", flush=True)
    print(f"Writing {OUT_PATH} ...", flush=True)
    with dask.config.set(scheduler="threads", num_workers=8):
        with ProgressBar():
            ds_out.to_zarr(OUT_PATH, mode="w")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build_baltic_boundary_forecasts()