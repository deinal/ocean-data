#!/usr/bin/env python3
"""
Build two products on the Baltic mask grid:

  1. data/baltic_forecasts.zarr
     Indexing axes (init_time, lead_time) from data/ifs_forecasts.zarr.
     Values from data/baltic_oper_data.zarr at time = init_time + lead_time.

  2. data/baltic_ifs_forecasts.zarr
     IFS atmospheric forecasts regridded (bilinear) onto the mask grid,
     with land set to NaN via ds_mask.sla as a surface mask.

Both outputs share the exact latitude/longitude values from
data/baltic_mask.zarr so downstream code can align them without
float-precision surprises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

LEXICON = {
    "2m_temperature": "sotemair",
    "10m_u_component_of_wind": "sowinu10",
    "10m_v_component_of_wind": "sowinv10",
    "ssrd": "sosudosw",
    "strd": "sosudolw",
    "tprate": "sowaprec",
    "2d": "sod2m",
    "mean_sea_level_pressure": "somslpre",
}

IFS_PATH = "data/ifs2024_daily_means.zarr"
ANA_PATH      = "data/baltic_oper_data.zarr"
MASK_PATH     = "data/baltic_mask.zarr"
OUT_PATH      = "data/baltic_forecasts.zarr"
IFS_OUT_PATH  = "data/baltic_ifs_forecasts.zarr"

GRID_TOL_DEG = 1e-8   # acceptable grid mismatch for assign_coords()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_timedelta64_lead(ds: xr.Dataset, name: str = "lead_time") -> xr.Dataset:
    lt = ds[name].values
    if lt.dtype.kind != "m":
        lt = pd.to_timedelta(lt).to_numpy()
    return ds.assign_coords({name: lt.astype("timedelta64[ns]")})


def _assign_grid_from_mask(
    ds: xr.Dataset, ds_mask: xr.Dataset, name: str,
) -> xr.Dataset:
    """Force ds.lat/lon to exactly equal the mask's lat/lon.

    Requires that `ds` is already on the same grid as the mask up to
    `GRID_TOL_DEG`. Fails loudly otherwise.
    """
    for dim in ("latitude", "longitude"):
        if ds.sizes[dim] != ds_mask.sizes[dim]:
            raise RuntimeError(
                f"{name}: {dim} size {ds.sizes[dim]} != mask {ds_mask.sizes[dim]}"
            )
        d = float(np.max(np.abs(
            ds[dim].values.astype("f8") - ds_mask[dim].values.astype("f8")
        )))
        if d > GRID_TOL_DEG:
            raise RuntimeError(
                f"{name}: {dim} differs from mask by {d:g} deg (>{GRID_TOL_DEG})"
            )
    return ds.assign_coords(
        latitude=ds_mask.latitude.values,
        longitude=ds_mask.longitude.values,
    )


# ---------------------------------------------------------------------------
# main builders
# ---------------------------------------------------------------------------

def build_baltic_forecasts(ds_mask: xr.Dataset) -> None:
    """Truth store: analysis values indexed by IFS (init_time, lead_time)."""
    ds_ifs = xr.open_zarr(IFS_PATH, decode_timedelta=True)
    ds_ifs = _ensure_timedelta64_lead(ds_ifs, "lead_time")

    init_times = pd.DatetimeIndex(ds_ifs.init_time.values)
    lead_times = pd.TimedeltaIndex(ds_ifs.lead_time.values)
    lead_vals  = lead_times.to_numpy().astype("timedelta64[ns]")

    print(
        f"[truth] IFS axes: {len(init_times)} init_time(s) "
        f"[{init_times.min().date()} .. {init_times.max().date()}], "
        f"{len(lead_times)} lead_time(s) "
        f"[{lead_times.min()} .. {lead_times.max()}]",
        flush=True,
    )

    ds_ana = xr.open_zarr(ANA_PATH, decode_timedelta=True)
    ana_times = pd.to_datetime(ds_ana.time.values).normalize()
    ds_ana = ds_ana.assign_coords(time=ana_times.values)

    # Force analysis grid to exactly match the mask grid (fixes float drift).
    ds_ana = _assign_grid_from_mask(ds_ana, ds_mask, name="baltic_oper_data_complete")

    needed_times = pd.DatetimeIndex(
        sorted({it + lt for it in init_times for lt in lead_times})
    )
    ana_set = set(pd.DatetimeIndex(ds_ana.time.values))
    missing = [t for t in needed_times if t not in ana_set]
    assert not missing, (
        f"{len(missing)} required valid_time(s) missing from {ANA_PATH}, "
        f"e.g. {[str(t.date()) for t in missing[:10]]}"
    )
    print(f"[truth] all {len(needed_times)} required valid_times present.", flush=True)

    per_init = []
    for it in init_times:
        needed = [it + lt for lt in lead_times]
        chunk = ds_ana.sel(time=needed).rename(time="lead_time")
        chunk = chunk.assign_coords(lead_time=lead_vals)
        chunk = chunk.expand_dims(init_time=[np.datetime64(it, "ns")])
        per_init.append(chunk)

    ds_out = xr.concat(per_init, dim="init_time")

    # valid datetime per (init_time, lead_time)
    ds_out = ds_out.assign_coords(time=ds_out.init_time + ds_out.lead_time)

    chunks = {"init_time": 1, "lead_time": -1}
    for d in ("depth", "latitude", "longitude"):
        if d in ds_out.dims:
            chunks[d] = -1
    ds_out = ds_out.chunk(chunks).unify_chunks()

    print(f"[truth] output dims: {dict(ds_out.sizes)}", flush=True)
    print(f"[truth] writing {OUT_PATH} ...", flush=True)
    with dask.config.set(scheduler="threads", num_workers=8):
        with ProgressBar():
            ds_out.to_zarr(OUT_PATH, mode="w")
    print(f"[truth] wrote {OUT_PATH}", flush=True)


def build_baltic_ifs_forecasts(ds_mask: xr.Dataset) -> None:
    """IFS forecasts, bilinear-interpolated onto the mask grid, land NaN."""
    ds_ifs = xr.open_zarr(IFS_PATH, decode_timedelta=True)
    ds_ifs = _ensure_timedelta64_lead(ds_ifs, "lead_time")

    rename_dict = {k: v for k, v in LEXICON.items() if k in ds_ifs.data_vars}
    ds_ifs = ds_ifs.rename(rename_dict)

    print("Fixing longitude...", flush=True)
    lon_new = np.where(ds_ifs.longitude > 180, ds_ifs.longitude - 360, ds_ifs.longitude)
    ds_ifs = ds_ifs.assign_coords(longitude=lon_new)
    # sort for monotonic coords
    ds_ifs = ds_ifs.sortby("latitude").sortby("longitude")
    # roll so -180 is first
    ds_ifs = ds_ifs.roll(longitude=-1, roll_coords=True)
    lon_vals = ds_ifs.longitude.values.astype(np.float64)
    lon_vals[0] = -180.0
    ds_ifs = ds_ifs.assign_coords(longitude=lon_vals)

    print(
        f"[ifs] source dims: {dict(ds_ifs.sizes)}; "
        f"regridding to mask grid "
        f"lat[{ds_mask.sizes['latitude']}] lon[{ds_mask.sizes['longitude']}]",
        flush=True,
    )

    # Bilinear spatial regrid onto the mask grid.
    ds_ifs_rg = ds_ifs.interp(
        latitude=ds_mask.latitude.values,
        longitude=ds_mask.longitude.values,
        method="linear",
    )

    # Surface mask from ds_mask.sla: True over ocean, False over land -> NaN.
    surface_mask = ds_mask["sla"].astype(bool)
    ds_ifs_rg = ds_ifs_rg.where(surface_mask)

    # valid datetime per (init_time, lead_time)
    ds_ifs_rg = ds_ifs_rg.assign_coords(
        time=ds_ifs_rg.init_time + ds_ifs_rg.lead_time
    )

    chunks = {"init_time": 1, "lead_time": -1, "latitude": -1, "longitude": -1}
    ds_ifs_rg = ds_ifs_rg.chunk(chunks).unify_chunks()

    print(f"[ifs] output dims: {dict(ds_ifs_rg.sizes)}", flush=True)
    print(f"[ifs] writing {IFS_OUT_PATH} ...", flush=True)
    with dask.config.set(scheduler="threads", num_workers=8):
        with ProgressBar():
            ds_ifs_rg.to_zarr(IFS_OUT_PATH, mode="w")
    print(f"[ifs] wrote {IFS_OUT_PATH}", flush=True)


def main() -> None:
    ds_mask = xr.open_zarr(MASK_PATH)
    print(
        f"[mask] dims: {dict(ds_mask.sizes)}, vars: {list(ds_mask.data_vars)}",
        flush=True,
    )

    build_baltic_forecasts(ds_mask)
    build_baltic_ifs_forecasts(ds_mask)


if __name__ == "__main__":
    main()