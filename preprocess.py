import os
import xarray as xr
import numpy as np
from datetime import date

# ──────────────── USER PARAMETERS ────────────────
raw_dir  = "data/baltic/raw"
prep_dir = "data/baltic/prep"
os.makedirs(prep_dir, exist_ok=True)

static_grid = xr.open_zarr("data/baltic/static/bathymetry.zarr")
target_lat   = static_grid.latitude
target_lon   = static_grid.longitude
surface_mask = static_grid.mask.isel(depth=0).astype(bool)

# ──────────────── REUSEABLE PROCESSING FUNCS ────────────────
def interpolate_to_grid(ds):
    return ds.interp(latitude=target_lat, longitude=target_lon, method="linear")

def get_depth_mask(ds, max_depth=150, step=5):
    if "depth" not in ds.coords:
        return None
    depths = ds.depth.values
    mask   = depths <= max_depth
    return depths[mask][::step]

def process_3d_subset(ds):
    sel = get_depth_mask(ds)
    return ds.sel(depth=sel) if sel is not None else ds

def process_wave(ds):
    # Match wave dataset to the surface_mask grid (has 1 extra pixel in lat and lon)
    ds = ds.sel(
        latitude=ds.latitude[:-1], # drop max latitude
        longitude=ds.longitude[1:] # drop min longitude
    )
    ds = ds.assign_coords(
        latitude=target_lat,
        longitude=target_lon,
    )

    for var in ["VHM0", "VMDR", "VTM10"]:
        ds[var] = ds[var].where(~surface_mask | ~np.isnan(ds[var]), 0)
    
    ds = ds.rename({"VHM0": "swh", "VTM10": "mwp"})
    
    ang = np.deg2rad(ds["VMDR"])
    ds["uw"] = np.sin(ang)
    ds["vw"] = np.cos(ang)
    ds = ds.drop_vars("VMDR")

    ds = ds.resample(time="1D").mean()

    return ds

# ──────────────── GENERIC ZARR PROCESSOR ────────────────
def prep_zarr(name,
              start_date: date,
              end_date:   date,
              custom_proc=None,
              include_vars=None,
              merge_sla: bool=False):
    """
    name       : base filename (no .zarr)
    start_date : datetime.date
    end_date   : datetime.date
    custom_proc: optional function(ds)->ds
    include_vars: list of data_vars to keep
    merge_sla  : if True, also load bal_sla_ana.zarr, daily-aggregate it, and merge in
    """

    src = os.path.join(raw_dir,  f"{name}.zarr")
    dst = os.path.join(prep_dir, f"{name}.zarr")

    # load & time-slice
    ds = xr.open_zarr(src).sel(
        time=slice(np.datetime64(start_date),
                   np.datetime64(end_date))
    )

    # drop unwanted vars
    if include_vars:
        ds = ds[[v for v in include_vars if v in ds.data_vars]]

    # any custom fn
    if custom_proc:
        ds = custom_proc(ds)

    # reanalysis differ very slightly from static/analysis/forecast grid
    ds = ds.assign_coords(latitude=target_lat, longitude=target_lon)

    # aggregate daily data for analysis sla
    if merge_sla:
        # pull SLA, slice & resample it
        sla = (
            xr.open_zarr(os.path.join(raw_dir, "bal_sla_ana.zarr"))
              .sel(time=slice(np.datetime64(start_date),
                              np.datetime64(end_date)))
              .resample(time="1D")
              .mean()
        )

        # merge daily-SLA into your dataset
        ds = xr.merge([ds, sla])
    
    # rechunk
    if "depth" in ds.dims:
        ds = ds.chunk({
            "time": 30,
            "depth": ds.sizes["depth"],
            "latitude": ds.sizes["latitude"],
            "longitude": ds.sizes["longitude"],
        })
    else:
        ds = ds.chunk({
            "time": 30,
            "latitude": ds.sizes["latitude"],
            "longitude": ds.sizes["longitude"],
        })

    # write out
    ds.to_zarr(dst, mode="w")
    print(f"Wrote {dst}")


# reanalysis
prep_zarr("era5",
          start_date=date(2000, 1, 1),
          end_date=date(2000, 1, 22),
          custom_proc=interpolate_to_grid)

prep_zarr("bal_phy_rea",
          start_date=date(2000, 1, 1),
          end_date=date(2000, 1, 22),
          custom_proc=process_3d_subset)

prep_zarr("bal_bgc_rea",
          start_date=date(2000, 1, 1),
          end_date=date(2000, 1, 22),
          include_vars=["chl","o2","zsd"],
          custom_proc= process_3d_subset)

prep_zarr("bal_wav_rea",
          start_date=date(2000, 1, 1),
          end_date=date(2000, 1, 22),
          custom_proc=process_wave)


# analysis
prep_zarr("bal_phy_ana",
          start_date=date(2023, 1, 1),
          end_date=date(2023, 1, 22),
          custom_proc=process_3d_subset,
          merge_sla=True)

prep_zarr("bal_bgc_ana",
          start_date=date(2023, 1, 1),
          end_date=date(2023, 1, 22),
          include_vars=["chl","o2","zsd"],
          custom_proc=process_3d_subset)

prep_zarr("bal_wav_ana",
          start_date=date(2023, 1, 1),
          end_date=date(2023, 1, 22),
          custom_proc=process_wave)

