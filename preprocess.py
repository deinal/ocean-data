import os
import xarray as xr
import numpy as np
from datetime import date

# ──────────────── USER PARAMETERS ────────────────
MAX_DEPTH = 104
STEP = 1

raw_dir  = "data/baltic/raw"
prep_dir = "../baltic-sea-forecasting/data/prep"
os.makedirs(prep_dir, exist_ok=True)

static_grid = xr.open_zarr("data/baltic/static/bathymetry.zarr")
target_lat   = static_grid.latitude
target_lon   = static_grid.longitude
surface_mask = static_grid.mask.isel(depth=0).astype(bool)

# ──────────────── REUSEABLE PROCESSING FUNCS ────────────────
def process_era5(ds):
    ds = ds.interp(latitude=target_lat, longitude=target_lon, method="linear")
    return ds.resample(time="1D").mean()

def process_3d_subset(ds):
    if "depth" not in ds.coords:
        return ds
    else:
        return ds.sel(depth=ds.depth[ds.depth <= MAX_DEPTH]).isel(depth=slice(None, None, STEP))

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
    
    encoding = {}
    for var in ds.data_vars:
        dims = ds[var].dims
        chunk_sizes = []
        for d in dims:
            if d == "time":
                chunk_sizes.append(30)
            else:
                chunk_sizes.append(ds.sizes[d])
        encoding[var] = {"chunks": tuple(chunk_sizes)}

    # write out
    ds.to_zarr(dst, mode="w", encoding=encoding)
    print(f"Wrote {dst}")


# reanalysis
prep_zarr("era5",
          start_date=date(2000, 1, 1),
          end_date=date(2020, 12, 31),
          custom_proc=process_era5)

prep_zarr("bal_phy_rea",
          start_date=date(2000, 1, 1),
          end_date=date(2020, 12, 31),
          custom_proc=process_3d_subset)

prep_zarr("bal_bgc_rea",
          start_date=date(2000, 1, 1),
          end_date=date(2020, 12, 31),
          include_vars=["chl","o2","zsd"],
          custom_proc=process_3d_subset)

prep_zarr("bal_wav_rea",
          start_date=date(2000, 1, 1),
          end_date=date(2020, 12, 31),
          custom_proc=process_wave)


# analysis
prep_zarr("bal_phy_ana",
          start_date=date(2023, 1, 1),
          end_date=date(2023, 1, 30),
          custom_proc=process_3d_subset,
          merge_sla=True)

prep_zarr("bal_bgc_ana",
          start_date=date(2023, 1, 1),
          end_date=date(2023, 1, 30),
          include_vars=["chl","o2","zsd"],
          custom_proc=process_3d_subset)

prep_zarr("bal_wav_ana",
          start_date=date(2023, 1, 1),
          end_date=date(2023, 1, 30),
          custom_proc=process_wave)


# Create feature mask
variable_depth = {
    "mlotst": False,
    "siconc": False,
    "sithick": False,
    "sla": False,
    "mwp": False,
    "swh": False,
    "uw": False,
    "vw": False,
    "zsd": False,
    "so": True,
    "thetao": True,
    "uo": True,
    "vo": True,
    "chl": True,
    "o2": True,
}

# Selected depth levels
mask_selection = static_grid["mask"].sel(depth=static_grid.depth[static_grid.depth <= MAX_DEPTH]).isel(depth=slice(None, None, STEP))

# Surface mask
surface_mask = static_grid["mask"].isel(depth=0)

# Build output dataset
feature_mask_dict = {}

for var, has_depth in variable_depth.items():
    if has_depth:
        feature_mask_dict[var] = mask_selection.copy(deep=False)
    else:
        feature_mask_dict[var] = surface_mask.copy(deep=False)

# Combine as Dataset
feature_mask_ds = xr.Dataset(data_vars=feature_mask_dict)

# Save to Zarr
feature_mask_ds = feature_mask_ds.chunk({"depth": -1, "latitude": -1, "longitude": -1})
feature_mask_ds.to_zarr(os.path.join(prep_dir, "mask.zarr"), mode="w")
print(feature_mask_ds)

# Store static variables
static_ds = static_grid.sel(depth=static_grid.depth[static_grid.depth <= MAX_DEPTH]).isel(depth=slice(None, None, STEP))
static_ds = static_ds.chunk({"depth": -1, "latitude": -1, "longitude": -1})
static_ds.to_zarr(os.path.join(prep_dir, "static.zarr"), mode="w")
print(static_ds)