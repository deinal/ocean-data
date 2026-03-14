
# Fetch GFS data from dynamical.org

Requires zarr v3, hence its own subproject.

```
pdm install
```

Run `pdm run python fetch_gfs.py` to download 15-day GFS forecasts with first day Wednesdays of 2024.

Then export the zarr v3 to netcdf using `pdm run python zarr_to_netcdf.py` to use in `ocean-data` with zarr v2.

You can specify output directories for the data as it fits. Also num workers are by default 8.