# ocean-data

## Installation

```
pdm venv create --with-pip
pdm install
```

All scripts write to `data/` by default (`--base-dir` / `--out-dir` to change).

## Global data

### 1. Reanalysis (GLORYS, 1993–2024)

```
python fetch_global_static.py                 # global_static.zarr, global_mask.zarr
python fetch_global_data.py --start 1993-01-01 --end 2024-12-31   # global_data.zarr
python fetch_global_data.py --start 2025-01-01 --end 2025-01-31   # renamed output to global_data_jan_2025.zarr for additional data
python regrid_global.py --num-workers 100     # global_{data,mask,static}_{1_4,1}.zarr
```

### 2. Analysis (GLO12 operational, 2023-01 – 2025-01)

```
python fetch_global_analysis_data.py          # global_analysis_data.zarr
python fetch_global_analysis_static.py        # global_analysis_static.zarr, global_analysis_mask.zarr
python fetch_oceanbench_inputs.py             # glo12_nowcasts.zarr (OceanBench, 2024 Tuesdays)
python regrid_global_analysis.py              # global_analysis_{data,mask,static}_1_4.zarr, glo12_nowcasts_1_4.zarr
python process_global_analysis.py             # global_analysis_forecasts.zarr (init_time x lead_time -1..10 days)
```

### 3. Forecast ground truth (2024 weekly inits)

`global_analysis_mondays_2024.zarr` (the 53 Mondays of 2024, taken from the GLO12 analysis).
```
python regrid_global_additional.py            # {global_data_jan_2025,glo12_nowcasts,global_analysis_mondays_2024}_{1_4,1}.zarr
python process_global_data.py                 # global_forecasts_truth.zarr
```

### 4. Atmospheric forcing

ERA5 (ARCO, hourly → daily). The fetch may be split into several runs (`arco_era5_1.zarr`, `arco_era5_2.zarr`, ...), which `combine_era5.py` merges:
```
python fetch_arco_era5.py --start-date 1993-01-01 --end-date 2024-12-31
python combine_era5.py                        # arco_era5.zarr
python regrid_era5.py --num-workers 100       # global_era5_{1_4,1}.zarr, global_analysis_era5_1_4.zarr, baltic_era5.zarr
```
`regrid_era5.py` also regrids onto `baltic_mask.zarr`, so the Baltic static data must exist first.

IFS 2024 forecasts:
```
python process_ifs_2024_data.py               # ifs2024_daily_means.zarr
python regrid_ifs.py                          # ifs_analysis_forecasts.zarr
```
`regrid_ifs.py` has hardcoded `OUT_PATH`/`MASK_PATH`. Swap the commented lines to produce `ifs_forecasts.zarr` on `global_mask_1_4.zarr`.

## Fetch Baltic data

```
python fetch_baltic_static.py
python fetch_baltic_data.py --start-year 2022 --end-year 2023
```

```
python fetch_boundary_static.py
python fetch_boundary_data.py --start-year 2022 --end-year 2023

For ERA5 static is loaded after data
```
python fetch_era5_data.py --start-year 2022 --end-year 2023
python fetch_era5_static.py
```

Daily cronjob
```
0 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_weather_states.py >> /scratch/project_2004522/dholmber/ocean-data/weather.log 2>&1

15 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_global_forecast.py >> /scratch/project_2004522/dholmber/ocean-data/global.log 2>&1

30 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_baltic_forecast.py >> /scratch/project_2004522/dholmber/ocean-data/baltic.log 2>&1
```

Process weather states

https://huggingface.co/ecmwf/aifs-single/blob/main/run_AIFS_v0_2_1.ipynb


## fetch weatherquest data

fetch an interval as
```
python fetch_wq_ocean.py --start 1993-01-01 --end 1999-12-31
```

fetch static
```
python fetch_wq_static.py
```

combine intervals (hardcoded dates here) and replace nans with zeros for ice variables
```
python combine_wq_data.py
```

create zarrs for weekly means (starting at 1=Mon, 2=Tue, etc.)
```
python create_wq_weekly.py --weekday 1
```