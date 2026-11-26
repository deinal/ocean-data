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

## Baltic data

### 1. Reanalysis (BAL MY, 1993–2024)

```
python fetch_baltic_static.py                 # baltic_static.zarr, baltic_mask.zarr
python fetch_baltic_data.py --start 1993-01-01 --end 2024-12-31   # baltic_data.zarr
```

### 2. Operational analysis (BAL ANFC, 2023-01 onwards)

`baltic_oper_mask.zarr` exists already.
```
python fetch_baltic_oper_data.py --start 2023-01-01 --end 2026-04-20   # baltic_oper_data.zarr
```

### 3. Forecast ground truth and IFS forcing (2024 weekly inits)

Needs `ifs2024_daily_means.zarr` from the global IFS step.
```
python process_baltic_forecasts.py            # baltic_forecasts.zarr, baltic_ifs_forecasts.zarr
```

ERA5 forcing (`baltic_era5.zarr`) comes from `regrid_era5.py`, see Global data step 4.

### 4. Boundary (North Sea strip, 5–9°E, 55–60°N)

Cropped from the global GLORYS data, so run Global data step 1 first.
```
python create_baltic_boundary.py --num-workers 100   # baltic_boundary_{data,static,mask}.zarr
```

Boundary forecasts: lead -1 day from `global_analysis_mondays_2024.zarr`, lead 0 from `glo12_nowcasts.zarr`, leads 1–10 days from the GLO12 forecasts.
```
python fetch_glo12_forecasts.py               # glo12_forecasts.zarr
python process_baltic_boundary.py             # baltic_boundary_forecasts.zarr
```

### 5. Daily cronjob to fetch Baltic forecasts

```
0 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_weather_states.py >> /scratch/project_2004522/dholmber/ocean-data/weather.log 2>&1

15 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_global_forecast.py >> /scratch/project_2004522/dholmber/ocean-data/global.log 2>&1

30 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_baltic_forecast.py >> /scratch/project_2004522/dholmber/ocean-data/baltic.log 2>&1
```

## OceanBench evaluation

Global forecasts from the OceanBench challengers, 2024 weekly inits, leads 1–10 days.
All are written as `{challenger}_global_forecasts.zarr` on the 1/4° grid with depths `[0, 47, 92, 222, 318, 541]` m.

GloNet:
```
python fetch_oceanbench_challengers.py --challenger glonet   # glonet_global_forecasts.zarr
```

Wenhai and XiHe: fetch the raw forecasts (depths < 600 m) first, then select or interpolate the target depths and regrid:
```
python fetch_raw_forecast.py --challenger wenhai      # wenhai_raw_forecasts.zarr
python process_wenhai_xihe.py --challenger wenhai     # wenhai_global_forecasts.zarr
python fetch_raw_forecast.py --challenger xihe
python process_wenhai_xihe.py --challenger xihe
```

GLO12 forecasts over the Baltic (5–35°E, 50–70°N), interpolated to the Baltic grid for comparison with the Baltic model:
```
python fetch_glo12_baltic_forecasts.py                # glo12_baltic_forecasts.zarr
python interpolate_glo12_baltic_forecasts.py          # glo12_baltic_forecasts_interp.zarr
```

## Hugging Face

The Baltic zarrs are published as [deinal/njord-baltic-data](https://huggingface.co/datasets/deinal/njord-baltic-data). They were staged and uploaded with:
```
python stage_hf_baltic.py
python upload_hf_baltic.py
```

It can be downloaded to a local data directory with:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deinal/njord-baltic-data",
    repo_type="dataset",
    local_dir="njord_baltic_data"
)
```

This will yield a local data folder that can be used with [njord](https://github.com/deinal/njord).
