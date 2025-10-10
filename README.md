# ocean-data

## Installation

```
pdm venv create --with-pip
pdm install
```

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