# ocean-data

Daily cronjob
```
0 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_weather_states.py >> /scratch/project_2004522/dholmber/ocean-data/weather.log 2>&1

15 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_global_forecast.py >> /scratch/project_2004522/dholmber/ocean-data/global.log 2>&1

30 17 * * * /appl/soft/geo/geoconda/3.10.9_20231117/bin/python /scratch/project_2004522/dholmber/ocean-data/fetch_baltic_forecast.py >> /scratch/project_2004522/dholmber/ocean-data/baltic.log 2>&1
```

Process weather states

https://huggingface.co/ecmwf/aifs-single/blob/main/run_AIFS_v0_2_1.ipynb
