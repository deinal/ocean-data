import datetime
import os
from ecmwf.opendata import Client

# Initialize the ECMWF Open Data client
client = Client(source="ecmwf")

# Define base output directory
BASE_PATH = "/scratch/project_2004522/dholmber/ocean-data/ecmwf_init"
os.makedirs(BASE_PATH, exist_ok=True)

# Parameters
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Define date range
now = client.latest()
start_date = now - datetime.timedelta(days=3)
end_date = now - datetime.timedelta(days=1)
print(f"Downloading data from {start_date.strftime('%Y-%m-%d-%H')} to {end_date.strftime('%Y-%m-%d-%H')}")

# Retrieval function
def get_open_data(client, param, start_date, end_date, group, levelist=[]):
    current_date = start_date
    while current_date <= end_date:
        try:
            grib_file = os.path.join(BASE_PATH, f"{group}_{current_date.strftime('%Y%m%d%H')}.grib2")

            request_params = {
                "date": current_date,
                "param": param,
                "levelist": levelist,
                "target": grib_file,
            }

            print(f"Requesting data for {current_date.strftime('%Y-%m-%d')} to {grib_file}")
            client.retrieve(**request_params)
            print(f"Saved: {grib_file}")

        except Exception as e:
            print(f"Failed to retrieve data for {current_date.strftime('%Y-%m-%d')}: {e}")

        # Increment date by 6 hours
        current_date += datetime.timedelta(hours=6)

# Retrieve surface parameters
get_open_data(client, param=PARAM_SFC, levelist=[], start_date=start_date, end_date=end_date, group="sfc")

# Retrieve pressure level parameters
get_open_data(client, param=PARAM_PL, levelist=LEVELS, start_date=start_date, end_date=end_date, group="pl")
