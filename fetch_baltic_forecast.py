import os
from datetime import datetime, timedelta
import copernicusmarine as cm

# Base output directory
base_output_directory = "/scratch/project_2004522/dholmber/ocean-data/forecasts"

os.makedirs(base_output_directory, exist_ok=True)

# Define dates
today = datetime.utcnow()
date_string = today.strftime("%Y%m%d")
start_date = (today - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")
end_date = (today + timedelta(days=5)).strftime("%Y-%m-%dT23:59:59")
start_date_7_10 = (today + timedelta(days=6)).strftime("%Y-%m-%dT00:00:00")
end_date_7_10 = (today + timedelta(days=9)).strftime("%Y-%m-%dT23:59:59")

# Physics 3D
cm.subset(
    dataset_id="cmems_mod_bal_phy_anfc_P1D-m",
    variables=["so", "thetao", "uo", "vo"],
    start_datetime=start_date,
    end_datetime=end_date,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_phy",
    output_filename=f"ana_3d_{date_string}",
    file_format="netcdf",
)

# Physics 2D
cm.subset(
    dataset_id="cmems_mod_bal_phy_anfc_P1D-m",
    variables=["mlotst", "siconc", "sithick"],
    start_datetime=start_date,
    end_datetime=end_date,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_phy",
    output_filename=f"ana_2d_{date_string}",
    file_format="netcdf",
)

# Physics 3D 7-10 Days
cm.subset(
    dataset_id="cmems_mod_bal_phy_anfc_7-10days_P1D-m",
    variables=["so", "thetao", "uo", "vo"],
    start_datetime=start_date_7_10,
    end_datetime=end_date_7_10,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_phy",
    output_filename=f"ana_7-10days_3d_{date_string}",
    file_format="netcdf",
)

# Physics 2D 7-10 Days
cm.subset(
    dataset_id="cmems_mod_bal_phy_anfc_7-10days_P1D-m",
    variables=["mlotst", "siconc", "sithick"],
    start_datetime=start_date_7_10,
    end_datetime=end_date_7_10,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_phy",
    output_filename=f"ana_7-10days_2d_{date_string}",
    file_format="netcdf",
)

# SLA
cm.subset(
    dataset_id="cmems_mod_bal_phy_anfc_PT1H-i",
    variables=["sla"],
    start_datetime=start_date,
    end_datetime=end_date,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_phy",
    output_filename=f"ana_sla_{date_string}",
    file_format="netcdf",
)

# SLA 7-10 Days
cm.subset(
    dataset_id="cmems_mod_bal_phy_anfc_7-10days_PT1H-i",
    variables=["sla"],
    start_datetime=start_date_7_10,
    end_datetime=end_date_7_10,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_phy",
    output_filename=f"ana_7-10days_sla_{date_string}",
    file_format="netcdf",
)

# Biogeochemistry 3D
cm.subset(
    dataset_id="cmems_mod_bal_bgc_anfc_P1D-m",
    variables=["chl", "nh4", "no3", "o2", "pH", "po4", "spCO2", "zsd"],
    start_datetime=start_date,
    end_datetime=end_date,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_bgc",
    output_filename=f"ana_3d_{date_string}",
    file_format="netcdf",
)

# Biogeochemistry 7-10 Days
cm.subset(
    dataset_id="cmems_mod_bal_bgc_anfc_7-10days_P1D-m",
    variables=["chl", "nh4", "no3", "o2", "pH", "po4", "spCO2", "zsd"],
    start_datetime=start_date_7_10,
    end_datetime=end_date_7_10,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_bgc",
    output_filename=f"ana_7-10days_3d_{date_string}",
    file_format="netcdf",
)

# Hourly nppv
cm.subset(
    dataset_id="cmems_mod_bal_bgc-pp_anfc_P1D-i",
    variables=["nppv"],
    start_datetime=start_date,
    end_datetime=end_date,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_bgc",
    output_filename=f"ana_nppv_{date_string}",
    file_format="netcdf",
)

# Hourly nppv 7-10 Days
cm.subset(
    dataset_id="cmems_mod_bal_bgc-pp_anfc_7-10days_P1D-i",
    variables=["nppv"],
    start_datetime=start_date_7_10,
    end_datetime=end_date_7_10,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_bgc",
    output_filename=f"ana_7-10days_nppv_{date_string}",
    file_format="netcdf",
)

# Waves
cm.subset(
    dataset_id="cmems_mod_bal_wav_anfc_PT1H-i",
    variables=["VCMX", "VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW", "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW", "VMXL", "VPED", "VSDX", "VSDY", "VTM01_SW1", "VTM01_SW2", "VTM01_WW", "VTM02", "VTM10", "VTPK"],
    start_datetime=start_date,
    end_datetime=end_date,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_wav",
    output_filename=f"ana_2d_{date_string}",
    file_format="netcdf",
)

# Waves 7-10 Days
cm.subset(
    dataset_id="cmems_mod_bal_wav_anfc_7-10days_PT1H-i",
    variables=["VCMX", "VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW", "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW", "VMXL", "VPED", "VSDX", "VSDY", "VTM01_SW1", "VTM01_SW2", "VTM01_WW", "VTM02", "VTM10", "VTPK"],
    start_datetime=start_date_7_10,
    end_datetime=end_date_7_10,
    force_download=True,
    output_directory=f"{base_output_directory}/bal_wav",
    output_filename=f"ana_7-10days_2d_{date_string}",
    file_format="netcdf",
)
