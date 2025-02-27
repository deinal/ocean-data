from datetime import datetime, timedelta
import copernicusmarine as cm

# Base output directory
base_output_directory = "/scratch/project_2004522/dholmber/ocean-data/forecasts"

# Define dates
today = datetime.utcnow()
date_string = today.strftime("%Y%m%d")
start_date = (today - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")
end_date = (today + timedelta(days=9)).strftime("%Y-%m-%dT23:59:59")

# Physics 3D
cm.subset(
    dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
    variables=["so"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_phy",
    output_filename=f"ana_so_{date_string}",
    file_format="netcdf",
)
cm.subset(
    dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    variables=["thetao"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_phy",
    output_filename=f"ana_thetao_{date_string}",
    file_format="netcdf",
)
cm.subset(
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    variables=["uo"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_phy",
    output_filename=f"ana_uo_{date_string}",
    file_format="netcdf",
)
cm.subset(
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    variables=["vo"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_phy",
    output_filename=f"ana_vo_{date_string}",
    file_format="netcdf",
)

# Physics 2D
cm.subset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    variables=["mlotst", "siconc", "sithick", "usi", "vsi", "zos"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_phy",
    output_filename=f"ana_2d_{date_string}",
    file_format="netcdf",
)

# Biogeochemistry
cm.subset(
    dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
    variables=["nppv", "o2"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_bgc",
    output_filename=f"ana_bio_{date_string}",
    file_format="netcdf",
)
cm.subset(
    dataset_id="cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    variables=["no3", "po4", "si"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_bgc",
    output_filename=f"ana_nut_{date_string}",
    file_format="netcdf",
)
cm.subset(
    dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
    variables=["chl"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_bgc",
    output_filename=f"ana_chl_{date_string}",
    file_format="netcdf",
)

# Waves
cm.subset(
    dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
    variables=["VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW", "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW", "VPED", "VSDX", "VSDY", "VTM01_SW1", "VTM01_SW2", "VTM01_WW", "VTM02", "VTM10", "VTPK"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=-4,
    maximum_longitude=12,
    minimum_latitude=48,
    maximum_latitude=66,
    force_download=True,
    output_directory=f"{base_output_directory}/glo_wav",
    output_filename=f"ana_{date_string}",
    file_format="netcdf",
)
