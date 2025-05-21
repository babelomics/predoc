# This file removes omop codes present in the config file if needed
# Any further cleaning of data should be done here

import os

import pandas as pd
import yaml
from dotenv import load_dotenv

# == Parameters =======================================================
print("Reading parameters...")
load_dotenv()
# Define entrada y salida de archivos
main_dir = os.environ.get("DATA_PATH")
input_dir = f"{main_dir}/omop/raw/"
output_dir = f"{main_dir}/omop/rare/"
os.makedirs(output_dir, exist_ok=True)

# Load config file
config_file_path = f"./params.yaml"
with open(config_file_path, "r", encoding="utf-8") as config_file:
    config_data = yaml.safe_load(config_file)
for param, value in config_data.items():
    print(f" {param}: {value}")

# == Retrieve configuration ===========================================
ban_codes = config_data["ban_codes"]

# == Retrieve data ====================================================
print("Retrieving data...")
tables = {
    f.split(".")[0]: pd.read_parquet(f"{input_dir}{f}")
    for f in os.listdir(input_dir)
    if f.endswith(".parquet")
}

print("Filtering data...")
cohort = tables["PERSON"].copy()

# Filter by ban_list
# Create a ban_list with ppl that have it
filt = tables["CONDITION_OCCURRENCE"]["condition_source_concept_id"].isin(ban_codes)
ban_list = tables["CONDITION_OCCURRENCE"].loc[filt, "person_id"].values
# Filter the cohort
cohort = cohort[~cohort["person_id"].isin(ban_list)]

print("Saving data...")
# Reduce all tables to cohort patients
for key, table in tables.items():
    try:
        table = table[table["person_id"].isin(cohort["person_id"].to_numpy())]
        table.to_parquet(f"{output_dir}{key}.parquet")
    except KeyError:
        pass
# Bring concept and provider as well for next step
tables["CONCEPT"].to_parquet(f"{output_dir}CONCEPT.parquet")
tables["PROVIDER"].to_parquet(f"{output_dir}PROVIDER.parquet")

print("Done.")
