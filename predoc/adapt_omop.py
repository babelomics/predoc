"""
2024-11-08
For a detailed description of this code, go to:
predoc/notebooks/adapt_omop_guide.ipynb
"""
# %% Start-up cell
import os

import pandas as pd
import yaml
from dotenv import load_dotenv

# == Parameters =======================================================
print("Reading parameters...")
load_dotenv()
# Define entrada y salida de archivos
main_dir = os.environ.get("LOCAL_DATA_DIR")
input_dir = f"{main_dir}/omop/rare/"
output_dir = f"{main_dir}/omop/done/"
os.makedirs(output_dir, exist_ok=True)

# Load config file
config_file_path = "./params.yaml"
with open(config_file_path, "r", encoding="utf-8") as config_file:
    config_data = yaml.safe_load(config_file)
for param, value in config_data.items():
    print(f" {param}: {value}")

# == Retrieve data ====================================================
print("Retrieving data...")
tables = {
    f.split(".")[0]: pd.read_parquet(f"{input_dir}{f}")
    for f in os.listdir(input_dir)
    if f.endswith(".parquet")
}

# == Perform the adaptation ===========================================
# %% Patients and main diagnosis
print("Processing pats dataframe...")

# Filter those that do have the defined target concept id
pats = tables["CONDITION_OCCURRENCE"]
filt = pats["condition_source_concept_id"] == config_data["target_diagnosis_concept_id"]
pats = pats.loc[filt, ["person_id", "condition_source_value", "condition_start_date"]]
pats = pats.drop_duplicates(subset=["person_id", "condition_start_date"])

# Escogemos el index
pats = pats.set_index("person_id")

# Traemos las personas con su fecha de nacimiento
person = tables["PERSON"][["person_id", "birth_datetime"]]
person = person.set_index("person_id")

# Hacemos el merge
pats = person.merge(pats, on="person_id", how="left")

# Seleccionamos columnas
pats = pats[["birth_datetime", "condition_start_date"]]

# Rename condition_start_date to avoid future collisions
pats = pats.rename(
    {
        "condition_start_date": "main_condition_start_date",
    },
    axis=1,
)

# Make sure dates are dates
pats["birth_datetime"] = pd.to_datetime(pats["birth_datetime"])
pats["main_condition_start_date"] = pd.to_datetime(pats["main_condition_start_date"])

# Save
pats = pats.reset_index()
pats.to_parquet(f"{output_dir}pats.parquet")

# %% process comormidities info
print("Processing bps dataframe...")
# -- Retrieve list of codes from BPS Pathology
BPS_codes = tables["CONCEPT"][
    (tables["CONCEPT"]["vocabulary_id"] == "BPS Pathology")
    & (tables["CONCEPT"]["domain_id"] == "Condition")
]

BPS_codes = BPS_codes[["concept_id", "concept_name"]]

# -- Get a subset of CONDITION_OCCURRENCE
bps_index = tables["CONDITION_OCCURRENCE"]["condition_source_concept_id"].isin(
    BPS_codes["concept_id"]
)
bps = tables["CONDITION_OCCURRENCE"].loc[bps_index]
# Keep only relevant columns
bps = bps[["person_id", "condition_start_date", "condition_source_value"]]

# Make sure dates are dates
bps["condition_start_date"] = pd.to_datetime(bps["condition_start_date"])

# Save
bps.to_parquet(f"{output_dir}bps.parquet")


# %% diag
print("Processing diag dataframe...")
# -- Retrieve list of codes from BPS Pathology
BPS_codes = tables["CONCEPT"][
    (tables["CONCEPT"]["vocabulary_id"] == "BPS Pathology")
    & (tables["CONCEPT"]["domain_id"] == "Condition")
]
BPS_codes = BPS_codes[["concept_id", "concept_name"]]

# -- Get a subset of CONDITION_OCCURRENCE
bps_index = tables["CONDITION_OCCURRENCE"]["condition_source_concept_id"].isin(
    BPS_codes["concept_id"]
)
diag = tables["CONDITION_OCCURRENCE"].loc[~bps_index]
# Keep only relevant columns
diag = diag[["person_id", "condition_start_date", "condition_source_value"]]

# Make sure dates are dates
diag["condition_start_date"] = pd.to_datetime(diag["condition_start_date"])

# Save
diag.to_parquet(f"{output_dir}diag.parquet")

# %% espe
print("Processing espe dataframe...")
# To get rid of the warning
pd.options.mode.copy_on_write = True

# Identify the table
espe = tables["VISIT_OCCURRENCE"]
provider_table = tables["PROVIDER"]
# Select only relevant columns
espe = espe[["person_id", "visit_start_date", "provider_id"]]

# Map provider_id to provider_name
provider_map = dict(zip(provider_table["provider_id"], provider_table["provider_name"]))
espe["provider_name"] = espe["provider_id"].map(provider_map)

# Set the index
espe = espe.set_index("person_id")

# Only need visits with a especialidad
espe = espe.dropna(subset="provider_id")
espe = espe.replace("aparato digestivo", "digestivo")

# Make sure dates are dates and ints are ints
espe["visit_start_date"] = pd.to_datetime(espe["visit_start_date"])
espe["provider_id"] = espe["provider_id"].astype("int")

# Save
espe = espe.reset_index()
espe.to_parquet(f"{output_dir}espe.parquet")

# %% analy
print("Processing analy dataframe...")
# Identify the table
analy = tables["MEASUREMENT"]
# Select only relevant columns
analy = analy[
    ["person_id", "measurement_date", "measurement_source_value", "value_source_value"]
]

# Make sure dates are dates
analy["measurement_date"] = pd.to_datetime(analy["measurement_date"])
# Make sure numeric are numeric
analy["value_source_value"] = pd.to_numeric(analy["value_source_value"])

# Save
analy.to_parquet(f"{output_dir}analy.parquet")

print("Done.")

# %%
