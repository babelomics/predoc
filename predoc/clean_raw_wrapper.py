"""
Wrapper function calling different functions which preprocess raw dataset into usable one
"""
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

from predoc.clean_raw_functions import filtered_txt, map_cie_code, normalice_titles
from predoc.datasets import data_dir
from predoc.disease_lists_and_groupings import mapping


def clean_data(
    directory,
    known="yes",
    train="yes",
    seed=3,
    train_year_min=2017,
    train_year_max=2021,
    cie="",
    group="",
    min_age=35,
    max_age=200,
    ref_year=0,
):
    """
    Wrapper function which cleans (remove cumbersome characters, convert dataypes etc.) the five datasets used in training an ovarian cancer predictor as well as predicting on new data. For training, index points to controls can be included.
    """

    # To get rid of the warning
    pd.options.mode.copy_on_write = True

    ### load raw data
    all_pats_raw = pd.read_parquet(
        f"{data_dir}/{directory}/pats.parquet"
    ).drop_duplicates()
    all_bps_raw = pd.read_parquet(
        f"{data_dir}/{directory}/bps.parquet"
    ).drop_duplicates()
    all_diag_raw = pd.read_parquet(
        f"{data_dir}/{directory}/diag.parquet"
    ).drop_duplicates()
    all_analy_raw = pd.read_parquet(
        f"{data_dir}/{directory}/analy.parquet"
    ).drop_duplicates()
    all_espe_raw = pd.read_parquet(
        f"{data_dir}/{directory}/espe.parquet"
    ).drop_duplicates()

    ### clean patient demographic data and filter by age limits
    all_pats = filtered_txt(
        all_pats_raw,
        column_date=["birth_datetime", "main_condition_start_date"],
        nuhsa_column="person_id",
        selected_column=["birth_datetime", "main_condition_start_date"],
        column_numeric=0,
    )

    all_pats_cases = all_pats[~all_pats["main_condition_start_date"].isna()]
    all_pats_cases["cohort"] = 1

    all_pats_cases["age"] = (
        (
            all_pats_cases["main_condition_start_date"]
            - all_pats_cases["birth_datetime"]
        ).dt.days
        / 365.25
    ).astype(int)
    all_pats_cases = all_pats_cases[
        (all_pats_cases["age"] >= min_age) & (all_pats_cases["age"] <= max_age)
    ]

    all_pats_controls = all_pats[(all_pats["main_condition_start_date"].isna())]
    all_pats_controls["cohort"] = -1

    # Save files for control
    if train == "yes":
        savedir = f"{data_dir}/omop/train/"
        os.makedirs(savedir, exist_ok=True)
        all_pats_cases.to_parquet(f"{savedir}control_all_pats_cases.parquet")
        all_pats_controls.to_parquet(f"{savedir}control_all_pats_controls.parquet")
        all_analy_raw.to_parquet(f"{savedir}control_all_analy_raw.parquet")

    ### set index points for controls in case of training
    if train == "yes":
        np.random.seed(seed)

        # ------------------------------------------------------------------------------
        # Remove duplicate patients so .nunique makes sense later
        all_pats_controls = (
            all_pats_controls.reset_index()
            .drop_duplicates(subset=["person_id"])
            .set_index("person_id")
        )
        # ------------------------------------------------------------------------------

        train_dates = all_pats_cases[
            (all_pats_cases["main_condition_start_date"].dt.year >= train_year_min)
            & (all_pats_cases["main_condition_start_date"].dt.year <= train_year_max)
        ]["main_condition_start_date"]
        random_sample = np.random.choice(
            train_dates, size=all_pats_controls.index.nunique()
        )
        all_pats_controls["main_condition_start_date"] = random_sample

    if (train == "yes") & (ref_year != 0):
        random.seed(seed)
        start_date = date(ref_year, 6, 1)
        end_date = date(ref_year, 12, 31)

        random_dates = []
        num_dates = all_pats_controls.index.nunique()

        random_days = np.random.randint(0, (end_date - start_date).days + 1, num_dates)

        # Convert the random days to timedelta objects
        random_timedeltas = [timedelta(days=int(day)) for day in random_days]

        # Create an array of random dates by adding timedeltas to the start date
        random_dates = [start_date + delta for delta in random_timedeltas]

        # Convert the list of random dates to a DatetimeArray
        random_dates_array = np.array(random_dates, dtype="datetime64[D]")

        all_pats_controls["main_condition_start_date"] = random_dates_array

    all_pats = pd.concat([all_pats_cases, all_pats_controls])

    all_pats.index.names = ["person_id"]

    ### clean chronic diseases dataset
    all_bps = filtered_txt(
        all_bps_raw,
        column_date=["condition_start_date"],
        nuhsa_column="person_id",
        selected_column=["condition_start_date", "condition_source_value"],
        column_numeric=0,
        column_factor="condition_source_value",
    )

    all_bps.condition_source_value = normalice_titles(all_bps.condition_source_value)
    all_bps.index.names = ["person_id"]

    ### clean symptoms and diagnostics dataset. If indicated, change diagnoses from ICD code to natural language.
    if len(cie) == 0:
        all_diag_raw = map_cie_code(all_diag_raw, mapping)
        all_diag_raw.loc[
            all_diag_raw[
                all_diag_raw["condition_source_concept_id"].str.len() == 3
            ].index,
            "condition_source_concept_id",
        ] = all_diag_raw.loc[
            all_diag_raw[
                all_diag_raw["condition_source_concept_id"].str.len() == 3
            ].index,
            "condition_source_value",
        ]
    all_diag = filtered_txt(
        all_diag_raw,
        column_date=["condition_start_date"],
        nuhsa_column="person_id",
        selected_column=[
            "person_id",
            "condition_start_date",
            "condition_source_value",
        ],
        column_factor="condition_source_value",
        index=False,
    ).set_index("person_id")

    ### if grouping is indicated, group ICD codes into root codes
    if len(group) > 0:
        all_diag["condition_source_concept_id"] = [
            s.split(".")[0] for s in all_diag["condition_source_concept_id"]
        ]

    all_diag.index.names = ["person_id"]

    ### clean specialist visits dataset
    all_espe = filtered_txt(
        df=all_espe_raw,
        column_date=["visit_start_date"],
        nuhsa_column="person_id",
        selected_column=["visit_start_date", "provider_name"],
        column_factor="provider_name",
        date_format="%d/%m/%y",
    )

    all_espe.provider_name = normalice_titles(all_espe.provider_name)
    all_espe.index.names = ["person_id"]

    ### clean analytics dataset
    all_analy = filtered_txt(
        all_analy_raw,
        column_date=["measurement_date"],
        nuhsa_column="person_id",
        selected_column=[
            "measurement_date",
            "measurement_source_value",
            "value_source_value",
            "person_id",
        ],
        column_factor="measurement_source_value",
        column_numeric=0,  # En los datos OMOP ya viene numerico
        index=False,
    ).set_index("person_id")

    all_analy.measurement_source_value = normalice_titles(
        all_analy.measurement_source_value
    )
    all_analy.index.names = ["person_id"]

    if train == "yes":
        all_pats.to_parquet(f"{savedir}control_all_pats.parquet")
        all_bps.to_parquet(f"{savedir}control_all_bps.parquet")
        all_diag.to_parquet(f"{savedir}control_all_diag.parquet")
        all_espe.to_parquet(f"{savedir}control_all_espe.parquet")
        all_analy.to_parquet(f"{savedir}control_all_analy.parquet")

    return all_pats, all_bps, all_diag, all_espe, all_analy
