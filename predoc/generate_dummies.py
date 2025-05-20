"""
Wrapper function to generate datasets fed into model
"""

from datetime import timedelta

import pandas as pd

from predoc.disease_lists_and_groupings import dict_laura_roman, dict_laura_roman_bps
from predoc.dummies_model_funcs import days_to_ov, dummies_mod, get_data_analyses


def generate_dummies(
    pats,
    bps,
    diag,
    espe,
    analy,
    history,
    horizon,
    col_to_add_value=0,
    col_to_add="dump_date",
    min_age=35,
    max_age=200,
    cie="",
    diag_categorical=False,
    espe_categorical=False,
    chron_age=False,
    group="",
):
    """
    Wrapper function which transforms dataframes from long to wide format.
    """

    ### Add grouped chronic diseases to chronic diseases dataset
    bps.index.names = ["person_id"]
    bps_groups = bps.copy().reset_index("person_id")
    bps = bps.reset_index("person_id")
    bps_groups["condition_source_value"] = bps_groups["condition_source_value"].replace(
        dict_laura_roman_bps
    )
    bps = pd.concat((bps, bps_groups)).drop_duplicates()
    bps = bps.set_index("person_id")

    ### Add grouped symptoms and diagnoses to symptoms and diagnoses dataset when these are in natural language
    diag_groups = diag.copy().reset_index("person_id")
    diag = diag.reset_index("person_id")
    diag_groups["condition_source_value"] = diag_groups[
        "condition_source_value"
    ].replace(dict_laura_roman)
    diag = pd.concat((diag, diag_groups)).drop_duplicates(
        ["person_id", "condition_start_date", "condition_source_value"]
    )
    diag = diag.set_index("person_id")

    ### "col_to_add_value" is True for prediction, in which case all data is taken relative to the dump date. Otherwise, data is taken relative to the index point
    if col_to_add_value == 0:
        bps = pats.merge(bps, left_index=True, right_index=True, how="left")
        diag = pats.merge(diag, left_index=True, right_index=True, how="left")
        espe = pats.merge(espe, left_index=True, right_index=True, how="left")
        analy = pats.merge(analy, left_index=True, right_index=True, how="left")

        ### get days between index point and diagnosis, specialist visit or analytic
        bps = days_to_ov(
            bps,
            "days_to_diagnosis",
            "main_condition_start_date",
            "condition_start_date",
        )
        diag = days_to_ov(
            diag,
            "days_to_diagnosis",
            "main_condition_start_date",
            "condition_start_date",
        )
        espe = days_to_ov(
            espe, "days_to_diagnosis", "main_condition_start_date", "visit_start_date"
        )
        analy = days_to_ov(
            analy, "days_to_diagnosis", "main_condition_start_date", "measurement_date"
        )

        ### if indicated, replace the presence of chronic diseases from binary to the age at which the patient was diagnosed with the disease, otherwise convert to wide format
        if chron_age:
            bps["age"] = (
                (bps["condition_start_date"] - bps["birth_datetime"]).dt.days / 365.25
            ).astype(int)
            bps_dum = bps.reset_index()[
                ["person_id", "condition_source_value", "age"]
            ].pivot_table(
                index="person_id", columns="condition_source_value", values="age"
            )
            bps_dum.columns.name = ""
        else:
            bps_dum = dummies_mod(
                bps,
                "condition_source_value",
                "days_to_diagnosis",
                history=False,
                horizon=horizon,
                categorical=True,
            ).fillna(0)

        ### convert diagnoses to wide format. Diagnoses can either be binary or continuous variables indicating the number of times the patient has had the symptom or diagnosis
        if diag_categorical:
            diag_dum = dummies_mod(
                diag,
                "condition_source_value",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
                categorical=diag_categorical,
            ).fillna(0)

        else:
            diag_dum = dummies_mod(
                diag,
                "condition_source_value",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
            ).fillna(0)

        ### convert specialist visits to wide format. Specialist visits can either be binary or continuous variables indicating the number of times the patient has visited the specialist
        if espe_categorical:
            special_dum = dummies_mod(
                espe,
                "provider_name",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
                categorical=espe_categorical,
            ).fillna(0)
        else:
            special_dum = dummies_mod(
                espe,
                "provider_name",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
            ).fillna(0)

        ### convert analytics to wide format
        data_analyses = get_data_analyses(
            analy,
            ["person_id", "measurement_source_value"],
            "value_source_value",
            "days_to_diagnosis",
            horizon=horizon,
            history=history,
        )

        ### assign age to patients in reference to their index point
        pats = pats.assign(
            age=(pats.main_condition_start_date - timedelta(days=int(horizon))).dt.year
            - pats.birth_datetime.dt.year
        )

        # Merge into single df
        all_dum = (
            bps_dum.merge(diag_dum, right_index=True, left_index=True, how="outer")
            .merge(special_dum, right_index=True, left_index=True, how="outer")
            .merge(data_analyses, right_index=True, left_index=True, how="outer")
            .merge(
                pats[["main_condition_start_date", "cohort"]],
                right_index=True,
                left_index=True,
                how="outer",
            )
            .merge(pats[["age"]], left_index=True, right_index=True, how="outer")
        )

        all_dum = all_dum[(all_dum["age"] >= min_age) & (all_dum["age"] <= max_age)]

    else:
        bps = pats.merge(bps, left_index=True, right_index=True, how="left")
        diag = pats.merge(diag, left_index=True, right_index=True, how="left")
        espe = pats.merge(espe, left_index=True, right_index=True, how="left")
        analy = pats.merge(analy, left_index=True, right_index=True, how="left")

        ### done for predictions, Here, the added column is the dump date
        bps[col_to_add] = col_to_add_value
        diag[col_to_add] = col_to_add_value
        espe[col_to_add] = col_to_add_value
        analy[col_to_add] = col_to_add_value

        ### get days between index point and diagnosis, specialist visit or analytic
        bps = days_to_ov(bps, "days_to_diagnosis", col_to_add, "condition_start_date")
        diag = days_to_ov(diag, "days_to_diagnosis", col_to_add, "condition_start_date")
        espe = days_to_ov(espe, "days_to_diagnosis", col_to_add, "visit_start_date")
        analy = days_to_ov(analy, "days_to_diagnosis", col_to_add, "measurement_date")

        ### if indicated, replace the presence of chronic diseases from binary to the age at which the patient was diagnosed with the disease, otherwise convert to wide format
        if chron_age:
            bps["age"] = (
                (bps["condition_start_date"] - bps["birth_datetime"]).dt.days / 365.25
            ).astype(int)
            bps_dum = bps.reset_index()[
                ["person_id", "condition_source_value", "age"]
            ].pivot_table(
                index="person_id", columns="condition_source_value", values="age"
            )
            bps_dum.columns.name = ""
        else:
            bps_dum = dummies_mod(
                bps,
                "condition_source_value",
                "days_to_diagnosis",
                history=False,
                horizon=horizon,
                categorical=True,
            ).fillna(0)

        ### convert diagnoses to wide format. Diagnoses can either be binary or continuous variables indicating the number of times the patient has had the symptom or diagnosis
        if diag_categorical:
            diag_dum = dummies_mod(
                diag,
                "condition_source_value",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
                categorical=diag_categorical,
            ).fillna(0)
        else:
            diag_dum = dummies_mod(
                diag,
                "condition_source_value",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
            ).fillna(0)

        ### convert specialist visits to wide format. Specialist visits can either be binary or continuous variables indicating the number of times the patient has visited the specialist
        if espe_categorical:
            special_dum = dummies_mod(
                espe,
                "provider_name",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
                categorical=espe_categorical,
            ).fillna(0)
        else:
            special_dum = dummies_mod(
                espe,
                "provider_name",
                "days_to_diagnosis",
                history=history,
                horizon=horizon,
            ).fillna(0)

        ### convert analytics to wide format
        data_analyses = get_data_analyses(
            analy,
            ["person_id", "measurement_source_value"],
            "value_source_value",
            "days_to_diagnosis",
            horizon=horizon,
            history=history,
        )

        ### assign age to patients in reference to their index point
        pats = pats.assign(
            age=(col_to_add_value - timedelta(days=int(horizon))).year
            - pats.birth_datetime.dt.year
        )
        # Merge into single df
        all_dum = (
            bps_dum.merge(diag_dum, right_index=True, left_index=True, how="outer")
            .merge(special_dum, right_index=True, left_index=True, how="outer")
            .merge(data_analyses, right_index=True, left_index=True, how="outer")
            .merge(
                pats[["main_condition_start_date", "cohort"]],
                right_index=True,
                left_index=True,
                how="outer",
            )
            .merge(pats[["age"]], left_index=True, right_index=True, how="outer")
        )

    return all_dum
