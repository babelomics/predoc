"""
Functions which preprocess raw dataset into usable one
"""

import collections.abc

import pandas as pd


def normalice_titles(columns):
    """
    Given a vector of column names as strings, normalizes all the strings so they have the same format.

    """
    return (
        columns.str.lower()
        .str.replace("de_", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(",", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.replace("__", "_")
        .replace(
            "filtrado_glomerular_1_73_m^2_estimado",
            "filtrado_glomerular_estimado",
            regex=False,
        )
    )


def filtered_txt(
    df,
    column_date,
    nuhsa_column,
    selected_column,
    column_numeric=0,
    column_factor=0,
    index=True,
    date_format="%Y%m%d",
    errors="coerce",
):
    """
    Given a dataframe, normalizes columns and variable names.

    """
    # See if it is vector or a string
    if isinstance(column_date, collections.abc.Sequence) == True:
        for i in column_date:
            df[i] = pd.to_datetime(df[i], format=date_format, errors=errors)
    else:
        df[column_date] = pd.to_datetime(
            df[column_date], format=date_format, errors=errors
        )

    if column_factor == 0:
        pass
    else:
        df[column_factor] = df[column_factor].astype(str).astype("category")
        df[column_factor] = (
            df[column_factor]
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace(",", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("_#", "", regex=False)
            .str.replace("__", "_", regex=False)
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )

    if column_numeric == 0:
        pass
    else:
        df[column_numeric] = pd.to_numeric(
            df[column_numeric].str.replace(",", ".", regex=False), errors="coerce"
        )

    if index == True:
        df = df.set_index(nuhsa_column)

    df = df[selected_column]
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(",", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )

    return df


def map_cie_code(df, mapping_dict, col_to_map="COD_CIE_NORMALIZADO"):
    """
    Transform diagnoses in ICD code to natural language
    """
    df[col_to_map] = df[col_to_map].replace(mapping_dict)

    df_codes = df[df[col_to_map].str.contains(".", regex=False)]
    df_alphas = df[~df[col_to_map].str.contains(".", regex=False)]

    mapping_dict = {key[:3]: mapping_dict[key] for key in mapping_dict}

    df_codes[col_to_map] = df_codes[col_to_map].str.slice(0, 3)
    df_codes[col_to_map] = df_codes[col_to_map].replace(mapping_dict)

    df = pd.concat([df_codes, df_alphas])

    return df
