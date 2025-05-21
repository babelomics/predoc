"""
Functions to generate dataset fed into model
"""

import numpy as np
import pandas as pd


def dummies_mod(
    df, column_dum, column_days, history=False, horizon=30, categorical=False
):
    """
    Given a dataframe contaning columns of factors it transforms this column into dummies variables
    return the dataframe of this dummies
    """
    cols = set(df[column_dum].unique())

    if history:
        df = df[
            (df[column_days].dt.days >= horizon)
            & (df[column_days].dt.days <= history + horizon)
        ]
    else:
        df = df[df[column_days].dt.days >= horizon]

    dum = (
        (
            pd.get_dummies(
                df[[column_dum]], prefix="", prefix_sep="", columns=[column_dum]
            )
        )
        .groupby(df.index)
        .sum()
    )

    if categorical:
        dum[dum > 1] = 1

    missing_cols = list(set(dum.columns).symmetric_difference(cols))
    D = {missing_cols[i]: np.nan for i in range(len(missing_cols))}

    dum = dum.join(pd.DataFrame(D, index=dum.index)).fillna(0).astype(int)
    """
    if categorical:
        dum = dum.replace({0:"no", 1:"yes"})
    """
    return dum


def days_to_ov(df, col_out, col_in_one, col_in_two):
    """
    Let df be a dataframe containing features values with two columns of dates,
    calculate the days between this two columns and return df with only difference
    bigger than 0
    """

    df[col_out] = df[col_in_one] - df[col_in_two]
    df = df[df[col_out].dt.days >= 0]

    return df


def get_predics(df, column_cohort, year=None, column_selection=None):
    """Given a dataframe containing feature values, a column indicating if samples belong to a
    positive or negative class and optionally a datetime column to select subsamples of either class.
    If the latter is provided, a year for selection must be provided. This function
    returns X: feature values and y: sample classes compatible with an estimator
    """

    if year:
        df = df[df[column_selection].dt.year == year]
        X = df.drop([column_selection, column_cohort], axis=1)
    else:
        X = df.drop(column_cohort, axis=1)

    y = df[[column_cohort]]

    return X, y


def get_predics_df(X, y, predics, col_to_add_name=None, col_to_add_value=None):
    """Given dataframes X containing feature variables with index sample names, y class labels with index sample names
    and predics the prediction probabilities provided by an estimator. This function log transforms the prediction
    probabilities and returns a dataframe with both the probabilities and the log transformed probabilities. Optionally,
    adds an additional column.
    """

    log_probs = np.log(predics)
    log_probs_df = pd.DataFrame(
        log_probs, columns=["Ovarian cancer log probability"]
    ).set_index(X.index)
    log_probs_df["Ovarian cancer probability"] = predics
    if col_to_add_name:
        log_probs_df[col_to_add_name] = col_to_add_value

    return log_probs_df


def get_data_analyses(df, list_columns_to_group, value_col, days_col, horizon, history):
    """
    Given a dataframe contaning columns of factors  and a column of values of analitics
    it transforms the rows of the column factor into columns and assing the value for the
    min, max and mean (each one is a different colum )return the dataframe
    """

    name = list_columns_to_group[1]
    cols = set(df[name].unique())

    df = df[
        (df[days_col].dt.days > horizon) & (df[days_col].dt.days <= horizon + history)
    ]
    missing_cols = list(set(df[name].unique()).symmetric_difference(cols))
    D = {missing_cols[i]: np.nan for i in range(len(missing_cols))}

    meds = (
        df.groupby(list_columns_to_group)
        .mean(value_col)
        .reset_index(level=list_columns_to_group[1])
    )
    meds = meds.pivot_table(
        values=value_col,
        index=meds.index,
        columns=list_columns_to_group[1],
        dropna=False,
    )
    meds = meds.join(pd.DataFrame(D, index=meds.index)).add_suffix("_mean")

    mins = (
        df.groupby(list_columns_to_group)
        .min(value_col)
        .reset_index(level=list_columns_to_group[1])
    )
    mins = mins.pivot_table(
        values=value_col,
        index=mins.index,
        columns=list_columns_to_group[1],
        dropna=False,
    )
    mins = mins.join(pd.DataFrame(D, index=meds.index)).add_suffix("_min")

    maxs = (
        df.groupby(list_columns_to_group)
        .max(value_col)
        .reset_index(level=list_columns_to_group[1])
    )
    maxs = maxs.pivot_table(
        values=value_col,
        index=maxs.index,
        columns=list_columns_to_group[1],
        dropna=False,
    )
    maxs = maxs.join(pd.DataFrame(D, index=meds.index)).add_suffix("_max")

    all_data = meds.merge(mins, left_index=True, right_index=True).merge(
        maxs, left_index=True, right_index=True, how="outer"
    )

    return all_data
