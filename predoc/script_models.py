# -*- coding: utf-8 -*-
"""
Separate the datasets into train, validation and test sets
"""


import pandas as pd
from sklearn.model_selection import train_test_split


def train_split(
    df,
    column_date,
    answer,
    year_test,
    year_min_train,
    year_max_train,
    month_min_val,
    month_max_val,
    val_prop,
    test_prop,
    random_state=3,
):
    """
    Given a dataframe df, divides it in three subdatasets: train, validation and test with X containing samples
    and feature values and y containing class labels. Firstly makes split seperately on cases and controls
    because case split is carried out according to diagnosis date and control split is carried out based on
    desired proportions of controls for each set. Training set contains both learning set and validation set.
    """

    max_date = pd.to_datetime(f"{year_max_train}-{month_min_val}-01")

    ##Split cases and controls
    cases = df[df[answer] == 1]
    controls = df[(df[answer] == -1) | (df[answer] == -2)]

    ##Split cases according to training years and validation months.
    ##Validation months always belong to last training year.
    X_cases_val = cases[
        (cases[column_date].dt.year == year_max_train)
        & (cases[column_date].dt.month >= month_min_val)
        & (cases[column_date].dt.month <= month_max_val)
    ]

    y_cases_val = cases[
        (cases[column_date].dt.year == year_max_train)
        & (cases[column_date].dt.month >= month_min_val)
        & (cases[column_date].dt.month <= month_max_val)
    ][[answer]]

    X_cases_train = cases[
        (cases[column_date].dt.year >= year_min_train) & (cases[column_date] < max_date)
    ]

    y_cases_train = cases[
        (cases[column_date].dt.year >= year_min_train) & (cases[column_date] < max_date)
    ][[answer]]

    X_cases_test = cases[(cases[column_date].dt.year == year_test)]
    y_cases_test = cases[(cases[column_date].dt.year == year_test)][[answer]]

    ##Carry out control split according to desired

    (
        X_control_train_,
        X_control_test,
        y_control_train_,
        y_control_test,
    ) = train_test_split(
        controls, controls[[answer]], test_size=test_prop, random_state=random_state
    )

    X_control_train, X_control_val, y_control_train, y_control_val = train_test_split(
        X_control_train_,
        X_control_train_[[answer]],
        test_size=(val_prop / (1 - val_prop)),
        random_state=random_state,
    )

    X_train = pd.concat([X_cases_train, X_control_train])
    y_train = pd.concat([y_cases_train, y_control_train])

    X_val = pd.concat([X_cases_val, X_control_val])
    y_val = pd.concat([y_cases_val, y_control_val])

    X_test = pd.concat([X_cases_test, X_control_test])
    y_test = pd.concat([y_cases_test, y_control_test])

    return X_train, X_val, y_train, y_val, X_test, y_test


# ---------------------------------------------------------
