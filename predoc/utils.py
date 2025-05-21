# -*- coding: utf-8 -*-
"""
Functions for loading and saving models and datasets 
"""
import os
from pathlib import Path

import joblib
import pandas as pd

from predoc.datasets import data_dir


def load_data(filename, folder, delimiter="|", sheet_name=None, engine=None):
    """
    Takes parquet, csv or txt file, opens and returns it
    """

    file_path = Path(data_dir, folder, filename)

    if file_path.suffix == ".parquet":
        try:
            data = pd.read_parquet(file_path)
            return data
        except FileNotFoundError:
            return "File not found"
    elif (file_path.suffix == ".csv") | (file_path.suffix == ".txt"):
        try:
            data = pd.read_csv(file_path, delimiter=delimiter)
            return data
        except FileNotFoundError:
            return "File not found"
    elif file_path.suffix == ".xls":
        try:
            data = pd.read_excel(file_path, engine=engine, sheet_name=sheet_name)
            return data
        except FileNotFoundError:
            return "File not found"
    else:
        return "load_data function cannot handle this file format at the moment"


def save_data(df, folder, filename, filetype):
    """
    Given a dataframe, saves it to a specified output path.
    """

    file_path = Path(data_dir, folder, filename)

    if filetype == "parquet":
        df.to_parquet(file_path)
    elif filetype == "csv":
        df.to_csv(file_path)
    elif filetype == "txt":
        df.to_csv(file_path, sep="|")
    else:
        print("File type not supported or path not found")


# save the models in
def save_model(model, model_name, folder):
    """
    Given a model, saves it to a specified output path.
    """

    file_path = Path(data_dir, folder, model_name)
    file_path_no_mod = Path(data_dir, folder)

    if not os.path.exists(file_path_no_mod):
        os.makedirs(file_path_no_mod)

    joblib.dump(model, open(file_path, "wb"))

    return model_name + " saved to " + folder


def load_model(model_name, folder):
    """
    Loads a model from a specified path.

    """
    file_path = Path(data_dir, folder, model_name)

    model = joblib.load(open(file_path, "rb"))

    return model
