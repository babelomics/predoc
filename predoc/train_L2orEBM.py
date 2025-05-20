#!/usr/bin/env python
# coding: utf-8

import sys

from predoc.clean_raw_wrapper import clean_data
from predoc.datasets import data_dir
from predoc.dictionaries_columns import (
    analy_columns_no_std_selected,
    bps_group_columns,
    cie_nogroup,
    special_columns,
    static_variables,
)
from predoc.generate_dummies import generate_dummies
from predoc.script_models import train_split
from predoc.utils import save_data, save_model

try:
    import argparse
    import os

    import numpy as np
    import pandas as pd
    from sklearn import set_config
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

    pd.options.mode.chained_assignment = None

    from interpret.glassbox import ExplainableBoostingClassifier

except ImportError as e:
    print(
        "This program requires Python 3.1+, numpy, pandas, scikit-learn and InterpretML",
        file=sys.stderr,
    )
    raise ImportError(e)


"""
Trains either an L2 logistic regression or EBM with 0 interactions from raw bps data. Outputs three subdatasets corresponding to training, validation and test sets. 

Accepts 33 parameters and saves to a desired output path.

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a specified classifier (EBM, ENet, GBC or RBF) with a given time history and horizon"
    )
    parser.add_argument("--horizon", type=int, help="Time horizon")

    parser.add_argument("--history", type=int, help="Time history")

    parser.add_argument("--training_year_min", type=int, help="Min training date year")

    parser.add_argument("--training_year_max", type=int, help="Max training date year")

    parser.add_argument("--year_test", type=int, help="Test year")

    parser.add_argument(
        "--validation_month_min", type=int, help="Min validation date month"
    )
    parser.add_argument(
        "--validation_month_max", type=int, help="Max validation date month"
    )

    parser.add_argument(
        "--control_validation_prop",
        type=float,
        help="Proportion of control to include in validation set",
    )

    parser.add_argument(
        "--control_test_prop",
        type=float,
        help="Proportion of control to include in test set",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Either 'ENet' for L2 regularized logistic regression, 'EBM' for explainable boosting classifier, 'GBC' for gradient boosting classifier, 'BRF' for balanced random forest 'dum' for dummy classifier",
    )

    parser.add_argument(
        "--data_path", type=str, help="Path to data within data directory"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path within data directory to output data and model",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        const="",
        default="",
        nargs="?",
        help="Suffix to add to output files",
    )

    parser.add_argument("--seed", type=int, const=3, default=3, nargs="?", help="Seed")

    parser.add_argument(
        "--min_age",
        type=int,
        const=35,
        default=35,
        nargs="?",
        help="Minimum age for training",
    )
    parser.add_argument(
        "--max_age",
        type=int,
        const=200,
        default=200,
        nargs="?",
        help="Maximum age for training",
    )

    parser.add_argument(
        "--interactions",
        type=int,
        const=0,
        default=0,
        nargs="?",
        help="Number of interactions for EBM",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        const=92,
        default=92,
        nargs="?",
        help="Number of CPU cores to use in parallelizable tasks",
    )

    args = parser.parse_args()

    full_out_path = os.path.join(data_dir, args.output_path)

    if not os.path.exists(full_out_path):
        os.makedirs(full_out_path)

    # ------Get raw data and generate dataset------

    pats, bps, diag, espe, analy = clean_data(
        args.data_path,
        seed=args.seed,
        cie="yes",
        min_age=args.min_age,
        max_age=args.max_age,
        ref_year=0,
        train_year_min=args.training_year_min,
        train_year_max=args.training_year_max,
    )

    analy = analy[analy["value_source_value"] >= 0]

    dumis_system = generate_dummies(
        pats,
        bps,
        diag,
        espe,
        analy,
        history=args.history,
        horizon=args.horizon,
        min_age=args.min_age,
        max_age=args.max_age,
        diag_categorical=True,
        espe_categorical=False,
        chron_age=False,
    )

    dumis_system.columns = [
        col.lower().replace(" ", "_").replace("__", "_") for col in dumis_system.columns
    ]

    analy_columns_no_std_selected = [
        col.lower().replace(" ", "_") for col in analy_columns_no_std_selected
    ]

    bps_group_columns = [col.lower().replace(" ", "_") for col in bps_group_columns]

    special_columns = [col.lower().replace(" ", "_") for col in special_columns]
    diag_col = dumis_system.columns.intersection(cie_nogroup)
    chron_col = dumis_system.columns.intersection(bps_group_columns)

    ##analytics without standard deviation
    cols_analytics = dumis_system.columns.intersection(analy_columns_no_std_selected)
    espe_col = dumis_system.columns.intersection(special_columns)

    # Save files for control
    dumis_system = dumis_system.reindex(sorted(dumis_system.columns), axis=1)

    # ------Split dataset and fit model------
    print("Splitting dataset")
    X_train, X_val, y_train, y_val, X_test, y_test = train_split(
        dumis_system,
        column_date="main_condition_start_date",
        answer="cohort",
        year_test=args.year_test,
        year_min_train=args.training_year_min,
        year_max_train=args.training_year_max,
        month_min_val=args.validation_month_min,
        month_max_val=args.validation_month_max,
        val_prop=args.control_validation_prop,
        test_prop=args.control_test_prop,
        random_state=args.seed,
    )

    X_train = X_train.replace({0: np.nan})
    X_train = X_train.dropna(axis=1, how="all")

    chron_col = X_train.columns.intersection(chron_col)
    diag_col = X_train.columns.intersection(diag_col)

    save_data(
        X_train,
        args.output_path,
        "control_train_" + args.suffix + ".parquet",
        "parquet",
    )

    save_data(
        X_val, args.output_path, "control_val_" + args.suffix + ".parquet", "parquet"
    )

    save_data(
        X_test, args.output_path, "control_test_" + args.suffix + ".parquet", "parquet"
    )

    # threshold = args.feature_presence # Removed argument?

    set_config(transform_output="pandas")

    analy_transformer = Pipeline(
        steps=[
            ("log_transform", FunctionTransformer(np.log1p)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[("num_imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )

    continuous_transformer = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("escaler", MinMaxScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("analy_tranf", analy_transformer, cols_analytics),
            ("chron_tranf", categorical_transformer, chron_col),
            ("diag_tranf", categorical_transformer, diag_col),
            ("espe_tranf", continuous_transformer, espe_col),
            ("statics_tranf", continuous_transformer, static_variables),
        ],
        remainder="drop",
    )

    if args.model == "ENet":
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "estimator",
                    LogisticRegressionCV(
                        max_iter=1000,
                        n_jobs=args.n_jobs,
                        scoring="average_precision",
                        class_weight="balanced",
                        random_state=args.seed,
                        l1_ratios=np.linspace(0.1, 0.99, 10),
                        cv=10,
                        solver="saga",
                        penalty="elasticnet",
                    ),
                ),
            ],
        )

    elif args.model == "EBM":
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "estimator",
                    ExplainableBoostingClassifier(
                        interactions=args.interactions, random_state=args.seed
                    ),
                ),
            ],
        )

    elif args.model == "GBC":
        param_grid = {
            "estimator__learning_rate": [1, 0.5, 0.25, 0.1, 0.05, 0.01],
            "estimator__n_estimators": [1, 2, 4, 8, 16, 32, 64, 100, 200],
            "estimator__max_depth": [i for i in range(1, 32)],
        }

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "estimator",
                    GradientBoostingClassifier(),
                ),
            ],
        )

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        grid_search = GridSearchCV(
            pipeline, param_grid, n_jobs=args.n_jobs, cv=kfold, verbose=0
        )
        print("Running GBC cross-validation")
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_

    else:
        raise Exception("Selected model not currently supported")

    print("Fitting model")

    model.fit(X_train, y_train.values)

    name_m = (
        args.model
        + "_"
        + str(args.history)
        + "history-"
        + str(args.horizon)
        + "horizon_"
        + args.suffix
        + ".sav"
    )

    save_model(model, name_m, args.output_path)

    print(args.model, "saved")
