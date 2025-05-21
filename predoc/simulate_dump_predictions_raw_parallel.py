#!/usr/bin/env python
import os
import sys

from predoc.clean_raw_wrapper import clean_data
from predoc.datasets import data_dir
from predoc.dummies_model_funcs import get_predics, get_predics_df
from predoc.generate_dummies import generate_dummies
from predoc.utils import load_model, save_data

try:
    import argparse
    import warnings

    import pandas as pd
    from joblib import Parallel, delayed

    pd.options.mode.chained_assignment = None
except ImportError as e:
    print("This program requires Python 3.1+, numpy and pandas", file=sys.stderr)
    raise ImportError(e)


"""
Simualates BPS data dumps on the 28th of each month in specified years.
Accepts 23 parameters and saves output to given output folder.

"""

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message=".*does not have valid feature names.*", category=UserWarning
    )

    parser = argparse.ArgumentParser(
        description="Predicts monthly ovarian cancer probability"
    )

    parser.add_argument("--horizon", type=int, help="Time horizon")
    parser.add_argument("--history", type=int, help="Time history")

    parser.add_argument("--data_path", type=str, help="Data path within data directory")
    parser.add_argument("--model_name", type=str, help="Model path within data folder")
    parser.add_argument(
        "--filter_analy",
        type=str,
        const="",
        default="",
        nargs="?",
        help="Take only patients with analytics or not",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        const="",
        default="",
        nargs="?",
        help="Option to generate prediction dataframe with metadata for evaluation",
    )
    parser.add_argument(
        "--full_data_path",
        type=str,
        const="",
        default="",
        nargs="?",
        help="If creating metadata dataframe, path of full data to associate global_ovarian_cancer_truth etc.",
    )
    parser.add_argument(
        "--output_path", type=str, help="Path within data directory to output data"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path within data directory with the model data"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        const="",
        default="",
        nargs="?",
        help="Suffix to add to output files",
    )
    parser.add_argument(
        "--year_min", type=int, default=2017, help="start year to create dump"
    )
    parser.add_argument("--year_max", type=int, default=2022, help="Last year for dump")

    parser.add_argument(
        "--month_min",
        type=int,
        default=1,
        help="start month to create dump (every year it gonna have this start)",
    )
    parser.add_argument(
        "--month_max",
        type=int,
        default=12,
        help="Last month to create dump(every year it gonna have this end)",
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

    model_name = args.model_name
    model_30 = load_model(model_name, args.model_path)
    os.makedirs(f"{data_dir}/{args.output_path}", exist_ok=True)

    # ------Get raw data and generate dataset------

    pats, bps, diag, espe, analy = clean_data(args.data_path, cie="yes", train="")

    analy = analy[analy["value_source_value"] >= 0]

    dates = [
        f"{i}-0{j}-28" if j < 10 else f"{i}-{j}-28"
        for i in range(args.year_min, args.year_max + 1)
        for j in range(args.month_min, args.month_max + 1)
    ]

    def predict_time_series(
        date,
        col_to_add_value,
        pats=pats,
        bps=bps,
        diag=diag,
        espe=espe,
        analy=analy,
        history=args.history,
        diag_categorical=True,
        espe_categorical=False,
        col_to_add="dump_date",
        chron_age=False,
    ):
        dumis_system = generate_dummies(
            pats,
            bps,
            diag,
            espe,
            analy,
            history=history,
            horizon=0,
            diag_categorical=diag_categorical,
            espe_categorical=espe_categorical,
            col_to_add=col_to_add,
            col_to_add_value=pd.to_datetime(date),
            chron_age=chron_age,
        )

        dumis_system.columns = [
            col.lower().replace(" ", "_").replace("__", "_")
            for col in dumis_system.columns
        ]

        model_features = model_30.feature_names_in_
        missing_cols = [
            col for col in model_features if col not in dumis_system.columns
        ]

        dumis_system = dumis_system.reindex(
            columns=dumis_system.columns.tolist() + missing_cols
        )

        dumis_system = dumis_system[model_30.feature_names_in_]

        dumis_system["cohort"] = -1

        dumis_system = dumis_system.reindex(sorted(dumis_system.columns), axis=1)

        X, y = get_predics(dumis_system, "cohort")

        predics = model_30.predict_proba(X)[:, -1]

        predics_df = get_predics_df(
            X,
            y,
            predics,
            col_to_add_name="Dump date",
            col_to_add_value=pd.to_datetime(date),
        )

        predics_df.index.name = "patient_id"
        predics_df = predics_df.reset_index()

        predics_df.columns = [
            col.lower().replace(" ", "_") for col in predics_df.columns
        ]

        predics_df = predics_df.set_index("patient_id")

        # get prediction coefficients
        columns = model_30.named_steps["estimator"].feature_names_in_

        try:
            coef = model_30.named_steps["estimator"].predict_and_contrib(
                model_30.named_steps["preprocessor"].transform(X)
            )[-1]
            df = pd.DataFrame(coef, columns=columns)
            df.index = X.index
            df["proba"] = model_30.predict_proba(X)[:, -1]
            df["dump_date"] = date
            predics_df = df
        except AttributeError:
            pass

        pats_meta = pd.read_parquet(f"{data_dir}/{args.full_data_path}/pats.parquet")
        pats_meta = pats_meta[["person_id", "main_condition_start_date"]].set_index(
            "person_id"
        )
        pats_meta = pats_meta[~pats_meta.index.duplicated()]

        pats_meta.to_parquet(f"{data_dir}/{args.output_path}/pats_meta.parquet")

        predics_df.loc[
            pats_meta[
                ~pats_meta["main_condition_start_date"].isna()
            ].index.intersection(predics_df.index.unique()),
            "global_ovarian_cancer_truth",
        ] = 1
        predics_df["global_ovarian_cancer_truth"] = predics_df[
            "global_ovarian_cancer_truth"
        ].fillna(-1)
        predics_df.loc[
            pats_meta[
                ~pats_meta["main_condition_start_date"].isna()
            ].index.intersection(predics_df.index.unique()),
            "ovarian_cancer_truth_date",
        ] = pats_meta["main_condition_start_date"]

        predics_df = predics_df[
            (predics_df["ovarian_cancer_truth_date"] >= predics_df["dump_date"])
            | predics_df["ovarian_cancer_truth_date"].isna()
        ]

        try:
            coefs_df = predics_df[[i for i in predics_df.columns if "tranf" in i]]

            
            save_data(
            coefs_df,
            args.output_path,
            "coefs_"
            + str(args.horizon)
            + "horizon_"
            + str(args.history)
            + "history_"
            + date
            + "_meta"
            + args.suffix
            + ".parquet",
            "parquet")
            

            predics_df = predics_df.drop(coefs_df.columns, axis=1)

            save_data(
                predics_df,
                args.output_path,
                "predics_"
                + str(args.horizon)
                + "horizon_"
                + str(args.history)
                + "history_"
                + date
                + "_meta"
                + args.suffix
                + ".parquet",
                "parquet",
            )

        except NameError:
            save_data(
                predics_df,
                args.output_path,
                "predics_"
                + str(args.horizon)
                + "horizon_"
                + str(args.history)
                + "history_"
                + date
                + "_meta"
                + args.suffix
                + ".parquet",
                "parquet",
            )

        print("done")

        return None

    with Parallel(n_jobs=args.n_jobs) as parallel:
        results = parallel(
            delayed(predict_time_series)(date=i, col_to_add_value=pd.to_datetime(i))
            for i in dates
        )
