#!usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from predoc.model_performance import get_days, get_metrics, get_predictions, get_y
from predoc.utils import load_data, save_data

pd.options.mode.chained_assignment = None


parser = argparse.ArgumentParser(
    description="Parse through a file containing directory names with probability trajectory files in order to generate metrics along varying thresholds"
)


parser.add_argument(
    "--directory", type=str, help="Path to directory with monthly predictions"
)
parser.add_argument(
    "--set", type=str, help="What dataset partitions to get metrics from"
)
parser.add_argument(
    "--all_controls",
    action="store_true",
    required=False,
    help="Get metrics with all controls or only those pertaining to the dataset partition",
)
parser.add_argument(
    "--min_age",
    type=int,
    const=50,
    default=50,
    nargs="?",
    help="Minimum age to evaluate",
)
parser.add_argument(
    "--max_age",
    type=int,
    const=200,
    default=100,
    nargs="?",
    help="Maximum age to evaluate",
)
parser.add_argument("--horizon", type=int, help="Time horizon")
parser.add_argument("--history", type=int, help="Time history")
parser.add_argument("--year_min", type=int, help="Min date")
parser.add_argument("--year_max", type=int, help="Max date")
parser.add_argument(
    "--suffix_out",
    type=str,
    const="",
    default="",
    nargs="?",
    help="Suffix to add to output file",
)
parser.add_argument(
    "--suffix_in",
    type=str,
    const="",
    default="",
    nargs="?",
    help="Aggregated probabilities file suffix",
)
parser.add_argument(
    "--prefix",
    type=str,
    const="predics",
    default="predics",
    nargs="?",
    help="Prefix of monthly prediction files",
)
parser.add_argument("--threshold", type=float, help="Decision threshold")
parser.add_argument(
    "--cases_year",
    type=int,
    const=2022,
    default=2022,
    nargs="?",
    help="Year on which cases were diagnosed",
)
parser.add_argument(
    "--data_path",
    type=str,
    const="raw/20231214",
    default="raw/20231214",
    nargs="?",
    help="Path to data within data directory",
)


args = parser.parse_args()


df_ = get_predictions(
    year_min=args.year_min,
    year_max=args.year_max,
    horizon=args.horizon,
    history=args.history,
    data_path=args.directory,
    meta="",
    prefix=args.prefix,
    suffix=args.suffix_in,
)

df_.index.name = "patient_id"
df_ = df_.rename(
    columns={"dump_date": "date_limit", "proba": "ovarian_cancer_probability"}
)
df_.date_limit = pd.to_datetime(df_.date_limit)
df_.loc[
    df_[df_["global_ovarian_cancer_truth"] == -1].index.unique(),
    "global_ovarian_cancer_truth",
] = 0

pats = load_data("01_Datos_Sociodemograficos.txt", args.data_path).set_index(
    "NUHSA_ENCRIPTADO"
)
pats["COD_FEC_NACIMIENTO"] = pd.to_datetime(pats["COD_FEC_NACIMIENTO"], format="%Y%m%d")


df_ = df_.merge(
    pats[["COD_FEC_NACIMIENTO"]], right_index=True, left_index=True, how="inner"
)
df_["age"] = ((df_["date_limit"] - df_["COD_FEC_NACIMIENTO"]).dt.days / 365.25).astype(
    int
)

df_ = df_.loc[
    df_[(df_["age"] >= args.min_age) & (df_["age"] <= args.max_age)].index.unique(),
]


if args.set == "Validation":
    df_set = load_data("control_val_.parquet", args.directory)
elif args.set == "Test 2022":
    df_set = load_data("control_test_.parquet", args.directory)

df_ = df_.loc[df_set.index.unique().intersection(df_.index.unique())]

if args.all_controls:
    cont = df_[df_["global_ovarian_cancer_truth"] == -1].drop(df_set.index.unique())
    df_ = pd.concat([cont, df_])

df_.index.name = "patient_id"


threshold = args.threshold

df_ = df_.dropna(subset=["ovarian_cancer_probability"])


df_["prediction"] = np.nan
df_.reset_index(inplace=True)
df_.loc[
    df_[
        df_["ovarian_cancer_probability"] >= threshold - 0.000000000000000001
    ].index.unique(),
    "prediction",
] = 1
df_["prediction"] = df_["prediction"].fillna(0)
df_ = df_.set_index("patient_id")


med_days, q25_days, q75_days = get_days(
    df_[
        (df_["prediction"] == 1)
        & (df_["ovarian_cancer_truth_date"].dt.year == args.cases_year)
    ]
)

y_true, y_pred, y_prob, pos_neg = get_y(df_)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

metrics = get_metrics(y_true, y_pred, y_prob)

metrics["Median days / 365"] = med_days / 365

metrics = metrics.melt(var_name="Metric").replace(
    {
        "binary_recall": "True Positive Rate",
        "false_pos_rate": "False Positive Rate",
        "binary_precision": "Positive Predictive Value",
        "f1_macro": r"$F1_{Macro}$",
        "brier_score": "Brier Score",
        "median_days/365": "Median days / 365",
        "matthews_corr_coef": "Matthew's correlation coefficient",
        "specificity": "Specificity",
    }
)

save_data(
    metrics, args.directory, f"metrics_aggregated{args.suffix_out}.parquet", "parquet"
)
print(f"Metrics saved to {args.directory}")
