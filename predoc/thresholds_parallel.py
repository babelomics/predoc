#!/usr/bin/env python
# coding: utf-8

# In[171]:

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from predoc.datasets import data_dir
from predoc.model_performance import get_days, get_predictions, set_threshold
from predoc.prg_home import precision_gain, recall_gain
from predoc.utils import load_data, save_data

# In[172]:


parser = argparse.ArgumentParser(
    description="Parse through a file containing directory names with probability trajectory files in order to generate metrics along varying thresholds"
)


parser.add_argument(
    "--directory", type=str, help="Path to directory with monthly predictions"
)

parser.add_argument(
    "--control_path", type=str, help="Path to directory with control_val_ file"
)

parser.add_argument("--output_path", type=str, help="Path to directory to save to")

parser.add_argument(
    "--set", type=str, help="What dataset partitions to get metrics from"
)
parser.add_argument(
    "--all_controls",
    action="store_true",
    required=False,
    help="Get metrics with all controls or only those pertaining to the dataset partition",
)
parser.add_argument("--horizon", type=int, help="Time horizon")
parser.add_argument("--history", type=int, help="Time history")
parser.add_argument("--year_min", type=int, help="Min date")
parser.add_argument("--year_max", type=int, help="Max date")
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
parser.add_argument(
    "--prefix",
    type=str,
    const="predics",
    default="predics",
    nargs="?",
    help="Prefix of monthly prediction files",
)
parser.add_argument(
    "--suffix_in",
    type=str,
    const="",
    default="",
    nargs="?",
    help="Suffix of monthly prediction files",
)
parser.add_argument(
    "--suffix_out",
    type=str,
    const="",
    default="",
    nargs="?",
    help="Suffix for output file",
)
parser.add_argument(
    "--desired_recall",
    type=float,
    const=0.0,
    default=0.0,
    nargs="?",
    help="Write to file the maximum threshold which obtains a recall equal or greater than the passed argument",
)
parser.add_argument(
    "--n_jobs",
    type=int,
    const=92,
    default=92,
    nargs="?",
    help="Number of CPU cores to use in parallelizable tasks",
)
parser.add_argument(
    "--data_path",
    type=str,
    const="raw/20231214",
    default="raw/20231214",
    nargs="?",
    help="Path to data within data directory",
)
parser.add_argument(
    "--resolution",
    type=int,
    const=2,
    default=2,
    nargs="?",
    help="Number of decimals for intervals between thresholds. I.e. --resolution=2 results in thresholds 0,01, 0,02, 0.03...",
)
parser.add_argument(
    "--dum",
    action="store_true",
    default=False,
    help="Get metrics accross threholds of randomly assigned probabilities",
)

args = parser.parse_args()
os.makedirs(f"{data_dir}/{args.output_path}", exist_ok=True)

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

pats = pd.read_parquet(f"{data_dir}/{args.data_path}/pats.parquet")
pats = pats[["person_id", "birth_datetime"]].set_index("person_id")
pats = pats[~pats.index.duplicated()]

df_ = df_.merge(
    pats[["birth_datetime"]], right_index=True, left_index=True, how="inner"
)
df_["age"] = ((df_["date_limit"] - df_["birth_datetime"]).dt.days / 365.25).astype(int)

df_ = df_.loc[
    df_[(df_["age"] >= args.min_age) & (df_["age"] <= args.max_age)].index.unique(),
]

if args.set == "Validation":
    df_set = load_data("control_val_.parquet", args.control_path)
elif args.set == "Test 2022":
    df_set = load_data("control_test_.parquet", args.control_path)
else:
    print("Indicate 'Validation' or 'Test 2022'")

df_ = df_.loc[df_set.index.unique().intersection(df_.index.unique())]

if args.all_controls:
    cont = df_[df_["global_ovarian_cancer_truth"] == -1].drop(df_set.index.unique())
    df_ = pd.concat([cont, df_])

df_.index.name = "patient_id"

if args.dum:
    df_["ovarian_cancer_probability"] = np.random.rand(len(df_))
    df_["ovarian_cancer_probability"] = df_["ovarian_cancer_probability"].round(2)


def calculate_precision_recall(df, threshold):
    df["prediction"] = np.nan
    df.reset_index(inplace=True)
    df.loc[
        df[df["ovarian_cancer_probability"] >= threshold - 0.0000000001].index.unique(),
        "prediction",
    ] = 1
    df = df.set_index("patient_id")
    df["prediction"] = df["prediction"].fillna(0)

    pos = df[df["prediction"] == 1].sort_values("date_limit")
    pos = pos[~pos.index.duplicated(keep="first")]

    neg = df.loc[pos.index.symmetric_difference(df.index)].sort_values("date_limit")
    neg = neg[~neg.index.duplicated(keep="last")]

    pos_neg = pd.concat([pos, neg])

    y_true = pos_neg["global_ovarian_cancer_truth"]
    y_pred = pos_neg["prediction"]
    pos_neg["ovarian_cancer_probability"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    one_min_fpr = 1 - (fp / (fp + tn))
    fpr = fp / (fp + tn)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    prec_gain = precision_gain(tp=tp, tn=tn, fn=fn, fp=fp)
    rec_gain = recall_gain(tp=tp, tn=tn, fn=fn, fp=fp)

    try:
        med_days, q25_days, q75_days = get_days(
            df[(df["global_ovarian_cancer_truth"] == 1) & (df["prediction"] == 1)]
        )

    except IndexError:
        med_days = np.nan

    return precision, recall, one_min_fpr, fpr, prec_gain, rec_gain, med_days, threshold


min_ = df_.ovarian_cancer_probability.min()
max_ = df_.ovarian_cancer_probability.max()


def warn(*args, **kwargs):
    pass


# Suppress the warning
warnings.warn = warn

thr = [
    i / int("1" + args.resolution * "0")
    for i in range(0, int("1" + args.resolution * "0") + 1)
    if ((i / int("1" + args.resolution * "0")) + 0.00000000000001 >= min_)
    and ((i / int("1" + args.resolution * "0")) - 0.000000000000001 <= max_)
]

# Parallelize the iterations using joblib
with Parallel(n_jobs=args.n_jobs) as parallel:
    results = parallel(delayed(calculate_precision_recall)(df_, i) for i in thr)

metrics_thresholds = set(results)


metrics_thresholds = pd.DataFrame(
    metrics_thresholds,
    columns=[
        "Precision",
        "Recall",
        "Sensitivity",
        "False Positive Rate",
        "Precision gain",
        "Recall gain",
        "Median days",
        "Threshold",
    ],
)


if args.dum:
    save_data(
        metrics_thresholds,
        args.output_path,
        f"metrics_thresholds{args.suffix_out}_dum.parquet",
        "parquet",
    )
else:
    save_data(
        metrics_thresholds,
        args.output_path,
        f"metrics_thresholds{args.suffix_out}.parquet",
        "parquet",
    )


if args.desired_recall > 0.0:
    threshold = set_threshold(metrics_thresholds, args.desired_recall)
    print(f"{data_dir}/{args.directory}/threshold.txt")

    with open(f"{data_dir}/{args.output_path}/threshold.txt", "w") as f:
        f.write(str(threshold))

    if args.dum:
        with open(f"{data_dir}/{args.output_path}/threshold_dum.txt", "w") as f:
            f.write(str(threshold))

    print(f"Threshold saved to {args.output_path}")
