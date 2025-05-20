from __future__ import division

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score as gmean
from imblearn.metrics import make_index_balanced_accuracy as iba
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from statsmodels.stats.multitest import multipletests

from predoc.utils import load_data


def get_predictions(
    year_min,
    year_max,
    horizon,
    history,
    data_path,
    suffix="",
    meta="no_",
    prefix="predics",
):
    """This function retrieves predictions from an estimator pertaining to given dates and
    concatenates them into a single dataframe.
    """
    dates = [
        f"{i}-0{j}-28" if j < 10 else f"{i}-{j}-28"
        for i in range(year_min, year_max + 1)
        for j in range(1, 13)
    ]

    with Parallel(n_jobs=-1) as parallel:
        results = parallel(
            delayed(load_data)(
                filename=f"{prefix}_{horizon}horizon_{history}history_{i}_{meta}meta{suffix}.parquet",
                folder=data_path,
            )
            for i in dates
        )

    df = pd.concat(results)

    return df


def get_days(df):
    """Function to compute prediction earliness in days"""

    df = df.sort_values("date_limit")
    df = df[~df.index.duplicated(keep="first")]
    med_days = (df["ovarian_cancer_truth_date"] - df["date_limit"]).dt.days.median()
    q25_days = np.quantile(
        (df["ovarian_cancer_truth_date"] - df["date_limit"]).dt.days, q=0.25
    )
    q75_days = np.quantile(
        (df["ovarian_cancer_truth_date"] - df["date_limit"]).dt.days, q=0.75
    )

    return med_days, q25_days, q75_days


def get_y(df):
    """Function to get the true label, predicted label and predicted probability"""

    pos = df[df["prediction"] == 1].sort_values("date_limit")
    pos = pos[~pos.index.duplicated(keep="first")]

    neg = df.loc[pos.index.symmetric_difference(df.index)].sort_values("date_limit")
    neg = neg[~neg.index.duplicated(keep="last")]

    pos_neg = pd.concat([pos, neg])

    y_true = pos_neg["global_ovarian_cancer_truth"]
    y_pred = pos_neg["prediction"]
    y_prob = pos_neg["ovarian_cancer_probability"]

    return y_true, y_pred, y_prob, pos_neg


def get_metrics(y_true, y_pred, y_prob):
    """Function to compute various metrics and return them as a dataframe"""

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)

    tnr = tn / (tn + fp)
    npv = tn / (tn + fn)
    fdr = fp / (tp + fp)

    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)

    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
    precision_macro = precision_score(y_true=y_true, y_pred=y_pred, average="macro")

    mcc = matthews_corrcoef(y_true, y_pred)

    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")

    br_score = brier_score_loss(y_true=y_true, y_prob=y_prob)

    geometric_mean = gmean(y_true, y_pred, average="binary")

    balanced_gmean = iba(alpha=0.1, squared=True)(gmean)
    iba_gmean = balanced_gmean(y_true, y_pred, average="binary")

    specificity = 1 - fpr

    log_loss_value = log_loss(y_true, y_prob)

    metrics_df = pd.DataFrame(
        {
            "binary_recall": recall,
            "false_pos_rate": fpr,
            "binary_precision": precision,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "brier_score": br_score,
            "false_negative_rate": fnr,
            "true_negative_rate": tnr,
            "negative_predictive_value": npv,
            "false_discovery_rate": fdr,
            "matthews_corr_coef": mcc,
            "geometric_mean": geometric_mean,
            "balanced_geometric_mean": iba_gmean,
            "log_loss": log_loss_value,
            "specificity": specificity,
        },
        index=[0],
    )
    return metrics_df


def set_threshold(df, recall):
    """Function to set a threshold which returns a desired recall"""

    df = df[df["Recall"] >= recall - 0.004999]
    thr = df.sort_values("Threshold").iloc[-1]["Threshold"]
    print(thr)
    return thr


def calc_area(df, x_, y_):
    """Function to calculate the area undera curve using the trapezoidal rule"""

    y = np.array(df[y_])
    x = np.array(df[x_])

    # Sort the data by ascending order of x axis
    sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
    sorted_y = [y[i] for i in sorted_indices]
    sorted_x = [x[i] for i in sorted_indices]

    # Calculate area under the curve using the trapezoidal rule
    area = 0
    for i in range(1, len(sorted_x)):
        height = (sorted_y[i] + sorted_y[i - 1]) / 2
        width = sorted_x[i] - sorted_x[i - 1]
        area += height * width

    return area


def calc_av_precision(df):
    """Function to calculate the average precision as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight"""

    AP = []
    thre = df.sort_values("Recall").reset_index()
    for i in thre.index:
        score = (
            thre.iloc[int(i) + 1]["Recall"] - thre.iloc[int(i)]["Recall"]
        ) * thre.iloc[int(i) + 1]["Precision"]
        AP.append(score)
        if i == len(thre) - 2:
            break
    return sum(AP)


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


def fdr(pvalues):
    """
    Function to apply Benjamini-Hochberg FDR p-value correction for multiple hypothesis testing.

    Args:
    pvalues (array-like): array of p-values to correct.

    Returns:
    array: corrected p-values.
    """
    return multipletests(pvalues, alpha=0.05, method="fdr_bh")[1]


def t_stat_various(df, metric, train, test):
    """Compute corrected t-test on various groups"""

    D = {}
    for i in df.dropna(subset="Threshold").groupby("History"):
        D[i[0]] = i[1][metric].values

    D_diff = {}
    for key1, i in D.items():
        for key2, j in D.items():
            if (key1 == key2) or (f"{key2}-{key1}" in D_diff.keys()):
                pass
            else:
                diff = i - j
                D_diff[f"{key1}-{key2}"] = diff

    df = pd.DataFrame(D_diff)

    t_stud = df.apply(
        lambda x: compute_corrected_ttest(
            x, len(df) - 1, train.index.nunique(), test.index.nunique()
        )
    )

    t_stud = t_stud.T.rename(columns={0: "t-student", 1: "p-val"})

    t_stud["p-val_corrected"] = fdr(t_stud["p-val"])

    t_stud.loc[
        t_stud[t_stud["p-val_corrected"] > 1].index.unique(), "p-val_corrected"
    ] = 1

    t_stud.reset_index(inplace=True)

    return t_stud
