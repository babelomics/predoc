import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from paretoset import paretoset
from scipy.stats import weightedtau

from predoc.clean_raw_functions import filtered_txt
from predoc.dictionaries_columns import translations_dict
from predoc.model_performance import calc_area, calc_av_precision, get_predictions
from predoc.utils import load_data, load_model, save_data
from predoc.datasets import data_dir


def age_limit(
    predictions_df,
    data_path=f"{data_dir}/omop/done/",
    demo_file="pats.parquet",
    threshold=0.15,
):
    ## load and clean file containing patient diagnosis and birth date. Merge with predictions dataframe
    pats = load_data(demo_file, data_path)
    pats = filtered_txt(
        pats,
        column_date=["birth_datetime", "main_condition_start_date"],
        nuhsa_column="person_id",
        selected_column=["birth_datetime", "main_condition_start_date"],
    )

    df = predictions_df.merge(pats, right_index=True, left_index=True).reset_index()

    ### remove predictions carried out for patients < 50 years
    df["dump_date"] = pd.to_datetime(df["dump_date"])
    df["age"] = ((df["dump_date"] - df["birth_datetime"]).dt.days / 365.25).astype(int)
    df = df[df["age"] >= 50]

    return df


def set_predictions(predictions_df, threshold=0.15):
    ## set prediction labels to 1 if specified threshold is surpassed
    predictions_df.loc[
        predictions_df[predictions_df["proba"] >= threshold].index.unique(), "pred"
    ] = 1
    predictions_df["pred"].fillna(-1, inplace=True)
    predictions_df.set_index("person_id", inplace=True)

    pos = predictions_df[(predictions_df["pred"] == 1)].sort_values("dump_date")
    pos = pos[~pos.index.duplicated(keep="first")]

    neg = predictions_df[(predictions_df["pred"] == -1)].sort_values("dump_date")
    neg = neg[~neg.index.duplicated(keep="last")]

    neg = neg.drop(pos.index.unique().intersection(neg.index.unique()))

    predictions_df = pd.concat([pos, neg])

    return predictions_df


# TODO: Adapt cols in this function to OMOP
def earliness_test(
    data_path,
    suffix_out="",
    year_min=2018,
    year_max=2022,
    horizon=30,
    history=180,
    seed=3,
):
    ## load dataframe with predictions
    df = get_predictions(
        year_min,
        year_max,
        horizon,
        history,
        data_path=f"{data_path}/omop/simulate",  # simulation_path
        suffix="",
        meta="",
        prefix="predics",
    )

    ## keep patients diagnosed in 2022 (test year)
    df = df[(df["ovarian_cancer_truth_date"].dt.year == 2022)]

    ## remove patients below age limit
    df = age_limit(df)

    ## set prediction labels
    df = set_predictions(df)

    ## set days to diagnosis
    df["days_to_diag"] = (df["main_condition_start_date"] - df["dump_date"]).dt.days

    ## keep first positive prediction and group predictions over 360 days together
    pos = df[df["pred"] == 1]
    pos = pos.sort_values("dump_date")
    pos = pos[~pos.index.duplicated(keep="first")]
    pos["grouped_days"] = np.where(pos["days_to_diag"] > 400, 400, pos["days_to_diag"])

    ## plot fixed size earliness distribution plot
    fac = 1.2
    fig, ax = plt.subplots(figsize=(8 / fac, 4.5 / fac))

    sns.histplot(pos, x="grouped_days", bins=10)
    sns.set_context("paper")
    sns.despine(top=True, right=True)
    ax.set_xticks([i for i in range(20, 400, 40)])

    ticks = list(ax.get_xticks()[0:-1])
    ticks = list(map(str, ticks))
    ticks.append(">360")
    ax.set_xticklabels(ticks)

    ax.set_xlabel("Early prediction in days", labelpad=10.0, fontsize=14)
    ax.set_ylabel("Count", labelpad=10.0, fontsize=14)

    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=12)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=12)

    ax.grid(visible=True, alpha=0.4)

    plt.tight_layout()

    pos.reset_index()[["days_to_diag"]].rename(
        columns={"days_to_diag": "Earliness in days"}
    ).to_excel("earliness_test.xlsx")

    plt.savefig(f"days_to_prediction{suffix_out}.png", dpi=600)
    plt.savefig(f"days_to_prediction{suffix_out}.pdf", dpi=600)

    return None


def curves_test(data_path, curves_file="metrics_thresholds_test_parallel.parquet"):
    df = load_data(curves_file, f"{data_path}/omop/validate")
    df.loc[len(df)] = {"Precision": 1, "Recall": 0}

    ## plot fixed size metrics curves
    fig, ax = plt.subplots(1, 3, figsize=(8, 2), gridspec_kw={"wspace": 0.5})

    sns.lineplot(data=df, x="False Positive Rate", y="Recall", linewidth=2, ax=ax[0])
    ax[0].plot([0, 1], [0, 1], color="red", linestyle="--")

    for i in range(len(ax)):
        ax[i].grid(visible=True, alpha=0.4)
        sns.despine(top=True, right=True, ax=ax[i])

    sns.lineplot(df, x="Recall", y="Precision", legend=None, linewidth=2, ax=ax[1])
    sns.lineplot(df, x="Recall", y="Median days", legend=None, linewidth=2, ax=ax[2])

    ax[1].axhline(0.02, color="red", linestyle="--")

    ax[0].text(
        -0.4,
        1,
        "A.",
        fontweight="bold",
        fontsize="medium",
        horizontalalignment="center",
    )
    ax[0].text(
        1.25,
        1,
        "B.",
        fontweight="bold",
        fontsize="medium",
        horizontalalignment="center",
    )
    ax[0].text(
        2.8, 1, "C.", fontweight="bold", fontsize="medium", horizontalalignment="center"
    )

    ax[0].set_ylabel("True Positive Rate")

    ax[1].set_ylim(0, 1)

    plt.tight_layout()

    df.dropna(subset="Threshold").to_excel("ebm_30_180_test.xlsx", index=False)

    plt.savefig("metrics_composed_figure.png", dpi=600, bbox_inches="tight")
    plt.savefig("metrics_composed_figure.pdf", dpi=600, bbox_inches="tight")

    return None


def get_mean_abs_score(data_path, model_name, top=10):
    model = load_model(model_name, f"{data_path}/omop/train")

    ## extract variable names and mean absolute scores
    vals = {}
    for i, j in zip(
        model["estimator"].explain_global().data()["names"],
        model["estimator"].explain_global().data()["scores"],
    ):
        vals[i] = j
    vals = pd.DataFrame(vals, index=[0])
    vals = pd.melt(vals, var_name="Feature", value_name="Weighted mean absolute score")

    ## change variable names to nice format
    vals = vals.replace(translations_dict)

    ## extract top 10 variables
    top_vals = vals.nlargest(top, "Weighted mean absolute score")

    top_vals.to_excel(f"top{top}_mean_scores.xlsx")

    return top_vals

# %%dd

data_path =f"{data_dir}/omop/simulate/"
single="5278413491"
single_file=True
year_min=2018
year_max=2022
horizon=30
history=180

def get_feature_trajectories(
    data_path,
    single,
    single_file=True,
    year_min=2018,
    year_max=2022,
    horizon=30,
    history=180,
):
    if single_file:
        coefs = load_data(
            f"feature_trajectories_{year_min}_{year_max}.parquet", f"{data_path}/omop/simulate"
        )
    else:
        coefs = get_predictions(
            year_min,
            year_max,
            horizon,
            history,
            data_path,
            suffix="",
            meta="",
            prefix="coefs",
        )
        save_data(
            coefs,
            f"{data_path}/omop/simulate",
            f"feature_trajectories_{year_min}_{year_max}.parquet",
            "parquet",
        )

    coefs["dump_date"] = pd.to_datetime(coefs["dump_date"])
    coefs["days_to_diag"] = (
        (coefs["ovarian_cancer_truth_date"] - coefs["dump_date"]).dt.days
    ) * -1
    coefs["days_to_diag"].fillna(
        (pd.to_datetime("20221228", format="%Y%m%d") - coefs["dump_date"]).dt.days * -1,
        inplace=True,
    )

    ## extract probability and selected feature trajectory of a single selected patient
    feature_trajectories = coefs.loc[[single]][
        [
            "analy_tranf__linfocitos_porcentaje_mean",
            "analy_tranf__colesterol_min",
            "diag_tranf__r10.13",
            "proba",
            "days_to_diag",
        ]
    ]
    feature_trajectories = pd.melt(
        feature_trajectories,
        id_vars=["proba", "days_to_diag"],
        value_vars=[
            "analy_tranf__linfocitos_porcentaje_mean",
            "analy_tranf__colesterol_min",
            "diag_tranf__r10.13",
        ],
    )

    feature_trajectories.reset_index().to_excel("probability_feature_trajectory.xlsx")

    return feature_trajectories


def interpretability(
    data_path, 
    data_path, 
    single,
    model_name="EBM_180history-30horizon_.sav", 
    top=10, 
    single_file=True,
):
    vals = get_mean_abs_score(data_path=f"{data_path}/omop/train", model_name=model_name)
    feature_trajectories = get_feature_trajectories(f"{data_path}/omop/simulate", single=single, single_file=single_file)

    ## plot fixed size interpretability plot
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={"hspace": 0.5})

    sns.barplot(
        vals,
        x="Weighted mean absolute score",
        y="Feature",
        color="gray",
        ax=ax[0],
        width=0.6,
    )
    sns.lineplot(
        feature_trajectories,
        y="proba",
        x="days_to_diag",
        ax=ax[1],
        legend=False,
        color="orange",
    )
    sns.scatterplot(
        feature_trajectories,
        y="proba",
        x="days_to_diag",
        color="orange",
        legend=False,
        ax=ax[1],
    )

    ax[0].set_yticklabels(labels=ax[0].get_yticklabels(), fontsize=12)
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), fontsize=12)

    ax[0].set_xlabel(xlabel=ax[0].get_xlabel(), fontsize=14)

    ax[0].set_ylabel("")
    ax2 = ax[1].twinx()

    hue_colors = ["purple", "blue", "green"]

    hue_order = [
        "analy_tranf__linfocitos_porcentaje_mean",
        "diag_tranf__r10.13",
        "analy_tranf__colesterol_min",
    ]
    sns.lineplot(
        feature_trajectories,
        y="value",
        markers=["o"],
        linestyle="dashdot",
        x="days_to_diag",
        ax=ax2,
        hue="variable",
        palette=hue_colors,
        hue_order=hue_order,
        legend=False,
    )

    ax2.set_ylabel("Feature score", rotation=270, labelpad=15)

    solid_line = Line2D([0], [0], linestyle="solid", color="orange")
    dashdot_line1 = Line2D([0], [0], linestyle="dashdot", color="purple")
    dashdot_line2 = Line2D([0], [0], linestyle="dashdot", color="blue")
    dashdot_line3 = Line2D([0], [0], linestyle="dashdot", color="green")

    ax[1].legend(
        handles=[solid_line, dashdot_line1, dashdot_line2, dashdot_line3],
        labels=[
            "Prediction probability",
            "Mean lymphocyte\npercentage score",
            "Epigastric pain score",
            "Minimum cholesterol score",
        ],
        loc="upper left",
        fontsize=12,
    )

    ax[1].set_xlabel("Days to end of evaluation", fontsize=14)
    ax[1].set_ylabel("Prediction probability", fontsize=14)

    ax[1].set_yticklabels(labels=ax[1].get_yticklabels(), fontsize=12)
    ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), fontsize=12)

    ax2.set_ylabel(ax2.get_ylabel(), fontsize=14)

    ax[1].axhline(0.15, color="red", linestyle=":")

    for i in range(len(ax)):
        ax[i].grid(visible=True, alpha=0.4)
        sns.despine(top=True, right=True, ax=ax[i])

    bbox = ax[1].get_position()
    ax[0].get_position()

    ax[1].set_position([bbox.x0 - 0.4, 0.05, bbox.width + 0.4, bbox.height + 0.08])
    sns.despine(top=True, right=True, ax=ax2)

    ax[0].text(
        -0.085,
        0,
        "A.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )
    ax[0].text(
        -0.085,
        13,
        "B.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )

    plt.savefig(
        "absolute_scores_and_feature_trajectories.png", dpi=600, bbox_inches="tight"
    )
    plt.savefig(
        "absolute_scores_and_feature_trajectories.pdf", dpi=600, bbox_inches="tight"
    )

    return None


def join_seeds(
    data_path,
    set_,
    horizon=30,
    history=180,
    year_min=2018,
    year_max=2022,
    out_path="sandbox/voliva",
    threshold=0.15,
):
    if set_ == "test":
        keep = load_data(
            "control_test_.parquet", f"{data_path}/ebm_{horizon}_{history}_seed1/simulate"
        ).index.unique()
    elif set_ == "val":
        keep = load_data(
            "control_val_.parquet", f"{data_path}/ebm_{horizon}_{history}_seed1/simulate"
        ).index.unique()

    D = {}
    for i in range(1, 101):
        print(f"{data_path}/ebm_{horizon}_{history}_seed{i}/simulate")
        df = get_predictions(
            year_min,
            year_max,
            horizon,
            history,
            f"{data_path}/ebm_{horizon}_{history}_seed{i}/simulate",
            suffix="",
            meta="",
            prefix="predics",
        )
        df = df.loc[keep.intersection(df.index.unique())]

        df = age_limit(df)
        df = set_predictions(df, threshold=threshold)
        D[i] = df

    seeds = pd.concat(D).reset_index()
    seeds = seeds.rename(columns={"level_0": "Seed"})

    save_data(
        seeds,
        data_path,
        f"all_probabilities_seeds_{horizon}horizon_{history}history_{set_}_threshold{threshold}.parquet",
        "parquet",
    )

    return seeds


def prob_diff(seeds):
    seeds = seeds.rename(columns={"proba": "ovarian_cancer_probability"})

    s1 = seeds[seeds["Seed"] == 1]
    seeds.loc[s1.index.unique(), "Probability at seed 1"] = s1[
        "ovarian_cancer_probability"
    ]
    seeds["Difference in probability"] = (
        seeds["Probability at seed 1"] - seeds["ovarian_cancer_probability"]
    )

    return seeds


def get_probabilities_seeds(
    data_path,
    min_year,
    max_year,
    n_jobs=92,
    seed_file=True,
    horizon=30,
    history=180,
    set_="test",
    threshold=0.15,
):
    if seed_file:
        seeds = load_data(
            f"all_probabilities_seeds_{horizon}horizon_{history}history_{set_}_threshold{threshold}.parquet",
            f"{data_path}/simulate",
        )
    else:
        seeds = join_seeds(data_path, set_=set_, threshold=threshold)

    seeds = seeds.set_index("index")

    return seeds


def get_metrics_seeds(
    data_path,
    metrics_file="metrics_aggregated_30horizon_180history_test.parquet",
    horizon=30,
    history=180,
):
    D = {}
    for i in range(1, 101):
        met = load_data(metrics_file, f"{data_path}/ebm_{horizon}_{history}_seed{i}/simulate")
        if isinstance(met, str):
            pass
        else:
            D[i] = met
    metrics = (
        pd.concat(D)
        .reset_index()
        .drop("level_1", axis=1)
        .rename(columns={"level_0": "Seed"})
    )

    return metrics


def get_curves_seeds(
    data_path="sandbox/voliva",
    thresholds_file="metrics_thresholds_test_parallel.parquet",
):
    D = {}
    for i in range(1, 101):
        cur = load_data(thresholds_file, f"{data_path}/ebm_30_180_seed{i}/validate")
        if isinstance(cur, str):
            pass
        else:
            D[i] = cur
    curves = (
        pd.concat(D)
        .reset_index()
        .drop("level_1", axis=1)
        .rename(columns={"level_0": "Seed"})
    )

    auroc = pd.DataFrame(
        curves.groupby("Seed").apply(
            lambda x: calc_area(x, x_="False Positive Rate", y_="Recall")
        )
    ).rename(columns={0: "AUROC"})
    auroc = auroc.reset_index().melt(id_vars="Seed", var_name="Metric")

    auprc = pd.DataFrame(
        curves.dropna(subset="Threshold")
        .groupby("Seed")
        .apply(lambda x: calc_av_precision(x))
    ).rename(columns={0: "AUPRC"})
    auprc = auprc.reset_index().melt(id_vars="Seed", var_name="Metric")

    return auroc, auprc


def get_scores_seeds(data_path):
    D_mods = {}
    for i in range(1, 101):
        mod = load_model(
            "EBM_180history-30horizon_.sav", f"{data_path}/ebm_30_180_seed{i}/train"
        )
        D_scores = {}
        for p, j in zip(
            mod["estimator"].explain_global().data()["names"],
            mod["estimator"].explain_global().data()["scores"],
        ):
            D_scores[p] = j
        df_scores = pd.DataFrame(D_scores, index=[0])
        D_mods[i] = df_scores

    vals = (
        pd.concat(D_mods)
        .reset_index()
        .drop("level_1", axis=1)
        .rename(columns={"level_0": "Seed"})
    )
    vals.set_index("Seed", inplace=True)
    vals = vals.T

    return vals


def get_corr_values(scores):
    correlation_values = {}
    for col1 in scores.columns:
        # tau_array = []
        for col2 in scores.columns:
            if col2 >= col1 + 1:
                # Step 2: Compute Kendall’s τ rank correlation coefficient
                tau, _ = weightedtau(scores[col1], scores[col2])
                # tau_array.append(tau)
                correlation_values[f"{col1}_{col2}"] = tau
    tau_df = pd.DataFrame(
        correlation_values, index=["Kendall's weighted \u03c4 coefficient"]
    ).T

    return tau_df


def save_stability(metrics, seeds, tau):
    seeds = seeds.reset_index()[
        [
            "Seed",
            "ovarian_cancer_probability",
            "global_ovarian_cancer_truth",
            "Probability at seed 1",
            "Difference in probability",
        ]
    ]
    seeds = seeds[seeds["Seed"] != 1]

    seeds_cases = seeds[seeds["global_ovarian_cancer_truth"] == 1].drop(
        "global_ovarian_cancer_truth", axis=1
    )
    seeds_controls = seeds[seeds["global_ovarian_cancer_truth"] == -1].drop(
        "global_ovarian_cancer_truth", axis=1
    )

    seeds_controls = (
        seeds_controls.groupby("Seed")
        .apply(lambda x: x.sample(n=10000))
        .reset_index(drop=True)
    )

    with pd.ExcelWriter("ebm_stability.xlsx") as writer:
        # Write each DataFrame to a separate sheet
        metrics.to_excel(writer, sheet_name="Metrics stability", index=False)
        seeds_cases.to_excel(
            writer, sheet_name="Probability difference cases", index=False
        )
        seeds_controls.dropna().to_excel(
            writer, sheet_name="Probability difference controls", index=False
        )
        tau.to_excel(writer, sheet_name="Kendall's weighted tau", index=True)

    return None


def stability(
    data_path,
    min_year=2018,
    max_year=2022,
    n_jobs=92,
    seed_file=True,
    horizon=30,
    history=180,
    metrics_file="metrics_aggregated_val.parquet",
    thresholds_file="metrics_thresholds_val_parallel.parquet",
    set_="test",
    threshold=0.15,
):
    probabilities = get_probabilities_seeds(
        f"{data_path}/simulate",
        min_year,
        max_year,
        n_jobs=92,
        seed_file=seed_file,
        horizon=30,
        history=180,
        set_=set_,
        threshold=threshold,
    )
    probabilities = prob_diff(probabilities)

    metrics = get_metrics_seeds(
        f"{data_path}/validate", metrics_file=metrics_file, horizon=horizon, history=history
    )
    metrics = metrics[
        metrics["Metric"].isin(
            [
                "True Positive Rate",
                "False Positive Rate",
                "Positive Predictive Value",
                "$F1_{Macro}$",
                "Median days / 365",
            ]
        )
    ]
    auroc, auprc = get_curves_seeds(data_path, thresholds_file=thresholds_file)
    metrics = pd.concat([metrics, auroc, auprc])

    scores = get_scores_seeds(data_path)
    scores_tau = get_corr_values(scores)
    ## create fixed size plot
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], wspace=0.25, hspace=0.5)

    # Subplot 1
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.set_axisbelow(True)

    sns.histplot(
        probabilities[
            (probabilities["Seed"] != 1)
            & (probabilities["global_ovarian_cancer_truth"] == 1)
        ],
        x="Difference in probability",
        ax=ax1,
        bins=50,
    )

    ax1.grid(visible=True, alpha=0.4)
    sns.despine(top=True, right=True, ax=ax1)

    ax1.set_ylabel(ylabel=ax1.get_ylabel(), fontsize=14)
    ax1.set_xlabel(xlabel="Difference in probability (cases)", fontsize=14)

    ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontsize=12)
    ax1.set_xticklabels(labels=ax1.get_xticklabels(), fontsize=12, rotation=20)

    ax1.tick_params(axis="x", pad=0)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-4, 4))

    ax1.yaxis.set_major_formatter(formatter)

    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Subplot 2
    ax2 = fig.add_subplot(gs[1, 0])

    ax2.set_axisbelow(True)

    sns.histplot(
        probabilities[
            (probabilities["Seed"] != 1)
            & (probabilities["global_ovarian_cancer_truth"] == -1)
        ],
        x="Difference in probability",
        ax=ax2,
        bins=50,
    )

    ax2.grid(visible=True, alpha=0.4)
    sns.despine(top=True, right=True, ax=ax2)

    ax2.set_ylabel(ylabel=ax2.get_ylabel(), fontsize=14)
    ax2.set_xlabel(xlabel="Difference in probability (controls)", fontsize=14)

    ax2.set_yticklabels(labels=ax2.get_yticklabels(), fontsize=12)
    ax2.set_xticklabels(labels=ax2.get_xticklabels(), fontsize=12, rotation=20)

    ax2.tick_params(axis="x", pad=0)

    ax2.yaxis.set_major_formatter(formatter)

    ax4 = fig.add_subplot(gs[2, 0])

    ax4.set_axisbelow(True)

    ax4.grid(visible=True, alpha=0.4, zorder=-1)

    sns.histplot(scores_tau, x="Kendall's weighted \u03c4 coefficient", ax=ax4)

    sns.despine(top=True, right=True, ax=ax4)

    ax4.set_xticklabels(
        labels=[f"{i:.3f}" for i in ax4.get_xticks()], fontsize=12, rotation=20
    )
    ax4.set_yticklabels(
        labels=[
            str(label) if index != 0 else ""
            for index, label in enumerate(ax4.get_yticks())
        ],
        fontsize=12,
    )

    ax4.set_ylabel(ylabel=ax4.get_ylabel(), fontsize=14)
    ax4.set_xlabel(xlabel=ax4.get_xlabel(), fontsize=14)

    ax4.tick_params(axis="x", pad=0)

    # Subplot 3 spanning two rows
    ax3 = fig.add_subplot(gs[:, 1])
    sns.boxplot(
        metrics,
        x="Metric",
        y="value",
        ax=ax3,
        order=[
            "True Positive Rate",
            "False Positive Rate",
            "Positive Predictive Value",
            "$F1_{Macro}$",
            "AUROC",
            "AUPRC",
            "Median days / 365",
        ],
    )

    ax3.set_xticklabels(
        labels=["TPR", "FPR", "PPV", "F1", "AUROC", "AUPRC", "Earliness"], rotation=45
    )

    ax3.set_ylabel("Value", fontsize=14)
    ax3.set_xlabel(xlabel=ax3.get_xlabel(), fontsize=14)

    ax3.set_yticklabels(labels=ax3.get_yticklabels(), fontsize=12)
    ax3.set_xticklabels(labels=ax3.get_xticklabels(), fontsize=12)

    ax3.tick_params(axis="x", pad=0)

    ax3.grid(visible=True, alpha=0.4)
    sns.despine(top=True, right=True, ax=ax3)

    ax1.text(
        -1.35,
        31000,
        "A.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )
    ax1.text(
        -1.35,
        -14000,
        "B.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )
    ax1.text(
        -1.35,
        -59000,
        "C.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )
    ax1.text(
        1.15,
        31000,
        "D.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )

    # plt.tight_layout()

    plt.savefig("stability_composed_figure.png", bbox_inches="tight", dpi=600)
    plt.savefig("stability_composed_figure.pdf", bbox_inches="tight", dpi=600)

    save_stability(metrics=metrics, seeds=probabilities, tau=scores_tau)

    return None


def get_fixed_horizon_various_history(
    data_path,
    horizon=30,
    history=[i for i in range(30, 390, 30)],
    seeds=[i for i in range(1, 101)],
    metrics_file="metrics_thresholds_val_parallel.parquet",
):
    history = history
    seeds = seeds

    D = {}

    for x in range(len(history)):
        for y in range(len(seeds)):
            df = load_data(
                metrics_file, f"{data_path}/ebm_{horizon}_{history[x]}_seed{seeds[y]}/validate"
            )
            if not isinstance(df, str):
                avprec = calc_av_precision(df.dropna(subset=["Threshold"]))
                df["AUPRC"] = avprec

                D[f"{history[x]}-{seeds[y]}"] = df

    his_hor = (
        pd.concat(D)
        .reset_index()
        .rename(columns={"level_0": "History"})
        .drop("level_1", axis=1)
    )

    his_hor["Seed"] = his_hor["History"].apply(lambda x: x.split("-")[1])
    his_hor["History"] = his_hor["History"].apply(lambda x: x.split("-")[0])
    his_hor["History"] = his_hor["History"].astype(int)

    his_hor.to_excel(f"{horizon}horizon_various_history_seeds.xlsx")

    return his_hor


def get_med_days_seed(his_hor, recall=0.58):
    med_days_df = (
        his_hor[his_hor["Recall"] >= recall - 0.004999]
        .sort_values("Threshold", ascending=False)
        .groupby(["History", "Seed"])
        .first()
        .reset_index()
    )
    med_days_df["Median days"] = med_days_df["Median days"] / 365.65

    return med_days_df


def distribution_auprc_med_days(
    data_path,
    horizon=30,
    history=[i for i in range(30, 390, 30)],
    seeds=[i for i in range(1, 101)],
    recall=0.58,
):
    his_hor = get_fixed_horizon_various_history(
        f"{data_path}/validate", horizon=horizon, history=history, seeds=seeds
    )

    med_days = get_med_days_seed(his_hor, recall=recall)

    fac = 1.2
    fig, ax = plt.subplots(figsize=(10 / fac, 7 / fac))
    sns.boxplot(
        pd.melt(
            med_days[["AUPRC", "Seed", "History", "Median days"]],
            id_vars=["Seed", "History"],
        ),
        y="value",
        x="History",
        hue="variable",
        width=0.5,
        fliersize=2,
    )
    # sns.stripplot(x="History", y="Average Precision", jitter=False, data=his_hor.drop_duplicates(subset=["History", "Seed"]), color="black", alpha=0.5, size=4)

    sns.despine(top=True, right=True)

    plt.ylabel("Score", fontsize=14)
    plt.xlabel(ax.get_xlabel(), fontsize=14)

    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=12)

    # plt.legend(title="Metric", loc=[0.4,0.79], fontsize=12, title_fontsize=14)

    plt.legend(title="Metric", fontsize=12, title_fontsize=14)

    # ax.grid(visible=True, alpha=0.4)
    # plt.grid(True, which='minor', axis='both')

    # Set minor ticks on the x-axis
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0.5, 12, 1)))
    ax.tick_params(which="minor", length=0)

    # Set grid lines between xticks (minor grid)
    ax.grid(which="minor", linestyle="-", linewidth="0.5", color="black", alpha=0.4)
    ax.grid(
        which="minor",
        axis="y",
        linestyle="-",
        linewidth="0.5",
        color="black",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig("30_horizon_seeds_auprc.png", dpi=300, bbox_inches="tight")
    plt.savefig("30_horizon_seeds_auprc.pdf", dpi=300, bbox_inches="tight")

    return None


def get_various_horizon_various_history(
    data_path,
    history=[i for i in range(30, 390, 30)],
    horizon=[i for i in range(0, 270, 30)],
    metrics_file="metrics_thresholds_val.parquet",
    recall=0.58,
):
    D = {}
    D_prec_rec = {}

    avprec_array = []

    for x in range(len(history)):
        for y in range(len(horizon)):
            df = load_data(
                metrics_file, f"{data_path}ebm_{horizon[y]}_{history[x]}/validate"
            )
            if not isinstance(df, str):
                df = df.append(
                    {"Precision gain": df["Precision gain"].max(), "Recall gain": 0},
                    ignore_index=True,
                )

                try:
                    avprec = calc_av_precision(df.dropna(subset=["Threshold"]))

                except IndexError:
                    avprec = 0

                avprec_array.append(avprec)

            else:
                avprec = 0
                avprec_array.append(avprec)

            if isinstance(df, str):
                print(f"{horizon[y]}-{history[x]}")
            D[f"{horizon[y]}-{history[x]}"] = df

        D_prec_rec[history[x]] = avprec_array
        avprec_array = []

    his_hor_all = (
        pd.concat(D).reset_index().rename(columns={"level_0": "Horizon-History"})
    )
    his_hor_all = his_hor_all[
        ~((his_hor_all["Precision"] == 0) & (his_hor_all["Recall"] == 1))
    ]
    heat_prec_rec = pd.DataFrame(D_prec_rec, index=horizon)

    heat_prec_rec.to_excel("auprc_heatmap.xlsx")
    his_hor_all.to_excel("various_history_varios_horizon_auprc_heatmap.xlsx")

    return his_hor_all, heat_prec_rec


def get_pareto(med_days_df):
    med_days_df = med_days_df.groupby("History").mean().reset_index()
    med_days_df["History"] = med_days_df["History"].astype(str)

    mask_auprc = paretoset(med_days_df[["Median days", "AUPRC"]], sense=["max", "max"])

    med_days_df["Pareto set"] = np.nan
    med_days_df.loc[med_days_df[mask_auprc].index.unique(), "Pareto set"] = "Pareto set"
    med_days_df.fillna("Non-pareto set", inplace=True)

    med_days_df[["History", "AUPRC", "Median days", "Pareto set"]].to_excel(
        f"median_days_seeds.xlsx"
    )

    return med_days_df, mask_auprc


def heat_pareto(
    data_path,
    horizon=30,
    history=[i for i in range(0, 390, 30)],
    seeds=[i for i in range(1, 101)],
    recall=0.58,
):
    his_hor_all, heat_prec_rec = get_various_horizon_various_history(f"{data_path}/simulate")
    his_hor_30 = get_fixed_horizon_various_history(
        f"{data_path}/simulate", horizon=horizon, history=history, seeds=seeds
    )

    med_days = get_med_days_seed(his_hor_30, recall=recall)
    med_days_pareto, mask_auprc = get_pareto(med_days)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(
        heat_prec_rec.T,
        annot=True,
        fmt=f".3f",
        ax=ax[0],
        annot_kws={"fontsize": 14},
        cbar_kws={"location": "top", "shrink": 1.56},
    )

    ax[0].invert_yaxis()
    ax[0].set_xlabel("Horizon", fontsize=16, labelpad=5)
    ax[0].set_ylabel("History", fontsize=16, labelpad=5)

    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), fontsize=14)
    ax[0].set_yticklabels(labels=ax[0].get_yticklabels(), fontsize=14, rotation=0)

    cbar = ax[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    sns.scatterplot(
        med_days_pareto,
        y="AUPRC",
        x="Median days",
        hue="Pareto set",
        ax=ax[1],
        style="Pareto set",
        hue_order=["Pareto set", "Non-pareto set"],
        s=100,
        palette=[
            sns.color_palette("colorblind")[1],
            sns.color_palette("colorblind")[0],
        ],
    )

    ax[1].set_ylabel(ylabel=ax[1].get_ylabel(), fontsize=16)
    ax[1].set_xlabel("Earliness", fontsize=16)

    ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), fontsize=14)
    ax[1].set_yticklabels(labels=ax[1].get_yticklabels(), fontsize=14)

    ax[1].legend(title="", fontsize=14, loc="lower left")

    # Add labels for specific points (adjust indices as needed)
    for i, row in med_days_pareto[mask_auprc].iterrows():
        plt.text(
            row["Median days"],
            row["AUPRC"] - 0.0015,
            row["History"],
            ha="center",
            va="center",
            fontsize=14,
        )

    plt.tight_layout()

    bbox = ax[0].get_position()
    bbox2 = ax[1].get_position()

    ax[0].set_position([bbox.x0 - 0.1, bbox.y0, bbox.width + 0.2, bbox.height])
    ax[1].set_position([bbox2.x0 + 0.015, bbox2.y0, bbox2.width, bbox.height])

    sns.despine(top=True, right=True, ax=ax[0])
    sns.despine(top=True, right=True, ax=ax[1])

    ax[0].text(
        -0.8,
        12.5,
        "A.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )
    ax[0].text(
        9.6,
        12.5,
        "B.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )

    ax[1].grid(visible=True, alpha=0.4)

    plt.savefig("heatmap_prec_rec_240.png", dpi=600, bbox_inches="tight")
    plt.savefig("heatmap_prec_rec_240.pdf", dpi=600, bbox_inches="tight")

    return None


def curves_ablation(
    data_path,
    chron="chron",
    diag="diag",
    analy="analy",
    espe="espe",
    age="age",
    all_="ebm_seed3",
    metrics_file="metrics_thresholds_30horizon_180history_test.parquet",
):
    sets = [chron, diag, analy, espe, age, all_]
    nice = [
        "Chronic comorbidities",
        "Diagnostics",
        "Analytics",
        "Specialty",
        "Age",
        "All datasets",
    ]

    D = {}
    for i, j in zip(sets, nice):
        df = load_data(metrics_file, f"{data_path}/{i}/validate")
        df.loc[len(df)] = {"Recall": 1, "False Positive Rate": 1}
        df.loc[len(df)] = {"Precision": 1, "Recall": 0}
        df.loc[len(df)] = {"Precision gain": 1, "Recall gain": 0}
        df.loc[len(df)] = {"Median days": 0, "Recall": 0}

        D[j] = df

    sets = pd.concat(D).reset_index().rename(columns={"level_0": "Left out dataset"})

    hue_colors = {
        "Chronic comorbidities": "orange",
        "Diagnostics": "red",
        "Analytics": "green",
        "Specialty": "blue",
        "Age": "purple",
        "All datasets": "black",
    }

    fig, ax = plt.subplots(
        2, 2, figsize=(8, 7), gridspec_kw={"wspace": 0.5, "hspace": 0.4}
    )

    sns.lineplot(
        data=sets,
        x="False Positive Rate",
        y="Recall",
        hue="Left out dataset",
        palette=hue_colors,
        linewidth=1,
        ax=ax[0, 0],
        legend=None,
    )
    ax[0, 0].plot([0, 1], [0, 1], color="lightgray", linestyle="--")

    for i in range(0, 2):
        for j in range(0, 2):
            if (i == 1) and (j == 1):
                pass
            else:
                ax[i, j].grid(visible=True, alpha=0.4)
                sns.despine(top=True, right=True, ax=ax[i, j])
                ax[i, j].set_xticklabels(labels=ax[i, j].get_xticklabels(), fontsize=12)
                ax[i, j].set_yticklabels(labels=ax[i, j].get_yticklabels(), fontsize=12)
                ax[i, j].set_ylabel(ylabel=ax[i, j].get_ylabel(), fontsize=14)
                ax[i, j].set_xlabel(xlabel=ax[i, j].get_xlabel(), fontsize=14)

    sns.lineplot(
        sets.reset_index(),
        x="Recall",
        y="Precision",
        legend=None,
        palette=hue_colors,
        hue="Left out dataset",
        linewidth=1,
        ax=ax[0, 1],
    )
    sns.lineplot(
        sets.reset_index().dropna(subset="Median days"),
        x="Recall",
        y="Median days",
        legend=None,
        palette=hue_colors,
        hue="Left out dataset",
        linewidth=1,
        ax=ax[1, 0],
    )
    sns.lineplot(
        sets.reset_index(),
        x="Recall gain",
        y="Precision gain",
        legend=None,
        palette=hue_colors,
        hue="Left out dataset",
        linewidth=1,
        ax=ax[1, 1],
    )
    ax[1, 1].plot([1, 0], [0, 1], color="lightgray", linestyle="--")

    ax[0, 1].axhline(0.02, color="lightgray", linestyle="--")

    ax[0, 0].text(
        -0.4, 1, "A.", fontweight="bold", fontsize="large", horizontalalignment="center"
    )
    ax[0, 0].text(
        1.25, 1, "B.", fontweight="bold", fontsize="large", horizontalalignment="center"
    )
    ax[0, 0].text(
        -0.4,
        -0.4,
        "C.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )
    ax[0, 0].text(
        1.25,
        -0.4,
        "D.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )

    ax[0, 0].set_ylabel("True Positive Rate", fontsize=14)

    ax[0, 1].set_ylim(0, 1)

    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1500)

    chron_line = Line2D([0], [0], linestyle="solid", color="orange")
    diag_line = Line2D([0], [0], linestyle="solid", color="red")
    analy_line = Line2D([0], [0], linestyle="solid", color="green")
    espe_line = Line2D([0], [0], linestyle="solid", color="blue")
    age_line = Line2D([0], [0], linestyle="solid", color="purple")
    all_line = Line2D([0], [0], linestyle="solid", color="black")

    plt.legend(
        title="Left out dataset",
        title_fontsize=14,
        handles=[all_line, diag_line, analy_line, chron_line, espe_line, age_line],
        labels=[
            "All datasets",
            "Diagnostics",
            "Analytics",
            "Chronic comorbidities",
            "Specialty",
            "Age",
        ],
        bbox_to_anchor=(0.9, -0.3),
        frameon=False,
        fontsize=12,
        ncol=3,
    )

    ax[1, 0].set_yticklabels(labels=[i for i in range(0, 1600, 200)], fontsize=12)

    ax[1, 1].set_yticklabels(labels=ax[1, 1].get_yticklabels(), fontsize=12)
    ax[1, 1].grid(visible=True, alpha=0.4)
    sns.despine(top=True, right=True, ax=ax[1, 1])
    ax[1, 1].set_xticklabels(labels=ax[1, 1].get_xticklabels(), fontsize=12)
    ax[1, 1].set_ylabel(ylabel=ax[1, 1].get_ylabel(), fontsize=14)
    ax[1, 1].set_xlabel(xlabel=ax[1, 1].get_xlabel(), fontsize=14)

    plt.tight_layout()

    plt.savefig("ablation.png", dpi=600, bbox_inches="tight")
    plt.savefig("ablation.pdf", dpi=600, bbox_inches="tight")

    sets.to_excel("ablation.xlsx")

    return None


def curves_algos(data_path):
    algos = ["gbc_seed3", "ebm_seed3"]

    D_algos = {}

    for i in algos:
        print(i)
        df = load_data(
            "metrics_thresholds_30horizon_180history_test.parquet",
            f"{data_path}/{i}/validate",
        )
        D_algos[i] = df
    enet = load_data(
        "metrics_thresholds_test_parallel.parquet", f"{data_path}/enet_seed1/validate"
    ).rename(columns={"Sensitivity": "1-FPR"})
    enet = enet[enet["Recall gain"] >= 0]
    D_algos["enet_seed1"] = enet

    algos = (
        pd.concat(D_algos)
        .reset_index()
        .drop("level_1", axis=1)
        .replace({"enet_seed1": "ENet", "ebm_seed3": "EBM", "gbc_seed3": "GBC"})
        .rename(columns={"level_0": "Model"})
        .reset_index()
    )
    print("algos together")

    algos = pd.concat(
        [algos, pd.DataFrame({"Model": "ENet", "Precision": 1, "Recall": 0}, index=[0])]
    )
    algos = pd.concat(
        [algos, pd.DataFrame({"Model": "EBM", "Precision": 1, "Recall": 0}, index=[0])]
    )

    algos = pd.concat(
        [
            algos,
            pd.DataFrame(
                {
                    "Model": "EBM",
                    "Precision gain": algos["Precision gain"].max(),
                    "Recall gain": 0,
                },
                index=[0],
            ),
        ]
    )
    algos = pd.concat(
        [
            algos,
            pd.DataFrame(
                {
                    "Model": "ENet",
                    "Precision gain": algos["Precision gain"].max(),
                    "Recall gain": 0,
                },
                index=[0],
            ),
        ]
    )
    algos = pd.concat(
        [
            algos,
            pd.DataFrame(
                {
                    "Model": "GBC",
                    "Precision gain": algos["Precision gain"].max(),
                    "Recall gain": 0,
                },
                index=[0],
            ),
        ]
    )

    algos = pd.concat(
        [
            algos,
            pd.DataFrame(
                {"Model": "ENet", "False Positive Rate": 0, "Recall": 0}, index=[0]
            ),
        ]
    )

    print("plotting")
    fig, ax = plt.subplots(
        2, 2, figsize=(8, 7), gridspec_kw={"wspace": 0.5, "hspace": 0.4}
    )

    sns.lineplot(
        data=algos.reset_index(),
        x="False Positive Rate",
        y="Recall",
        hue="Model",
        linewidth=1,
        ax=ax[0, 0],
        legend=None,
    )
    ax[0, 0].plot([0, 1], [0, 1], color="black", linestyle="--")

    for i in range(0, 2):
        for j in range(0, 2):
            if (i == 1) and (j == 1):
                pass
            else:
                ax[i, j].grid(visible=True, alpha=0.4)
                sns.despine(top=True, right=True, ax=ax[i, j])
                ax[i, j].set_xticklabels(labels=ax[i, j].get_xticklabels(), fontsize=12)
                ax[i, j].set_yticklabels(labels=ax[i, j].get_yticklabels(), fontsize=12)
                ax[i, j].set_ylabel(ylabel=ax[i, j].get_ylabel(), fontsize=14)
                ax[i, j].set_xlabel(xlabel=ax[i, j].get_xlabel(), fontsize=14)

    sns.lineplot(
        algos.reset_index(),
        x="Recall",
        y="Precision",
        hue="Model",
        linewidth=1,
        ax=ax[0, 1],
    )
    sns.lineplot(
        algos.reset_index(),
        x="Recall",
        y="Median days",
        legend=None,
        hue="Model",
        linewidth=1,
        ax=ax[1, 0],
    )
    sns.lineplot(
        algos.reset_index(),
        x="Recall gain",
        y="Precision gain",
        legend=None,
        hue="Model",
        linewidth=1,
        ax=ax[1, 1],
    )

    ax[0, 1].axhline(0.02, color="black", linestyle="--")

    ax[0, 0].text(
        -0.4, 1, "A.", fontweight="bold", fontsize="large", horizontalalignment="center"
    )
    ax[0, 0].text(
        1.25, 1, "B.", fontweight="bold", fontsize="large", horizontalalignment="center"
    )
    ax[0, 0].text(
        -0.4,
        -0.4,
        "C.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )
    ax[0, 0].text(
        1.25,
        -0.4,
        "D.",
        fontweight="bold",
        fontsize="large",
        horizontalalignment="center",
    )

    ax[0, 0].set_ylabel("True Positive Rate", fontsize=14)

    ax[0, 1].set_ylim(0, 1)

    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1500)

    ax[1, 1].plot(
        [1, 0],
        [0, algos.reset_index()["Precision gain"].max()],
        color="black",
        linestyle="--",
    )

    line1 = Line2D([0], [0], linestyle="solid", color="orange")
    line2 = Line2D([0], [0], linestyle="solid", color="red")
    line3 = Line2D([0], [0], linestyle="solid", color="green")

    # plt.legend(title="Model", title_fontsize=14, handles=[line1, line2, line3], labels=["EBM", "GBC", "ENet"],
    #           bbox_to_anchor=(0.4, -0.3), frameon=False, fontsize=12, ncol=4)

    ax[1, 0].set_yticklabels(labels=[i for i in range(0, 1600, 200)], fontsize=12)

    ax[1, 1].set_yticklabels(labels=ax[1, 1].get_yticklabels(), fontsize=12)
    ax[1, 1].grid(visible=True, alpha=0.4)
    sns.despine(top=True, right=True, ax=ax[1, 1])
    ax[1, 1].set_xticklabels(labels=ax[1, 1].get_xticklabels(), fontsize=12)
    ax[1, 1].set_ylabel(ylabel=ax[1, 1].get_ylabel(), fontsize=14)
    ax[1, 1].set_xlabel(xlabel=ax[1, 1].get_xlabel(), fontsize=14)

    plt.tight_layout()

    plt.savefig("algorithm_comparison.png", bbox_inches="tight", dpi=600)
    plt.savefig("algorithm_comparison.pdf", bbox_inches="tight", dpi=600)

    algos.to_excel("algorithm_comparison.xlsx")

    return None
