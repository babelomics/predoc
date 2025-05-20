import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from predoc.dictionaries_columns import translations_dict


def plot_prec_rec_gain(df, hue, suffix_out=""):
    fac = 1.2

    fig, ax = plt.subplots(figsize=(9 / fac, 9 / fac))

    sns.lineplot(df.reset_index(), x="Recall gain", y="Precision gain", hue=hue)
    sns.despine(top=True, right=True)
    sns.set_context("paper")

    plt.legend(fontsize=16)

    ax.set_ylabel(ylabel=ax.get_ylabel(), fontsize=18)
    ax.set_xlabel(xlabel=ax.get_xlabel(), fontsize=18)

    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)

    ax.grid(visible=True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"prec_rec_gain{suffix_out}.png", bbox_inches="tight")
    plt.savefig(f"prec_rec_gain{suffix_out}.pdf", bbox_inches="tight")

    return None


def plot_metrics_curves(
    df, hue, suffix_out="", markers=[".", "_", "o"], colors=["blue", "orange", "green"]
):
    fac = 1.2

    fig, ax = plt.subplots(figsize=(10.5 / fac, 10.5 / fac))

    ax0 = sns.lineplot(df, x="Recall", y="Precision", hue=hue, legend=None, linewidth=2)

    sns.lineplot(
        df,
        x="Recall",
        y="1-FPR",
        hue=hue,
        ax=ax0,
        linestyle="dotted",
        legend=None,
        linewidth=2,
    )

    ax1 = ax0.twinx()

    sns.lineplot(
        df,
        x="Recall",
        y="Median days",
        hue=hue,
        ax=ax1,
        linestyle="dashdot",
        legend=None,
        linewidth=2,
    )

    markers = markers
    colors = colors

    solid_line = Line2D([0], [0], linestyle="solid", color="black")
    dotted_line = Line2D([0], [0], linestyle="dotted", color="black")
    dashdot_line = Line2D([0], [0], linestyle="dashdot", color="black")

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")

    handles_list = [f("s", colors[i]) for i in range(len(df[hue].unique()))]
    handles_list.extend([[solid_line], [dotted_line], [dashdot_line]])

    handles = [handles_list[i][0] for i in range(len(handles_list))]
    labels = np.append(
        df[hue].unique(), ["Precision-Recall", "1-FPR", "Median days/365"]
    )

    plt.legend(
        handles, labels, framealpha=1, bbox_to_anchor=(1.02, 1.2), fontsize=16, ncol=2
    )

    sns.set_context("paper")

    ax0.spines["top"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    ax0.set_ylabel("Precision", fontsize=18)
    ax1.set_ylabel("Median days/365", fontsize=18)

    ax0.set_xlabel("Recall", fontsize=18)

    ax0.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)
    ax1.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)

    ax0.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)
    ax1.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)

    ax0.grid(visible=True, alpha=0.4)

    plt.tight_layout()

    plt.savefig(f"curves{suffix_out}.png", dpi=300)
    plt.savefig(f"curves{suffix_out}.pdf", dpi=300)

    return None


def plot_roc(df, hue, suffix_out=""):
    fac = 1.2

    fig, ax = plt.subplots(figsize=(9 / fac, 9 / fac))

    sns.lineplot(data=df, x="False Positive Rate", y="Recall", linewidth=2, hue=hue)
    sns.despine(top=True, right=True)

    ax.plot([0, 1], [0, 1], color="red", linestyle="--")

    plt.ylabel("True Positive Rate", fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=18)

    plt.legend(fontsize=16)

    sns.set_context("paper")

    ax.grid(visible=True, alpha=0.4)

    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)

    plt.tight_layout()

    plt.savefig(f"roc{suffix_out}.png", dpi=300)
    plt.savefig(f"roc{suffix_out}.pdf", dpi=300)

    return None


def plot_metrics(df, hue, order, suffix_out=""):
    fac = 1.2

    fig, ax = plt.subplots(figsize=(12 / fac, 9 / fac))

    sns.barplot(df, x="value", y="Metric", hue=hue, order=order)
    sns.despine(top=True, right=True)

    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i], fontsize=14, padding=2)

    plt.ylabel(ylabel="", fontsize=16)
    plt.xlabel(xlabel="Value", fontsize=18)

    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)

    sns.set_context("poster")
    plt.legend(loc=(0.3, 0.5), fontsize=16)

    ax.grid(visible=True, alpha=0.4)

    plt.tight_layout()

    plt.savefig(f"metrics_models{suffix_out}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"metrics_models{suffix_out}.pdf", bbox_inches="tight", dpi=300)

    return None


def plot_individual_proba_trajectories(
    df,
    y="proba",
    x="days_to_diag",
    hue="global_ovarian_cancer_truth",
    threshold=0.15,
    suffix_out="",
):
    fac = 1.2
    fig, ax = plt.subplots(1, 1, figsize=(16 / fac, 9 / fac))

    # sns.scatterplot(case_cont, x="end_date",y="ovarian_cancer_probability",  marker='^', s=50)

    sns.lineplot(
        df,
        y=y,
        x=x,
        hue=hue,
        hue_order=["Control", "Case"],
        markers=["o", "^"],
        style=hue,
        linewidth=2,
    )
    sns.set_context("paper")

    sns.despine(top=True, right=True)

    ax.axhline(threshold, color="red", linestyle="--")

    plt.legend(loc="upper left", fontsize=16)
    ax.set_xlabel("Days to end of evaluation", labelpad=10.0, fontsize=18)
    ax.set_ylabel("Ovarian cancer probability", labelpad=10.0, fontsize=18)
    # ax.set_yticks([i/100 for i in range(10, 2)])

    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)

    ax.grid(visible=True, alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        f"case_control_proba_trajectories{suffix_out}.png", bbox_inches="tight", dpi=300
    )
    plt.savefig(
        f"case_control_proba_trajectories{suffix_out}.pdf", bbox_inches="tight", dpi=300
    )

    return None


def plot_proba_feature_trajectory(
    df,
    feature,
    feature_legend_label,
    co=None,
    co_date=None,
    co_legend_label=None,
    suffix_out="",
):
    fac = 1.2
    fig, ax = plt.subplots(1, 1, figsize=(16 / fac, 9 / fac))

    sns.lineplot(
        df,
        y="ovarian_cancer_probability",
        x="days_to_diag",
        color="orange",
        legend=False,
    )
    sns.scatterplot(df, y="ovarian_cancer_probability", color="orange", legend=False)

    sns.set_context("talk")

    ax1 = ax.twinx()

    sns.lineplot(
        df, y=feature, markers=["o"], linestyle="dashdot", x="days_to_diag", ax=ax1
    )

    solid_line = Line2D([0], [0], linestyle="solid", color="orange")
    dashdot_line = Line2D([0], [0], linestyle="dashdot", color="blue")
    dashed_line = Line2D([0], [0], linestyle="dashed", color="red")

    ax.set_xlabel("Days to ovarian cancer diagnosis", labelpad=10.0, fontsize=16)
    ax.set_ylabel("Ovarian cancer probability", labelpad=10.0, fontsize=16)

    ax1.set_ylabel("Feature score", labelpad=10.0, fontsize=16)

    ax.spines["top"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    if co:
        ax.axvline(co_date, color="red", linestyle="--")

    if co:
        plt.legend(
            handles=[solid_line, dashdot_line, dashed_line],
            labels=[
                "Ovarian cancer probability",
                feature_legend_label,
                co_legend_label,
            ],
            loc="upper left",
            fontsize=16,
        )

    else:
        plt.legend(
            handles=[solid_line, dashdot_line, dashed_line],
            labels=["Ovarian cancer probability", feature_legend_label],
            loc="upper left",
            fontsize=16,
        )

    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=14)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=14)

    ax.grid(visible=True, alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        f"proba_feature_trajectory{suffix_out}.png", bbox_inches="tight", dpi=300
    )
    plt.savefig(
        f"proba_feature_trajectory{suffix_out}.pdf", bbox_inches="tight", dpi=300
    )

    return None


def plot_feature_score_by_prediction(df, figsize_x=16, figsize_y=9, suffix_out=""):
    fac = 1.2

    fig, ax = plt.subplots(figsize=(figsize_x / fac, figsize_y / fac))

    sns.despine(top=True, right=True)
    sns.set_context("paper")

    sns.violinplot(
        df.replace(translations_dict),
        x="Feature score",
        y="Feature",
        hue="Prediction",
        density_norm="count",
        dodge=True,
        hue_order=["Negative prediction", "Positive prediction"],
    )

    ax.axvline(0, color="black", alpha=0.3)
    plt.legend(fontsize=16, loc=(0.65, 0.88))
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)

    ax.set_ylabel(ylabel=ax.get_ylabel(), fontsize=18)
    ax.set_xlabel(xlabel=ax.get_xlabel(), fontsize=18)
    ax.grid(visible=True, alpha=0.4)

    plt.tight_layout()

    plt.savefig(f"violin_features{suffix_out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"violin_features{suffix_out}.pdf", dpi=300, bbox_inches="tight")

    return None


def plot_days_to_diag(df, days="days_to_diag", suffix_out=""):
    df["grouped_days"] = np.where(df[days] > 400, 400, df[days])

    fac = 1.2
    fig, ax = plt.subplots(figsize=(16 / fac, 9 / fac))

    sns.histplot(df, x="grouped_days", bins=10)
    sns.set_context("paper")
    sns.despine(top=True, right=True)
    ax.set_xticks([i for i in range(20, 400, 40)])

    ticks = list(ax.get_xticks()[0:-1])
    ticks = list(map(str, ticks))
    ticks.append(">360")
    ax.set_xticklabels(ticks)
    # plt.title("Distribution of early prediction days", fontsize=36)

    ax.set_xlabel("Early prediction in days", labelpad=10.0, fontsize=18)
    ax.set_ylabel("Count", labelpad=10.0, fontsize=18)

    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=16)

    ax.grid(visible=True, alpha=0.4)

    plt.tight_layout()

    plt.savefig(f"days_to_prediction{suffix_out}.png", dpi=300)
    plt.savefig(f"days_to_prediction{suffix_out}.pdf", dpi=300)

    return None
