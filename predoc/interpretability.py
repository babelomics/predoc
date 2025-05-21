import pandas as pd


def interpret_strip_df(df):
    pos = df[df["pred"] == 1].sort_values("date_limit")
    pos = pos[~pos.index.duplicated(keep="first")]

    neg = df.loc[pos.index.symmetric_difference(df.index)].sort_values("date_limit")
    neg = neg[~neg.index.duplicated(keep="last")]

    pos_neg = pd.concat([pos, neg])

    ##features starts with "tranf", this line is to keep only those columns
    feat = pos_neg[[i for i in pos_neg.columns if "tranf" in i]]

    feat = feat.merge(
        pos_neg[["global_ovarian_cancer_truth", "pred"]],
        right_index=True,
        left_index=True,
    )
    feat = pd.melt(
        feat.reset_index(),
        id_vars=["index", "global_ovarian_cancer_truth", "pred"],
        var_name="Feature",
        value_name="Feature score",
    )
    feat.set_index("index", inplace=True)

    return feat


def get_top10(df):
    df["abs"] = df["Feature score"].abs()
    sorted_df = (
        pd.DataFrame(df.groupby("Feature")["abs"].mean())
        .reset_index()
        .sort_values("abs", ascending=False)
    )

    sorted_df_top = sorted_df.head(10)["Feature"].tolist()

    return sorted_df_top
