"""Streamlit app: estimate a company's risk of bankruptcy 1-5 years ahead.

    streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from bankruptcy.config import FIGURES_DIR
from bankruptcy.features import definition, meaning
from bankruptcy.predict import BankruptcyPredictor

st.set_page_config(page_title="Bankruptcy Risk Predictor", page_icon="📊", layout="wide")

RAISES, LOWERS = "#e34948", "#2a78d6"


@st.cache_resource
def load_predictor():
    return BankruptcyPredictor()


predictor = load_predictor()
META = predictor.metadata


def years_label(h):
    return f"{h} year ahead" if h == 1 else f"{h} years ahead"


def pct(p):
    return f"{p:.1%}" if p >= 0.001 else "<0.1%"


# ===========================================================
# Sidebar: horizon + model card
# ===========================================================
with st.sidebar:
    st.header("Prediction horizon")
    years = st.radio("Predict bankruptcy…", predictor.horizons, format_func=years_label,
                     label_visibility="collapsed")
    m = META["horizons"][str(years)]
    test = m["metrics"]["test"]

    st.divider()
    st.subheader("This model")
    st.markdown(
        f"- Trained on **{m['n_train']:,}** companies (**{m['n_train_bankrupt']}** went bankrupt)\n"
        f"- Uses the **{len(m['features'])}** most useful financial ratios\n"
        f"- Flags a company when its probability is **≥ {m['threshold']:.0%}**"
    )
    c1, c2 = st.columns(2)
    c1.metric("Caught (recall)", f"{test['recall']:.0%}",
              help="Share of bankrupt test companies the model flagged.")
    c2.metric("Right (precision)", f"{test['precision']:.0%}",
              help="Share of flagged test companies that really went bankrupt.")
    c1.metric("F1 score", f"{test['f1']:.2f}", help="Balance of recall and precision (0 to 1).")
    c2.metric("ROC-AUC", f"{test['roc_auc']:.2f}",
              help="How well the model ranks bankrupt companies above healthy ones (0.5 = guessing, 1 = perfect).")
    st.caption(f"Scores measured on {m['n_test']:,} companies the model never saw during training.")

    st.divider()
    st.subheader("Empty boxes mean…")
    missing_mode = st.radio(
        "Empty boxes mean",
        ["I don't know the value", "Not reported in the accounts"],
        label_visibility="collapsed",
        help="In this data, a value missing from a company's accounts is often a warning sign "
             "(for example, no sales figure for last year). If you simply don't know a value, keep the first "
             "option: the model then uses a typical value instead.",
    )
    FILL = missing_mode == "I don't know the value"


st.title("📊 Bankruptcy Risk Predictor")
st.write(
    "Enter a company's financial ratios to estimate the probability that it goes bankrupt "
    f"**{years_label(years)}**. Models are LightGBM gradient-boosted trees trained on "
    "43,405 financial statements of Polish companies (UCI Machine Learning Repository)."
)

tab_one, tab_batch, tab_about = st.tabs(["🏢 Single company", "📁 Batch scoring (CSV)", "ℹ️ How it works"])

# ===========================================================
# Single company
# ===========================================================
with tab_one:
    feats = m["features"]

    def fill_example(h):
        choice = st.session_state[f"example_{h}"]
        values = META["horizons"][str(h)]["examples"].get(choice, {}).get("values", {})
        for f in META["horizons"][str(h)]["features"]:
            st.session_state[f"in_{h}_{f}"] = values.get(f)

    left, right = st.columns([3, 2])
    with left:
        st.selectbox(
            "Start from",
            ["Blank form", *m["examples"].keys()],
            key=f"example_{years}",
            on_change=fill_example,
            args=(years,),
            help="Examples are real companies from the test set, which the model never saw in training.",
        )
    with right:
        st.write("")
        st.caption("Leave a box empty if you don't know the value. The sidebar sets how empty boxes are treated.")

    inputs = {}
    cols = st.columns(2)
    for i, f in enumerate(feats):
        s = m["feature_stats"][f]
        key = f"in_{years}_{f}"
        default = {} if key in st.session_state else {"value": None}   # examples fill session_state
        with cols[i % 2]:
            inputs[f] = st.number_input(
                f"{meaning(f)} ({f})",
                **default,
                format="%.4f",
                key=key,
                placeholder="unknown",
                help=f"**{definition(f)}**  \nTypical range: {s['p05']:,.3f} to {s['p95']:,.3f} "
                     f"(median {s['median']:,.3f})",
            )

    if st.button("Predict bankruptcy risk", type="primary", use_container_width=True):
        res = predictor.predict(years, inputs, fill_missing=FILL).iloc[0]
        p, level = res["bankruptcy_probability"], res["risk_level"]
        n_missing = sum(v is None for v in inputs.values())

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Bankruptcy probability", pct(p))
        b.metric("Decision threshold", f"{m['threshold']:.0%}")
        c.metric("Risk level", level)
        st.progress(min(float(p), 1.0))

        if level == "High":
            st.error(f"⚠️ **Likely to go bankrupt {years_label(years)}.** "
                     f"The probability is above the {m['threshold']:.0%} threshold.")
        elif level == "Medium":
            st.warning(f"🟠 **Watch closely.** The probability is below the {m['threshold']:.0%} threshold, "
                       f"but above {m['risk_bands']['medium']:.0%}, so this company is not clearly safe.")
        else:
            st.success(f"✅ **Not likely to go bankrupt {years_label(years)}.** "
                       f"For comparison, about {m['base_rate']:.1%} of companies in the data went bankrupt.")
        if n_missing and FILL:
            st.caption(f"{n_missing} of {len(feats)} boxes were empty and filled with typical values.")
        elif n_missing:
            st.caption(f"{n_missing} of {len(feats)} boxes were empty and treated as not reported. "
                       "Missing values are often a warning sign in this data, so they can raise the risk.")

        st.subheader("Why this result?")
        ex = predictor.explain(years, inputs, fill_missing=FILL)
        top = ex.head(10).copy()
        top["effect"] = top["contribution"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
        top["label"] = top["meaning"] + " (" + top["feature"] + ")"
        top["shown_value"] = [
            "not reported" if pd.isna(v) else
            (f"typical value {v:,.4f}" if inputs[f] is None else f"{v:,.4f}")
            for f, v in zip(top["feature"], top["value"])
        ]
        chart = (
            alt.Chart(top)
            .mark_bar(cornerRadiusEnd=4, size=18)
            .encode(
                x=alt.X("contribution:Q", title="Push on the risk score (log-odds)"),
                y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=380)),
                color=alt.Color("effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                                            range=[RAISES, LOWERS]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("label:N", title="Ratio"),
                         alt.Tooltip("shown_value:N", title="Value entered"),
                         alt.Tooltip("contribution:Q", title="Push", format="+.3f")],
            )
            .properties(height=36 * len(top))
        )
        st.altair_chart(chart, use_container_width=True)

        up = ex[ex["contribution"] > 0].head(1)
        down = ex[ex["contribution"] < 0].head(1)
        notes = []
        if len(up):
            notes.append(f"The biggest factor **raising** the risk is *{up.iloc[0]['meaning'].lower()}* ({up.iloc[0]['feature']}).")
        if len(down):
            notes.append(f"The biggest factor **lowering** the risk is *{down.iloc[0]['meaning'].lower()}* ({down.iloc[0]['feature']}).")
        st.markdown(" ".join(notes))
        st.caption("Each bar is a SHAP value: how much that ratio moved this company's score away from the "
                   "average company. Red bars push towards bankruptcy, blue bars push away from it.")

# ===========================================================
# Batch scoring
# ===========================================================
with tab_batch:
    all_feats = sorted({f for h in predictor.horizons for f in predictor.features(h)},
                       key=lambda f: int(f[4:]))
    st.write("Score many companies at once for **all five horizons**. Upload a CSV with one row per company. "
             "Columns are the ratio names (`Attr1` … `Attr64`). An optional `company` column is kept as the label. "
             "Missing columns or empty cells follow the sidebar setting for empty boxes.")
    template = pd.DataFrame(columns=["company", *all_feats])
    st.download_button("Download CSV template", template.to_csv(index=False), "bankruptcy_template.csv",
                       "text/csv")
    up = st.file_uploader("Upload companies (CSV)", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        out = pd.DataFrame({"company": df["company"] if "company" in df else range(1, len(df) + 1)})
        for h in predictor.horizons:
            r = predictor.predict(h, df, fill_missing=FILL)
            out[f"probability_{h}y"] = r["bankruptcy_probability"].round(4).values
            out[f"prediction_{h}y"] = r["prediction"].values
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Download results", out.to_csv(index=False), "bankruptcy_predictions.csv", "text/csv")

# ===========================================================
# How it works
# ===========================================================
with tab_about:
    st.markdown(f"""
### In one paragraph
For each horizon there is a separate **LightGBM** model: a large group of small decision trees, where each tree
corrects the mistakes of the ones before it. The trees look at a company's financial ratios (profitability, debt,
liquidity, activity) and add up to a **risk score**. That score is converted into a **probability** that was
checked against real outcomes (calibrated). If the probability is at or above the horizon's **threshold**,
the company is flagged as likely to go bankrupt.

### The data
The Polish Companies Bankruptcy dataset (UCI) has 64 financial ratios per company, from the EMIS database.
Bankrupt companies were analysed over 2000–2012, and still-operating ones over 2007–2013. Each of the five files
covers a different gap between the financial statements and the bankruptcy outcome:

| Horizon | Data file | Companies | Went bankrupt |
|---|---|---|---|
""" + "\n".join(
        f"| {years_label(int(h))} | `{mm['source_file']}` | {mm['n_train'] + mm['n_test']:,} | "
        f"{mm['n_train_bankrupt'] + mm['n_test_bankrupt']:,} |"
        for h, mm in sorted(META["horizons"].items(), key=lambda kv: int(kv[0]))
    ) + """

### How the models were built
1. **Split.** 85% of companies for training, 15% locked away for the final test.
2. **Pick the ratios.** Cross-validation on the training part ranks all 64 ratios, and the top 20 are kept.
3. **Tune.** Optuna tries 40 LightGBM settings and keeps the one with the best cross-validated precision-recall score.
4. **Calibrate and set the threshold.** Out-of-fold predictions turn raw scores into honest probabilities and pick the threshold with the best F1.
5. **Test once** on the locked-away 15%.
""")
    st.markdown("### Test results")
    rows = []
    for h, mm in sorted(META["horizons"].items(), key=lambda kv: int(kv[0])):
        t = mm["metrics"]["test"]
        rows.append({"Horizon": years_label(int(h)), "Threshold": f"{mm['threshold']:.2f}",
                     "Recall": f"{t['recall']:.2f}", "Precision": f"{t['precision']:.2f}",
                     "F1": f"{t['f1']:.2f}", "ROC-AUC": f"{t['roc_auc']:.2f}", "PR-AUC": f"{t['pr_auc']:.2f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    fig_cmp = Path(FIGURES_DIR) / "old_vs_new.png"
    if fig_cmp.exists():
        st.image(str(fig_cmp), caption="New models (v2) compared with the original models (v1) on the same test companies")
    fig_imp = Path(FIGURES_DIR) / f"importance_{years}y.png"
    if fig_imp.exists():
        st.image(str(fig_imp), caption=f"Ratios that matter most for the {years_label(years)} model")
    st.caption("This tool is for education and research. It is not financial advice.")
