import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SRC_DIR = os.path.join(BASE_DIR, "src")

CLUSTERED_PATH = os.path.join(OUTPUTS_DIR, "clustered_clients.csv")
EVAL_PATH = os.path.join(OUTPUTS_DIR, "k_evaluation.csv")


def _ensure_pipeline_outputs():
    if os.path.exists(CLUSTERED_PATH) and os.path.exists(EVAL_PATH):
        return
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    cwd = os.getcwd()
    os.chdir(BASE_DIR)
    try:
        from clustering import run_pipeline, name_clusters
        import joblib

        result, artifacts = run_pipeline(n_clusters=5)
        names = name_clusters(result)
        result["segment_name"] = result["kmeans_cluster"].map(names)
        result.to_csv(CLUSTERED_PATH, index=False)
        artifacts["eval_df"].to_csv(EVAL_PATH, index=False)
        joblib.dump(artifacts["scaler"], os.path.join(OUTPUTS_DIR, "scaler.joblib"))
        joblib.dump(artifacts["kmeans_model"], os.path.join(OUTPUTS_DIR, "kmeans_model.joblib"))
        joblib.dump(artifacts["feature_cols"], os.path.join(OUTPUTS_DIR, "feature_cols.joblib"))
    finally:
        os.chdir(cwd)


_ensure_pipeline_outputs()

st.set_page_config(page_title="Parcl Buyer Intelligence", page_icon="🏢", layout="wide")

SEGMENT_COLORS = {
    "Corporate Buyers": "#2563EB",
    "Loan-Backed Buyers": "#F59E0B",
    "Slow-Accumulating Buyers": "#10B981",
    "Fast-Accumulating Buyers": "#EF4444",
    "Large-Portfolio Buyers": "#8B5CF6",
}

COUNTRY_ISO3 = {
    "USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU",
    "France": "FRA", "Belgium": "BEL", "Mexico": "MEX",
    "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK",
}

# Plain-language descriptions, written for a non-technical reader.
SEGMENT_BLURBS = {
    "Corporate Buyers": {
        "emoji": "🏢",
        "one_liner": "Companies, not individuals, buying property.",
        "detail": "Every client here is registered as a company rather than a person. "
                   "They buy a similar number of units and spend about the same as everyone "
                   "else — the thing that sets them apart is simply *who* they are, not how "
                   "they shop.",
    },
    "Loan-Backed Buyers": {
        "emoji": "🏦",
        "one_liner": "Bought their properties using a loan.",
        "detail": "These clients financed their purchase rather than paying cash. They look "
                   "similar to other buyers in every other way — same typical portfolio size, "
                   "same typical spend — financing is the one thing that distinguishes them.",
    },
    "Slow-Accumulating Buyers": {
        "emoji": "🐢",
        "one_liner": "Built their property portfolio gradually, over a longer period.",
        "detail": "It took these clients longer, on average, to go from their first purchase "
                   "to their last. They're not buying more property than anyone else — just "
                   "taking their time doing it.",
    },
    "Fast-Accumulating Buyers": {
        "emoji": "🐇",
        "one_liner": "Built their property portfolio more quickly.",
        "detail": "These clients moved from their first purchase to their last in a shorter "
                   "window than average. Same typical portfolio size as everyone else — they "
                   "just moved faster.",
    },
    "Large-Portfolio Buyers": {
        "emoji": "📦",
        "one_liner": "Own noticeably more properties than a typical client.",
        "detail": "A small group (about 2.5% of clients) who own 6 or more properties — more "
                   "than double the typical client's 3-4. This is the most distinct group in "
                   "the data.",
    },
}


@st.cache_data
def load_data():
    df = pd.read_csv(CLUSTERED_PATH)
    df["first_purchase"] = pd.to_datetime(df["first_purchase"])
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    return df


@st.cache_data
def load_eval():
    return pd.read_csv(EVAL_PATH)


df = load_data()
eval_df = load_eval()

# ---------------------------------------------------------------- SIDEBAR --
st.sidebar.title("🏢 Parcl Buyer Intelligence")
st.sidebar.caption("Buyer segmentation for real estate market intelligence")
st.sidebar.divider()

st.sidebar.subheader("Filters")
countries = st.sidebar.multiselect("Country", sorted(df["country"].unique()), default=[])
client_types = st.sidebar.multiselect("Client Type", sorted(df["client_type"].unique()), default=[])
segments = st.sidebar.multiselect("Segment", sorted(df["segment_name"].unique()), default=[])

f = df.copy()
if countries:
    f = f[f["country"].isin(countries)]
if client_types:
    f = f[f["client_type"].isin(client_types)]
if segments:
    f = f[f["segment_name"].isin(segments)]

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(f):,}** of **{len(df):,}** clients")
if len(f) == 0:
    st.sidebar.warning("No clients match the current filters. Try clearing one.")

# ------------------------------------------------------------------ HEADER --
st.title("Buyer Segmentation & Investment Profiling")
st.caption("Five buyer groups, found by letting the data speak rather than assuming who buyers are.")

if len(f) == 0:
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Explore Segments", "🧪 Methodology"])

# ---------------------------------------------------------- TAB 1: OVERVIEW --
with tab1:
    k1, k2, k3 = st.columns(3)
    k1.metric("Clients analyzed", f"{len(f):,}")
    k2.metric("Buyer segments found", "5")
    k3.metric("Avg. properties per client", f"{f['num_properties'].mean():.1f}")

    st.divider()
    st.subheader("The five buyer groups")

    cards = st.columns(5)
    for col, (name, info) in zip(cards, SEGMENT_BLURBS.items()):
        n = (f["segment_name"] == name).sum()
        pct = n / len(f) * 100 if len(f) else 0
        with col:
            st.markdown(f"### {info['emoji']}")
            st.markdown(f"**{name}**")
            st.caption(f"{n:,} clients · {pct:.0f}%")
            st.write(info["one_liner"])

    st.divider()
    c1, c2 = st.columns([1, 1.3])
    with c1:
        dist = f["segment_name"].value_counts().reset_index()
        dist.columns = ["segment", "count"]
        fig = px.pie(
            dist, names="segment", values="count", hole=0.5,
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="How many clients are in each group?",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### What makes each group different")
        for name, info in SEGMENT_BLURBS.items():
            with st.expander(f"{info['emoji']}  {name}"):
                st.write(info["detail"])

# --------------------------------------------------- TAB 2: EXPLORE SEGMENTS --
with tab2:
    st.subheader("Pick a segment to explore")
    selected_segment = st.selectbox(
        "Segment", sorted(f["segment_name"].unique()),
        format_func=lambda s: f"{SEGMENT_BLURBS[s]['emoji']}  {s}",
    )
    seg_df = f[f["segment_name"] == selected_segment]
    info = SEGMENT_BLURBS[selected_segment]

    st.info(f"**{info['one_liner']}**  {info['detail']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clients in this group", f"{len(seg_df):,}")
    c2.metric("Typical portfolio size", f"{seg_df['num_properties'].median():.0f} properties")
    c3.metric("Typical purchase amount", f"${seg_df['avg_ticket_size'].median():,.0f}")

    st.divider()
    c4, c5 = st.columns(2)
    with c4:
        geo = seg_df.groupby("country").size().reset_index(name="clients")
        geo["iso3"] = geo["country"].map(COUNTRY_ISO3)
        fig_geo = px.choropleth(
            geo, locations="iso3", locationmode="ISO-3", color="clients",
            hover_name="country", color_continuous_scale="Blues",
            title="Where are they located?",
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with c5:
        type_counts = seg_df["client_type"].value_counts().reset_index()
        type_counts.columns = ["client_type", "count"]
        fig_type = px.bar(
            type_counts, x="client_type", y="count",
            title="Individuals vs. companies",
            color_discrete_sequence=[SEGMENT_COLORS.get(selected_segment, "#2563EB")],
        )
        st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("##### A few clients in this group")
    sample_cols = ["client_id", "client_type", "country", "num_properties", "avg_ticket_size"]
    nice_names = {"client_id": "Client", "client_type": "Type", "country": "Country",
                  "num_properties": "Properties Owned", "avg_ticket_size": "Avg. Purchase ($)"}
    st.dataframe(
        seg_df[sample_cols].head(15).rename(columns=nice_names),
        use_container_width=True, hide_index=True,
    )

# ----------------------------------------------------------- TAB 3: METHODOLOGY --
with tab3:
    st.subheader("How these segments were found")
    st.caption(
        "This section is for anyone who wants to check the methodology. "
        "It's not required reading to use the dashboard."
    )

    with st.expander("Why 5 segments, and not 4 or 6?", expanded=False):
        st.markdown(
            "Two methods were used to pick the number of segments: the **elbow method** "
            "(does adding another segment meaningfully improve the model?) and the "
            "**silhouette score** (how well-separated are the segments?). On top of that, "
            "a second clustering algorithm (Hierarchical clustering) was run independently "
            "to check whether it agreed with the main one (K-Means) — strong agreement "
            "between two different methods is good evidence the groups are real, not just "
            "an artifact of one algorithm's choices."
        )
        c1, c2 = st.columns(2)
        with c1:
            fig_elbow = px.line(eval_df, x="k", y="inertia", markers=True, title="Elbow Method")
            fig_elbow.add_vline(x=5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_elbow, use_container_width=True)
        with c2:
            fig_sil = px.line(eval_df, x="k", y="silhouette", markers=True, title="Silhouette Score")
            fig_sil.add_vline(x=5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_sil, use_container_width=True)
        st.markdown(
            "**5 segments (highlighted above) was chosen because it's the point where a "
            "genuinely distinct group — the Large-Portfolio Buyers — separates cleanly from "
            "everyone else, and where the two clustering algorithms agree most strongly "
            "(Adjusted Rand Index = 0.689, compared to 0.365 at 4 segments). Beyond 5 "
            "segments, the model mostly re-slices the same variable into finer bands rather "
            "than finding new structure, and the two algorithms agree less.**"
        )

    with st.expander("What was tested and didn't hold up", expanded=False):
        st.markdown(
            "The original project brief suggested buyers might group by age, stated "
            "purchase intent (investment vs. personal use), or satisfaction score. Each "
            "of these was tested directly against actual buyer behavior (spend, portfolio "
            "size, financing) using correlation analysis, chi-square tests, and t-tests."
        )
        test_results = pd.DataFrame({
            "Hypothesis": [
                "Younger clients are more likely to use a loan",
                "Investment-purpose buyers spend more",
                "Companies buy more properties / spend more",
                "Happier clients (higher satisfaction) spend more",
                "Country/region affects buying behavior",
            ],
            "Result": ["Not supported", "Not supported", "Not supported", "Not supported", "Not supported"],
            "Evidence": [
                "correlation ≈ -0.03",
                "correlation ≈ -0.02",
                "correlation ≈ 0.01",
                "correlation ≈ 0.00–0.07",
                "chi-square p = 0.43 (no significant link)",
            ],
        })
        st.dataframe(test_results, use_container_width=True, hide_index=True)
        st.markdown(
            "Instead, the segments above are built from what *did* hold up: whether a "
            "client is a company, whether they used a loan, and how many properties they "
            "bought and how quickly."
        )

    with st.expander("A bug we found and fixed along the way", expanded=False):
        st.markdown(
            "An earlier version of this analysis misread the purchase dates in the data "
            "(reading `MM-DD-YYYY` as `DD-MM-YYYY`). That bug made it look like a large "
            "group of clients bought their entire portfolio in about a week — a "
            "'Rapid Portfolio Investors' segment. After fixing the date format, that "
            "pattern disappeared; it was an artifact of the bug, not a real buyer type. "
            "This is mentioned here because catching and correcting it is part of the "
            "process, not something to hide."
        )

st.divider()
st.caption(
    "Built for Parcl Co. Limited × Unified Mentor — Machine Learning Based Buyer "
    "Segmentation and Investment Profiling for Real Estate Market Intelligence."
)