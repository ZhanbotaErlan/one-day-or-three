from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# The app.py file is inside crossmarket_signals.
# The results and figures folders are in the parent research folder.
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

st.set_page_config(
    page_title="CrossMarket Signals",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    main = pd.read_csv(RESULTS / "final_main_results.csv")
    thresholds = pd.read_csv(RESULTS / "final_threshold_results.csv")
    years = pd.read_csv(RESULTS / "final_year_results.csv")
    paths = pd.read_csv(RESULTS / "final_return_paths.csv")
    calm = pd.read_csv(RESULTS / "strict_calm_benchmark.csv")

    strict_events = pd.read_csv(
        BASE / "data" / "processed" / "market_comparison_strict.csv"
    )
    calendar_audit = pd.read_csv(
        BASE / "data" / "processed" / "market_comparison_calendar_audit.csv"
    )

    return main, thresholds, years, paths, calm, strict_events, calendar_audit
    

try:
    main, thresholds, years, paths, calm, strict_events, calendar_audit = load_data()
except FileNotFoundError as error:
    st.error(f"Could not find a results file: {error}")
    st.info(
        "Check that crossmarket_signals is inside market_research "
        "and that the results folder is beside it."
    )
    st.stop()


METRIC_LABELS = {
    "gap_day1": "Day 1 gap",
    "gap_day3": "Day 3 gap",
    "gap_change_day1_to_day3": "Change in gap, Day 1 to Day 3",
}

METRIC_ORDER = [
    "gap_day1",
    "gap_day3",
    "gap_change_day1_to_day3",
]


def get_main_row(metric):
    rows = main.loc[main["metric"] == metric]
    return rows.iloc[0]


def pp(value):
    return f"{value:+.3f} pp"


# ---------- Page styling ----------

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f6fa;
        color: #172033;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .hero {
        background: linear-gradient(120deg, #10233f, #183b5b 65%, #176b70);
        padding: 3rem 3.2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-label {
        color: #9edbd4;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: white;
        font-size: 3.1rem;
        margin: .6rem 0;
    }
    .hero p {
        color: #dce8f2;
        max-width: 780px;
        font-size: 1.05rem;
        line-height: 1.65;
    }
    .result-card {
        background: white;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 135px;
        box-shadow: 0 5px 18px rgba(25, 42, 70, .04);
    }
    .result-label {
        color: #69758a;
        font-size: .88rem;
    }
    .result-value {
        color: #14243b;
        font-size: 1.85rem;
        font-weight: 750;
        margin-top: .55rem;
    }
    .result-note {
        color: #748096;
        font-size: .77rem;
        margin-top: .55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Header ----------

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Independent market research · 2017–2025</div>
        <h1>One Day or Three?</h1>
        <p>
            Hong Kong and Kazakhstan equities after sharp declines
            in the U.S. stock market.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Explore local-market return paths, sell-off thresholds, "
    "year-by-year robustness, and the quiet-day benchmark."
)


# ---------- Overview and interactive return paths ----------

overview_tab, event_tab, audit_tab, threshold_tab, year_tab, calm_tab, methods_tab, materials_tab, sources_tab = st.tabs(
    [
        "Overview",
        "Event Explorer",
        "Calendar Audit",
        "Sell-off thresholds",
        "Year robustness",
        "Quiet-day benchmark",
        "Methodology",
        "Figures & Downloads",
        "Data & Code",
    ]
)

with overview_tab:
    st.subheader("Main findings")

    main_rows = [get_main_row(metric) for metric in METRIC_ORDER]
    columns = st.columns(3)

    for column, row in zip(columns, main_rows):
        with column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        {METRIC_LABELS[row["metric"]]}
                    </div>
                    <div class="result-value">
                        {row["estimate_pp"]:+.3f} pp
                    </div>
                    <div class="result-note">
                        95% cluster-bootstrap CI:
                        [{row["ci_lower_pp"]:+.3f},
                        {row["ci_upper_pp"]:+.3f}] pp
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.info(
        "The Day 1 average gap is positive. The confidence interval for "
        "the Day 1 to Day 3 change includes zero, so the analysis does not "
        "establish that the gap narrows."
    )

    st.markdown("### Mean cumulative return after a sell-off")

    selected_markets = st.multiselect(
        "Choose markets to display",
        options=sorted(paths["market"].unique()),
        default=sorted(paths["market"].unique()),
    )

    visible_paths = paths.loc[paths["market"].isin(selected_markets)]

    path_figure = px.line(
        visible_paths,
        x="local_session",
        y="mean_return_percent",
        color="market",
        markers=True,
        labels={
            "local_session": "Local trading session",
            "mean_return_percent": "Mean cumulative return (%)",
            "market": "Market",
        },
        hover_data={"events": True},
    )
    path_figure.update_layout(
        template="plotly_white",
        legend_title_text="",
        xaxis=dict(tickmode="array", tickvals=[1, 2, 3]),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(path_figure, use_container_width=True)

    st.caption(
        "Returns are cumulative from each market’s baseline close. "
        "The chart uses the 65-event calendar-filtered sample."
    )

    st.markdown("### Main estimates")
    main_table = main.copy()
    main_table["measure"] = main_table["metric"].map(METRIC_LABELS)
    st.dataframe(
        main_table[
            [
                "measure",
                "events",
                "clusters",
                "estimate_pp",
                "ci_lower_pp",
                "ci_upper_pp",
            ]
        ].rename(
            columns={
                "measure": "Measure",
                "events": "Events",
                "clusters": "Overlap groups",
                "estimate_pp": "Estimate (pp)",
                "ci_lower_pp": "95% CI lower (pp)",
                "ci_upper_pp": "95% CI upper (pp)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
# ---------- Event Explorer ----------

with event_tab:
    st.subheader("Explore an individual U.S. sell-off event")

    event_view = strict_events.copy()
    event_view["us_event_date"] = pd.to_datetime(event_view["us_event_date"])
    event_view["year"] = event_view["us_event_date"].dt.year

    available_years = sorted(event_view["year"].unique())
    year_filter = st.selectbox(
        "Filter by year",
        options=["All years"] + [str(year) for year in available_years],
        key="event_year_filter",
    )

    if year_filter != "All years":
        event_view = event_view.loc[
            event_view["year"] == int(year_filter)
        ]

    event_dates = (
        event_view["us_event_date"]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    selected_date = st.selectbox(
        "Select a U.S. sell-off date",
        options=event_dates,
        key="selected_event_date",
    )

    event = event_view.loc[
        event_view["us_event_date"].dt.strftime("%Y-%m-%d")
        == selected_date
    ].iloc[0]

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric(
        "S&P 500 daily return",
        f'{event["sp500_return_percent"]:.2f}%',
    )
    metric2.metric("Day 1 KASE–HSI gap", f'{event["gap_day1"] * 100:+.3f} pp')
    metric3.metric("Day 3 KASE–HSI gap", f'{event["gap_day3"] * 100:+.3f} pp')

    event_path = pd.DataFrame(
        [
            {
                "Market": "Hang Seng Index",
                "Local session": 0,
                "Cumulative return (%)": 0,
            },
            {
                "Market": "Hang Seng Index",
                "Local session": 1,
                "Cumulative return (%)": event["hsi_return_day1"] * 100,
            },
            {
                "Market": "Hang Seng Index",
                "Local session": 2,
                "Cumulative return (%)": event["hsi_return_day2"] * 100,
            },
            {
                "Market": "Hang Seng Index",
                "Local session": 3,
                "Cumulative return (%)": event["hsi_return_day3"] * 100,
            },
            {
                "Market": "KASE Index",
                "Local session": 0,
                "Cumulative return (%)": 0,
            },
            {
                "Market": "KASE Index",
                "Local session": 1,
                "Cumulative return (%)": event["kase_return_day1"] * 100,
            },
            {
                "Market": "KASE Index",
                "Local session": 2,
                "Cumulative return (%)": event["kase_return_day2"] * 100,
            },
            {
                "Market": "KASE Index",
                "Local session": 3,
                "Cumulative return (%)": event["kase_return_day3"] * 100,
            },
        ]
    )

    event_figure = px.line(
        event_path,
        x="Local session",
        y="Cumulative return (%)",
        color="Market",
        markers=True,
        labels={
            "Local session": "Local trading session",
            "Cumulative return (%)": "Cumulative return (%)",
        },
    )
    event_figure.update_layout(template="plotly_white")
    event_figure.update_xaxes(tickmode="array", tickvals=[0, 1, 2, 3])
    st.plotly_chart(event_figure, use_container_width=True)

    st.caption(
        f'HSI dates: baseline {event["hsi_baseline_date"]}, '
        f'Day 1 {event["hsi_day1_date"]}, Day 3 {event["hsi_day3_date"]}. '
        f'KASE dates: baseline {event["kase_baseline_date"]}, '
        f'Day 1 {event["kase_day1_date"]}, Day 3 {event["kase_day3_date"]}.'
    )

# ---------- Calendar Audit ----------

with audit_tab:
    st.subheader("How the calendar filter changes the event sample")

    audit_view = calendar_audit.copy()
    audit_view["passes_calendar_filter"] = (
        audit_view["passes_calendar_filter"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    kept = int(audit_view["passes_calendar_filter"].sum())
    excluded = len(audit_view) - kept

    count1, count2, count3 = st.columns(3)
    count1.metric("Candidate events", len(audit_view))
    count2.metric("Retained events", kept)
    count3.metric("Excluded events", excluded)

    audit_choice = st.selectbox(
        "Show events",
        ["All events", "Retained events", "Excluded events"],
        key="audit_filter",
    )

    if audit_choice == "Retained events":
        audit_view = audit_view.loc[
            audit_view["passes_calendar_filter"]
        ]
    elif audit_choice == "Excluded events":
        audit_view = audit_view.loc[
            ~audit_view["passes_calendar_filter"]
        ]

    audit_dates = audit_view["us_event_date"].astype(str).tolist()
    selected_audit_date = st.selectbox(
        "Select an event",
        audit_dates,
        key="selected_audit_date",
    )

    audit_event = audit_view.loc[
        audit_view["us_event_date"].astype(str) == selected_audit_date
    ].iloc[0]

    st.write(
        f'For this event, the filter counted '
        f'{int(audit_event["hsi_us_sessions"])} U.S. session(s) '
        f'for HSI and {int(audit_event["kase_us_sessions"])} '
        f'for KASE before Day 1.'
    )

    st.write(
        "The event is "
        + (
            "retained."
            if audit_event["passes_calendar_filter"]
            else "excluded because the session-count rule is not met in both markets."
        )
    )

    audit_columns = [
        "us_event_date",
        "sp500_return_percent",
        "hsi_baseline_date",
        "hsi_day1_date",
        "hsi_us_sessions",
        "kase_baseline_date",
        "kase_day1_date",
        "kase_us_sessions",
        "passes_calendar_filter",
    ]

    st.dataframe(
        audit_view[audit_columns].rename(
            columns={
                "us_event_date": "U.S. event date",
                "sp500_return_percent": "S&P 500 return (%)",
                "hsi_baseline_date": "HSI baseline",
                "hsi_day1_date": "HSI Day 1",
                "hsi_us_sessions": "U.S. sessions before HSI Day 1",
                "kase_baseline_date": "KASE baseline",
                "kase_day1_date": "KASE Day 1",
                "kase_us_sessions": "U.S. sessions before KASE Day 1",
                "passes_calendar_filter": "Retained",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

# ---------- Threshold sensitivity ----------

with threshold_tab:
    st.subheader("Does the result depend on the sell-off threshold?")

    threshold_metrics = st.selectbox(
        "Choose the estimate",
        options=METRIC_ORDER,
        format_func=lambda metric: METRIC_LABELS[metric],
        key="threshold_metric",
    )

    threshold_data = thresholds.loc[
        thresholds["metric"] == threshold_metrics
    ].sort_values("threshold_percent")

    threshold_figure = go.Figure()
    threshold_figure.add_trace(
        go.Scatter(
            x=threshold_data["threshold_percent"].astype(str) + "%",
            y=threshold_data["estimate_pp"],
            mode="markers+lines",
            name=METRIC_LABELS[threshold_metrics],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(
                    threshold_data["ci_upper_pp"]
                    - threshold_data["estimate_pp"]
                ),
                arrayminus=(
                    threshold_data["estimate_pp"]
                    - threshold_data["ci_lower_pp"]
                ),
                visible=True,
            ),
            marker=dict(size=10, color="#16877d"),
            line=dict(color="#16877d", width=3),
        )
    )
    threshold_figure.add_hline(y=0, line_dash="dash", line_color="#778399")
    threshold_figure.update_layout(
        template="plotly_white",
        xaxis_title="S&P 500 sell-off threshold",
        yaxis_title="Estimate (percentage points)",
        showlegend=False,
    )
    st.plotly_chart(threshold_figure, use_container_width=True)

    st.write(
        "The Day 1 estimate is positive across all three thresholds. "
        "The Day 3 and change estimates have confidence intervals that "
        "include zero."
    )

    st.dataframe(
        thresholds.rename(
            columns={
                "threshold_percent": "Threshold (%)",
                "metric": "Measure",
                "events": "Events",
                "clusters": "Overlap groups",
                "estimate_pp": "Estimate (pp)",
                "ci_lower_pp": "95% CI lower (pp)",
                "ci_upper_pp": "95% CI upper (pp)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ---------- Leave-one-year-out robustness ----------

with year_tab:
    st.subheader("Leave-one-year-out: Day 1 gap")

    year_data = years.loc[years["metric"] == "gap_day1"].sort_values(
        "excluded_year"
    )

    year_figure = go.Figure()
    year_figure.add_trace(
        go.Scatter(
            x=year_data["excluded_year"].astype(str),
            y=year_data["estimate_pp"],
            mode="markers",
            name="Day 1 gap",
            error_y=dict(
                type="data",
                symmetric=False,
                array=year_data["ci_upper_pp"] - year_data["estimate_pp"],
                arrayminus=year_data["estimate_pp"] - year_data["ci_lower_pp"],
                visible=True,
            ),
            marker=dict(size=10, color="#21618c"),
        )
    )
    year_figure.add_hline(y=0, line_dash="dash", line_color="#778399")
    year_figure.update_layout(
        template="plotly_white",
        xaxis_title="Excluded year",
        yaxis_title="Day 1 gap (percentage points)",
        showlegend=False,
    )
    st.plotly_chart(year_figure, use_container_width=True)

    st.write(
        "The Day 1 estimate remains positive after excluding each year "
        "in turn. The estimates vary across exclusions."
    )

    st.dataframe(
        year_data[
            [
                "excluded_year",
                "events",
                "clusters",
                "estimate_pp",
                "ci_lower_pp",
                "ci_upper_pp",
            ]
        ].rename(
            columns={
                "excluded_year": "Excluded year",
                "events": "Events",
                "clusters": "Overlap groups",
                "estimate_pp": "Day 1 gap (pp)",
                "ci_lower_pp": "95% CI lower (pp)",
                "ci_upper_pp": "95% CI upper (pp)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ---------- Quiet-day benchmark ----------

with calm_tab:
    st.subheader("Sell-off days compared with quiet days")

    available_metrics = list(calm["metric"].unique())
    selected_calm_metric = st.selectbox(
        "Choose benchmark estimate",
        options=available_metrics,
        key="calm_metric",
    )

    calm_data = calm.loc[calm["metric"] == selected_calm_metric].sort_values(
        "block_length"
    )

    block_options = sorted(calm_data["block_length"].unique())
    selected_block = st.select_slider(
        "Circular block length (U.S. trading sessions)",
        options=block_options,
        value=20 if 20 in block_options else block_options[0],
    )

    selected_row = calm_data.loc[
        calm_data["block_length"] == selected_block
    ].iloc[0]

        # Compare sell-off and quiet-day gap levels side by side.
    main_d1 = get_main_row("gap_day1")
    main_d3 = get_main_row("gap_day3")
    main_change = get_main_row("gap_change_day1_to_day3")

    def get_excess(metric_fragment):
        matches = calm.loc[
            (calm["block_length"] == selected_block)
            & calm["metric"].str.contains(
                metric_fragment,
                case=False,
                regex=False,
            )
        ]

        if matches.empty:
            st.error(
                f'Could not find "{metric_fragment}". '
                f'Available metrics: {", ".join(calm["metric"].unique())}'
            )
            st.stop()

        return matches.iloc[0]

    excess_d1 = get_excess("Day 1 excess")
    excess_d3 = get_excess("Day 3 excess")
    excess_change = get_excess("change")
    benchmark_comparison = pd.DataFrame(
        [
            {
                "Horizon": "Day 1",
                "Sell-off gap (pp)": main_d1["estimate_pp"],
                "Weighted quiet-day gap (pp)": (
                    main_d1["estimate_pp"] - excess_d1["estimate_pp"]
                ),
                "Excess gap (pp)": excess_d1["estimate_pp"],
                "Excess 95% CI": (
                    f'[{excess_d1["ci_lower_pp"]:+.3f}, '
                    f'{excess_d1["ci_upper_pp"]:+.3f}]'
                ),
            },
            {
                "Horizon": "Day 3",
                "Sell-off gap (pp)": main_d3["estimate_pp"],
                "Weighted quiet-day gap (pp)": (
                    main_d3["estimate_pp"] - excess_d3["estimate_pp"]
                ),
                "Excess gap (pp)": excess_d3["estimate_pp"],
                "Excess 95% CI": (
                    f'[{excess_d3["ci_lower_pp"]:+.3f}, '
                    f'{excess_d3["ci_upper_pp"]:+.3f}]'
                ),
            },
            {
                "Horizon": "Change, Day 1 to Day 3",
                "Sell-off gap (pp)": main_change["estimate_pp"],
                "Weighted quiet-day gap (pp)": (
                    main_change["estimate_pp"]
                    - excess_change["estimate_pp"]
                ),
                "Excess gap (pp)": excess_change["estimate_pp"],
                "Excess 95% CI": (
                    f'[{excess_change["ci_lower_pp"]:+.3f}, '
                    f'{excess_change["ci_upper_pp"]:+.3f}]'
                ),
            },
        ]
    )

    st.markdown("### Sell-off and quiet-day gaps")
    st.caption(
        "The quiet-day benchmark uses 672 eligible observations and is "
        "weighted to match the sell-off events by year."
    )
    st.dataframe(
        benchmark_comparison,
        hide_index=True,
        use_container_width=True,
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Estimated excess gap", f'{selected_row["estimate_pp"]:+.3f} pp')
    k2.metric("95% CI lower", f'{selected_row["ci_lower_pp"]:+.3f} pp')
    k3.metric("95% CI upper", f'{selected_row["ci_upper_pp"]:+.3f} pp')

    calm_figure = go.Figure()
    calm_figure.add_trace(
        go.Scatter(
            x=calm_data["block_length"],
            y=calm_data["estimate_pp"],
            mode="markers+lines",
            error_y=dict(
                type="data",
                symmetric=False,
                array=calm_data["ci_upper_pp"] - calm_data["estimate_pp"],
                arrayminus=calm_data["estimate_pp"] - calm_data["ci_lower_pp"],
                visible=True,
            ),
            marker=dict(size=10, color="#a45a52"),
            line=dict(color="#a45a52", width=3),
        )
    )
    calm_figure.add_hline(y=0, line_dash="dash", line_color="#778399")
    calm_figure.update_layout(
        template="plotly_white",
        xaxis_title="Block length (U.S. trading sessions)",
        yaxis_title="Excess gap (percentage points)",
        showlegend=False,
    )
    st.plotly_chart(calm_figure, use_container_width=True)

    st.caption(
        "The benchmark is weighted by the yearly distribution of sell-off "
        "events. Block lengths of 10, 20, and 40 sessions are shown."
    )

    st.dataframe(
        calm.rename(
            columns={
                "block_length": "Block length",
                "metric": "Measure",
                "estimate_pp": "Estimate (pp)",
                "ci_lower_pp": "95% CI lower (pp)",
                "ci_upper_pp": "95% CI upper (pp)",
                "valid_repetitions": "Valid repetitions",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ---------- Methodology and formulas ----------

with methods_tab:
    st.subheader("Methodology and formulas")

    with st.expander("1. U.S. daily return and sell-off definition", expanded=True):
        st.latex(
            r"r^{US}_t=100\left(\frac{P^{US}_t}{P^{US}_{t-1}}-1\right)"
        )
        st.latex(r"\text{Sell-off event if }r^{US}_t\leq -2\%")

    with st.expander("2. Calendar filter"):
        st.latex(
            r"C_{m,i}=\sum_t\mathbf{1}(b_{m,i}\leq t<a_{m,i})"
        )
        st.latex(
            r"\text{Retain }i\text{ if }"
            r"C_{\mathrm{HSI},i}=C_{\mathrm{KASE},i}=1"
        )
        st.caption(
            "The filter retains 65 of 85 candidate events. "
            "These form 47 overlapping-window groups."
        )

    with st.expander("3. Cumulative returns and KASE–HSI gap"):
        st.latex(
            r"R_{m,i,h}=100\left(\frac{P_{m,i,h}}{P_{m,i,0}}-1\right),"
            r"\qquad h\in\{1,3\}"
        )
        st.latex(
            r"D_{i,h}=R_{\mathrm{KASE},i,h}-R_{\mathrm{HSI},i,h}"
        )
        st.latex(
            r"\Delta D_i=D_{i,3}-D_{i,1},"
            r"\qquad \overline{\Delta D}=\frac{1}{N}\sum_i\Delta D_i"
        )
        st.caption(
            "Returns are percentages; gaps and changes in gaps are "
            "percentage points (pp)."
        )

    with st.expander("4. Quiet-day benchmark"):
        st.latex(r"\text{Quiet day if }|r^{US}_t|\leq 0.5\%")
        st.latex(
            r"\overline{D}^{\,Q}_h=\sum_yw_y\overline{D}^{\,Q}_{y,h},"
            r"\qquad w_y=\frac{n_y}{\sum_y n_y}"
        )
        st.latex(
            r"E_h=\overline{D}^{\,S}_h-\overline{D}^{\,Q}_h,"
            r"\qquad \Delta E=E_3-E_1"
        )

    with st.expander("5. Resampling and sensitivity checks"):
        st.write(
            "Main event-gap intervals use a cluster bootstrap of the "
            "47 overlapping-window groups with 10,000 repetitions. "
            "Quiet-day benchmark intervals use a circular block bootstrap "
            "with a main block length of 20 U.S. sessions; lengths 10 and "
            "40 are sensitivity checks."
        )
        st.write(
            "Threshold sensitivity uses S&P 500 return cutoffs of "
            "−1.5%, −2.0%, and −2.5%. Leave-one-year-out estimates "
            "recalculate the Day 1 gap after excluding each year."
        )

    st.warning(
        "These are conditional comparisons of local-currency price-index "
        "returns. They do not establish causality or represent an investment strategy."
    )
# ---------- Figures and downloads ----------

with materials_tab:
    st.subheader("Research figures")

    figure_details = [
        (
            "Figure 1 · Mean cumulative return paths",
            "figure_1_return_paths",
            "Average cumulative returns for the two local indices after sell-off events.",
        ),
        (
            "Figure 2 · Return gaps and confidence intervals",
            "figure_2_gaps_and_intervals",
            "KASE–HSI gaps and quiet-day-adjusted gaps at Day 1, Day 3, and the change.",
        ),
        (
            "Figure 3 · Sell-off threshold sensitivity",
            "figure_3_threshold_sensitivity",
            "Day 1 estimates across the three sell-off thresholds.",
        ),
    ]

    for title, filename, description in figure_details:
        with st.expander(title):
            st.write(description)

            png_path = FIGURES / f"{filename}.png"
            pdf_path = FIGURES / f"{filename}.pdf"

            if png_path.exists():
                st.image(str(png_path), use_container_width=True)
                with open(png_path, "rb") as file:
                    st.download_button(
                        "Download PNG",
                        data=file.read(),
                        file_name=f"{filename}.png",
                        mime="image/png",
                        key=f"{filename}_png",
                    )

            if pdf_path.exists():
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        "Download PDF",
                        data=file.read(),
                        file_name=f"{filename}.pdf",
                        mime="application/pdf",
                        key=f"{filename}_pdf",
                    )

    st.divider()
    st.subheader("Download result data")
    st.caption("These CSV files contain the values used in the website’s tables and charts.")

    downloads = [
        ("Main results", main, "final_main_results.csv"),
        ("Threshold sensitivity", thresholds, "final_threshold_results.csv"),
        ("Leave-one-year-out results", years, "final_year_results.csv"),
        ("Mean return paths", paths, "final_return_paths.csv"),
        ("Quiet-day benchmark", calm, "strict_calm_benchmark.csv"),
        ("Calendar-filtered events", strict_events, "market_comparison_strict.csv"),
        ("Calendar audit", calendar_audit, "market_comparison_calendar_audit.csv"),
    ]

    for label, dataframe, filename in downloads:
        st.download_button(
            label=f"Download {label}",
            data=dataframe.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
        ) 

# ---------- Data and reproducibility ----------

with sources_tab:
    st.subheader("Data sources")

    st.markdown(
        """
        - **S&P 500:** FRED daily series, used to identify U.S. sell-off events.
        - **Hang Seng Index:** Yahoo Finance historical index data.
        - **KASE Index:** Kazakhstan Stock Exchange historical index archive.
        """
    )

    st.caption(
        "The analysis uses local-currency price-index values. "
        "It does not include dividends or convert returns into a common currency."
    )

    st.divider()
    st.subheader("Analysis scripts")
    st.write(
        "Download the Python scripts used to prepare the data, construct "
        "event samples, estimate results, and create figures."
    )

    script_names = [
        "download_data.py",
        "identify_events.py",
        "clean_kase_data.py",
        "match_hsi_events.py",
        "match_kase_events.py",
        "audit_event_windows.py",
        "compare_markets.py",
        "bootstrap_confidence_intervals.py",
        "threshold_robustness.py",
        "robustness_by_year.py",
        "strict_calm_benchmark.py",
        "finalize_analysis.py",
    ]

    for script_name in script_names:
        script_path = BASE / script_name
        if script_path.exists():
            st.download_button(
                label=f"Download {script_name}",
                data=script_path.read_bytes(),
                file_name=script_name,
                mime="text/x-python",
                key=f"script_{script_name}",
            )        