"""
app.py — Web Traffic Forecasting Dashboard
Multi-section Streamlit app with 60-30-10 color rule, timeline controls,
festival spike analysis, and ML forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

from data_loader import (
    load_amazon, load_web_traffic, load_web_traffic_2026,
    load_ride_bookings, load_shoppers,
    aggregate_amazon_daily, aggregate_web_daily, aggregate_rides_daily,
    compute_festival_lift, get_festivals_in_range, FESTIVAL_DATES,
)
from forecaster import forecast_prophet, forecast_arima, seasonal_decompose

# ========================================================================
# Page config & custom CSS
# ========================================================================
st.set_page_config(
    page_title="Traffic Forecasting Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 60-30-10 palette
# 60 %  — Background:   #0E1117 (Streamlit dark default)
# 30 %  — Surface/cards: #1A1D23
# 10 %  — Accent:        #4FC3F7 (light blue)
ACCENT = "#4FC3F7"
SURFACE = "#1A1D23"
TEXT_PRIMARY = "#E0E0E0"
TEXT_SECONDARY = "#9E9E9E"
POSITIVE = "#66BB6A"
NEGATIVE = "#EF5350"
FESTIVAL_BAND = "rgba(79, 195, 247, 0.08)"

st.markdown(f"""
<style>
    /* --- 60-30-10 tokens --- */
    :root {{
        --accent: {ACCENT};
        --surface: {SURFACE};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--surface);
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid #2A2D33;
        border-radius: 8px;
        padding: 16px 20px;
    }}
    div[data-testid="stMetric"] label {{
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-weight: 600;
    }}

    /* Section headers */
    .section-header {{
        color: var(--accent);
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid var(--accent);
        display: inline-block;
    }}

    /* Page title */
    .page-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }}
    .page-subtitle {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }}

    /* KPI row inside festival cards */
    .festival-card {{
        background: var(--surface);
        border-left: 3px solid var(--accent);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .festival-card h4 {{
        margin: 0 0 4px 0;
        color: var(--text-primary);
        font-size: 0.95rem;
    }}
    .lift-positive {{ color: {POSITIVE}; font-weight: 600; }}
    .lift-negative {{ color: {NEGATIVE}; font-weight: 600; }}

    /* Tab styling */
    button[data-baseweb="tab"] {{
        font-size: 0.85rem !important;
    }}

    /* Hide Streamlit branding */
    #MainMenu, footer {{ visibility: hidden; }}

    /* Chart descriptions */
    .chart-desc {{
        color: var(--text-secondary);
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: -8px;
        margin-bottom: 1.2rem;
        padding: 10px 14px;
        background: var(--surface);
        border-radius: 6px;
        border-left: 2px solid #2A2D33;
    }}
</style>
""", unsafe_allow_html=True)


# ========================================================================
# Plotly layout defaults (dark, no gradient)
# ========================================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRIMARY, size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor="#2A2D33", zeroline=False),
    yaxis=dict(gridcolor="#2A2D33", zeroline=False),
)


def apply_layout(fig, height=420, **kwargs):
    layout = {**PLOTLY_LAYOUT, "height": height, **kwargs}
    fig.update_layout(**layout)
    return fig


def add_festival_bands(fig, start_date, end_date):
    """Add shaded vertical bands for festivals in the visible range."""
    fests = get_festivals_in_range(start_date, end_date)
    for name, fs, fe in fests:
        fig.add_vrect(
            x0=fs, x1=fe,
            fillcolor=FESTIVAL_BAND,
            line_width=0,
            annotation_text=name,
            annotation_position="top left",
            annotation=dict(font_size=9, font_color=TEXT_SECONDARY),
        )
    return fig


# ========================================================================
# Data loading (cached)
# ========================================================================
@st.cache_data(show_spinner=False)
def get_amazon():
    return load_amazon()

@st.cache_data(show_spinner=False)
def get_web_traffic():
    return load_web_traffic()

@st.cache_data(show_spinner=False)
def get_web_2026():
    return load_web_traffic_2026()

@st.cache_data(show_spinner=False)
def get_rides():
    return load_ride_bookings()



# ========================================================================
# Sidebar — Navigation & Global controls
# ========================================================================
st.sidebar.markdown(f'<div class="page-title">Traffic Forecast</div>', unsafe_allow_html=True)
st.sidebar.markdown(f'<div class="page-subtitle">Web Traffic Forecasting using Time Series</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    [
        "E-Commerce Analysis",
        "Web Traffic Analysis",
        "Ride Bookings Analysis",
        "Forecasting Lab",

    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")


# ========================================================================
# Helper — timeline range controls (reused per section)
# ========================================================================
def timeline_control(key_prefix, min_date, max_date, default_start=None, default_end=None):
    """Render a date range selector and return (start, end) as date objects."""
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input(
            "From", value=default_start or min_date,
            min_value=min_date, max_value=max_date, key=f"{key_prefix}_start",
        )
    with c2:
        end = st.date_input(
            "To", value=default_end or max_date,
            min_value=min_date, max_value=max_date, key=f"{key_prefix}_end",
        )
    return start, end



# ========================================================================
# PAGE 2 — E-Commerce Deep Dive
# ========================================================================
if page == "E-Commerce Analysis":
    st.markdown('<div class="page-title">E-Commerce Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Amazon sales trends, category breakdown, and festival spikes</div>', unsafe_allow_html=True)

    amz = get_amazon()

    # Sidebar filters
    categories = ["All"] + sorted(amz["Category"].dropna().unique().tolist())
    sel_cat = st.sidebar.selectbox("Category", categories, key="amz_cat")
    if sel_cat != "All":
        amz = amz[amz["Category"] == sel_cat]


    amz_daily = aggregate_amazon_daily(amz)
    mn = amz_daily["date"].min().date()
    mx = amz_daily["date"].max().date()

    # --- Timeline ---
    st.markdown('<div class="section-header">Sales Over Time</div>', unsafe_allow_html=True)
    ts, te = timeline_control("amz_sales", mn, mx)
    mask = (amz_daily["date"].dt.date >= ts) & (amz_daily["date"].dt.date <= te)
    filtered = amz_daily[mask].copy()

    tab_rev, tab_orders, tab_qty = st.tabs(["Revenue", "Orders", "Quantity"])

    for tab, col_name, label in [
        (tab_rev, "revenue", "Revenue ($)"),
        (tab_orders, "orders", "Orders"),
        (tab_qty, "quantity", "Quantity"),
    ]:
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered["date"], y=filtered[col_name],
                mode="lines", name=label,
                line=dict(color=ACCENT, width=1.5),
            ))
            if len(filtered) >= 7:
                filtered[f"ma7_{col_name}"] = filtered[col_name].rolling(7).mean()
                fig.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[f"ma7_{col_name}"],
                    mode="lines", name="7-Day Avg",
                    line=dict(color="#FFFFFF", width=2),
                ))
            add_festival_bands(fig, ts, te)
            apply_layout(fig, height=380, xaxis_title="Date", yaxis_title=label)
            st.plotly_chart(fig, use_container_width=True)

            # Dynamic description per tab
            if len(filtered) > 0:
                avg_v = filtered[col_name].mean()
                peak_v = filtered[col_name].max()
                filter_ctx = f" for {sel_cat}" if sel_cat != "All" else ""
                peak_d = filtered.loc[filtered[col_name].idxmax(), "date"].strftime("%b %d, %Y")
                st.markdown(f'<div class="chart-desc">{label}{filter_ctx}: Average <b>{avg_v:,.0f}</b> per day, '
                            f'peak of <b>{peak_v:,.0f}</b> on {peak_d} across the selected {(te - ts).days}-day window. '
                            f'The white line smooths daily noise using a 7-day rolling mean.</div>',
                            unsafe_allow_html=True)

    st.markdown("---")

    # --- Category Breakdown ---

        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        cat_df = amz.groupby("Category")["TotalAmount"].sum().reset_index()
        cat_df = cat_df.sort_values("TotalAmount", ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            x=cat_df["TotalAmount"], y=cat_df["Category"],
            orientation="h", marker_color=ACCENT,
        ))
        apply_layout(fig, height=360, xaxis_title="Revenue ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        top_cat = cat_df.iloc[-1]["Category"] if len(cat_df) > 0 else "N/A"
        st.markdown(f'<div class="chart-desc">Top revenue-generating category: <b>{top_cat}</b>. '
                    f'</div>',
                    unsafe_allow_html=True)



    # --- Monthly Heatmap ---
    st.markdown('<div class="section-header">Monthly Revenue Heatmap</div>', unsafe_allow_html=True)
    amz_copy = amz.copy()
    amz_copy["year"] = amz_copy["date"].dt.year
    amz_copy["month"] = amz_copy["date"].dt.month
    heat_df = amz_copy.groupby(["year", "month"])["TotalAmount"].sum().reset_index()
    heat_pivot = heat_df.pivot(index="year", columns="month", values="TotalAmount").fillna(0)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=month_labels[:heat_pivot.shape[1]],
        y=heat_pivot.index.astype(str),
        colorscale=[[0, SURFACE], [1, ACCENT]],
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Revenue: $%{z:,.0f}<extra></extra>",
    ))
    apply_layout(fig, height=280, xaxis_title="Month", yaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="chart-desc">Color intensity represents total monthly revenue. '
                'Brighter cells (Oct-Nov) typically correspond to Diwali/Navratri festival season, '
                'the strongest retail period in India.</div>', unsafe_allow_html=True)

    # --- Festival Lift ---
    st.markdown('<div class="section-header">Festival Impact</div>', unsafe_allow_html=True)
    lift_df = compute_festival_lift(amz_daily, "revenue")
    if not lift_df.empty:
        fig = go.Figure(go.Bar(
            x=lift_df["festival"], y=lift_df["lift_pct"],
            marker_color=[POSITIVE if v > 0 else NEGATIVE for v in lift_df["lift_pct"]],
            text=[f"{v:+.1f}%" for v in lift_df["lift_pct"]],
            textposition="outside",
        ))
        apply_layout(fig, height=340, xaxis_title="", yaxis_title="% Lift vs Baseline")
        st.plotly_chart(fig, use_container_width=True)
        best = lift_df.iloc[0]
        st.markdown(f'<div class="chart-desc">Strongest festival: <b>{best["festival"]}</b> with '
                    f'<b>{best["lift_pct"]:+.1f}%</b> lift over the baseline of ${best["baseline"]:,.0f}/day. '
                    f'Green bars = positive impact, red = negative. Lift is computed by comparing the full festival '
                    f'week average against non-festival daily average.</div>', unsafe_allow_html=True)


# ========================================================================
# PAGE 3 — Web Traffic Analysis
# ========================================================================
elif page == "Web Traffic Analysis":
    st.markdown('<div class="page-title">Web Traffic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Visit trends, traffic sources, devices, and conversions</div>', unsafe_allow_html=True)

    web = get_web_traffic()
    web_daily = aggregate_web_daily(web)
    mn = web_daily["date"].min().date()
    mx = web_daily["date"].max().date()

    # --- Timeline ---
    st.markdown('<div class="section-header">Daily Visits</div>', unsafe_allow_html=True)
    ts, te = timeline_control("web_visits", mn, mx)
    mask = (web_daily["date"].dt.date >= ts) & (web_daily["date"].dt.date <= te)
    filtered = web_daily[mask].copy()

    tab_visits, tab_bounce, tab_conv = st.tabs(["Visits", "Bounce Rate", "Conversion Rate"])

    for tab, col, label in [
        (tab_visits, "visits", "Visits"),
        (tab_bounce, "bounce_rate", "Bounce Rate (%)"),
        (tab_conv, "conversion_rate", "Conversion Rate (%)"),
    ]:
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered["date"], y=filtered[col],
                mode="lines", name=label,
                line=dict(color=ACCENT, width=1.5),
            ))
            if len(filtered) >= 7:
                filtered[f"ma7_{col}"] = filtered[col].rolling(7).mean()
                fig.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[f"ma7_{col}"],
                    mode="lines", name="7-Day Avg",
                    line=dict(color="#FFFFFF", width=2),
                ))
            add_festival_bands(fig, ts, te)
            apply_layout(fig, height=380, xaxis_title="Date", yaxis_title=label)
            st.plotly_chart(fig, use_container_width=True)

            # Dynamic description
            if len(filtered) > 0:
                avg_v = filtered[col].mean()
                st.markdown(f'<div class="chart-desc">{label} over {(te - ts).days} days. '
                            f'Average: <b>{avg_v:,.1f}</b>. The 7-day moving average (white) reveals the underlying trend '
                            f'by smoothing out daily volatility.</div>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Traffic Source Breakdown</div>', unsafe_allow_html=True)
        # Filter by timeline
        web_filt = web[(web["date"].dt.date >= ts) & (web["date"].dt.date <= te)]
        src = web_filt.groupby("traffic_source").size().reset_index(name="count")
        fig = go.Figure(go.Pie(
            labels=src["traffic_source"], values=src["count"],
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Set2),
        ))
        apply_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
        top_src = src.sort_values("count", ascending=False).iloc[0]["traffic_source"] if len(src) > 0 else "N/A"
        st.markdown(f'<div class="chart-desc">Dominant traffic source: <b>{top_src}</b>. '
                    f'Organic search = Google/Bing, Social = social media platforms, Direct = typed URL.</div>',
                    unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Device Distribution</div>', unsafe_allow_html=True)
        dev = web_filt.groupby("device_type").size().reset_index(name="count")
        fig = go.Figure(go.Pie(
            labels=dev["device_type"], values=dev["count"],
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=[ACCENT, "#B0BEC5", "#78909C"]),
        ))
        apply_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
        top_dev = dev.sort_values("count", ascending=False).iloc[0]["device_type"] if len(dev) > 0 else "N/A"
        st.markdown(f'<div class="chart-desc">Most-used device: <b>{top_dev}</b>. '
                    f'High mobile share indicates a need for mobile-first website design.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top Domains by Monthly Visits</div>', unsafe_allow_html=True)
    web2026 = get_web_2026()
    top_n = st.slider("Show top N domains", 5, 30, 15, key="top_domains")
    top = web2026.nlargest(top_n, "monthly_visits")[["global_rank", "domain", "category", "monthly_visits", "bounce_rate_pct"]]
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.markdown(f'<div class="chart-desc">Showing the top {top_n} globally-ranked domains by monthly visits '
                f'from the 2026 dataset. Bounce rate indicates the % of single-page sessions.</div>',
                unsafe_allow_html=True)


# ========================================================================
# PAGE 4 — Ride Bookings Analysis
# ========================================================================
elif page == "Ride Bookings Analysis":
    st.markdown('<div class="page-title">Ride Bookings Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">NCR ride demand, vehicle types, and booking trends</div>', unsafe_allow_html=True)

    rides = get_rides()
    rides_daily = aggregate_rides_daily(rides)
    mn = rides_daily["date"].min().date()
    mx = rides_daily["date"].max().date()

    # --- KPIs ---
    rk1, rk2, rk3 = st.columns(3)
    completed = rides[rides["Booking Status"] == "Completed"]
    rk1.metric("Total Bookings", f"{len(rides):,}")
    rk2.metric("Completed Rides", f"{len(completed):,}")
    rk3.metric("Avg Booking Value", f"Rs {rides['Booking Value'].mean():,.0f}")

    # --- Timeline ---
    st.markdown('<div class="section-header">Daily Bookings Trend</div>', unsafe_allow_html=True)
    ts, te = timeline_control("ride_trend", mn, mx)
    mask = (rides_daily["date"].dt.date >= ts) & (rides_daily["date"].dt.date <= te)
    filtered = rides_daily[mask].copy()

    tab_book, tab_val = st.tabs(["Bookings Count", "Booking Value"])
    for tab, col, label in [
        (tab_book, "bookings", "Bookings"),
        (tab_val, "booking_value", "Revenue (Rs)"),
    ]:
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered["date"], y=filtered[col],
                mode="lines", name=label, line=dict(color=ACCENT, width=1.5),
            ))
            if len(filtered) >= 7:
                filtered[f"ma7_{col}"] = filtered[col].rolling(7).mean()
                fig.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[f"ma7_{col}"],
                    mode="lines", name="7-Day Avg", line=dict(color="#FFFFFF", width=2),
                ))
            add_festival_bands(fig, ts, te)
            apply_layout(fig, height=380, xaxis_title="Date", yaxis_title=label)
            st.plotly_chart(fig, use_container_width=True)

            # Dynamic description
            if len(filtered) > 0:
                avg_v = filtered[col].mean()
                st.markdown(f'<div class="chart-desc">{label} over {(te - ts).days} days in Delhi-NCR. '
                            f'Average: <b>{avg_v:,.0f}</b>/day. Festival bands show periods of expected '
                            f'higher ride demand due to celebrations and travel.</div>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Vehicle Type Breakdown</div>', unsafe_allow_html=True)
        vt = rides.groupby("Vehicle Type").size().reset_index(name="count")
        fig = go.Figure(go.Bar(
            x=vt["count"], y=vt["Vehicle Type"],
            orientation="h", marker_color=ACCENT,
        ))
        apply_layout(fig, height=340, xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        top_vt = vt.sort_values("count", ascending=False).iloc[0]["Vehicle Type"] if len(vt) > 0 else "N/A"
        st.markdown(f'<div class="chart-desc">Most popular vehicle type: <b>{top_vt}</b>. '
                    f'Distribution reflects rider preferences in the Delhi-NCR region.</div>',
                    unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Booking Status Distribution</div>', unsafe_allow_html=True)
        bs = rides.groupby("Booking Status").size().reset_index(name="count")
        fig = go.Figure(go.Pie(
            labels=bs["Booking Status"], values=bs["count"],
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Set2),
        ))
        apply_layout(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)
        completed_pct = (bs.loc[bs["Booking Status"] == "Completed", "count"].sum() / bs["count"].sum() * 100) if len(bs) > 0 else 0
        st.markdown(f'<div class="chart-desc">Completion rate: <b>{completed_pct:.1f}%</b>. '
                    f'High cancellation or "No Driver Found" rates indicate supply-demand imbalance.</div>',
                    unsafe_allow_html=True)


# ========================================================================
# PAGE 6 — Forecasting Lab (PRIMARY FOCUS)
# ========================================================================
elif page == "Forecasting Lab":
    st.markdown('<div class="page-title">Forecasting Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Time-series forecasting with Prophet and ARIMA models</div>', unsafe_allow_html=True)

    # --- Controls ---
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        dataset_choice = st.selectbox("Dataset", [
            "Amazon Sales (Revenue)",
            "Amazon Sales (Orders)",
            "Web Traffic (Visits)",
            "Ride Bookings (Count)",
            "Ride Bookings (Value)",
        ], key="fc_dataset")
    with fc2:
        model_choice = st.selectbox("Model", ["Prophet", "ARIMA / SARIMA"], key="fc_model")
    with fc3:
        horizon = st.slider("Forecast Horizon (days)", 7, 180, 30, key="fc_horizon")

    # --- Prepare data ---
    if "Amazon" in dataset_choice:
        amz = get_amazon()
        amz_daily = aggregate_amazon_daily(amz)
        if "Revenue" in dataset_choice:
            df_fc = amz_daily[["date", "revenue"]].rename(columns={"date": "ds", "revenue": "y"})
            value_label = "Revenue ($)"
        else:
            df_fc = amz_daily[["date", "orders"]].rename(columns={"date": "ds", "orders": "y"})
            value_label = "Orders"
        df_for_model = amz_daily
        date_col = "date"
        val_col = "revenue" if "Revenue" in dataset_choice else "orders"
    elif "Web Traffic" in dataset_choice:
        web = get_web_traffic()
        web_daily = aggregate_web_daily(web)
        df_fc = web_daily[["date", "visits"]].rename(columns={"date": "ds", "visits": "y"})
        value_label = "Visits"
        df_for_model = web_daily
        date_col = "date"
        val_col = "visits"
    else:
        rides = get_rides()
        rides_daily = aggregate_rides_daily(rides)
        if "Value" in dataset_choice:
            df_fc = rides_daily[["date", "booking_value"]].rename(columns={"date": "ds", "booking_value": "y"})
            value_label = "Booking Value (Rs)"
            val_col = "booking_value"
        else:
            df_fc = rides_daily[["date", "bookings"]].rename(columns={"date": "ds", "bookings": "y"})
            value_label = "Bookings"
            val_col = "bookings"
        df_for_model = rides_daily
        date_col = "date"

    # --- Timeline control for historical view ---
    st.markdown('<div class="section-header">Historical Data Range</div>', unsafe_allow_html=True)
    mn = df_fc["ds"].min().date()
    mx = df_fc["ds"].max().date()
    ts, te = timeline_control("fc_hist", mn, mx)
    mask_hist = (df_fc["ds"].dt.date >= ts) & (df_fc["ds"].dt.date <= te)
    df_fc_filtered = df_fc[mask_hist].copy()

    st.markdown("---")

    # --- Run forecast ---
    if st.button("Run Forecast", type="primary", use_container_width=True):
        with st.spinner("Training model... this may take a moment."):
            if model_choice == "Prophet":
                forecast, metrics, model = forecast_prophet(
                    df_for_model, date_col, val_col, periods=horizon,
                )
                # Build chart
                fig = go.Figure()
                # Historical
                fig.add_trace(go.Scatter(
                    x=df_fc_filtered["ds"], y=df_fc_filtered["y"],
                    mode="lines", name="Actual",
                    line=dict(color="#B0BEC5", width=1.5),
                ))
                # Forecast
                last_hist_date = df_fc["ds"].max()
                forecast_portion = forecast[forecast["ds"] > last_hist_date]
                fig.add_trace(go.Scatter(
                    x=forecast_portion["ds"], y=forecast_portion["yhat"],
                    mode="lines", name="Forecast",
                    line=dict(color=ACCENT, width=2.5),
                ))
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_portion["ds"], y=forecast_portion["yhat_upper"],
                    mode="lines", line=dict(width=0), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_portion["ds"], y=forecast_portion["yhat_lower"],
                    mode="lines", fill="tonexty",
                    fillcolor="rgba(79, 195, 247, 0.15)",
                    line=dict(width=0), name="Confidence Interval",
                ))
                # Festival bands on forecast
                if len(forecast_portion) > 0:
                    f_start = forecast_portion["ds"].min().date()
                    f_end = forecast_portion["ds"].max().date()
                    add_festival_bands(fig, ts, f_end)

                apply_layout(fig, height=460, xaxis_title="Date", yaxis_title=value_label)
                st.plotly_chart(fig, use_container_width=True)

                # Live forecast description
                if len(forecast_portion) > 0:
                    pred_end = forecast_portion.iloc[-1]["yhat"]
                    pred_start = forecast_portion.iloc[0]["yhat"]
                    direction = "upward" if pred_end > pred_start else "downward" if pred_end < pred_start else "flat"
                    fests = get_festivals_in_range(forecast_portion["ds"].min().date(), forecast_portion["ds"].max().date())
                    fest_txt = f' Upcoming festivals in forecast window: {", ".join(set(f[0] for f in fests))}.' if fests else ''
                    st.markdown(f'<div class="chart-desc">Prophet forecasts a <b>{direction}</b> trend over the next '
                                f'<b>{horizon} days</b>. Predicted end value: <b>{pred_end:,.0f}</b>. '
                                f'The shaded band is the 80% confidence interval — wider bands indicate higher uncertainty.{fest_txt}</div>',
                                unsafe_allow_html=True)

            else:
                # ARIMA
                forecast_df, hist_df, metrics = forecast_arima(
                    df_for_model, date_col, val_col, periods=horizon,
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_fc_filtered["ds"], y=df_fc_filtered["y"],
                    mode="lines", name="Actual",
                    line=dict(color="#B0BEC5", width=1.5),
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"], y=forecast_df["yhat"],
                    mode="lines", name="ARIMA Forecast",
                    line=dict(color=ACCENT, width=2.5),
                ))
                if len(forecast_df) > 0:
                    f_start = forecast_df["ds"].min().date()
                    f_end = forecast_df["ds"].max().date()
                    add_festival_bands(fig, ts, f_end)

                apply_layout(fig, height=460, xaxis_title="Date", yaxis_title=value_label)
                st.plotly_chart(fig, use_container_width=True)

                # Live ARIMA description
                if len(forecast_df) > 0:
                    pred_end = forecast_df.iloc[-1]["yhat"]
                    direction = "upward" if forecast_df["yhat"].iloc[-1] > forecast_df["yhat"].iloc[0] else "downward"
                    st.markdown(f'<div class="chart-desc">ARIMA/SARIMA forecasts a <b>{direction}</b> trend over '
                                f'<b>{horizon} days</b>. Final predicted value: <b>{pred_end:,.0f}</b>. '
                                f'ARIMA uses autoregression (past values) and moving average (past errors) '
                                f'with differencing to handle non-stationary data.</div>',
                                unsafe_allow_html=True)

            # --- Metrics ---
            st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE", f"{metrics['MAE']:,.2f}")
            mc2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
            mc3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

            # Metrics description
            mape_quality = "excellent" if metrics["MAPE"] < 10 else "good" if metrics["MAPE"] < 20 else "moderate" if metrics["MAPE"] < 30 else "high"
            rmse_vs_mae = "consistent errors" if metrics["RMSE"] < metrics["MAE"] * 1.3 else "some large outlier errors"
            st.markdown(f'<div class="chart-desc">MAE (Mean Absolute Error) = avg error in original units. '
                        f'RMSE penalizes large errors more — {rmse_vs_mae}. '
                        f'MAPE of <b>{metrics["MAPE"]:.1f}%</b> is <b>{mape_quality}</b> '
                        f'(under 10% = excellent, 10-20% = good, over 30% = poor). '
                        f'Metrics computed on a 20% held-out test set.</div>', unsafe_allow_html=True)

            # --- Seasonal Decomposition ---
            st.markdown('<div class="section-header">Seasonal Decomposition</div>', unsafe_allow_html=True)
            try:
                series = df_fc_filtered.set_index("ds")["y"].dropna()
                if len(series) >= 14:
                    decomp = seasonal_decompose(series, period=7)
                    comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Trend", "Seasonal", "Residual"])
                    for comp_tab, comp_key, comp_label in [
                        (comp_tab1, "trend", "Trend"),
                        (comp_tab2, "seasonal", "Seasonal"),
                        (comp_tab3, "residual", "Residual"),
                    ]:
                        with comp_tab:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=decomp[comp_key].index, y=decomp[comp_key].values,
                                mode="lines", name=comp_label,
                                line=dict(color=ACCENT if comp_key != "residual" else "#78909C", width=1.5),
                            ))
                            apply_layout(fig, height=280, xaxis_title="Date", yaxis_title=comp_label)
                            st.plotly_chart(fig, use_container_width=True)
                            descs = {"trend": "Shows the long-term direction after removing seasonality. Upward = growing, flat = stable.",
                                     "seasonal": "Repeating weekly pattern extracted via additive decomposition. Peaks show high-activity days (e.g. weekends).",
                                     "residual": "Random noise left after removing trend and seasonality. Large spikes here indicate unusual events or anomalies."}
                            st.markdown(f'<div class="chart-desc">{descs[comp_key]}</div>', unsafe_allow_html=True)
                else:
                    st.info("Not enough data points for seasonal decomposition (need 14+).")
            except Exception as e:
                st.warning(f"Decomposition unavailable: {e}")

    else:
        # Show preview before running
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_fc_filtered["ds"], y=df_fc_filtered["y"],
            mode="lines", name="Historical",
            line=dict(color=ACCENT, width=1.5),
        ))
        add_festival_bands(fig, ts, te)
        apply_layout(fig, height=400, xaxis_title="Date", yaxis_title=value_label)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Select your dataset, model, and horizon above, then click **Run Forecast**.")



