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
    load_ride_bookings, aggregate_amazon_daily, aggregate_web_daily, 
    aggregate_rides_daily, compute_festival_lift, get_festivals_in_range, 
    FESTIVAL_DATES,
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
        "Comparative Insights",
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
# PAGE 1 — E-Commerce Deep Dive
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
                peak_d = filtered.loc[filtered[col_name].idxmax(), "date"].strftime("%b %d, %Y")
                filter_ctx = f" for {sel_cat}" if sel_cat != "All" else ""
                st.markdown(f'<div class="chart-desc">{label}{filter_ctx}: Average <b>{avg_v:,.0f}</b> per day, '
                            f'peak of <b>{peak_v:,.0f}</b> on {peak_d} across the selected {(te - ts).days}-day window. '
                            f'The white line smooths daily noise using a 7-day rolling mean.</div>',
                            unsafe_allow_html=True)

    st.markdown("---")

    # --- Category Breakdown ---
    c1, c2 = st.columns(2)
    with c1:
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
                    f'Bars show cumulative revenue across the full dataset.</div>',
                    unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Top 10 Products</div>', unsafe_allow_html=True)
        prod_df = amz.groupby("ProductName")["TotalAmount"].sum().reset_index()
        prod_df = prod_df.sort_values("TotalAmount", ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            x=prod_df["TotalAmount"], y=prod_df["ProductName"],
            orientation="h", marker_color="#B0BEC5",
        ))
        apply_layout(fig, height=360, xaxis_title="Revenue ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        top_prod = prod_df.iloc[-1]["ProductName"] if len(prod_df) > 0 else "N/A"
        st.markdown(f'<div class="chart-desc">Highest-revenue product: <b>{top_prod}</b>. '
                    f'Shows the top 10 individual products by total sales value.</div>',
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
# PAGE 2 — Web Traffic Analysis
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
# PAGE 3 — Ride Bookings Analysis
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
