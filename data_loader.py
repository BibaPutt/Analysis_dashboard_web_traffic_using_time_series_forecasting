"""
data_loader.py
Centralized data loading, cleaning, festival tagging, and aggregation utilities.
"""

import pandas as pd
import numpy as np
from datetime import date

# ---------------------------------------------------------------------------
# Festival calendar — realistic full-week windows
# (3 days before the festival + festival day(s) + 3 days after)
# This captures the pre-festival shopping rush and post-festival returns/traffic.
# ---------------------------------------------------------------------------
FESTIVAL_DATES = {
    "Republic Day":    [(date(2022, 1, 23), date(2022, 1, 29)),
                        (date(2023, 1, 23), date(2023, 1, 29)),
                        (date(2024, 1, 23), date(2024, 1, 29)),
                        (date(2025, 1, 23), date(2025, 1, 29))],
    "Holi":            [(date(2022, 3, 15), date(2022, 3, 21)),
                        (date(2023, 3, 5),  date(2023, 3, 11)),
                        (date(2024, 3, 22), date(2024, 3, 28)),
                        (date(2025, 3, 11), date(2025, 3, 17))],
    "Independence Day":[(date(2022, 8, 12), date(2022, 8, 18)),
                        (date(2023, 8, 12), date(2023, 8, 18)),
                        (date(2024, 8, 12), date(2024, 8, 18)),
                        (date(2025, 8, 12), date(2025, 8, 18))],
    "Navratri":        [(date(2022, 9, 23), date(2022, 10, 8)),
                        (date(2023, 10, 12), date(2023, 10, 27)),
                        (date(2024, 9, 30),  date(2024, 10, 15)),
                        (date(2025, 9, 19),  date(2025, 10, 4))],
    "Dussehra":        [(date(2022, 10, 2), date(2022, 10, 9)),
                        (date(2023, 10, 21), date(2023, 10, 28)),
                        (date(2024, 10, 9),  date(2024, 10, 16)),
                        (date(2025, 9, 29),  date(2025, 10, 6))],
    "Diwali":          [(date(2022, 10, 19), date(2022, 10, 29)),
                        (date(2023, 11, 7),  date(2023, 11, 17)),
                        (date(2024, 10, 26), date(2024, 11, 6)),
                        (date(2025, 10, 16), date(2025, 10, 26))],
    "Christmas":       [(date(2022, 12, 22), date(2022, 12, 28)),
                        (date(2023, 12, 22), date(2023, 12, 28)),
                        (date(2024, 12, 22), date(2024, 12, 28)),
                        (date(2025, 12, 22), date(2025, 12, 28))],
    "Eid":             [(date(2022, 4, 29), date(2022, 5, 7)),
                        (date(2023, 4, 18), date(2023, 4, 26)),
                        (date(2024, 4, 7),  date(2024, 4, 15)),
                        (date(2025, 3, 27), date(2025, 4, 4))],
}


def get_festival(dt):
    """Return the festival name if a date falls inside a festival window, else None."""
    if isinstance(dt, pd.Timestamp):
        d = dt.date()
    else:
        d = dt
    for fest, windows in FESTIVAL_DATES.items():
        for start, end in windows:
            if start <= d <= end:
                return fest
    return None


def get_festivals_in_range(start_date, end_date):
    """Return list of (festival_name, start, end) that overlap with a date range."""
    results = []
    for fest, windows in FESTIVAL_DATES.items():
        for ws, we in windows:
            if ws <= end_date and we >= start_date:
                results.append((fest, max(ws, start_date), min(we, end_date)))
    return results


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_amazon(path="Amazon.csv"):
    """Load Amazon ecommerce data."""
    df = pd.read_csv(path, parse_dates=["OrderDate"])
    df = df.rename(columns={"OrderDate": "date"})
    df["festival"] = df["date"].apply(get_festival)
    df["is_festival"] = df["festival"].notna()
    return df


def load_web_traffic(path="global_web_traffic_dataset.csv"):
    """Load web traffic visit-level data."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["festival"] = df["date"].apply(get_festival)
    df["is_festival"] = df["festival"].notna()
    return df


def load_web_traffic_2026(path="global_web_traffic_2026.csv"):
    """Load domain-level web traffic rankings."""
    df = pd.read_csv(path)
    return df


def load_ride_bookings(path="ncr_ride_bookings.csv"):
    """Load NCR ride bookings data."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"})
    df["festival"] = df["date"].apply(get_festival)
    df["is_festival"] = df["festival"].notna()
    return df


def load_shoppers(path="online_shoppers_intention.csv"):
    """Load online shoppers intention data."""
    df = pd.read_csv(path)
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    df["month_num"] = df["Month"].map(month_map)
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_amazon_daily(df):
    """Daily aggregate: total revenue, total orders, avg order value."""
    agg = df.groupby("date").agg(
        revenue=("TotalAmount", "sum"),
        orders=("OrderID", "count"),
        quantity=("Quantity", "sum"),
    ).reset_index()
    agg["avg_order_value"] = agg["revenue"] / agg["orders"]
    agg["festival"] = agg["date"].apply(get_festival)
    agg["is_festival"] = agg["festival"].notna()
    return agg.sort_values("date")


def aggregate_web_daily(df):
    """Daily aggregate: visits, avg session duration, bounce rate, conversion rate."""
    agg = df.groupby("date").agg(
        visits=("visit_id", "count"),
        avg_session_dur=("session_duration_sec", "mean"),
        bounce_rate=("bounce", "mean"),
        conversion_rate=("conversion", "mean"),
    ).reset_index()
    agg["bounce_rate"] = agg["bounce_rate"] * 100
    agg["conversion_rate"] = agg["conversion_rate"] * 100
    agg["festival"] = agg["date"].apply(get_festival)
    agg["is_festival"] = agg["festival"].notna()
    return agg.sort_values("date")


def aggregate_rides_daily(df):
    """Daily aggregate: total bookings, total booking value, completed rides."""
    agg = df.groupby("date").agg(
        bookings=("Booking ID", "count"),
        booking_value=("Booking Value", "sum"),
        avg_distance=("Ride Distance", "mean"),
    ).reset_index()
    agg["festival"] = agg["date"].apply(get_festival)
    agg["is_festival"] = agg["festival"].notna()
    return agg.sort_values("date")


def compute_festival_lift(daily_df, metric_col):
    """Compute % lift during each festival vs. the non-festival baseline."""
    baseline = daily_df.loc[~daily_df["is_festival"], metric_col].mean()
    if baseline == 0:
        return pd.DataFrame()
    results = []
    for fest in daily_df["festival"].dropna().unique():
        fest_mean = daily_df.loc[daily_df["festival"] == fest, metric_col].mean()
        lift = ((fest_mean - baseline) / baseline) * 100
        results.append({"festival": fest, "avg_value": round(fest_mean, 2),
                        "baseline": round(baseline, 2), "lift_pct": round(lift, 1)})
    return pd.DataFrame(results).sort_values("lift_pct", ascending=False)
