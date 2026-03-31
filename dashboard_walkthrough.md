# Complete Dashboard Walkthrough — Every Concept Explained

This is your full preparation guide. Read it top to bottom. For each graph, there's a **"What to tell ma'am"** section — use those exact talking points.

---

## Project Summary (One-liner you should memorize)

> "This is a Streamlit-based interactive dashboard that uses **Time Series Analysis and Machine Learning** to analyze historical web traffic, e-commerce sales, ride bookings, and shopper behavior — and then **forecast future trends** using Prophet and ARIMA models, while highlighting the impact of Indian festivals on these metrics."

---

## Tech Stack (if asked)

| Tech | Why we used it |
|---|---|
| **Python** | Core language for Data Science |
| **Streamlit** | Rapid web dashboard framework — no HTML/JS needed |
| **Pandas** | Data loading, cleaning, aggregation |
| **NumPy** | Numerical computations (Z-scores, etc.) |
| **Plotly** | Interactive charts (hover, zoom, pan) |
| **Prophet (Meta)** | Time series forecasting with seasonality |
| **statsmodels** | ARIMA/SARIMA + seasonal decomposition |
| **scikit-learn** | Evaluation metrics (MAE, RMSE) |

---

## Datasets Used (5 CSVs)

### 1. Amazon.csv (E-Commerce)
- **~100K+ rows** of Amazon orders from 2022-2023
- Columns: `OrderDate`, `ProductName`, `Category`, `Brand`, `TotalAmount`, `Quantity`, `PaymentMethod`, `OrderStatus`, `City`, `State`, `Country`
- **Use**: Main dataset for sales trends, festival spikes, and forecasting

### 2. global_web_traffic_dataset.csv (Website Visits)
- **~1000+ rows** of individual website visits from 2025
- Columns: `date`, `country`, `traffic_source`, `device_type`, `browser`, `page`, `session_duration_sec`, `pages_viewed`, `bounce`, `conversion`
- **Use**: Web traffic trend analysis, source/device breakdowns

### 3. global_web_traffic_2026.csv (Domain Rankings)
- **~1000 rows** of top global website rankings
- Columns: `global_rank`, `domain`, `category`, `monthly_visits`, `bounce_rate_pct`, `avg_session_duration_s`
- **Use**: Benchmarking top websites

### 4. ncr_ride_bookings.csv (Ride Hailing)
- **~200K+ rows** of ride bookings in NCR (Delhi-NCR) from 2024
- Columns: `Date`, `Booking ID`, `Booking Status`, `Vehicle Type`, `Booking Value`, `Ride Distance`, `Driver Ratings`
- **Use**: Ride demand trends and festival mobility patterns

### 5. online_shoppers_intention.csv (Shopper Behavior)
- **~12K rows** of online shopping sessions
- Columns: `Month`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `Revenue`, `VisitorType`
- **Use**: Understanding purchase intent and visitor behavior

---

## Page-by-Page Deep Dive

---

### PAGE 1: Executive Overview

**What it shows:** High-level KPIs (Total Revenue, Total Orders, Average Order Value, Web Visits, Ride Bookings) plus a Revenue Trend line chart with festival bands.

**What to tell ma'am:**
> "This is the executive summary page. It gives a bird's-eye view of all our key performance indicators across all five datasets. The revenue trend chart shows daily sales with a 7-day moving average overlaid in white, and the light-blue shaded regions are festival periods. Below, we have festival impact cards showing the percentage lift each festival brings compared to the non-festival baseline."

#### Concepts used:
- **KPI (Key Performance Indicator)**: A measurable value that shows how effectively a goal is being achieved. Here: revenue, orders, conversion.
- **7-Day Moving Average**: Instead of showing raw noisy daily data, we average the last 7 days at each point. This smooths out daily fluctuations and reveals the true underlying trend. Formula: `MA(t) = (value(t) + value(t-1) + ... + value(t-6)) / 7`
- **Festival Lift %**: Compares the average daily metric during festival week vs. non-festival days. Formula: `Lift = ((festival_avg - baseline_avg) / baseline_avg) x 100`. A lift of +15% means festival days generate 15% more than usual.

---

### PAGE 2: E-Commerce Deep Dive

**What it shows:** Amazon sales data with tabs for Revenue / Orders / Quantity, category breakdown bars, top products, a monthly heatmap, and festival impact bars.

**Sidebar filters:** Category and Brand dropdowns — these filter ALL charts on this page.

**What to tell ma'am (for each visual):**

#### Sales Over Time (Line Chart with tabs)
> "This chart shows daily sales trends. The blue line is the raw daily value, the white line is the 7-day moving average. The shaded vertical bands mark festival weeks. We can switch between Revenue, Orders, and Quantity using the tabs at the top. The date range controls let us zoom into any time period."

> (If a filter is selected): "Right now I've filtered for the [Category] category / [Brand] brand, so we're seeing trends specific to that segment."

#### Revenue by Category (Horizontal Bar)
> "This horizontal bar chart shows total revenue by product category, sorted from lowest to highest. It tells us which categories drive the most business."

#### Top 10 Products (Horizontal Bar)
> "Similar to categories, but at the individual product level. These are the top 10 revenue-generating products."

#### Monthly Revenue Heatmap
> "This heatmap shows revenue intensity across months and years. Darker cells mean lower revenue, brighter cells mean higher. You can visually spot which months are consistently strong — for example, October-November (Diwali season) tends to be brighter."

**Concept — Heatmap:** A matrix visualization where color intensity represents value magnitude. We use a two-color scale from dark (low) to accent blue (high). Great for spotting patterns across two dimensions (month x year).

#### Festival Impact (Bar Chart)
> "This bar chart shows the percentage lift each festival brings to sales. Green bars mean positive lift (festival boosts sales), red bars mean negative lift. For example, if Diwali shows +20%, it means daily sales during the Diwali week are 20% higher than the non-festival baseline."

---

### PAGE 3: Web Traffic Analysis

**What it shows:** Daily web visits with tabs (Visits / Bounce Rate / Conversion Rate), traffic source donut, device distribution donut, and top domains table.

**What to tell ma'am:**

#### Daily Visits (Line Chart with tabs)
> "This tracks daily website visits over time. We also show bounce rate and conversion rate as separate tabs. Bounce rate is the percentage of visitors who leave after viewing just one page — high bounce means poor engagement. Conversion rate is the percentage of visits that resulted in a desired action (purchase, signup)."

**Key definitions:**
- **Bounce Rate**: Visitor lands on a page and leaves without clicking anything. `Bounce Rate = (single-page visits / total visits) x 100`. Lower is better.
- **Conversion Rate**: `Conversions / Total Visits x 100`. Measures how effective the site is at driving desired actions.
- **Session Duration**: How long a user stays on the website in seconds.

#### Traffic Source Donut
> "This donut chart breaks down where visitors come from — Organic Search (Google), Social Media, Direct (typed URL), Referral (from another site), or Email campaigns. It helps us understand which acquisition channels are strongest."

#### Device Distribution Donut
> "Shows the split between Desktop, Mobile, and Tablet users. This matters for UX decisions — if 60% are mobile, the site needs to be mobile-first."

#### Top Domains Table
> "This is from the 2026 global rankings dataset. It shows the top websites by monthly visits along with their bounce rates. We use it as a benchmark."

---

### PAGE 4: Ride Bookings Analysis

**What it shows:** NCR ride booking trends, vehicle type breakdown, booking status funnel.

**What to tell ma'am:**

#### Daily Bookings Trend
> "This shows how many rides were booked each day in the Delhi-NCR region. The festival bands help us see if ride demand increases during celebrations — people travel more during Diwali, for office parties during Christmas, etc."

#### Vehicle Type Breakdown
> "This bar chart shows the distribution of ride types — Auto, eBike, Go Sedan, Premier Sedan, Bike. It reveals customer preferences."

#### Booking Status Distribution
> "This donut shows the funnel: Completed, Incomplete, Cancelled by Customer, Cancelled by Driver, No Driver Found. A high 'No Driver Found' rate indicates supply-side issues."

---

### PAGE 5: Shopper Behavior

**What it shows:** Conversion by month with SpecialDay overlay, bounce vs exit scatter, visitor type comparison.

**What to tell ma'am:**

#### Revenue Sessions by Month (Dual-Axis Chart)
> "The blue bars show conversion rate by month. The white line on the secondary y-axis shows the average 'Special Day' index. Special Day is a proximity measure to holidays like Valentine's Day or Mother's Day — values closer to 1 mean the session happened very close to a special day. We're checking if special days correlate with higher conversions."

**Concept — SpecialDay:** A float between 0 and 1 in the dataset. 1.0 means the session happened exactly on a special day (like Valentine's, Mother's Day). 0.0 means it's far from any special day. It's calculated as `1 - (days_to_nearest_special_day / lookback_window)`.

#### Bounce vs Exit Rates (Scatter Plot)
> "Each dot is a session. X-axis is bounce rate, Y-axis is exit rate. Blue dots are sessions that resulted in revenue, grey are non-revenue. We'd expect revenue sessions to cluster in the lower-left (low bounce, low exit), which confirms that engagement drives purchases."

**Key difference:**
- **Bounce Rate**: Leaving from the FIRST page (entered and left immediately)
- **Exit Rate**: Percentage of views where a page was the LAST page visited. Exit rate can be high for a thank-you page (that's normal), but high exit rate on a product page is bad.

#### Visitor Type Performance
> "Bars show conversion rate for Returning Visitors vs New Visitors vs Others. Typically, returning visitors convert better because they already know the brand."

---

### PAGE 6: Forecasting Lab (MAIN PAGE)

This is the core of the project. Know this page inside out.

**What it shows:** Choose a dataset, choose a model (Prophet or ARIMA), set a horizon, and generate a forecast with confidence intervals, model metrics, and seasonal decomposition.

**What to tell ma'am:**

> "This is the heart of the project. We select a dataset — say Amazon Sales Revenue — choose a forecasting model like Prophet, set how many days ahead we want to predict, and click Run Forecast. The model trains on 80% of the historical data, evaluates on the remaining 20%, and then retrains on the full data to generate the final forecast."

---

#### PROPHET (Meta's Prophet)

**What it is:**
Prophet is an open-source time-series forecasting library by Meta (Facebook). It's designed for business forecasting where data has:
- **Trends** (overall upward/downward direction)
- **Seasonality** (weekly, yearly repeating patterns)
- **Holidays/events** (sudden spikes)

**How it works (simplified):**
```
y(t) = trend(t) + seasonality(t) + holidays(t) + error(t)
```
- **Trend**: Uses a piecewise linear model. It automatically detects "changepoints" where the trend shifts (e.g., a sudden growth after a marketing campaign).
- **Seasonality**: Uses Fourier series to model weekly and yearly patterns. Fourier series means it decomposes repeating patterns into sine and cosine waves of different frequencies.
- **Changepoint detection**: Prophet automatically identifies points where the trend changes slope. The `changepoint_prior_scale` parameter controls how flexible this is (we use 0.05 = somewhat conservative).

**Why we chose it:**
- Handles missing data gracefully
- Doesn't require data to be evenly spaced
- Provides uncertainty/confidence intervals out of the box
- Designed for business data with seasonal patterns

**If ma'am asks "Why not just use linear regression?":**
> "Linear regression assumes a single straight line, which can't capture seasonality or changing trends. Prophet models the trend as a piecewise function that can change direction at detected changepoints, and adds Fourier-based seasonality on top."

---

#### ARIMA / SARIMA

**What it is:**
ARIMA = **A**uto**R**egressive **I**ntegrated **M**oving **A**verage

- **AR (AutoRegressive)**: Current value depends on past values. [y(t) = c + φ1*y(t-1) + φ2*y(t-2) + ... + error](file:///c:/Users/purpo/Documents/Anubis/project/traffic-dashboard/app.py#145-149)
- **I (Integrated)**: We difference the series to make it stationary. If original series has a trend, differencing removes it. `d=1` means we subtract each value from the previous.
- **MA (Moving Average)**: Current value depends on past errors. [y(t) = c + θ1*error(t-1) + θ2*error(t-2) + ...](file:///c:/Users/purpo/Documents/Anubis/project/traffic-dashboard/app.py#145-149)

**ARIMA(p, d, q):**
- `p` = number of AR terms (how many past values to look at)
- `d` = differencing order (how many times we difference)
- `q` = number of MA terms (how many past errors to consider)

We use **SARIMA** = Seasonal ARIMA, which adds seasonal components: [(P, D, Q, s)](file:///c:/Users/purpo/Documents/Anubis/project/traffic-dashboard/app.py#181-184) where `s=7` for weekly seasonality.

**What is stationarity?**
A time series is **stationary** if its statistical properties (mean, variance) don't change over time. Most real-world data is non-stationary (has trends). ARIMA handles this through differencing (`I` component).

**If ma'am asks "How do you choose p, d, q?":**
> "We use default values of (1,1,1) which work well for most daily business data. In production, you'd use ACF/PACF plots or auto_arima to find optimal values. The `d=1` handles the trend, `p=1` captures short-term momentum, and `q=1` incorporates recent forecast errors."

---

#### Model Evaluation Metrics

After forecasting, we show three metrics:

| Metric | Formula | What it means | Good value |
|---|---|---|---|
| **MAE** | `mean(abs(actual - predicted))` | Average absolute error in original units. If MAE=500 for revenue, predictions are off by $500 on average. | Lower = better. Context-dependent. |
| **RMSE** | `sqrt(mean((actual - predicted)^2))` | Like MAE but penalizes large errors more (because of squaring). Always >= MAE. | Lower = better. Close to MAE means errors are consistent. |
| **MAPE** | `mean(abs((actual - predicted) / actual)) x 100` | Error as a percentage. MAPE=10% means predictions are off by 10% on average. | <10% = excellent, 10-20% = good, >30% = poor |

**If ma'am asks "What's the difference between MAE and RMSE?":**
> "Both measure prediction error, but RMSE penalizes large errors more heavily because it squares the differences before averaging. If RMSE is much larger than MAE, it means there are some big outlier errors. If they're close, the model's errors are evenly distributed."

---

#### Seasonal Decomposition

After the forecast, we decompose the historical data into three components:

1. **Trend**: The long-term direction (going up, down, or flat). Extracted by averaging out the seasonal fluctuations.
2. **Seasonal**: The repeating weekly pattern. For example, sales might spike every weekend and dip on Mondays.
3. **Residual**: What's left after removing trend and seasonality. These are random fluctuations, noise, or unusual events.

We use **Additive Decomposition**: `Observed = Trend + Seasonal + Residual`

**If ma'am asks "When would you use multiplicative instead?":**
> "When the seasonal fluctuations grow proportionally with the trend. For example, if sales double and the seasonal swings also double, that's multiplicative. If the swings stay the same size regardless of the level, that's additive. Our data shows relatively constant swings, so additive is appropriate."

---

#### Confidence Intervals (Prophet)

The shaded band around the Prophet forecast represents the **80% confidence interval**: we're 80% confident the actual value will fall within this band. Wider bands = more uncertainty. Bands tend to widen further into the future because uncertainty increases with distance.

---

### PAGE 7: Comparative Insights

**What it shows:** Cross-dataset festival impact table, grouped bar comparison, and anomaly detection.

#### Festival Impact Comparison Table
> "This table shows the percentage lift each festival brings across all three time-series datasets (Amazon Revenue, Web Visits, Ride Bookings) side by side. It helps us see if festivals have a universal impact or if they affect some domains more than others."

#### Anomaly Detection (Z-Score)

**What is Z-Score?**
Z-Score measures how many standard deviations a value is from the mean.

```
Z = (value - mean) / standard_deviation
```

- Z = 0: exactly average
- Z = 1: one standard deviation above average
- Z = 2: two standard deviations above (only ~2.5% of data is this extreme)
- Z = -2: two standard deviations below

We flag a data point as an **anomaly** if `|Z| > threshold` (default: 2.5 sigma).

**What to tell ma'am:**
> "We compute the Z-score for each daily data point. Points beyond the threshold (marked as red diamonds) are statistical anomalies — unusually high or low values. These often correspond to festivals, flash sales, system outages, or data errors. The threshold slider lets us adjust sensitivity — lower threshold catches more anomalies, higher threshold catches only extreme ones."

**If ma'am asks "Why Z-score and not IQR?":**
> "Both are valid. Z-score assumes roughly normal distribution and uses mean/standard deviation. IQR (Interquartile Range) uses median and quartiles, making it more robust to extreme outliers. We chose Z-score because it gives us a continuous measure of how unusual each point is, and the threshold is intuitive (2.5 sigma = 99.4% confidence)."

---

## Festival Analysis — Why Full-Week Windows?

We don't just mark the exact festival day. We use a **full-week window** (3 days before + festival day(s) + 3 days after) because:

1. **Pre-festival rush**: People shop 2-3 days before Diwali for gifts, decorations, clothes
2. **Day-of**: The actual festival day might see reduced e-commerce (people celebrate) but high ride bookings (travel)
3. **Post-festival**: Returns, exchanges, post-sale clearances, back-to-work shopping
4. **Delivery lead times**: Online orders placed 3 days before are intended for festival use

This realistic window captures the true economic impact of each festival.

---

## Design Principles

### 60-30-10 Color Rule
- **60% — Background (#0E1117)**: The dominant dark color, creates visual space
- **30% — Surface (#1A1D23)**: Cards, sidebar, secondary containers
- **10% — Accent (#4FC3F7)**: Interactive elements, chart lines, highlights, section borders

This ratio prevents visual fatigue and creates a professional, clean look.

### Why no gradients?
Gradients can distort data perception in charts — a gradient fill on a bar chart makes it harder to accurately read values. Flat solid colors ensure data accuracy.

---

## Dynamic Descriptions

Every chart in the dashboard has a **context-aware description** below it that updates based on:
- The selected date range
- The selected category/brand filter
- Computed statistics from the visible data
- Festival spikes detected in the visible range

This means when you change a filter or date range, the description auto-updates to describe what you're currently seeing.

---

## Common Questions Ma'am Might Ask

**Q: "Why did you use Streamlit and not Flask/Django?"**
> "Streamlit is purpose-built for data science dashboards. It lets us create interactive visualizations with pure Python — no HTML/CSS/JS needed. Flask and Django are general-purpose web frameworks that would require building the frontend separately."

**Q: "What is time series data?"**
> "Any data that has observations recorded at regular time intervals. Our data has daily observations of sales, visits, and bookings. The key property is that observations are ordered in time and often correlated — today's sales are related to yesterday's sales."

**Q: "What is stationarity and why does it matter?"**
> "A stationary series has a constant mean and variance over time. ARIMA requires stationarity because its mathematical formulas assume stable statistical properties. Non-stationary data (with trends or changing variance) will give poor forecasts. The 'I' in ARIMA handles this through differencing."

**Q: "What are changepoints in Prophet?"**
> "Points in time where the trend changes direction — for example, sales might grow steadily for 6 months, then suddenly plateau or decline. Prophet automatically detects these shifts using a Bayesian approach."

**Q: "What is the Fourier series in seasonality?"**
> "It's a mathematical technique that represents any periodic (repeating) pattern as a sum of sine and cosine waves. So weekly seasonality might be: [seasonal(t) = a1*sin(2*pi*t/7) + b1*cos(2*pi*t/7) + a2*sin(4*pi*t/7) + ...](file:///c:/Users/purpo/Documents/Anubis/project/traffic-dashboard/forecaster.py#15-30) More terms = more complex seasonal pattern captured."

**Q: "How does caching work in the dashboard?"**
> "We use `@st.cache_data` — when a function runs the first time, Streamlit stores its return value. On re-runs (when user interacts), it returns the cached result instead of re-reading the CSV. This makes the dashboard fast."

**Q: "What is a donut chart vs pie chart?"**
> "Same thing, but with a hole in the center (`hole=0.5`). The hole makes it easier to add a label in the center and reduces the bias of comparing slice angles, making it slightly more readable."

**Q: "What is a dual-axis chart?"**
> "A chart with two y-axes — one on the left, one on the right — each with its own scale. We use this when plotting two metrics with very different ranges (like conversion % on left and special day index on right). Without dual axes, the smaller-valued metric would appear flat."

**Q: "What is the confidence interval showing?"**
> "It's the uncertainty band. The model says 'I'm 80% confident the real value will be somewhere in this shaded region.' Wider bands mean more uncertainty. Bands get wider as we forecast further into the future because prediction becomes harder."

**Q: "Why are some festival lifts negative?"**
> "A negative lift means the metric actually decreased during that festival week compared to the baseline. This can happen if the festival causes a reduction in activity — for example, Independence Day might reduce e-commerce activity as people attend events rather than shop online."
