# --------------------------------------------------------------------------- #
# SEM FORECAST Streamlit App – Enhanced with Dashboard and Table Views         #
# --------------------------------------------------------------------------- #
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import io
from scipy import stats
from sklearn.linear_model import LinearRegression
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="SEM Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------- #
# Configuration and Helper Functions                                           #
# --------------------------------------------------------------------------- #
ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
REQ_COLS = {
    "date", "orders", "clicks", "impressions", "cost",
    "brand_cost", "nonbrand_cost", "promo_flag",
}
BASE_REGS = ["brand_cost", "nonbrand_cost", "promo_flag"]

def calculate_trendline(x_values, y_values):
    """Calculate linear trendline for given x,y values"""
    if len(x_values) < 2 or len(y_values) < 2:
        return None, None, None
    
    # Convert to numpy arrays and handle pandas Series properly
    x_vals = np.array(x_values)
    y_vals = np.array(y_values)
    
    if len(x_vals) == 0 or len(y_vals) == 0:
        return None, None, None
    
    # Convert dates to numeric values for regression
    try:
        # Get the first date value properly
        first_date = pd.to_datetime(x_vals[0])
        x_numeric = np.array([(pd.to_datetime(x) - first_date).days for x in x_vals]).reshape(-1, 1)
    except (ValueError, TypeError):
        return None, None, None
    
    # Remove NaN values
    mask = ~np.isnan(y_vals)
    if np.sum(mask) < 2:
        return None, None, None
    
    x_clean = x_numeric[mask]
    y_clean = y_vals[mask]
    
    try:
        # Fit linear regression
        model = LinearRegression()
        model.fit(x_clean, y_clean)
        
        # Calculate trend values for all x points
        trend_values = model.predict(x_numeric)
        
        # Calculate R-squared
        y_pred = model.predict(x_clean)
        if len(y_clean) > 1 and np.var(y_clean) > 0:
            r_squared = 1 - (np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2))
        else:
            r_squared = 0.0
        
        return trend_values, model.coef_[0], r_squared
    except Exception:
        return None, None, None

def add_simple_trendline(fig, x_values, y_values, name_prefix, color, dash_style='dash'):
    """Add a simple trendline (hidden from legend)"""
    
    if len(x_values) < 2:
        return fig
    
    trend_values, slope, r_squared = calculate_trendline(x_values, y_values)
    
    if trend_values is not None:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=trend_values,
            mode='lines',
            name=f"{name_prefix} Trend",
            line=dict(color=color, width=2, dash=dash_style),
            opacity=0.7,
            showlegend=False,  # Hide from legend to reduce clutter
            hovertemplate=f"{name_prefix} Trend<br>Date: %{{x}}<br>Trend Value: %{{y:,.0f}}<extra></extra>"
        ))
    
    return fig

def filter_data_by_date_range(data, date_col, start_date, end_date):
    """Filter dataframe by date range"""
    if start_date is None or end_date is None:
        return data
    
    mask = (data[date_col] >= start_date) & (data[date_col] <= end_date)
    return data[mask]

def create_dynamic_trendline_chart(table, orders_fc, clicks_fc, impr_fc, chart_type="daily"):
    """Create charts with truly dynamic trendlines that recalculate based on user-selected range"""
    
    # Date range selector using Streamlit widgets
    st.subheader(f"📅 Select Date Range for {chart_type.title()} Analysis")
    
    # Get full date range
    all_dates = table['ds'].dropna()
    if not all_dates.empty:
        min_date = all_dates.min().date()
        max_date = all_dates.max().date()
        
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"{chart_type}_dynamic_start"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"{chart_type}_dynamic_end"
            )
        
        with col3:
            st.write("Quick Select:")
            col3a, col3b = st.columns(2)
            with col3a:
                if st.button("Last 30D", key=f"{chart_type}_quick_30"):
                    start_date = max_date - pd.Timedelta(days=30)
                    start_date = max(start_date, min_date)
            with col3b:
                if st.button("Last 90D", key=f"{chart_type}_quick_90"):
                    start_date = max_date - pd.Timedelta(days=90) 
                    start_date = max(start_date, min_date)
        
        # Convert to pandas timestamps for filtering
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Filter all data based on selected range
        filtered_table = table[(table['ds'] >= start_ts) & (table['ds'] <= end_ts)]
        filtered_orders_fc = orders_fc[(orders_fc['ds'] >= start_ts) & (orders_fc['ds'] <= end_ts)] if orders_fc is not None else None
        filtered_clicks_fc = clicks_fc[(clicks_fc['ds'] >= start_ts) & (clicks_fc['ds'] <= end_ts)] if clicks_fc is not None else None  
        filtered_impr_fc = impr_fc[(impr_fc['ds'] >= start_ts) & (impr_fc['ds'] <= end_ts)] if impr_fc is not None else None
        
        st.info(f"📊 Analyzing {len(filtered_table)} days of data from {start_date} to {end_date}")
        
        return filtered_table, filtered_orders_fc, filtered_clicks_fc, filtered_impr_fc, start_ts, end_ts
    
    return table, orders_fc, clicks_fc, impr_fc, None, None

def create_chart_with_hierarchical_controls(fig, full_data, chart_key):
    """Create chart with hierarchical filtering: Slider > Quick Select > Date Range"""
    
    # Configure chart with interactive controls
    fig.update_layout(
        dragmode='zoom',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=30, label="30D", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.05)
        ),
        # Store original data in layout for access
        annotations=[
            dict(
                text=f"<!-- Data: {full_data} -->",
                showarrow=False,
                x=0, y=0,
                xref="paper", yref="paper",
                font=dict(size=1, color="rgba(0,0,0,0)")
            )
        ]
    )
    
    return fig

def create_single_forecast_chart(data, metric, forecast_data=None, original_data=None, 
                               colors={'actual': 'black', 'forecast': 'blue', 'ci': 'rgba(0,100,80,0.2)'}, 
                               enhanced=False, title_prefix=""):
    """Create a single forecast chart for any metric"""
    actual_col = f"actual_{metric}"
    upper_col = f"{metric}_upper"
    lower_col = f"{metric}_lower"
    
    # Determine split point
    last_actual = data[data[actual_col].notna()]["ds"].max() if not data[data[actual_col].notna()].empty else None
    
    fig = go.Figure()
    
    # Historical data
    hist_data = data[data[actual_col].notna()]
    if not hist_data.empty:
        fig.add_trace(go.Scatter(
            x=hist_data["ds"],
            y=hist_data[actual_col],
            mode='lines+markers',
            name=f'Actual {metric.title()}',
            line=dict(color=colors['actual'], width=2)
        ))
    
    # Forecast data - use separate forecast_data if provided, otherwise use main data
    if forecast_data is not None and not forecast_data.empty:
        fc_data = forecast_data
        fc_col = metric
    else:
        fc_data = data[data["ds"] > last_actual] if last_actual is not None else data[data[metric].notna()]
        fc_col = metric
    
    if not fc_data.empty:
        fig.add_trace(go.Scatter(
            x=fc_data["ds"],
            y=fc_data[fc_col],
            mode='lines+markers',
            name=f'Forecast {metric.title()}',
            line=dict(color=colors['forecast'], width=2)
        ))
        
        # Confidence intervals
        if upper_col in fc_data.columns and lower_col in fc_data.columns:
            fig.add_trace(go.Scatter(
                x=fc_data["ds"],
                y=fc_data[upper_col],
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=fc_data["ds"],
                y=fc_data[lower_col],
                mode='lines',
                name='90% Confidence Interval',
                fill='tonexty',
                fillcolor=colors['ci'],
                line=dict(width=0)
            ))
    
    # Add trendlines
    if not hist_data.empty and len(hist_data) >= 2:
        fig = add_simple_trendline(fig, hist_data["ds"], hist_data[actual_col], f"Historical {metric.title()}", "gray")
    if not fc_data.empty and len(fc_data) >= 2:
        trendline_color = colors['forecast'].replace('blue', 'lightblue').replace('green', 'lightgreen').replace('red', 'lightcoral')
        fig = add_simple_trendline(fig, fc_data["ds"], fc_data[fc_col], f"Forecast {metric.title()}", trendline_color)
    
    # Configure with hierarchical controls if enhanced
    if enhanced and original_data is not None:
        fig = create_chart_with_hierarchical_controls(fig, original_data, metric)
    
    fig.update_layout(
        title=f"{title_prefix}{metric.title()} Forecast with Dynamic Trendlines",
        xaxis_title="Date",
        yaxis_title=metric.title(),
        hovermode='x unified'
    )
    
    return fig

def create_enhanced_forecast_charts(filtered_table, filtered_orders_fc, filtered_clicks_fc, filtered_impr_fc, 
                                  original_table, original_orders_fc, original_clicks_fc, original_impr_fc):
    """Create forecast charts with enhanced hierarchical filtering capabilities"""
    colors_map = {
        'orders': {'actual': 'black', 'forecast': 'blue', 'ci': 'rgba(0,100,80,0.2)'},
        'clicks': {'actual': 'black', 'forecast': 'green', 'ci': 'rgba(0,100,80,0.2)'},
        'impressions': {'actual': 'black', 'forecast': 'red', 'ci': 'rgba(0,100,80,0.2)'}
    }
    
    fig_orders = create_single_forecast_chart(filtered_table, 'orders', None, original_table, 
                                           colors_map['orders'], True, "Hierarchical ")
    fig_clicks = create_single_forecast_chart(filtered_table, 'clicks', filtered_clicks_fc, original_clicks_fc,
                                           colors_map['clicks'], True, "Hierarchical ")
    fig_impressions = create_single_forecast_chart(filtered_table, 'impressions', filtered_impr_fc, original_impr_fc,
                                                colors_map['impressions'], True, "Hierarchical ")
    
    return fig_orders, fig_clicks, fig_impressions

@st.cache_data
def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load and validate uploaded dataframe"""
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXT:
        st.error(f"Unsupported file type: {ext}")
        return None
    
    try:
        if ext == ".csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=0)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None
    
    # Clean column names by stripping whitespace and normalizing
    df.columns = df.columns.str.strip()
    
    missing = REQ_COLS - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        return None
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    if df.empty:
        st.error("No valid data rows after processing")
        return None
    
    return df

def prophet_forecast(df_train, target, regs, df_future):
    """Generate Prophet forecast for a target variable"""
    df_model = df_train[["ds", target] + regs].rename(columns={target: "y"})
    
    m = Prophet(
        changepoint_prior_scale=0.5,
        seasonality_mode="multiplicative",
        interval_width=0.90,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    
    for r in regs:
        m.add_regressor(r)
    
    m.fit(df_model)
    
    fut = df_future[["ds"] + regs]
    fc = m.predict(fut)
    
    return (
        fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(columns={
            "yhat": target,
            "yhat_lower": f"{target}_lower",
            "yhat_upper": f"{target}_upper"})
    )

@st.cache_data
def run_sem_pipeline(df_full):
    """Main forecasting pipeline"""
    df_full = df_full.copy()
    df_full.rename(columns={"date": "ds"}, inplace=True)
    
    hist = df_full[df_full["orders"].notna()].copy()
    future = df_full[df_full["orders"].isna()].copy()
    
    if future.empty:
        st.warning("No future data found (rows with null orders)")
        return None, None, None, None
    
    last_hist = hist["ds"].max()
    
    # Training window = last 730 days
    train = hist[hist["ds"] >= last_hist - pd.Timedelta(days=730)].copy()
    
    # Generate forecasts
    clicks_fc = prophet_forecast(train, "clicks", BASE_REGS, future)
    impr_fc = prophet_forecast(train, "impressions", BASE_REGS, future)
    
    # Orders forecast using predicted clicks/impressions
    fut_orders = (
        future.drop(columns=["clicks", "impressions"])
        .merge(clicks_fc[["ds", "clicks"]], on="ds")
        .merge(impr_fc[["ds", "impressions"]], on="ds")
    )
    orders_fc = prophet_forecast(
        train, "orders", BASE_REGS + ["clicks", "impressions"], fut_orders
    )
    
    # Combined results table
    table = (
        hist[["ds", "cost", "orders", "clicks", "impressions"]]
        .rename(columns={"orders": "actual_orders", "clicks": "actual_clicks", "impressions": "actual_impressions"})
        .merge(orders_fc[["ds", "orders", "orders_lower", "orders_upper"]], on="ds", how="outer")
        .merge(clicks_fc[["ds", "clicks", "clicks_lower", "clicks_upper"]], on="ds", how="outer")
        .merge(impr_fc[["ds", "impressions", "impressions_lower", "impressions_upper"]], on="ds", how="outer")
        .merge(future[["ds", "cost"]].rename(columns={"cost": "planned_cost"}), on="ds", how="outer")
        .sort_values("ds")
    )
    
    return table, orders_fc, clicks_fc, impr_fc

def create_forecast_charts(table, orders_fc, clicks_fc, impr_fc):
    """Create interactive Plotly charts with dynamic trendlines"""
    colors_map = {
        'orders': {'actual': 'black', 'forecast': 'blue', 'ci': 'rgba(0,100,80,0.2)'},
        'clicks': {'actual': 'black', 'forecast': 'green', 'ci': 'rgba(0,128,0,0.2)'},
        'impressions': {'actual': 'black', 'forecast': 'red', 'ci': 'rgba(255,0,0,0.2)'}
    }
    
    fig_orders = create_single_forecast_chart(table, 'orders', None, table, colors_map['orders'], True)
    fig_clicks = create_single_forecast_chart(table, 'clicks', clicks_fc, table, colors_map['clicks'], True)
    fig_impressions = create_single_forecast_chart(table, 'impressions', impr_fc, table, colors_map['impressions'], True)
    
    return fig_orders, fig_clicks, fig_impressions

def calculate_summary_metrics(table):
    """Calculate useful summary metrics"""
    last_actual = table[table["actual_orders"].notna()]["ds"].max()
    forecast_data = table[table["ds"] > last_actual].copy()
    
    if forecast_data.empty:
        return {}
    
    metrics = {
        "forecast_period_days": len(forecast_data),
        "total_forecast_orders": forecast_data["orders"].sum(),
        "avg_daily_orders": forecast_data["orders"].mean(),
        "total_planned_cost": forecast_data["planned_cost"].sum() if "planned_cost" in forecast_data.columns else 0,
        "forecast_cpa": (forecast_data["planned_cost"].sum() / forecast_data["orders"].sum()) if forecast_data["orders"].sum() > 0 else 0,
        "confidence_range_orders": (forecast_data["orders_upper"].sum() - forecast_data["orders_lower"].sum()),
    }
    
    return metrics

def aggregate_data_by_period(table, period='W'):
    """Aggregate daily data to specified period (W=weekly, M=monthly)"""
    aggregated_table = table.copy()
    
    # Add period start date
    period_col = 'period_start'
    aggregated_table[period_col] = aggregated_table['ds'].dt.to_period(period).dt.start_time
    
    # Sum columns - but handle NaN values properly
    sum_cols = ['actual_orders', 'orders', 'actual_clicks', 'clicks', 'actual_impressions', 
                'impressions', 'cost', 'planned_cost', 'orders_lower', 'orders_upper', 
                'clicks_lower', 'clicks_upper', 'impressions_lower', 'impressions_upper']
    
    # Group by period and aggregate
    aggregated_data = aggregated_table.groupby(period_col).apply(
        lambda group: pd.Series({
            col: group[col].sum() if group[col].notna().any() else np.nan
            for col in sum_cols if col in group.columns
        })
    ).reset_index()
    
    aggregated_data.rename(columns={period_col: 'ds'}, inplace=True)
    
    return aggregated_data

def aggregate_weekly_data(table):
    """Aggregate daily data to weekly view"""
    return aggregate_data_by_period(table, 'W')

def aggregate_monthly_data(table):
    """Aggregate daily data to monthly view"""
    return aggregate_data_by_period(table, 'M')

def calculate_marketing_metrics(display_table):
    """Calculate CPO, CPC, and CPM metrics for a display table"""
    def get_combined_value(row, actual_col, forecast_col):
        """Combine actual and forecast values, handling historical, current, and future periods"""
        actual_exists = actual_col in display_table.columns
        forecast_exists = forecast_col in display_table.columns
        
        actual_val = row.get(actual_col, np.nan) if actual_exists else np.nan
        forecast_val = row.get(forecast_col, np.nan) if forecast_exists else np.nan
        
        # Convert to numeric, treating 0 as a valid value
        actual_val = pd.to_numeric(actual_val, errors='coerce')
        forecast_val = pd.to_numeric(forecast_val, errors='coerce')
        
        # If both values exist and are not null, add them (mid-month scenario)
        if pd.notnull(actual_val) and pd.notnull(forecast_val):
            return actual_val + forecast_val
        # If only actual exists and is valid, use it (historical months)
        elif pd.notnull(actual_val):
            return actual_val
        # If only forecast exists and is valid, use it (future months)
        elif pd.notnull(forecast_val):
            return forecast_val
        # If neither exists or both are invalid, return nan
        else:
            return np.nan
    
    def calculate_metric(row, numerator_cols, denominator_cols, multiplier=1):
        """Generic metric calculation function"""
        combined_cost = get_combined_value(row, numerator_cols[0], numerator_cols[1])
        if pd.isnull(combined_cost) or combined_cost == 0:
            return np.nan
        combined_volume = get_combined_value(row, denominator_cols[0], denominator_cols[1])
        return (combined_cost / combined_volume) * multiplier if combined_volume > 0 else np.nan
    
    # Check if we have cost columns available
    has_cost = 'cost' in display_table.columns or 'planned_cost' in display_table.columns
    
    if has_cost:
        # CPO (Cost Per Order)
        if 'orders' in display_table.columns or 'actual_orders' in display_table.columns:
            display_table['CPO'] = display_table.apply(
                lambda row: calculate_metric(row, ('cost', 'planned_cost'), ('actual_orders', 'orders')), 
                axis=1
            )
        
        # CPC (Cost Per Click)
        if 'clicks' in display_table.columns or 'actual_clicks' in display_table.columns:
            display_table['CPC'] = display_table.apply(
                lambda row: calculate_metric(row, ('cost', 'planned_cost'), ('actual_clicks', 'clicks')), 
                axis=1
            )
        
        # CPM (Cost Per Thousand Impressions)  
        if 'impressions' in display_table.columns or 'actual_impressions' in display_table.columns:
            display_table['CPM'] = display_table.apply(
                lambda row: calculate_metric(row, ('cost', 'planned_cost'), ('actual_impressions', 'impressions'), 1000), 
                axis=1
            )
    
    return display_table

def apply_budget_scenario(table, brand_adjustment, nonbrand_adjustment):
    """Apply budget scenario adjustments to the forecast table - ensures Prophet regressor columns are updated"""
    scenario_table = table.copy()
    
    # Apply adjustments to future periods only
    future_mask = scenario_table["actual_orders"].isna()
    
    # Prophet model uses brand_cost and nonbrand_cost as regressors, so we MUST have these columns
    # If they don't exist, create them from available cost data
    if 'brand_cost' not in scenario_table.columns or 'nonbrand_cost' not in scenario_table.columns:
        # Determine which cost column to use as basis
        cost_col = None
        if 'planned_cost' in scenario_table.columns:
            cost_col = 'planned_cost'
        elif 'cost' in scenario_table.columns:
            cost_col = 'cost'
        
        if cost_col is not None:
            # Calculate brand share from historical data if possible
            hist_data = scenario_table[scenario_table["actual_orders"].notna()]
            if len(hist_data) > 0 and 'brand_cost' in hist_data.columns and 'nonbrand_cost' in hist_data.columns:
                # Use actual historical brand share
                total_brand_hist = hist_data['brand_cost'].sum()
                total_nonbrand_hist = hist_data['nonbrand_cost'].sum()
                total_hist = total_brand_hist + total_nonbrand_hist
                brand_share = total_brand_hist / total_hist if total_hist > 0 else 0.7
            else:
                # Default assumption: 70% brand, 30% non-brand
                brand_share = 0.7
            
            # Create the columns for entire dataset
            scenario_table['brand_cost'] = scenario_table[cost_col] * brand_share
            scenario_table['nonbrand_cost'] = scenario_table[cost_col] * (1 - brand_share)
        else:
            # If no cost data available, create zero columns (forecast will fail but won't crash)
            scenario_table['brand_cost'] = 0
            scenario_table['nonbrand_cost'] = 0
    
    # Apply adjustments to the Prophet regressor columns (this is the key fix!)
    if brand_adjustment != 0:
        scenario_table.loc[future_mask, 'brand_cost'] *= (1 + brand_adjustment / 100)
    
    if nonbrand_adjustment != 0:
        scenario_table.loc[future_mask, 'nonbrand_cost'] *= (1 + nonbrand_adjustment / 100)
    
    # Update other cost columns to maintain consistency
    if 'brand_cost' in scenario_table.columns and 'nonbrand_cost' in scenario_table.columns:
        new_total_cost = scenario_table.loc[future_mask, 'brand_cost'] + scenario_table.loc[future_mask, 'nonbrand_cost']
        
        # Update all cost tracking columns
        if 'cost' in scenario_table.columns:
            scenario_table.loc[future_mask, 'cost'] = new_total_cost
        if 'planned_cost' in scenario_table.columns:
            scenario_table.loc[future_mask, 'planned_cost'] = new_total_cost
    
    return scenario_table

def calculate_historical_elasticities(table, recency_days=14, decay_half_life=30, confidence_level=0.9, n_bootstrap=1000):
    """Calculate brand vs non-brand elasticities from historical data with recency weighting and confidence intervals"""
    
    # Get historical data only
    hist_data = table[table["actual_orders"].notna()].copy()
    
    if len(hist_data) < 10:  # Need sufficient data points
        # Return default elasticities if insufficient data
        return {
            'brand_elasticity': 0.15,      # Brand typically has higher elasticity
            'nonbrand_elasticity': 0.08,   # Non-brand typically lower elasticity  
            'brand_share': 0.7,            # Assume brand is 70% of spend
            'baseline_orders_per_brand_dollar': 0.02,
            'baseline_orders_per_nonbrand_dollar': 0.01,
            'recency_weighted': False,
            'confidence_intervals': False
        }
    
    # Sort by date and calculate recency weights
    hist_data = hist_data.sort_values('ds')
    latest_date = hist_data['ds'].max()
    
    # Calculate days back from latest date
    hist_data['days_back'] = (latest_date - hist_data['ds']).dt.days
    
    # Exponential decay weighting: weight = exp(-days_back * ln(2) / half_life)
    # This gives 50% weight at half_life days back, 25% at 2x half_life, etc.
    hist_data['recency_weight'] = np.exp(-hist_data['days_back'] * np.log(2) / decay_half_life)
    
    # Additional boost for very recent data (last recency_days)
    recent_boost = 1.5  # 50% boost for recent data
    hist_data.loc[hist_data['days_back'] <= recency_days, 'recency_weight'] *= recent_boost
    
    # Normalize weights so they sum to original data length (maintains scale)
    total_weight = hist_data['recency_weight'].sum()
    if total_weight > 0:
        hist_data['recency_weight'] = hist_data['recency_weight'] * len(hist_data) / total_weight
    
    # Calculate weighted correlations and elasticities
    elasticities = {}
    
    def weighted_correlation(x, y, weights):
        """Calculate weighted correlation coefficient"""
        if len(x) < 2 or weights.sum() == 0:
            return 0
        
        # Weighted means
        wx = np.average(x, weights=weights)
        wy = np.average(y, weights=weights)
        
        # Weighted covariance and variances
        cov = np.average((x - wx) * (y - wy), weights=weights)
        var_x = np.average((x - wx) ** 2, weights=weights)
        var_y = np.average((y - wy) ** 2, weights=weights)
        
        if var_x <= 0 or var_y <= 0:
            return 0
        
        return cov / np.sqrt(var_x * var_y)
    
    def bootstrap_elasticity(cost_data, orders_data, weights, n_bootstrap=1000):
        """Calculate bootstrap confidence intervals for elasticity"""
        if len(cost_data) < 10 or cost_data.var() == 0:
            return 0, 0, 0  # elasticity, lower_ci, upper_ci
        
        elasticities = []
        n_samples = len(cost_data)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Sample data and weights
            boot_cost = cost_data.iloc[indices].values
            boot_orders = orders_data.iloc[indices].values  
            boot_weights = weights.iloc[indices].values
            
            # Calculate correlation for this bootstrap sample
            corr = weighted_correlation(boot_cost, boot_orders, boot_weights)
            
            # Convert to elasticity (same logic as main calculation)
            if 'brand' in str(cost_data.name).lower():
                elasticity = abs(corr) * 0.22  # Brand multiplier
            else:
                elasticity = abs(corr) * 0.17  # Non-brand multiplier
            
            elasticities.append(elasticity)
        
        elasticities = np.array(elasticities)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(elasticities, lower_percentile)
        upper_ci = np.percentile(elasticities, upper_percentile)
        mean_elasticity = np.mean(elasticities)
        
        return mean_elasticity, lower_ci, upper_ci
    
    # Brand elasticity calculation with recency weighting and confidence intervals
    if 'brand_cost' in hist_data.columns and hist_data['brand_cost'].var() > 0:
        # Use separate brand_cost column if available
        brand_corr = weighted_correlation(
            hist_data['brand_cost'].values,
            hist_data['actual_orders'].values,
            hist_data['recency_weight'].values
        )
        
        # Calculate bootstrap confidence intervals
        hist_data['brand_cost'].name = 'brand_cost'  # Ensure name is set for bootstrap function
        brand_elasticity_boot, brand_lower_ci, brand_upper_ci = bootstrap_elasticity(
            hist_data['brand_cost'], hist_data['actual_orders'], hist_data['recency_weight'], n_bootstrap
        )
        
        # Debug: Add fallback to simple correlation if weighted fails
        if brand_corr == 0 or np.isnan(brand_corr):
            brand_corr_simple = hist_data['brand_cost'].corr(hist_data['actual_orders'])
            brand_corr = brand_corr_simple if not np.isnan(brand_corr_simple) else 0
        
        brand_mean_orders = np.average(hist_data['actual_orders'], weights=hist_data['recency_weight'])
        brand_mean_cost = np.average(hist_data['brand_cost'], weights=hist_data['recency_weight'])
        
        if brand_mean_orders > 0 and brand_mean_cost > 0:
            brand_elasticity = abs(brand_corr) * 0.22
            elasticities['baseline_orders_per_brand_dollar'] = brand_mean_orders / brand_mean_cost
        else:
            brand_elasticity = 0.15
            elasticities['baseline_orders_per_brand_dollar'] = 0.02
            brand_lower_ci = 0.05
            brand_upper_ci = 0.25
        
        # Store confidence intervals
        elasticities['brand_elasticity_lower_ci'] = brand_lower_ci
        elasticities['brand_elasticity_upper_ci'] = brand_upper_ci
        
        # Debug info
        elasticities['debug_brand_corr'] = brand_corr
        elasticities['debug_brand_mean_orders'] = brand_mean_orders
        elasticities['debug_brand_mean_cost'] = brand_mean_cost
    elif 'cost' in hist_data.columns and hist_data['cost'].var() > 0:
        # Fallback: use total cost as proxy for brand cost correlation
        total_cost_corr = weighted_correlation(
            hist_data['cost'].values,
            hist_data['actual_orders'].values,
            hist_data['recency_weight'].values
        )
        
        # Bootstrap for total cost (proxy for brand)
        hist_data['cost'].name = 'total_cost'
        _, total_lower_ci, total_upper_ci = bootstrap_elasticity(
            hist_data['cost'], hist_data['actual_orders'], hist_data['recency_weight'], n_bootstrap
        )
        
        total_mean_orders = np.average(hist_data['actual_orders'], weights=hist_data['recency_weight'])
        total_mean_cost = np.average(hist_data['cost'], weights=hist_data['recency_weight'])
        
        if total_mean_orders > 0 and total_mean_cost > 0:
            # Brand is typically more elastic than total, so boost the correlation
            brand_elasticity = abs(total_cost_corr) * 0.28  # Higher multiplier for brand
            elasticities['baseline_orders_per_brand_dollar'] = total_mean_orders / total_mean_cost
            # Adjust confidence intervals for brand (higher than total cost)
            elasticities['brand_elasticity_lower_ci'] = total_lower_ci * 1.2
            elasticities['brand_elasticity_upper_ci'] = total_upper_ci * 1.2
        else:
            brand_elasticity = 0.15
            elasticities['baseline_orders_per_brand_dollar'] = 0.02
            elasticities['brand_elasticity_lower_ci'] = 0.05
            elasticities['brand_elasticity_upper_ci'] = 0.25
    else:
        brand_elasticity = 0.15
        elasticities['baseline_orders_per_brand_dollar'] = 0.02
        elasticities['brand_elasticity_lower_ci'] = 0.05
        elasticities['brand_elasticity_upper_ci'] = 0.25
    
    # Non-brand elasticity calculation with recency weighting and confidence intervals
    if 'nonbrand_cost' in hist_data.columns and hist_data['nonbrand_cost'].var() > 0:
        # Use separate nonbrand_cost column if available
        nonbrand_corr = weighted_correlation(
            hist_data['nonbrand_cost'].values,
            hist_data['actual_orders'].values,
            hist_data['recency_weight'].values
        )
        
        # Calculate bootstrap confidence intervals
        hist_data['nonbrand_cost'].name = 'nonbrand_cost'
        nonbrand_elasticity_boot, nonbrand_lower_ci, nonbrand_upper_ci = bootstrap_elasticity(
            hist_data['nonbrand_cost'], hist_data['actual_orders'], hist_data['recency_weight'], n_bootstrap
        )
        
        # Debug: Add fallback to simple correlation if weighted fails
        if nonbrand_corr == 0 or np.isnan(nonbrand_corr):
            nonbrand_corr_simple = hist_data['nonbrand_cost'].corr(hist_data['actual_orders'])
            nonbrand_corr = nonbrand_corr_simple if not np.isnan(nonbrand_corr_simple) else 0
        
        nonbrand_mean_orders = np.average(hist_data['actual_orders'], weights=hist_data['recency_weight'])
        nonbrand_mean_cost = np.average(hist_data['nonbrand_cost'], weights=hist_data['recency_weight'])
        
        if nonbrand_mean_orders > 0 and nonbrand_mean_cost > 0:
            nonbrand_elasticity = abs(nonbrand_corr) * 0.17
            elasticities['baseline_orders_per_nonbrand_dollar'] = nonbrand_mean_orders / nonbrand_mean_cost
        else:
            nonbrand_elasticity = 0.08
            elasticities['baseline_orders_per_nonbrand_dollar'] = 0.01
            nonbrand_lower_ci = 0.02
            nonbrand_upper_ci = 0.15
        
        # Store confidence intervals
        elasticities['nonbrand_elasticity_lower_ci'] = nonbrand_lower_ci
        elasticities['nonbrand_elasticity_upper_ci'] = nonbrand_upper_ci
        
        # Debug info
        elasticities['debug_nonbrand_corr'] = nonbrand_corr
        elasticities['debug_nonbrand_mean_orders'] = nonbrand_mean_orders
        elasticities['debug_nonbrand_mean_cost'] = nonbrand_mean_cost
    elif 'cost' in hist_data.columns and hist_data['cost'].var() > 0:
        # Fallback: use total cost as proxy for nonbrand cost correlation  
        total_cost_corr = weighted_correlation(
            hist_data['cost'].values,
            hist_data['actual_orders'].values,
            hist_data['recency_weight'].values
        )
        
        # Use the same bootstrap results from earlier if available
        if 'cost' in hist_data.columns:
            hist_data['cost'].name = 'total_cost_nonbrand'
            _, total_lower_ci_nb, total_upper_ci_nb = bootstrap_elasticity(
                hist_data['cost'], hist_data['actual_orders'], hist_data['recency_weight'], n_bootstrap
            )
        
        total_mean_orders = np.average(hist_data['actual_orders'], weights=hist_data['recency_weight'])
        total_mean_cost = np.average(hist_data['cost'], weights=hist_data['recency_weight'])
        
        if total_mean_orders > 0 and total_mean_cost > 0:
            # Non-brand is typically less elastic than total
            nonbrand_elasticity = abs(total_cost_corr) * 0.15  # Lower multiplier for non-brand
            elasticities['baseline_orders_per_nonbrand_dollar'] = total_mean_orders / total_mean_cost
            # Adjust confidence intervals for non-brand (lower than total cost)
            elasticities['nonbrand_elasticity_lower_ci'] = total_lower_ci_nb * 0.8
            elasticities['nonbrand_elasticity_upper_ci'] = total_upper_ci_nb * 0.8
        else:
            nonbrand_elasticity = 0.08
            elasticities['baseline_orders_per_nonbrand_dollar'] = 0.01
            elasticities['nonbrand_elasticity_lower_ci'] = 0.02
            elasticities['nonbrand_elasticity_upper_ci'] = 0.15
    else:
        nonbrand_elasticity = 0.08
        elasticities['baseline_orders_per_nonbrand_dollar'] = 0.01
        elasticities['nonbrand_elasticity_lower_ci'] = 0.02
        elasticities['nonbrand_elasticity_upper_ci'] = 0.15
    
    # Calculate weighted brand share
    if 'brand_cost' in hist_data.columns and 'nonbrand_cost' in hist_data.columns:
        weighted_brand = np.average(hist_data['brand_cost'], weights=hist_data['recency_weight'])
        weighted_nonbrand = np.average(hist_data['nonbrand_cost'], weights=hist_data['recency_weight'])
        total_weighted_spend = weighted_brand + weighted_nonbrand
        brand_share = weighted_brand / total_weighted_spend if total_weighted_spend > 0 else 0.7
    else:
        brand_share = 0.7  # Default assumption
    
    # Calculate recency statistics for debugging
    recent_data_points = len(hist_data[hist_data['days_back'] <= recency_days])
    avg_weight_recent = hist_data[hist_data['days_back'] <= recency_days]['recency_weight'].mean() if recent_data_points > 0 else 0
    avg_weight_old = hist_data[hist_data['days_back'] > decay_half_life]['recency_weight'].mean() if len(hist_data[hist_data['days_back'] > decay_half_life]) > 0 else 0
    
    elasticities.update({
        'brand_elasticity': max(0.05, min(0.35, brand_elasticity)),  # Slightly higher cap due to recency focus
        'nonbrand_elasticity': max(0.02, min(0.25, nonbrand_elasticity)),  # Slightly higher cap due to recency focus
        'brand_share': brand_share,
        'recency_weighted': True,
        'recent_data_points': recent_data_points,
        'avg_weight_recent': avg_weight_recent,
        'avg_weight_old': avg_weight_old,
        'recency_days': recency_days,
        'decay_half_life': decay_half_life,
        'confidence_intervals': True,
        'confidence_level': confidence_level
    })
    
    return elasticities

def calculate_incremental_impact(baseline_table, scenario_table, elasticities):
    """Calculate incremental impact using elasticity curves with diminishing returns and confidence intervals"""
    
    baseline_future = baseline_table[baseline_table["actual_orders"].isna()].copy()
    scenario_future = scenario_table[scenario_table["actual_orders"].isna()].copy()
    
    if baseline_future.empty or scenario_future.empty:
        return 1.0, {}, 1.0, 1.0
    
    # CRITICAL FIX: Ensure baseline table has brand_cost and nonbrand_cost columns
    # The baseline table may not have these columns in future periods, so create them
    if 'brand_cost' not in baseline_future.columns or baseline_future['brand_cost'].sum() == 0:
        # Use the same logic as apply_budget_scenario to create baseline brand/nonbrand costs
        cost_col = None
        if 'planned_cost' in baseline_future.columns:
            cost_col = 'planned_cost'
        elif 'cost' in baseline_future.columns:
            cost_col = 'cost'
        
        if cost_col is not None:
            # Use historical brand share or default
            brand_share = elasticities.get('brand_share', 0.7)
            baseline_future['brand_cost'] = baseline_future[cost_col] * brand_share
            baseline_future['nonbrand_cost'] = baseline_future[cost_col] * (1 - brand_share)
    
    # Calculate spend changes
    brand_baseline = baseline_future['brand_cost'].sum() if 'brand_cost' in baseline_future.columns else 0
    brand_scenario = scenario_future['brand_cost'].sum() if 'brand_cost' in scenario_future.columns else 0
    
    nonbrand_baseline = baseline_future['nonbrand_cost'].sum() if 'nonbrand_cost' in baseline_future.columns else 0
    nonbrand_scenario = scenario_future['nonbrand_cost'].sum() if 'nonbrand_cost' in scenario_future.columns else 0
    
    # Calculate absolute spend changes
    brand_change = brand_scenario - brand_baseline
    nonbrand_change = nonbrand_scenario - nonbrand_baseline
    
    # Apply diminishing returns curve: impact = elasticity * ln(1 + % change)
    # Calculate point estimates and confidence intervals
    brand_impact = 0
    nonbrand_impact = 0
    brand_impact_lower = 0
    brand_impact_upper = 0
    nonbrand_impact_lower = 0
    nonbrand_impact_upper = 0
    
    # Get elasticity confidence intervals if available
    brand_elast = elasticities['brand_elasticity']
    brand_elast_lower = elasticities.get('brand_elasticity_lower_ci', brand_elast * 0.7)
    brand_elast_upper = elasticities.get('brand_elasticity_upper_ci', brand_elast * 1.3)
    
    nonbrand_elast = elasticities['nonbrand_elasticity']
    nonbrand_elast_lower = elasticities.get('nonbrand_elasticity_lower_ci', nonbrand_elast * 0.7)
    nonbrand_elast_upper = elasticities.get('nonbrand_elasticity_upper_ci', nonbrand_elast * 1.3)
    
    if brand_baseline > 0 and brand_change != 0:
        brand_pct_change = brand_change / brand_baseline
        # Diminishing returns: use log function to model saturation
        if brand_pct_change > 0:
            log_term = np.log(1 + brand_pct_change)
            brand_impact = brand_elast * log_term
            brand_impact_lower = brand_elast_lower * log_term
            brand_impact_upper = brand_elast_upper * log_term
        else:
            # For decreases, use linear relationship (no saturation effect)
            brand_impact = brand_elast * brand_pct_change
            brand_impact_lower = brand_elast_lower * brand_pct_change
            brand_impact_upper = brand_elast_upper * brand_pct_change
    
    if nonbrand_baseline > 0 and nonbrand_change != 0:
        nonbrand_pct_change = nonbrand_change / nonbrand_baseline
        if nonbrand_pct_change > 0:
            log_term_nb = np.log(1 + nonbrand_pct_change)
            nonbrand_impact = nonbrand_elast * log_term_nb
            nonbrand_impact_lower = nonbrand_elast_lower * log_term_nb
            nonbrand_impact_upper = nonbrand_elast_upper * log_term_nb
        else:
            nonbrand_impact = nonbrand_elast * nonbrand_pct_change
            nonbrand_impact_lower = nonbrand_elast_lower * nonbrand_pct_change
            nonbrand_impact_upper = nonbrand_elast_upper * nonbrand_pct_change
    
    # Total multiplicative impact (1 + impact) - point estimate and confidence intervals
    total_impact_factor = 1 + brand_impact + nonbrand_impact
    total_impact_factor_lower = 1 + brand_impact_lower + nonbrand_impact_lower
    total_impact_factor_upper = 1 + brand_impact_upper + nonbrand_impact_upper
    
    impact_details = {
        'brand_spend_change': brand_change,
        'nonbrand_spend_change': nonbrand_change,
        'brand_impact_pct': brand_impact * 100,
        'nonbrand_impact_pct': nonbrand_impact * 100,
        'total_impact_factor': total_impact_factor,
        'total_impact_factor_lower': total_impact_factor_lower,
        'total_impact_factor_upper': total_impact_factor_upper,
        'brand_baseline': brand_baseline,
        'nonbrand_baseline': nonbrand_baseline,
        # Confidence interval details
        'brand_impact_pct_lower': brand_impact_lower * 100,
        'brand_impact_pct_upper': brand_impact_upper * 100,
        'nonbrand_impact_pct_lower': nonbrand_impact_lower * 100,
        'nonbrand_impact_pct_upper': nonbrand_impact_upper * 100,
        # Add debugging details
        'brand_elasticity_used': elasticities.get('brand_elasticity', 0),
        'nonbrand_elasticity_used': elasticities.get('nonbrand_elasticity', 0),
        'brand_elasticity_lower_ci': brand_elast_lower,
        'brand_elasticity_upper_ci': brand_elast_upper,
        'nonbrand_elasticity_lower_ci': nonbrand_elast_lower,
        'nonbrand_elasticity_upper_ci': nonbrand_elast_upper,
        'brand_pct_change': brand_change / brand_baseline if brand_baseline > 0 else 0,
        'nonbrand_pct_change': nonbrand_change / nonbrand_baseline if nonbrand_baseline > 0 else 0,
        'raw_brand_impact': brand_impact,
        'raw_nonbrand_impact': nonbrand_impact,
        # Additional debug info
        'brand_log_term': np.log(1 + brand_change / brand_baseline) if brand_baseline > 0 and brand_change > 0 else 0,
        'nonbrand_log_term': np.log(1 + nonbrand_change / nonbrand_baseline) if nonbrand_baseline > 0 and nonbrand_change > 0 else 0,
        'sum_impacts': brand_impact + nonbrand_impact,
        # Debug the baseline creation
        'debug_baseline_had_brand_cost': 'brand_cost' in baseline_table[baseline_table["actual_orders"].isna()].columns,
        'debug_baseline_brand_sum_original': baseline_table[baseline_table["actual_orders"].isna()]['brand_cost'].sum() if 'brand_cost' in baseline_table[baseline_table["actual_orders"].isna()].columns else 'N/A'
    }
    
    return total_impact_factor, impact_details, total_impact_factor_lower, total_impact_factor_upper

def generate_scenario_forecast(scenario_table, baseline_table=None):
    """Generate new forecasts using elasticity-based modeling"""
    
    # Get future data from scenario table
    future_data = scenario_table[scenario_table["actual_orders"].isna()].copy()
    
    if future_data.empty or baseline_table is None:
        return None, None, None, {}
    
    # Calculate historical elasticities
    elasticities = calculate_historical_elasticities(baseline_table)
    
    # Calculate incremental impact with diminishing returns and confidence intervals
    impact_factor, impact_details, impact_factor_lower, impact_factor_upper = calculate_incremental_impact(baseline_table, scenario_table, elasticities)
    
    # Apply impact to baseline forecasts - use baseline forecast as starting point
    baseline_future = baseline_table[baseline_table["actual_orders"].isna()].copy()
    scenario_orders_fc = baseline_future[['ds']].copy()
    
    if 'orders' in baseline_future.columns and len(baseline_future) > 0:
        # Apply elasticity-based impact factor to baseline forecast
        scenario_orders_fc['orders'] = baseline_future['orders'] * impact_factor
        
        # Add elasticity confidence intervals
        scenario_orders_fc['orders_lower_elasticity'] = baseline_future['orders'] * impact_factor_lower
        scenario_orders_fc['orders_upper_elasticity'] = baseline_future['orders'] * impact_factor_upper
        
        # Combine with baseline Prophet confidence intervals if they exist
        if 'orders_upper' in baseline_future.columns and 'orders_lower' in baseline_future.columns:
            # Use wider of the two uncertainty sources
            baseline_lower = baseline_future['orders_lower'] * impact_factor
            baseline_upper = baseline_future['orders_upper'] * impact_factor
            
            # Take the most conservative (widest) bounds
            scenario_orders_fc['orders_lower'] = np.minimum(baseline_lower, scenario_orders_fc['orders_lower_elasticity'])
            scenario_orders_fc['orders_upper'] = np.maximum(baseline_upper, scenario_orders_fc['orders_upper_elasticity'])
        else:
            # Use only elasticity confidence intervals
            scenario_orders_fc['orders_lower'] = scenario_orders_fc['orders_lower_elasticity']
            scenario_orders_fc['orders_upper'] = scenario_orders_fc['orders_upper_elasticity']
    else:
        scenario_orders_fc['orders'] = 0
        scenario_orders_fc['orders_lower'] = 0
        scenario_orders_fc['orders_upper'] = 0
    
    # Clicks and impressions respond differently - they're more directly tied to spend
    clicks_impact = 1 + (impact_factor - 1) * 0.9  # 90% of orders impact
    impressions_impact = 1 + (impact_factor - 1) * 1.1  # 110% of orders impact (more responsive)
    
    scenario_clicks_fc = baseline_future[['ds']].copy()
    if 'clicks' in baseline_future.columns and len(baseline_future) > 0:
        scenario_clicks_fc['clicks'] = baseline_future['clicks'] * clicks_impact
        
        # Add confidence intervals if they exist
        if 'clicks_upper' in baseline_future.columns:
            scenario_clicks_fc['clicks_upper'] = baseline_future['clicks_upper'] * clicks_impact
        if 'clicks_lower' in baseline_future.columns:
            scenario_clicks_fc['clicks_lower'] = baseline_future['clicks_lower'] * clicks_impact
    else:
        scenario_clicks_fc['clicks'] = 0
    
    scenario_impr_fc = baseline_future[['ds']].copy()
    if 'impressions' in baseline_future.columns and len(baseline_future) > 0:
        scenario_impr_fc['impressions'] = baseline_future['impressions'] * impressions_impact
        
        # Add confidence intervals if they exist
        if 'impressions_upper' in baseline_future.columns:
            scenario_impr_fc['impressions_upper'] = baseline_future['impressions_upper'] * impressions_impact
        if 'impressions_lower' in baseline_future.columns:
            scenario_impr_fc['impressions_lower'] = baseline_future['impressions_lower'] * impressions_impact
    else:
        scenario_impr_fc['impressions'] = 0
    
    scenario_metrics = {
        'total_orders': scenario_orders_fc['orders'].sum(),
        'total_orders_lower': scenario_orders_fc['orders_lower'].sum() if 'orders_lower' in scenario_orders_fc.columns else 0,
        'total_orders_upper': scenario_orders_fc['orders_upper'].sum() if 'orders_upper' in scenario_orders_fc.columns else 0,
        'impact_factor': impact_factor,
        'impact_factor_lower': impact_factor_lower,
        'impact_factor_upper': impact_factor_upper,
        'elasticities': elasticities,
        'impact_details': impact_details
    }
    
    return scenario_orders_fc, scenario_clicks_fc, scenario_impr_fc, scenario_metrics

def create_scenario_comparison_chart(baseline_table, scenario_table, baseline_orders_fc, scenario_orders_fc):
    """Create comparison chart between baseline and scenario forecasts"""
    fig = go.Figure()
    
    # Historical data (same for both)
    hist_data = baseline_table[baseline_table["actual_orders"].notna()]
    if not hist_data.empty:
        fig.add_trace(go.Scatter(
            x=hist_data["ds"],
            y=hist_data["actual_orders"],
            mode='lines+markers',
            name='Historical Orders',
            line=dict(color='black', width=2)
        ))
    
    # Baseline forecast
    if baseline_orders_fc is not None and not baseline_orders_fc.empty:
        fig.add_trace(go.Scatter(
            x=baseline_orders_fc["ds"],
            y=baseline_orders_fc["orders"],
            mode='lines+markers',
            name='Baseline Forecast',
            line=dict(color='blue', width=2)
        ))
    
    # Scenario forecast
    if scenario_orders_fc is not None and not scenario_orders_fc.empty:
        fig.add_trace(go.Scatter(
            x=scenario_orders_fc["ds"],
            y=scenario_orders_fc["orders"],
            mode='lines+markers',
            name='Scenario Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence intervals for scenario forecast if available
        if 'orders_upper' in scenario_orders_fc.columns and 'orders_lower' in scenario_orders_fc.columns:
            fig.add_trace(go.Scatter(
                x=scenario_orders_fc["ds"],
                y=scenario_orders_fc["orders_upper"],
                mode='lines',
                name='Scenario Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=scenario_orders_fc["ds"],
                y=scenario_orders_fc["orders_lower"],
                mode='lines',
                name='Scenario 90% CI',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0)
            ))
    
    fig.update_layout(
        title="Baseline vs Scenario Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Orders",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)
    )
    
    return fig

def create_weekly_forecast_charts(weekly_table, title_suffix="Weekly"):
    """Create weekly forecast charts using consolidated chart creation function"""
    colors_map = {
        'orders': {'actual': 'black', 'forecast': 'blue', 'ci': 'rgba(0,100,80,0.2)'},
        'clicks': {'actual': 'black', 'forecast': 'green', 'ci': 'rgba(0,128,0,0.2)'},
        'impressions': {'actual': 'black', 'forecast': 'red', 'ci': 'rgba(255,0,0,0.2)'}
    }
    
    # Create charts with custom titles and axis labels for weekly view
    fig_orders = create_single_forecast_chart(weekly_table, 'orders', None, weekly_table, 
                                           colors_map['orders'], True, f"{title_suffix} ")
    fig_orders.update_layout(xaxis_title="Week Starting")
    
    fig_clicks = create_single_forecast_chart(weekly_table, 'clicks', None, weekly_table,
                                           colors_map['clicks'], True, f"{title_suffix} ")
    fig_clicks.update_layout(xaxis_title="Week Starting")
    
    fig_impressions = create_single_forecast_chart(weekly_table, 'impressions', None, weekly_table,
                                                colors_map['impressions'], True, f"{title_suffix} ")
    fig_impressions.update_layout(xaxis_title="Week Starting")
    
    return fig_orders, fig_clicks, fig_impressions

# --------------------------------------------------------------------------- #
# Streamlit App Layout                                                         #
# --------------------------------------------------------------------------- #

st.title("📈 SEM Forecast Dashboard")
st.markdown("Upload your SEM data to generate forecasts with interactive visualizations and detailed analysis.")

# Sidebar for file upload and controls
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="File must contain columns: date, orders, clicks, impressions, cost, brand_cost, nonbrand_cost, promo_flag"
)

if uploaded_file is not None:
    # Load and process data
    df = load_dataframe(uploaded_file)
    
    if df is not None:
        st.sidebar.success(f"✅ File loaded successfully! {len(df)} rows")
        
        # Show data info
        with st.sidebar.expander("📊 Data Summary"):
            st.write(f"**Date range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Historical records:** {df['orders'].notna().sum()}")
            st.write(f"**Future records:** {df['orders'].isna().sum()}")
        
        # Run forecasting pipeline
        with st.spinner("🔮 Generating forecasts..."):
            table, orders_fc, clicks_fc, impr_fc = run_sem_pipeline(df)
        
        if table is not None:
            # Calculate metrics
            metrics = calculate_summary_metrics(table)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Daily Dashboard", "📅 Weekly Dashboard", "📋 Data Tables", "🔮 Scenario Planning", "📈 Advanced Analytics"])
            
            with tab1:
                st.header("Daily Forecast Dashboard")
                
                # Dynamic date range selector and data filtering
                filtered_table, filtered_orders_fc, filtered_clicks_fc, filtered_impr_fc, start_ts, end_ts = create_dynamic_trendline_chart(
                    table, orders_fc, clicks_fc, impr_fc, "daily"
                )
                
                # Calculate metrics for filtered data
                filtered_metrics = calculate_summary_metrics(filtered_table)
                
                # Key metrics row
                if filtered_metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Forecast Period", f"{filtered_metrics['forecast_period_days']} days")
                    with col2:
                        st.metric("Total Forecast Orders", f"{filtered_metrics['total_forecast_orders']:,.0f}")
                    with col3:
                        st.metric("Avg Daily Orders", f"{filtered_metrics['avg_daily_orders']:,.0f}")
                    with col4:
                        if filtered_metrics['total_planned_cost'] > 0:
                            st.metric("Forecast CPA", f"${filtered_metrics['forecast_cpa']:,.2f}")
                        else:
                            st.metric("Total Planned Cost", "Not available")
                
                # Charts with hierarchical dynamic trendlines
                fig_orders, fig_clicks, fig_impressions = create_enhanced_forecast_charts(
                    filtered_table, filtered_orders_fc, filtered_clicks_fc, filtered_impr_fc,
                    table, orders_fc, clicks_fc, impr_fc
                )
                
                # Display charts
                st.plotly_chart(fig_orders, use_container_width=True, key="orders_chart")
                
                st.plotly_chart(fig_clicks, use_container_width=True, key="clicks_chart")
                st.plotly_chart(fig_impressions, use_container_width=True, key="impressions_chart")
                
                # Add hierarchical trendline info box
                st.success("🎯 **Hierarchical Dynamic Trendlines**: \n1️⃣ **Date Range Boxes** set the base data range \n2️⃣ **Quick Buttons** (30D, 3M) override to show recent trends \n3️⃣ **Range Slider** provides the most specific filtering - drag to see trends for any window!")
            
            with tab2:
                st.header("Weekly Forecast Dashboard")
                
                # Aggregate data to weekly view
                weekly_table = aggregate_weekly_data(table)
                
                # Dynamic date range selector for weekly data
                filtered_weekly_table, _, _, _, weekly_start_ts, weekly_end_ts = create_dynamic_trendline_chart(
                    weekly_table, None, None, None, "weekly"
                )
                
                # Weekly metrics for filtered data
                last_actual_weekly = filtered_weekly_table[filtered_weekly_table["actual_orders"].notna()]["ds"].max() if not filtered_weekly_table[filtered_weekly_table["actual_orders"].notna()].empty else None
                weekly_forecast_data = filtered_weekly_table[filtered_weekly_table["ds"] > last_actual_weekly].copy() if last_actual_weekly is not None else filtered_weekly_table[filtered_weekly_table["orders"].notna()].copy()
                
                if not weekly_forecast_data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Forecast Period", f"{len(weekly_forecast_data)} weeks")
                    with col2:
                        st.metric("Total Forecast Orders", f"{weekly_forecast_data['orders'].sum():,.0f}")
                    with col3:
                        st.metric("Avg Weekly Orders", f"{weekly_forecast_data['orders'].mean():,.0f}")
                    with col4:
                        if 'planned_cost' in weekly_forecast_data.columns and weekly_forecast_data['planned_cost'].sum() > 0:
                            weekly_cpa = weekly_forecast_data['planned_cost'].sum() / weekly_forecast_data['orders'].sum()
                            st.metric("Weekly Forecast CPA", f"${weekly_cpa:,.2f}")
                        else:
                            st.metric("Weekly Planned Cost", "Not available")
                
                # Weekly Charts with truly dynamic trendlines based on filtered data
                fig_orders_weekly, fig_clicks_weekly, fig_impressions_weekly = create_weekly_forecast_charts(filtered_weekly_table, "Weekly")
                
                st.plotly_chart(fig_orders_weekly, use_container_width=True, key="weekly_orders_chart")
                
                st.plotly_chart(fig_clicks_weekly, use_container_width=True, key="weekly_clicks_chart")
                st.plotly_chart(fig_impressions_weekly, use_container_width=True, key="weekly_impressions_chart")
                
                # Add weekly hierarchical trendline info
                st.success("🎯 **Weekly Hierarchical Trendlines**: Use Date Range → Quick Buttons → Range Slider for progressively more specific trend analysis!")
            
            with tab3:
                st.header("Forecast Data Tables")
                
                # View selector
                view_type = st.radio("Select View", ["Daily View", "Weekly View", "Monthly View"], horizontal=True)
                
                if view_type == "Weekly View":
                    # Weekly data table
                    weekly_table = aggregate_weekly_data(table)
                    display_table = weekly_table.copy()
                    time_label = "Week Starting"
                    file_suffix = "weekly"
                elif view_type == "Monthly View":
                    # Monthly data table
                    monthly_table = aggregate_monthly_data(table)
                    display_table = monthly_table.copy()
                    time_label = "Month Starting"
                    file_suffix = "monthly"
                else:
                    # Daily data table
                    display_table = table.copy()
                    time_label = "Date"
                    file_suffix = "daily"
                
                # Date range selector (applies to all views)
                if not display_table.empty:
                    min_date = display_table['ds'].min().date()
                    max_date = display_table['ds'].max().date()
                    
                    st.subheader("📅 Date Range Filter")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_date = st.date_input(
                            "Start Date",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date
                        )
                    
                    with col2:
                        end_date = st.date_input(
                            "End Date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date
                        )
                    
                    # Apply date range filter
                    display_table = display_table[
                        (display_table['ds'].dt.date >= start_date) & 
                        (display_table['ds'].dt.date <= end_date)
                    ]
                
                # Data type filters
                col1, col2 = st.columns(2)
                with col1:
                    show_historical = st.checkbox("Show Historical Data", value=True)
                with col2:
                    show_forecast = st.checkbox("Show Forecast Data", value=True)
                
                # Filter data based on selection
                if not show_historical:
                    display_table = display_table[display_table["actual_orders"].isna()]
                if not show_forecast:
                    display_table = display_table[display_table["actual_orders"].notna()]
                
                # Format the table for better display
                display_table = display_table.copy()
                if not display_table.empty:
                    if view_type == "Weekly View":
                        display_table["ds"] = display_table["ds"].dt.strftime("%Y-%m-%d (Week Start)")
                    elif view_type == "Monthly View":
                        display_table["ds"] = display_table["ds"].dt.strftime("%Y-%m (Month Start)")
                    else:
                        display_table["ds"] = display_table["ds"].dt.strftime("%Y-%m-%d")
                
                # Rename ds column for clarity
                display_table = display_table.rename(columns={"ds": time_label})
                
                # Round numeric columns
                numeric_cols = display_table.select_dtypes(include=[np.number]).columns
                display_table[numeric_cols] = display_table[numeric_cols].round(0)
                
                # Calculate CPO, CPC, and CPM metrics
                display_table = calculate_marketing_metrics(display_table)
                
                # Format cost columns
                for col in ["cost", "planned_cost"]:
                    if col in display_table.columns:
                        display_table[col] = display_table[col].apply(
                            lambda x: f"${x:,.0f}" if pd.notnull(x) else ""
                        )
                
                # Format CPO, CPC, and CPM columns
                for col in ["CPO", "CPC", "CPM"]:
                    if col in display_table.columns:
                        display_table[col] = display_table[col].apply(
                            lambda x: f"${x:.2f}" if pd.notnull(x) else ""
                        )
                
                st.dataframe(
                    display_table,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = display_table.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download CSV ({view_type})",
                    data=csv,
                    file_name=f'forecast_results_{file_suffix}_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            
            with tab4:
                st.header("🔮 Scenario Planning")
                
                st.markdown("**What-if Analysis**: Adjust budget scenarios to see forecast impact")
                
                with st.expander("📖 Methodology"):
                    st.markdown("""
                    **Elasticity-Based Modeling with Recency Weighting:**
                    
                    1. **Historical Analysis**: Calculates separate elasticities for brand vs non-brand spend based on your historical data
                    2. **Recency Weighting**: Recent performance is weighted more heavily:
                       - Last 14 days get 50% boost in importance
                       - Exponential decay with 30-day half-life (data loses 50% weight every 30 days)
                       - This ensures recent market conditions drive predictions vs old patterns
                    3. **Differential Impact**: Brand and non-brand changes have different impacts based on:
                       - Weighted historical correlation with orders
                       - Relative spend share in your recent budget mix
                       - Different elasticity coefficients
                    4. **Diminishing Returns**: Uses logarithmic curves for spend increases (no diminishing effect for decreases)
                    5. **Incremental Modeling**: Applies `elasticity × ln(1 + % change)` for realistic impact calculation
                    
                    This approach prioritizes recent performance patterns while accounting for differential channel efficiency.
                    """)
                
                # Scenario controls
                st.subheader("Budget Adjustment Scenarios")
                
                col1, col2 = st.columns(2)
                with col1:
                    brand_adjustment = st.slider(
                        "Brand Budget Change (%)", 
                        min_value=-50, max_value=100, value=0, step=5,
                        help="Adjust brand budget by percentage"
                    )
                with col2:
                    nonbrand_adjustment = st.slider(
                        "Non-Brand Budget Change (%)", 
                        min_value=-50, max_value=100, value=0, step=5,
                        help="Adjust non-brand budget by percentage"
                    )
                
                # Apply scenario if adjustments are made
                if brand_adjustment != 0 or nonbrand_adjustment != 0:
                    scenario_table = apply_budget_scenario(table, brand_adjustment, nonbrand_adjustment)
                    
                    # Generate scenario forecasts - pass baseline table for comparison
                    scenario_orders_fc, scenario_clicks_fc, scenario_impr_fc, scenario_metrics = generate_scenario_forecast(
                        scenario_table, table
                    )
                    
                    
                    # Display scenario comparison
                    st.subheader("📊 Scenario vs Baseline Comparison")
                    
                    # Create comparison chart
                    scenario_comparison_chart = create_scenario_comparison_chart(
                        table, scenario_table, orders_fc, scenario_orders_fc
                    )
                    st.plotly_chart(scenario_comparison_chart, use_container_width=True)
                    
                    # Scenario metrics comparison
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        baseline_total = orders_fc['orders'].sum() if orders_fc is not None else 0
                        scenario_total = scenario_orders_fc['orders'].sum() if scenario_orders_fc is not None else 0
                        impact = scenario_total - baseline_total
                        
                        # Get confidence interval for orders impact
                        scenario_lower = scenario_metrics.get('total_orders_lower', scenario_total)
                        scenario_upper = scenario_metrics.get('total_orders_upper', scenario_total)
                        impact_lower = scenario_lower - baseline_total
                        impact_upper = scenario_upper - baseline_total
                        
                        st.metric(
                            "Orders Impact", 
                            f"{impact:+,.0f}",
                            delta=f"{(impact/baseline_total*100):+.1f}%" if baseline_total > 0 else "N/A"
                        )
                    
                    with col2:
                        # Get future data for both scenarios
                        last_actual = table[table['actual_orders'].notna()]['ds'].max()  
                        baseline_future = table[table['ds'] > last_actual] if not pd.isna(last_actual) else table[table['actual_orders'].isna()]
                        scenario_future = scenario_table[scenario_table['ds'] > last_actual] if not pd.isna(last_actual) else scenario_table[scenario_table['actual_orders'].isna()]
                        
                        # Calculate costs using the best available cost column
                        def get_total_cost(df):
                            if 'planned_cost' in df.columns and df['planned_cost'].notna().any():
                                return df['planned_cost'].sum()
                            elif 'cost' in df.columns and df['cost'].notna().any():
                                return df['cost'].sum()
                            elif 'brand_cost' in df.columns and 'nonbrand_cost' in df.columns:
                                return df['brand_cost'].fillna(0).sum() + df['nonbrand_cost'].fillna(0).sum()
                            else:
                                return 0
                        
                        baseline_cost = get_total_cost(baseline_future)
                        scenario_cost = get_total_cost(scenario_future)
                        cost_impact = scenario_cost - baseline_cost
                        
                        st.metric(
                            "Cost Impact", 
                            f"${cost_impact:+,.0f}",
                            delta=f"{(cost_impact/baseline_cost*100):+.1f}%" if baseline_cost > 0 else "N/A"
                        )
                    
                    with col3:
                        if baseline_cost > 0 and scenario_cost > 0 and baseline_total > 0 and scenario_total > 0:
                            baseline_cpa = baseline_cost / baseline_total
                            scenario_cpa = scenario_cost / scenario_total
                            cpa_impact = scenario_cpa - baseline_cpa
                            st.metric(
                                "CPA Impact", 
                                f"${cpa_impact:+.2f}",
                                delta=f"{(cpa_impact/baseline_cpa*100):+.1f}%" if baseline_cpa > 0 else "N/A"
                            )
                        else:
                            st.metric("CPA Impact", "N/A", help="Insufficient data for CPA calculation")
                
                else:
                    st.info("👆 Adjust the budget sliders above to see scenario forecasts")
            
            with tab5:
                st.header("📈 Advanced Analytics")
                
                # Performance metrics
                if metrics:
                    st.subheader("📊 Forecast Confidence Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Confidence Range (Orders)", 
                            f"{metrics['confidence_range_orders']:,.0f}",
                            help="Total difference between upper and lower confidence bounds"
                        )
                    with col2:
                        if metrics['total_forecast_orders'] > 0:
                            confidence_pct = (metrics['confidence_range_orders'] / metrics['total_forecast_orders']) * 100
                            st.metric(
                                "Relative Uncertainty", 
                                f"{confidence_pct:.1f}%",
                                help="Confidence range as percentage of total forecast"
                            )
                
                # Historical vs Forecast comparison
                st.subheader("📈 Historical vs Forecast Trends")
                
                last_actual = table[table["actual_orders"].notna()]["ds"].max()
                recent_hist = table[
                    (table["actual_orders"].notna()) & 
                    (table["ds"] >= last_actual - pd.Timedelta(days=30))
                ]
                forecast_30d = table[
                    (table["ds"] > last_actual) & 
                    (table["ds"] <= last_actual + pd.Timedelta(days=30))
                ]
                
                if len(recent_hist) > 0 and len(forecast_30d) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Avg Orders (Last 30d Historical)", 
                            f"{recent_hist['actual_orders'].mean():,.0f}"
                        )
                    with col2:
                        st.metric(
                            "Avg Orders (Next 30d Forecast)", 
                            f"{forecast_30d['orders'].mean():,.0f}"
                        )
                
                # Show model parameters used
                st.subheader("🔧 Model Configuration")
                st.json({
                    "changepoint_prior_scale": 0.5,
                    "seasonality_mode": "multiplicative",
                    "confidence_interval": "90%",
                    "training_window": "730 days",
                    "regressors": BASE_REGS + ["clicks", "impressions"]
                })

else:
    st.info("👆 Please upload a CSV or Excel file to get started.")
    
    # Show sample data format
    with st.expander("📋 Required Data Format"):
        st.markdown("""
        Your file must contain these columns:
        - **date**: Date column (YYYY-MM-DD format)
        - **orders**: Historical orders (use null/empty for future dates to forecast)
        - **clicks**: Historical clicks
        - **impressions**: Historical impressions  
        - **cost**: Historical cost
        - **brand_cost**: Brand spend regressor
        - **nonbrand_cost**: Non-brand spend regressor
        - **promo_flag**: Promotional activity flag (0/1)
        
        For forecasting, leave future dates with null values in the 'orders', 'clicks', and 'impressions' columns,
        but fill in the regressor columns (brand_cost, nonbrand_cost, promo_flag, cost).
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Prophet • Enhanced SEM Forecasting Dashboard")