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
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
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

def create_enhanced_forecast_charts(filtered_table, filtered_orders_fc, filtered_clicks_fc, filtered_impr_fc, 
                                  original_table, original_orders_fc, original_clicks_fc, original_impr_fc):
    """Create forecast charts with enhanced hierarchical filtering capabilities"""
    
    # This approach creates charts with the filtered data but includes information about the full dataset
    # The user can then use Plotly's built-in controls to further filter within this dataset
    
    # Determine split point
    last_actual = filtered_table[filtered_table["actual_orders"].notna()]["ds"].max() if not filtered_table[filtered_table["actual_orders"].notna()].empty else None
    
    # Orders Chart
    fig_orders = go.Figure()
    
    # Historical data
    hist_data = filtered_table[filtered_table["actual_orders"].notna()]
    if not hist_data.empty:
        fig_orders.add_trace(go.Scatter(
            x=hist_data["ds"],
            y=hist_data["actual_orders"],
            mode='lines+markers',
            name='Actual Orders',
            line=dict(color='black', width=2)
        ))
    
    # Forecast data
    forecast_data = filtered_table[filtered_table["ds"] > last_actual] if last_actual is not None else filtered_table[filtered_table["orders"].notna()]
    if not forecast_data.empty:
        fig_orders.add_trace(go.Scatter(
            x=forecast_data["ds"],
            y=forecast_data["orders"],
            mode='lines+markers',  
            name='Forecast Orders',
            line=dict(color='blue', width=2)
        ))
        
        # Confidence intervals
        if 'orders_upper' in forecast_data.columns and 'orders_lower' in forecast_data.columns:
            fig_orders.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["orders_upper"],
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_orders.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["orders_lower"],
                mode='lines',
                name='90% Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(width=0)
            ))
    
    # Add trendlines for currently filtered data
    if not hist_data.empty and len(hist_data) >= 2:
        fig_orders = add_simple_trendline(fig_orders, hist_data["ds"], hist_data["actual_orders"], "Historical Orders", "gray")
    if not forecast_data.empty and len(forecast_data) >= 2:
        fig_orders = add_simple_trendline(fig_orders, forecast_data["ds"], forecast_data["orders"], "Forecast Orders", "lightblue")
    
    # Configure with hierarchical controls
    fig_orders = create_chart_with_hierarchical_controls(fig_orders, original_table, 'orders')
    fig_orders.update_layout(
        title="Orders Forecast with Hierarchical Dynamic Trendlines",
        xaxis_title="Date",
        yaxis_title="Orders",
        hovermode='x unified'
    )
    
    # Apply same pattern to other charts
    # Clicks Chart
    fig_clicks = go.Figure()
    hist_clicks = filtered_table[filtered_table["actual_clicks"].notna()]
    if not hist_clicks.empty:
        fig_clicks.add_trace(go.Scatter(
            x=hist_clicks["ds"],
            y=hist_clicks["actual_clicks"],
            mode='lines+markers',
            name='Actual Clicks',
            line=dict(color='black', width=2)
        ))
    
    if filtered_clicks_fc is not None and not filtered_clicks_fc.empty:
        fig_clicks.add_trace(go.Scatter(
            x=filtered_clicks_fc["ds"],
            y=filtered_clicks_fc["clicks"],
            mode='lines+markers',
            name='Forecast Clicks',
            line=dict(color='green', width=2)
        ))
    
    # Add trendlines
    if not hist_clicks.empty and len(hist_clicks) >= 2:
        fig_clicks = add_simple_trendline(fig_clicks, hist_clicks["ds"], hist_clicks["actual_clicks"], "Historical Clicks", "gray")
    if filtered_clicks_fc is not None and not filtered_clicks_fc.empty and len(filtered_clicks_fc) >= 2:
        fig_clicks = add_simple_trendline(fig_clicks, filtered_clicks_fc["ds"], filtered_clicks_fc["clicks"], "Forecast Clicks", "lightgreen")
    
    fig_clicks = create_chart_with_hierarchical_controls(fig_clicks, original_clicks_fc, 'clicks')
    fig_clicks.update_layout(
        title="Clicks Forecast with Hierarchical Dynamic Trendlines",
        xaxis_title="Date",
        yaxis_title="Clicks",
        hovermode='x unified'
    )
    
    # Impressions Chart
    fig_impressions = go.Figure()
    hist_impressions = filtered_table[filtered_table["actual_impressions"].notna()]
    if not hist_impressions.empty:
        fig_impressions.add_trace(go.Scatter(
            x=hist_impressions["ds"],
            y=hist_impressions["actual_impressions"],
            mode='lines+markers',
            name='Actual Impressions',
            line=dict(color='black', width=2)
        ))
    
    if filtered_impr_fc is not None and not filtered_impr_fc.empty:
        fig_impressions.add_trace(go.Scatter(
            x=filtered_impr_fc["ds"],
            y=filtered_impr_fc["impressions"],
            mode='lines+markers',
            name='Forecast Impressions',
            line=dict(color='red', width=2)
        ))
    
    # Add trendlines
    if not hist_impressions.empty and len(hist_impressions) >= 2:
        fig_impressions = add_simple_trendline(fig_impressions, hist_impressions["ds"], hist_impressions["actual_impressions"], "Historical Impressions", "gray")
    if filtered_impr_fc is not None and not filtered_impr_fc.empty and len(filtered_impr_fc) >= 2:
        fig_impressions = add_simple_trendline(fig_impressions, filtered_impr_fc["ds"], filtered_impr_fc["impressions"], "Forecast Impressions", "lightcoral")
    
    fig_impressions = create_chart_with_hierarchical_controls(fig_impressions, original_impr_fc, 'impressions')
    fig_impressions.update_layout(
        title="Impressions Forecast with Hierarchical Dynamic Trendlines",
        xaxis_title="Date",
        yaxis_title="Impressions",
        hovermode='x unified'
    )
    
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
    
    # Determine split point between historical and forecast data
    last_actual = table[table["actual_orders"].notna()]["ds"].max() if not table[table["actual_orders"].notna()].empty else None
    
    # Orders Chart
    fig_orders = go.Figure()
    
    # Historical data
    hist_data = table[table["actual_orders"].notna()]
    if not hist_data.empty:
        fig_orders.add_trace(go.Scatter(
            x=hist_data["ds"],
            y=hist_data["actual_orders"],
            mode='lines+markers',
            name='Actual Orders',
            line=dict(color='black', width=2)
        ))
    
    # Forecast data
    forecast_data = table[table["ds"] > last_actual] if last_actual is not None else table[table["orders"].notna()]
    if not forecast_data.empty:
        fig_orders.add_trace(go.Scatter(
            x=forecast_data["ds"],
            y=forecast_data["orders"],
            mode='lines+markers',
            name='Forecast Orders',
            line=dict(color='blue', width=2)
        ))
    
    # Confidence intervals
    fig_orders.add_trace(go.Scatter(
        x=forecast_data["ds"],
        y=forecast_data["orders_upper"],
        mode='lines',
        name='Upper CI',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig_orders.add_trace(go.Scatter(
        x=forecast_data["ds"],
        y=forecast_data["orders_lower"],
        mode='lines',
        name='90% Confidence Interval',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(width=0)
    ))
    
    # Add simple trendlines (hidden from legend)
    if not hist_data.empty and len(hist_data) >= 2:
        fig_orders = add_simple_trendline(fig_orders, hist_data["ds"], hist_data["actual_orders"], 
                                        "Historical Orders", "gray")
    if not forecast_data.empty and len(forecast_data) >= 2:
        fig_orders = add_simple_trendline(fig_orders, forecast_data["ds"], forecast_data["orders"], 
                                        "Forecast Orders", "lightblue")

    # Apply chart controls  
    fig_orders = create_chart_with_hierarchical_controls(fig_orders, table, 'orders')
    fig_orders.update_layout(
        title="Orders Forecast with Dynamic Trendlines",
        xaxis_title="Date",
        yaxis_title="Orders",
        hovermode='x unified'
    )
    
    # Clicks Chart
    fig_clicks = go.Figure()
    
    # Historical clicks
    hist_clicks = table[table["actual_clicks"].notna()]
    if not hist_clicks.empty:
        fig_clicks.add_trace(go.Scatter(
            x=hist_clicks["ds"],
            y=hist_clicks["actual_clicks"],
            mode='lines+markers',
            name='Actual Clicks',
            line=dict(color='black', width=2)
        ))
    
    # Forecast clicks
    if clicks_fc is not None and not clicks_fc.empty:
        fig_clicks.add_trace(go.Scatter(
            x=clicks_fc["ds"],
            y=clicks_fc["clicks"],
            mode='lines+markers',
            name='Forecast Clicks',
            line=dict(color='green', width=2)
        ))
    
    # Confidence intervals for clicks
    if clicks_fc is not None and not clicks_fc.empty and 'clicks_upper' in clicks_fc.columns:
        fig_clicks.add_trace(go.Scatter(
            x=clicks_fc["ds"],
            y=clicks_fc["clicks_upper"],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_clicks.add_trace(go.Scatter(
            x=clicks_fc["ds"],
            y=clicks_fc["clicks_lower"],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0,128,0,0.2)',
            line=dict(width=0),
            name='90% Confidence Interval'
        ))
    
    # Add simple trendlines (hidden from legend)
    if not hist_clicks.empty and len(hist_clicks) >= 2:
        fig_clicks = add_simple_trendline(fig_clicks, hist_clicks["ds"], hist_clicks["actual_clicks"], 
                                        "Historical Clicks", "gray")
    if clicks_fc is not None and not clicks_fc.empty and len(clicks_fc) >= 2:
        fig_clicks = add_simple_trendline(fig_clicks, clicks_fc["ds"], clicks_fc["clicks"], 
                                        "Forecast Clicks", "lightgreen")

    # Apply chart controls
    fig_clicks = create_chart_with_hierarchical_controls(fig_clicks, table, 'clicks')  
    fig_clicks.update_layout(
        title="Clicks Forecast with Dynamic Trendlines",
        xaxis_title="Date",
        yaxis_title="Clicks",
        hovermode='x unified'
    )
    
    # Impressions Chart
    fig_impressions = go.Figure()
    
    # Historical impressions
    hist_impressions = table[table["actual_impressions"].notna()]
    if not hist_impressions.empty:
        fig_impressions.add_trace(go.Scatter(
            x=hist_impressions["ds"],
            y=hist_impressions["actual_impressions"],
            mode='lines+markers',
            name='Actual Impressions',
            line=dict(color='black', width=2)
        ))
    
    # Forecast impressions
    if impr_fc is not None and not impr_fc.empty:
        fig_impressions.add_trace(go.Scatter(
            x=impr_fc["ds"],
            y=impr_fc["impressions"],
            mode='lines+markers',
            name='Forecast Impressions',
            line=dict(color='red', width=2)
        ))
    
    # Confidence intervals for impressions
    if impr_fc is not None and not impr_fc.empty and 'impressions_upper' in impr_fc.columns:
        fig_impressions.add_trace(go.Scatter(
            x=impr_fc["ds"],
            y=impr_fc["impressions_upper"],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_impressions.add_trace(go.Scatter(
            x=impr_fc["ds"],
            y=impr_fc["impressions_lower"],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0),
            name='90% Confidence Interval'
        ))
    
    # Add simple trendlines (hidden from legend)
    if not hist_impressions.empty and len(hist_impressions) >= 2:
        fig_impressions = add_simple_trendline(fig_impressions, hist_impressions["ds"], hist_impressions["actual_impressions"], 
                                             "Historical Impressions", "gray")
    if impr_fc is not None and not impr_fc.empty and len(impr_fc) >= 2:
        fig_impressions = add_simple_trendline(fig_impressions, impr_fc["ds"], impr_fc["impressions"], 
                                             "Forecast Impressions", "lightcoral")

    # Apply chart controls
    fig_impressions = create_chart_with_hierarchical_controls(fig_impressions, table, 'impressions')
    fig_impressions.update_layout(
        title="Impressions Forecast with Dynamic Trendlines",
        xaxis_title="Date",
        yaxis_title="Impressions",
        hovermode='x unified'
    )
    
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

def aggregate_weekly_data(table):
    """Aggregate daily data to weekly view"""
    weekly_table = table.copy()
    
    # Add week start date
    weekly_table['week_start'] = weekly_table['ds'].dt.to_period('W').dt.start_time
    
    # Define aggregation functions for different column types
    agg_funcs = {}
    
    # Sum columns - but handle NaN values properly
    sum_cols = ['actual_orders', 'orders', 'actual_clicks', 'clicks', 'actual_impressions', 
                'impressions', 'cost', 'planned_cost', 'orders_lower', 'orders_upper', 
                'clicks_lower', 'clicks_upper', 'impressions_lower', 'impressions_upper']
    
    for col in sum_cols:
        if col in weekly_table.columns:
            # Use sum with skipna=True, but convert 0 sums back to NaN where all values were NaN
            agg_funcs[col] = lambda x: x.sum() if x.notna().any() else np.nan
    
    # Group by week and aggregate
    weekly_data = weekly_table.groupby('week_start').apply(
        lambda group: pd.Series({
            col: group[col].sum() if group[col].notna().any() else np.nan
            for col in sum_cols if col in group.columns
        })
    ).reset_index()
    
    weekly_data.rename(columns={'week_start': 'ds'}, inplace=True)
    
    return weekly_data

def create_weekly_forecast_charts(weekly_table, title_suffix="Weekly"):
    """Create weekly forecast charts"""
    
    # Determine split point between historical and forecast data
    last_actual = weekly_table[weekly_table["actual_orders"].notna()]["ds"].max()
    
    # Orders Chart
    fig_orders = go.Figure()
    
    # Historical data
    hist_data = weekly_table[weekly_table["actual_orders"].notna()]
    if not hist_data.empty:
        fig_orders.add_trace(go.Scatter(
            x=hist_data["ds"],
            y=hist_data["actual_orders"],
            mode='lines+markers',
            name='Actual Orders',
            line=dict(color='black', width=2)
        ))
    
    # Forecast data
    forecast_data = weekly_table[weekly_table["ds"] > last_actual] if not pd.isna(last_actual) else weekly_table[weekly_table["orders"].notna()]
    if not forecast_data.empty:
        fig_orders.add_trace(go.Scatter(
            x=forecast_data["ds"],
            y=forecast_data["orders"],
            mode='lines+markers',
            name='Forecast Orders',
            line=dict(color='blue', width=2)
        ))
        
        # Confidence intervals
        if 'orders_upper' in forecast_data.columns and 'orders_lower' in forecast_data.columns:
            fig_orders.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["orders_upper"],
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_orders.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["orders_lower"],
                mode='lines',
                name='90% Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(width=0)
            ))
    
    # Add simple trendlines for weekly orders (hidden from legend)
    if not hist_data.empty and len(hist_data) >= 2:
        fig_orders = add_simple_trendline(fig_orders, hist_data["ds"], hist_data["actual_orders"], "Historical Orders", "gray")
    if not forecast_data.empty and len(forecast_data) >= 2:
        fig_orders = add_simple_trendline(fig_orders, forecast_data["ds"], forecast_data["orders"], "Forecast Orders", "lightblue")

    # Apply chart controls
    fig_orders = create_chart_with_hierarchical_controls(fig_orders, weekly_table, f'weekly_orders')
    fig_orders.update_layout(
        title=f"Orders Forecast - {title_suffix} View with Dynamic Trendlines", 
        xaxis_title="Week Starting",
        yaxis_title="Orders",
        hovermode='x unified'
    )
    
    # Clicks Chart
    fig_clicks = go.Figure()
    
    # Historical clicks data
    hist_clicks = weekly_table[weekly_table["actual_clicks"].notna()]
    if not hist_clicks.empty:
        fig_clicks.add_trace(go.Scatter(
            x=hist_clicks["ds"],
            y=hist_clicks["actual_clicks"],
            mode='lines+markers',
            name='Actual Clicks',
            line=dict(color='black', width=2)
        ))
    
    # Forecast clicks data
    clicks_forecast = weekly_table[weekly_table["ds"] > last_actual] if not pd.isna(last_actual) else weekly_table[weekly_table["clicks"].notna()]
    if not clicks_forecast.empty:
        fig_clicks.add_trace(go.Scatter(
            x=clicks_forecast["ds"],
            y=clicks_forecast["clicks"],
            mode='lines+markers',
            name='Forecast Clicks',
            line=dict(color='green', width=2)
        ))
        
        if 'clicks_upper' in clicks_forecast.columns and 'clicks_lower' in clicks_forecast.columns:
            fig_clicks.add_trace(go.Scatter(
                x=clicks_forecast["ds"],
                y=clicks_forecast["clicks_upper"],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_clicks.add_trace(go.Scatter(
                x=clicks_forecast["ds"],
                y=clicks_forecast["clicks_lower"],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,128,0,0.2)',
                line=dict(width=0),
                name='90% Confidence Interval'
            ))
    
    # Add simple trendlines for weekly clicks (hidden from legend)
    if not hist_clicks.empty and len(hist_clicks) >= 2:
        fig_clicks = add_simple_trendline(fig_clicks, hist_clicks["ds"], hist_clicks["actual_clicks"], "Historical Clicks", "gray")
    if not clicks_forecast.empty and len(clicks_forecast) >= 2:
        fig_clicks = add_simple_trendline(fig_clicks, clicks_forecast["ds"], clicks_forecast["clicks"], "Forecast Clicks", "lightgreen")

    # Apply chart controls
    fig_clicks = create_chart_with_hierarchical_controls(fig_clicks, weekly_table, f'weekly_clicks')
    fig_clicks.update_layout(
        title=f"Clicks Forecast - {title_suffix} View with Dynamic Trendlines",
        xaxis_title="Week Starting", 
        yaxis_title="Clicks",
        hovermode='x unified'
    )
    
    # Impressions Chart
    fig_impressions = go.Figure()
    
    # Historical impressions data
    hist_impressions = weekly_table[weekly_table["actual_impressions"].notna()]
    if not hist_impressions.empty:
        fig_impressions.add_trace(go.Scatter(
            x=hist_impressions["ds"],
            y=hist_impressions["actual_impressions"],
            mode='lines+markers',
            name='Actual Impressions',
            line=dict(color='black', width=2)
        ))
    
    # Forecast impressions data
    impr_forecast = weekly_table[weekly_table["ds"] > last_actual] if not pd.isna(last_actual) else weekly_table[weekly_table["impressions"].notna()]
    if not impr_forecast.empty:
        fig_impressions.add_trace(go.Scatter(
            x=impr_forecast["ds"],
            y=impr_forecast["impressions"],
            mode='lines+markers',
            name='Forecast Impressions',
            line=dict(color='red', width=2)
        ))
        
        if 'impressions_upper' in impr_forecast.columns and 'impressions_lower' in impr_forecast.columns:
            fig_impressions.add_trace(go.Scatter(
                x=impr_forecast["ds"],
                y=impr_forecast["impressions_upper"],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_impressions.add_trace(go.Scatter(
                x=impr_forecast["ds"],
                y=impr_forecast["impressions_lower"],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                name='90% Confidence Interval'
            ))
    
    # Add simple trendlines for weekly impressions (hidden from legend)
    if not hist_impressions.empty and len(hist_impressions) >= 2:
        fig_impressions = add_simple_trendline(fig_impressions, hist_impressions["ds"], hist_impressions["actual_impressions"], "Historical Impressions", "gray")
    if not impr_forecast.empty and len(impr_forecast) >= 2:
        fig_impressions = add_simple_trendline(fig_impressions, impr_forecast["ds"], impr_forecast["impressions"], "Forecast Impressions", "lightcoral")

    # Apply chart controls
    fig_impressions = create_chart_with_hierarchical_controls(fig_impressions, weekly_table, f'weekly_impressions')
    fig_impressions.update_layout(
        title=f"Impressions Forecast - {title_suffix} View with Dynamic Trendlines",
        xaxis_title="Week Starting",
        yaxis_title="Impressions", 
        hovermode='x unified'
    )
    
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
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Dashboard", "📅 Weekly Dashboard", "📋 Data Tables", "📈 Advanced Analytics"])
            
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_clicks, use_container_width=True, key="clicks_chart")
                with col2:
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_clicks_weekly, use_container_width=True, key="weekly_clicks_chart")
                with col2:
                    st.plotly_chart(fig_impressions_weekly, use_container_width=True, key="weekly_impressions_chart")
                
                # Add weekly hierarchical trendline info
                st.success("🎯 **Weekly Hierarchical Trendlines**: Use Date Range → Quick Buttons → Range Slider for progressively more specific trend analysis!")
            
            with tab3:
                st.header("Forecast Data Tables")
                
                # View selector
                view_type = st.radio("Select View", ["Daily View", "Weekly View"], horizontal=True)
                
                if view_type == "Weekly View":
                    # Weekly data table
                    weekly_table = aggregate_weekly_data(table)
                    display_table = weekly_table.copy()
                    time_label = "Week Starting"
                    file_suffix = "weekly"
                else:
                    # Daily data table
                    display_table = table.copy()
                    time_label = "Date"
                    file_suffix = "daily"
                
                # Add filters
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
                if view_type == "Weekly View":
                    display_table["ds"] = display_table["ds"].dt.strftime("%Y-%m-%d (Week Start)")
                else:
                    display_table["ds"] = display_table["ds"].dt.strftime("%Y-%m-%d")
                
                # Rename ds column for clarity
                display_table = display_table.rename(columns={"ds": time_label})
                
                # Round numeric columns
                numeric_cols = display_table.select_dtypes(include=[np.number]).columns
                display_table[numeric_cols] = display_table[numeric_cols].round(0)
                
                # Format cost columns
                for col in ["cost", "planned_cost"]:
                    if col in display_table.columns:
                        display_table[col] = display_table[col].apply(
                            lambda x: f"${x:,.0f}" if pd.notnull(x) else ""
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
                st.header("Advanced Analytics")
                
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