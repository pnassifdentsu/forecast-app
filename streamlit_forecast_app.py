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

def aggregate_monthly_data(table):
    """Aggregate daily data to monthly view"""
    monthly_table = table.copy()
    
    # Add month start date
    monthly_table['month_start'] = monthly_table['ds'].dt.to_period('M').dt.start_time
    
    # Sum columns - but handle NaN values properly
    sum_cols = ['actual_orders', 'orders', 'actual_clicks', 'clicks', 'actual_impressions', 
                'impressions', 'cost', 'planned_cost', 'orders_lower', 'orders_upper', 
                'clicks_lower', 'clicks_upper', 'impressions_lower', 'impressions_upper']
    
    # Group by month and aggregate
    monthly_data = monthly_table.groupby('month_start').apply(
        lambda group: pd.Series({
            col: group[col].sum() if group[col].notna().any() else np.nan
            for col in sum_cols if col in group.columns
        })
    ).reset_index()
    
    monthly_data.rename(columns={'month_start': 'ds'}, inplace=True)
    
    return monthly_data

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
    """Generate new forecasts using sophisticated elasticity-based modeling"""
    
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
                st.header("🔮 Scenario Planning")
                
                st.markdown("**What-if Analysis**: Adjust budget scenarios to see forecast impact")
                
                with st.expander("📖 Methodology"):
                    st.markdown("""
                    **Sophisticated Elasticity-Based Modeling with Recency Weighting:**
                    
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
                    
                    # Enhanced debug information showing elasticity analysis - Always visible for troubleshooting
                    st.subheader("🔍 Elasticity Analysis & Debug")
                    if True:  # Temporarily always show debug info
                        st.write("**Available columns in data:**", list(table.columns))
                        future_data = table[table["actual_orders"].isna()]
                        st.write(f"**Future periods found:** {len(future_data)} days")
                        
                        # Debug scenario calculations
                        st.write("**Scenario Debug:**")
                        st.write(f"Brand adjustment: {brand_adjustment}%")
                        st.write(f"Non-brand adjustment: {nonbrand_adjustment}%")
                        st.write("**Prophet Model Uses:** brand_cost, nonbrand_cost, promo_flag, clicks, impressions")
                        
                        # Check if adjustments were applied
                        baseline_future = table[table["actual_orders"].isna()]
                        scenario_future_debug = scenario_table[scenario_table["actual_orders"].isna()]
                        
                        if len(baseline_future) > 0 and len(scenario_future_debug) > 0:
                            for col in ['brand_cost', 'nonbrand_cost', 'cost', 'planned_cost']:
                                if col in baseline_future.columns:
                                    baseline_sum = baseline_future[col].sum()
                                    scenario_sum = scenario_future_debug[col].sum()
                                    change = scenario_sum - baseline_sum
                                    st.write(f"- {col}: ${baseline_sum:,.0f} → ${scenario_sum:,.0f} (${change:+,.0f})")
                        
                        # Check orders calculation
                        baseline_orders_sum = baseline_future['orders'].sum() if 'orders' in baseline_future.columns else 0
                        scenario_orders_sum = scenario_orders_fc['orders'].sum() if scenario_orders_fc is not None else 0
                        orders_change = scenario_orders_sum - baseline_orders_sum
                        st.write(f"**Orders Forecast:**")
                        st.write(f"- Baseline: {baseline_orders_sum:,.0f} orders")
                        st.write(f"- Scenario: {scenario_orders_sum:,.0f} orders")
                        st.write(f"- Incremental: {orders_change:+,.0f} orders")
                        if baseline_orders_sum > 0:
                            percent_change = (orders_change / baseline_orders_sum) * 100
                            st.write(f"- % Change: {percent_change:+.1f}%")
                        
                        # Show impact factor calculation
                        impact_factor = scenario_metrics.get('impact_factor', 1.0)
                        st.write(f"**Impact Factor Applied: {impact_factor:.3f}**")
                        if impact_factor == 1.0:
                            st.error("⚠️ Impact factor is 1.0 - no incremental effect calculated!")
                            st.write("This usually means:")
                            st.write("- No cost changes were detected")
                            st.write("- Elasticity calculations returned zero")
                            st.write("- Brand/non-brand columns are missing or empty")
                        else:
                            st.success(f"✅ Impact factor calculated: {impact_factor:.3f} ({(impact_factor-1)*100:+.1f}% change)")
                        
                        # Show elasticity calculations
                        if 'elasticities' in scenario_metrics:
                            elasticities = scenario_metrics['elasticities']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Brand Elasticity", f"{elasticities['brand_elasticity']:.1%}")
                                if elasticities.get('confidence_intervals', False):
                                    brand_lower = elasticities.get('brand_elasticity_lower_ci', 0)
                                    brand_upper = elasticities.get('brand_elasticity_upper_ci', 0)
                                    st.caption(f"95% CI: [{brand_lower:.1%} - {brand_upper:.1%}]")
                                st.metric("Brand Share", f"{elasticities['brand_share']:.1%}")
                                if 'debug_brand_corr' in elasticities:
                                    st.write(f"Brand correlation: {elasticities['debug_brand_corr']:.3f}")
                            with col2:
                                st.metric("Non-Brand Elasticity", f"{elasticities['nonbrand_elasticity']:.1%}")
                                if elasticities.get('confidence_intervals', False):
                                    nonbrand_lower = elasticities.get('nonbrand_elasticity_lower_ci', 0)
                                    nonbrand_upper = elasticities.get('nonbrand_elasticity_upper_ci', 0)
                                    st.caption(f"95% CI: [{nonbrand_lower:.1%} - {nonbrand_upper:.1%}]")
                                st.metric("Non-Brand Share", f"{(1-elasticities['brand_share']):.1%}")
                                if 'debug_nonbrand_corr' in elasticities:
                                    st.write(f"Non-brand correlation: {elasticities['debug_nonbrand_corr']:.3f}")
                            with col3:
                                st.metric("Elasticity Ratio", f"{elasticities['brand_elasticity']/elasticities['nonbrand_elasticity']:.1f}x")
                                if elasticities.get('confidence_intervals', False):
                                    st.caption(f"Bootstrap CI ({elasticities.get('confidence_level', 0.9)*100:.0f}%)")
                                
                            # Show detailed correlation debug
                            if 'debug_brand_corr' in elasticities:
                                st.write("**Correlation Debug:**")
                                st.write(f"- Brand: {elasticities['debug_brand_corr']:.4f} correlation → {elasticities['brand_elasticity']:.4f} elasticity")
                                st.write(f"- Non-brand: {elasticities['debug_nonbrand_corr']:.4f} correlation → {elasticities['nonbrand_elasticity']:.4f} elasticity")
                                
                            # Show recency weighting info if available
                            if elasticities.get('recency_weighted', False):
                                st.write("**📅 Recency Weighting Applied:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"Recent data points (≤{elasticities['recency_days']}d): {elasticities['recent_data_points']}")
                                with col2:
                                    st.write(f"Avg weight (recent): {elasticities['avg_weight_recent']:.1f}")
                                with col3:
                                    st.write(f"Avg weight (old): {elasticities['avg_weight_old']:.1f}")
                                
                                st.write(f"Decay half-life: {elasticities['decay_half_life']} days (50% weight reduction)")
                            else:
                                st.info("Using default elasticities (insufficient historical data)")
                            
                            # Show impact details if available
                            if 'impact_details' in scenario_metrics:
                                details = scenario_metrics['impact_details']
                                st.write("**Spend Change Analysis:**")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"- Brand baseline: ${details['brand_baseline']:,.0f}")
                                    st.write(f"- Brand change: ${details['brand_spend_change']:+,.0f}")
                                    st.write(f"- Brand % change: {details['brand_pct_change']:+.1%}")
                                    st.write(f"- Brand elasticity: {details['brand_elasticity_used']:.3f}")
                                    st.write(f"- Raw brand impact: {details['raw_brand_impact']:+.4f}")
                                    st.write(f"- Brand impact: {details['brand_impact_pct']:+.1f}%")
                                with col2:
                                    st.write(f"- Non-brand baseline: ${details['nonbrand_baseline']:,.0f}")
                                    st.write(f"- Non-brand change: ${details['nonbrand_spend_change']:+,.0f}")
                                    st.write(f"- Non-brand % change: {details['nonbrand_pct_change']:+.1%}")
                                    st.write(f"- Non-brand elasticity: {details['nonbrand_elasticity_used']:.3f}")
                                    st.write(f"- Raw non-brand impact: {details['raw_nonbrand_impact']:+.4f}")
                                    st.write(f"- Non-brand impact: {details['nonbrand_impact_pct']:+.1f}%")
                                
                                st.write(f"**Total Impact Factor:** {details['total_impact_factor']:.3f} ({(details['total_impact_factor']-1)*100:+.1f}%)")
                                
                                # Show confidence intervals for impact factor
                                if 'total_impact_factor_lower' in details and 'total_impact_factor_upper' in details:
                                    lower_factor = details['total_impact_factor_lower']
                                    upper_factor = details['total_impact_factor_upper']
                                    st.write(f"**Impact Factor 90% CI:** [{lower_factor:.3f} - {upper_factor:.3f}] ({(lower_factor-1)*100:+.1f}% to {(upper_factor-1)*100:+.1f}%)")
                                    
                                    # Show impact range in terms of orders
                                    baseline_total = orders_fc['orders'].sum() if orders_fc is not None else 0
                                    if baseline_total > 0:
                                        lower_orders = baseline_total * (lower_factor - 1)
                                        upper_orders = baseline_total * (upper_factor - 1)
                                        st.write(f"**Estimated Orders Impact Range:** {lower_orders:+,.0f} to {upper_orders:+,.0f} orders")
                                
                                # Show the math breakdown
                                st.write("**Impact Calculation Breakdown:**")
                                st.write(f"- Brand: {details['brand_elasticity_used']:.4f} × ln(1+{details['brand_pct_change']:.3f}) = {details['brand_elasticity_used']:.4f} × {details['brand_log_term']:.4f} = {details['raw_brand_impact']:.4f}")
                                st.write(f"- Non-brand: {details['nonbrand_elasticity_used']:.4f} × ln(1+{details['nonbrand_pct_change']:.3f}) = {details['nonbrand_elasticity_used']:.4f} × {details['nonbrand_log_term']:.4f} = {details['raw_nonbrand_impact']:.4f}")
                                st.write(f"- Sum of impacts: {details['sum_impacts']:.4f}")
                                st.write(f"- Final factor: 1 + {details['sum_impacts']:.4f} = {details['total_impact_factor']:.4f}")
                                
                                # Highlight the problem if elasticities are zero
                                if details['brand_elasticity_used'] == 0 and details['nonbrand_elasticity_used'] == 0:
                                    st.error("🚨 Both elasticities are ZERO! This means no historical correlation was found between spend and orders.")
                                elif details['brand_elasticity_used'] == 0:
                                    st.warning("⚠️ Brand elasticity is ZERO - no historical correlation found.")  
                                elif details['nonbrand_elasticity_used'] == 0:
                                    st.warning("⚠️ Non-brand elasticity is ZERO - no historical correlation found.")
                                elif details['sum_impacts'] == 0:
                                    st.error("🚨 Sum of impacts is ZERO despite non-zero elasticities! Check the calculation logic.")
                                    st.write(f"Debug: Baseline had brand_cost: {details.get('debug_baseline_had_brand_cost', 'Unknown')}")
                                    st.write(f"Debug: Original baseline brand sum: {details.get('debug_baseline_brand_sum_original', 'Unknown')}")
                    
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
                        if scenario_lower != scenario_total or scenario_upper != scenario_total:
                            st.caption(f"90% CI: [{impact_lower:+,.0f} to {impact_upper:+,.0f}]")
                    
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