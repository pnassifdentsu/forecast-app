# SEM Forecast Dashboard 📈

A comprehensive Streamlit application for Search Engine Marketing (SEM) forecasting with advanced analytics and visualization capabilities, including specialized New Customer Order (NCOS) forecasting.

## Features

- **Data Upload & Validation**: Support for CSV, XLS, and XLSX files
- **Prophet-based Forecasting**: Advanced time series forecasting using Facebook Prophet
- **NCOS Forecasting**: Dedicated forecasting for New Customer Orders alongside regular orders
- **Interactive Dashboard**: Dynamic visualizations with Plotly and metric toggle functionality
- **Dual-Metric Analysis**: Switch between Orders and NCOS views across all dashboards
- **Cost Efficiency Metrics**: CPO (Cost Per Order) and NCOS CPO calculations
- **Scenario Planning**: Model different marketing scenarios for both Orders and NCOS
- **Elasticity Analysis**: Advanced elasticity modeling with statistical analysis
- **Weekly/Daily Views**: Comprehensive time-based aggregation with NCOS support
- **Export Capabilities**: Download forecasts and visualizations

## Required Data Columns

Your data file must include these columns:
- `date`: Date column
- `orders`: Order/conversion data
- `ncos`: New Customer Orders (orders from customers who have never placed an order before)
- `clicks`: Click data
- `impressions`: Impression data
- `cost`: Total cost
- `brand_cost`: Brand campaign costs
- `nonbrand_cost`: Non-brand campaign costs
- `promo_flag`: Promotional flag (0/1)

## Key Metrics Calculated

- **Orders Metrics**: Total Orders, CPO (Cost Per Order), CPA (Cost Per Acquisition)
- **NCOS Metrics**: Total NCOS, NCOS CPO, NCOS CPA, New Customer Share
- **Performance Metrics**: CPC (Cost Per Click), CPM (Cost Per Mille)
- **Relationship Metrics**: NCOS Share, Returning Orders, Cost Efficiency Ratios

## Installation

1. Clone this repository:
```bash
git clone https://github.com/pnassifdentsu/forecast-app.git
cd forecast-app
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_forecast_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Application Workflow

1. **Upload Data**: Upload your SEM data file (CSV/Excel) with NCOS column
2. **Select Metrics**: Choose between Orders or NCOS (New Customer Orders) analysis
3. **Configure Parameters**: Set forecast period and optimization options
4. **Run Forecast**: Generate Prophet-based forecasts for both Orders and NCOS
5. **Analyze Results**: 
   - **Daily Dashboard**: Interactive charts with Orders/NCOS toggle
   - **Weekly Dashboard**: Aggregated weekly analysis with metric selection
   - **Data Tables**: Detailed tables with CPO and NCOS CPO calculations
   - **Scenario Planning**: Budget and market condition modeling
6. **Export Results**: Download forecasts and visualizations

## Dashboard Tabs

- **📊 Daily Dashboard**: Daily forecasts with Orders/NCOS toggle and dynamic trendlines
- **📅 Weekly Dashboard**: Weekly aggregated views with metric selection
- **📋 Data Tables**: Detailed tabular data with cost metrics (CPO, NCOS CPO, CPC, CPM)
- **🔮 Scenario Planning**: Budget scenarios and market condition modeling
- **📈 Advanced Analytics**: Model performance metrics and optimization details
- **🌍 Market Conditions**: Configure market condition multipliers

## Dependencies

- streamlit>=1.28.0
- pandas>=1.5.0
- numpy>=1.21.0
- plotly>=5.0.0
- prophet>=1.1.0
- openpyxl>=3.0.0
- xlrd>=2.0.0
- scikit-learn>=1.0.0
- scipy>=1.7.0

## Project Structure

```
forecast-app/
├── streamlit_forecast_app.py    # Main Streamlit application
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
└── test files/                 # Sample data and test files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is proprietary software developed for internal use.