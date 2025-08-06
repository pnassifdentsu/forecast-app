# SEM Forecast Dashboard 📈

A comprehensive Streamlit application for Search Engine Marketing (SEM) forecasting with advanced analytics and visualization capabilities.

## Features

- **Data Upload & Validation**: Support for CSV, XLS, and XLSX files
- **Prophet-based Forecasting**: Advanced time series forecasting using Facebook Prophet
- **Interactive Dashboard**: Dynamic visualizations with Plotly
- **Scenario Planning**: Model different marketing scenarios
- **Elasticity Analysis**: Advanced elasticity modeling with statistical analysis
- **Export Capabilities**: Download forecasts and visualizations

## Required Data Columns

Your data file must include these columns:
- `date`: Date column
- `orders`: Order/conversion data
- `clicks`: Click data
- `impressions`: Impression data
- `cost`: Total cost
- `brand_cost`: Brand campaign costs
- `nonbrand_cost`: Non-brand campaign costs
- `promo_flag`: Promotional flag (0/1)

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

1. **Upload Data**: Upload your SEM data file (CSV/Excel)
2. **Configure Parameters**: Set forecast period and scenario parameters
3. **Run Forecast**: Generate Prophet-based forecasts
4. **Analyze Results**: View interactive charts and statistical analysis
5. **Export Results**: Download forecasts and visualizations

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