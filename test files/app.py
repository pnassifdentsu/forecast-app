# app.py
from flask import Flask, render_template, request, jsonify
from datetime import timedelta
import warnings
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ---------- helper: werkzeug FileStorage → pandas DataFrame ----------
from pathlib import Path
from werkzeug.exceptions import BadRequest

ALLOWED_EXT = {".csv", ".xls", ".xlsx"}

def load_dataframe(upload) -> pd.DataFrame:
    """Read the uploaded file into a DataFrame and ensure a 'date' column."""
    filename = upload.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        raise BadRequest(f"Unsupported file type: {ext} (allowed: {ALLOWED_EXT})")

    try:
        if ext == ".csv":
            df = pd.read_csv(upload, parse_dates=["date"])
        else:                              # Excel (first sheet)
            df = pd.read_excel(upload, sheet_name=0, parse_dates=["date"])
    except Exception as exc:
        raise BadRequest(f"Could not read file: {exc}")

    if "date" not in df.columns:
        raise BadRequest("File must contain a 'date' column.")

    return df

# ---------------------------------------------------------------------
# 0️  LOAD RAW DATA ONCE (Excel → DataFrame)
# ---------------------------------------------------------------------
from pathlib import Path

# 0️  LOAD RAW DATA ONCE (CSV → DataFrame)
BASE_DIR  = Path(__file__).resolve().parent     # folder where app.py lives
FILE_PATH = BASE_DIR / "sample.csv"             # no back‑slash headaches
RAW_DF    = pd.read_csv(FILE_PATH, parse_dates=["date"])


# ---------------------------------------------------------------------
# 1️  CONFIG BUILDER (returns one dict to rule them all)
# ---------------------------------------------------------------------
def build_config(
    *,
    future_budgets: dict[tuple[int, int], float] | None = None,  # {(yr, mon): budget}
    forecast_days: int | None = None,
    date_col: str   = "date",
    target_col: str = "Orders",
    cost_col: str   = "cost",
    interval_width: float = 0.80,
    optimization_trials: int = 10,
    cv_initial: str = "180 days",
    cv_period: str  = "7 days",
    cv_horizon: str = "30 days",
) -> dict:
    """Return a config dict consumed by the rest of the pipeline."""
    cfg = {
        "date_col": date_col,
        "target_col": target_col,
        "cost_col": cost_col,
        "interval_width": interval_width,
        "optimization_trials": optimization_trials,
        "cv_initial": cv_initial,
        "cv_period": cv_period,
        "cv_horizon": cv_horizon,
        "future_budgets": future_budgets or {},
    }

    # Resolve forecast horizon ------------------------------------------
    if forecast_days is not None:                # explicit beats implicit
        cfg["forecast_days"] = forecast_days
    elif cfg["future_budgets"]:                  # derive from last budget
        last_hist_date          = RAW_DF[date_col].max()
        last_budget_year, last_budget_mon = max(cfg["future_budgets"].keys())
        last_budget_day = (pd.Timestamp(year=last_budget_year,
                                        month=last_budget_mon,
                                        day=1)
                           + pd.offsets.MonthEnd(1))
        cfg["forecast_days"] = (last_budget_day - last_hist_date).days
    else:                                        # default fallback
        cfg["forecast_days"] = 30

    return cfg


# ---------------------------------------------------------------------
# 2️  PRE‑PROCESS DATA (clean + derive Prophet frame)
# ---------------------------------------------------------------------
def preprocess(df: pd.DataFrame, cfg: dict):
    df = df.copy()
    df = df.dropna(subset=[cfg["date_col"]])
    dcol, tcol, ccol = cfg["date_col"], cfg["target_col"], cfg["cost_col"]

    # Basic hygiene -----------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df[dcol]):
        df[dcol] = pd.to_datetime(df[dcol])

    df.sort_values(dcol, inplace=True)
    df[tcol].fillna(method="ffill", inplace=True)
    df[ccol].fillna(method="ffill", inplace=True)

    # Prophet frame -----------------------------------------------------
    prophet_df           = df[[dcol, tcol]].rename(columns={dcol: "ds", tcol: "y"})
    prophet_df[ccol]     = df[ccol].values
    last_hist_date       = df[dcol].max()

    # Extra calendar cols for budget allocation ------------------------
    df["dayofweek"] = df[dcol].dt.dayofweek
    df["month"]     = df[dcol].dt.month
    df["year"]      = df[dcol].dt.year

    return {"df": df, "prophet_df": prophet_df, "last_date": last_hist_date}


# ---------------------------------------------------------------------
# 3️  BUDGET ALLOCATION (spread monthly targets over weekdays)
# ---------------------------------------------------------------------
def allocate_budget(pre, cfg):
    df           = pre["df"]
    last_date    = pre["last_date"]
    dcol         = cfg["date_col"]
    ccol         = cfg["cost_col"]
    horizon      = cfg["forecast_days"]
    budgets      = cfg["future_budgets"]

    # Historical weekday pattern (use recent data for mix %) -----------
    recent_hist      = df[df[dcol] >= pd.Timestamp("2025-01-01")]
    weekday_spend    = recent_hist.groupby("dayofweek")[ccol].sum()
    weekday_pct      = weekday_spend / weekday_spend.sum()

    # Skeleton of future dates -----------------------------------------
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]
    fut_df = pd.DataFrame({"ds": future_dates})
    fut_df["dayofweek"] = fut_df["ds"].dt.dayofweek
    fut_df["month"]     = fut_df["ds"].dt.month
    fut_df["year"]      = fut_df["ds"].dt.year
    fut_df[ccol]        = 0.0

    # Distribute each monthly budget over that month’s weekdays --------
    for (yr, mo), budget in budgets.items():
        mask = (fut_df["year"] == yr) & (fut_df["month"] == mo)
        month_rows = fut_df.loc[mask]
        if month_rows.empty or budget == 0:
            continue

        counts     = month_rows["dayofweek"].value_counts()
        total_pct  = sum(weekday_pct.get(dow, 0) * cnt for dow, cnt in counts.items())
        if total_pct == 0:
            continue

        scale = budget / total_pct
        for idx, row in month_rows.iterrows():
            fut_df.at[idx, ccol] = weekday_pct.get(row["dayofweek"], 0) * scale

    return fut_df


# ---------------------------------------------------------------------
# 4️  HYPERPARAM SEARCH (placeholder → returns static params)
# ---------------------------------------------------------------------
def get_prophet_params(cfg: dict) -> dict:
    return {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale":   10.0,
        "seasonality_mode":       "additive",
        "changepoint_range":      0.8,
        "interval_width":         cfg["interval_width"],
    }


# ---------------------------------------------------------------------
# 5️  FORECAST (Prophet fit + predict)
# ---------------------------------------------------------------------
def forecast(prophet_df, future_df, params, cfg):
    ccol = cfg["cost_col"]

    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=params["holidays_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        changepoint_range=params["changepoint_range"],
        interval_width=params["interval_width"],
    )

    model.add_regressor(ccol)
    model.fit(prophet_df)

    future_full            = model.make_future_dataframe(periods=len(future_df))
    future_full[ccol]      = pd.concat([prophet_df[ccol], future_df[ccol]], ignore_index=True)
    fcst                   = model.predict(future_full)
    fcst = fcst.merge(prophet_df[["ds", "y"]], on="ds", how="left")

    # Monthly roll‑up ---------------------------------------------------
    fcst["year"]  = fcst["ds"].dt.year
    fcst["month"] = fcst["ds"].dt.month
    mthly = (
        fcst.groupby(["year", "month"])
            .agg(actual=("y", "sum"), yhat=("yhat", "sum"), spend=(ccol, "sum"))
            .reset_index()
    )
    mthly["combined"] = mthly.apply(
        lambda r: r["actual"] if r["actual"] > 0 else r["yhat"], axis=1
    )
    return fcst, mthly


# ---------------------------------------------------------------------
# 6️  PIPELINE WRAPPER (one call → results dict)
# ---------------------------------------------------------------------
def run_pipeline(cfg, df=None):
    df = df.copy() if df is not None else RAW_DF.copy()

    pre              = preprocess(df, cfg)
    fut              = allocate_budget(pre, cfg)
    params           = get_prophet_params(cfg)
    fcst, monthly    = forecast(pre["prophet_df"], fut, params, cfg)

    return {
        "forecast": fcst.to_dict(orient="records"),
        "monthly":  monthly.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------
# 7️  FLASK ROUTES
# ---------------------------------------------------------------------
from flask import render_template_string   # already imported in earlier examples

@app.route("/")
def home():
    return render_template_string("""
        <!doctype html>
        <title>Forecast Uploader</title>
        <h2>Upload CSV and run forecast</h2>
        <form action="/forecast" method="post" enctype="multipart/form-data">
          <label>Data file (must have date,cost,Orders columns): </label>
          <input type="file" name="file" accept=".csv" required><br><br>
          <button type="submit">Run forecast</button>
        </form>
    """)


# @app.route("/")
# def home():
#     return render_template("index.html")


@app.route("/forecast", methods=["POST"])
def forecast_api():
    """
    Handles a browser form upload (multipart/form-data) and returns JSON.
    Required form parts
    -------------------
    • file           – CSV or Excel with columns: date, cost, Orders (default)
    Optional form parts
    -------------------
    • target_col     – name of the y column (default "Orders")
    • future_budgets – JSON list like [[2025,8,12000],[2025,9,14000]]
    """
    # ---- 1. validate & load the file ----
    if "file" not in request.files:
        raise BadRequest("Upload must include a file.")
    upload = request.files["file"]
    if not upload or upload.filename == "":
        raise BadRequest("No file selected.")
    df = load_dataframe(upload)                     # <-- we just added this helper

    # ---- 2. optional extras from the same form ----
    target_col = request.form.get("target_col", "Orders")

    budgets_field = request.form.get("future_budgets", "").strip()
    budgets: dict[tuple[int, int], float] = {}
    if budgets_field:
        try:
            budgets_list = json.loads(budgets_field)
            budgets = {(int(yr), int(mo)): float(bud) for yr, mo, bud in budgets_list}
        except Exception as exc:
            raise BadRequest(f"Could not parse future_budgets: {exc}")

    # ---- 3. run the pipeline on the *uploaded* DataFrame ----
    cfg = build_config(target_col=target_col, future_budgets=budgets)
    results = run_pipeline(cfg, df)                 # df overrides RAW_DF
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
