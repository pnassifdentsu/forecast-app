# --------------------------------------------------------------------------- #
# SEM FORECAST Flask App – upload, model, and visualise                       #
# --------------------------------------------------------------------------- #
from pathlib import Path
import io, base64, warnings

import matplotlib
matplotlib.use("Agg")          # off-screen rendering
import matplotlib.pyplot as plt

import pandas as pd
from flask import Flask, request, jsonify, render_template_string, url_for
from werkzeug.exceptions import BadRequest
from prophet import Prophet

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# 0.  File upload → DataFrame                                                 #
# --------------------------------------------------------------------------- #
ALLOWED_EXT = {".csv", ".xls", ".xlsx"}

REQ_COLS = {
    "date", "orders", "clicks", "impressions", "cost",
    "brand_cost", "nonbrand_cost", "promo_flag",
}

def load_dataframe(upload) -> pd.DataFrame:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise BadRequest(f"Unsupported file type: {ext}")

    df = (
        pd.read_csv(upload) if ext == ".csv"
        else pd.read_excel(upload, sheet_name=0)
    )

    missing = REQ_COLS - set(df.columns)
    if missing:
        raise BadRequest(f"Missing columns: {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


# --------------------------------------------------------------------------- #
# 1.  Prophet helper                                                           #
# --------------------------------------------------------------------------- #
BASE_REGS = ["brand_cost", "nonbrand_cost", "promo_flag"]

def prophet_forecast(df_train, target, regs, df_future):
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
    fc  = m.predict(fut)
    return (
        fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(columns={
            "yhat": target,
            "yhat_lower": f"{target}_lower",
            "yhat_upper": f"{target}_upper"})
    )


# --------------------------------------------------------------------------- #
# 2.  Main pipeline                                                            #
# --------------------------------------------------------------------------- #
def run_sem_pipeline(df_full):
    df_full = df_full.copy()
    df_full.rename(columns={"date": "ds"}, inplace=True)

    hist = df_full[df_full["orders"].notna()].copy()
    future = df_full[df_full["orders"].isna()].copy()
    last_hist = hist["ds"].max()

    # training window = last 730 days
    train = hist[hist["ds"] >= last_hist - pd.Timedelta(days=730)].copy()

    # clicks & impressions
    clicks_fc = prophet_forecast(train, "clicks", BASE_REGS, future)
    impr_fc   = prophet_forecast(train, "impressions", BASE_REGS, future)

    # orders (use predicted clicks/impr)
    fut_orders = (
        future.drop(columns=["clicks", "impressions"])
        .merge(clicks_fc[["ds", "clicks"]], on="ds")
        .merge(impr_fc[["ds", "impressions"]], on="ds")
    )
    orders_fc = prophet_forecast(
        train, "orders", BASE_REGS + ["clicks", "impressions"], fut_orders
    )

    # combined table (history + forecast for orders)
    table = (
        hist[["ds", "cost", "orders"]]
        .rename(columns={"orders": "actual_orders"})
        .merge(orders_fc[["ds", "orders", "orders_lower", "orders_upper"]],
                on="ds", how="outer")
        .merge(future[["ds", "cost"]]
               .rename(columns = {"cost" : "planned_cost"}), on="ds", how="outer")
        .sort_values("ds")
    )

    # plotting helper
    def fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    # Orders chart (90-day hist + forecast)
    start = last_hist - pd.Timedelta(days=90)
    fig1 = plt.figure(figsize=(12,4))
    mask_hist = (table["ds"] >= start) & table["actual_orders"].notna()
    plt.plot(table.loc[mask_hist, "ds"],
             table.loc[mask_hist, "actual_orders"],
             "k-", label="Actual Orders")
    mask_fc = orders_fc["ds"] >= start
    plt.plot(orders_fc.loc[mask_fc, "ds"],
             orders_fc.loc[mask_fc, "orders"],
             "b-", label="Forecast Orders")
    plt.fill_between(orders_fc.loc[mask_fc, "ds"],
                     orders_fc.loc[mask_fc, "orders_lower"],
                     orders_fc.loc[mask_fc, "orders_upper"],
                     color="b", alpha=0.3)
    plt.legend()
    orders_png = fig_to_b64(fig1)

    # ── Clicks chart (full horizon) ──────────────────────────────────
    fig2 = plt.figure(figsize=(12,4))
    plt.plot(clicks_fc["ds"], clicks_fc["clicks"], "g-")
    plt.fill_between(clicks_fc["ds"],
                     clicks_fc["clicks_lower"],
                     clicks_fc["clicks_upper"],
                     color="g", alpha=0.3)
    plt.ylabel("Clicks"); plt.xlabel("Date"); plt.title("Forecast – Clicks")
    clicks_png = fig_to_b64(fig2)

    # ── Impressions chart (full horizon) ─────────────────────────────
    fig3 = plt.figure(figsize=(12,4))
    plt.plot(impr_fc["ds"], impr_fc["impressions"], "r-")
    plt.fill_between(impr_fc["ds"],
                     impr_fc["impressions_lower"],
                     impr_fc["impressions_upper"],
                     color="r", alpha=0.3)
    plt.ylabel("Impressions"); plt.xlabel("Date"); plt.title("Forecast – Impressions")
    impr_png = fig_to_b64(fig3)

    return table, {
        "orders_png":   orders_png,
        "clicks_png":   clicks_png,
        "impr_png":     impr_png,
    }

import math, numpy as np

def clean_nans(obj):
    """Recursively convert NaN/NaT values to None or clean date strings."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp, pd.NaT.__class__)):
        return obj.strftime("%Y-%m-%d") if pd.notna(obj) else None
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_nans(v) for v in obj]
    return obj  # fallback for other types



# --------------------------------------------------------------------------- #
# 3.  Routes                                                                   #
# --------------------------------------------------------------------------- #
HTML_FORM = """
<!doctype html>
<title>SEM Forecast Uploader</title>
<h2>Upload a CSV or Excel file</h2>
<form id="upload-form" action="/forecast" method="post"
      enctype="multipart/form-data">
  <input type="file" name="file" accept=".csv,.xls,.xlsx" required><br><br>
  <button type="submit">Run forecast</button>
</form>

<p>Required columns: date, orders, clicks, impressions, cost,
   brand_cost, nonbrand_cost, promo_flag</p>

<hr>
<div id="results"></div>

<script>
document.querySelector("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const resp = await fetch("/forecast", { method: "POST", body: new FormData(e.target) });
  const data = await resp.json();

  const container = document.getElementById("results");
  container.innerHTML = "";

  /* ---------- render HTML table (last 60 rows) ---------- */
  const rows = data.table.slice(-60);
  const cols = ["ds", "cost", "actual_orders", "planned_cost",
                "orders", "orders_lower", "orders_upper"];

  const table = document.createElement("table");
  table.style.borderCollapse = "collapse";
  table.style.marginBottom   = "1rem";

  /* 1️⃣  header – FIXED (wrapped in back-ticks) */
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr>" + cols.map(c =>
      `\<th style="border:1px solid #ccc;padding:4px 6px;background:#f5f5f5">${c}</th>`
    ).join("") + "</tr>";
  table.appendChild(thead);

  /* 2️⃣  body rows – FIXED (wrapped in back-ticks) */
  const tbody = document.createElement("tbody");
  rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = cols.map(c =>
      `\<td style="border:1px solid #eee;padding:4px 6px;text-align:right">${r[c] ?? ""}</td>`
    ).join("");
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);

  /* ---------- charts ---------- */
  ["orders_png", "clicks_png", "impr_png"].forEach(id => {
    const src = data.plots[id];
    if (!src) return;
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + src;
    img.style.maxWidth   = "100%";
    img.style.marginBottom = "1rem";
    container.appendChild(img);
  });
});
</script>
"""

@app.route("/")
def home():
    return render_template_string(HTML_FORM)


@app.route("/forecast", methods=["POST"])
def forecast_route():
    if "file" not in request.files:
        raise BadRequest("Upload must include a file.")
    upload = request.files["file"]
    df = load_dataframe(upload)

    table, plots = run_sem_pipeline(df)

    # Round all numeric columns to 2 decimals (float/int only)
    for col in table.select_dtypes(include=["number"]).columns:
        table[col] = table[col].round()

    # Format cost column as $ string with commas and 2 decimals
    for col in ['cost', 'planned_cost']:
        if col in table.columns:
            table[col] = table[col].apply(
                lambda x: f"${x:,.2f}" if pd.notnull(x) else None
            )

    payload = {
        "table": clean_nans(table.to_dict(orient="records")),
        "plots": plots
    }
    return jsonify(payload)


@app.errorhandler(BadRequest)
def bad_request(e):
    return jsonify({"error": str(e)}), 400


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(debug=True)