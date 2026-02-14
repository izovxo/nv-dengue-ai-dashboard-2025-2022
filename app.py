import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nueva Vizcaya Dengue AI Dashboard", layout="wide")

# -------------------------
# Default file paths (repo)
# -------------------------
DEFAULT_XLSX = "data/Nueva_Vizcaya_Climate_Dengue_LSTM_Ready_2015_2022_ONLY.xlsx"
DEFAULT_PRED = "data/LSTM_Predictions_Test_2015_2022.csv"
DEFAULT_SHEET = "Integrated_Monthly_2015_2022"

CLIMATE_COLS = ["PRECIP_TOTAL_MM", "RH2M_PCT", "TEMP_MEAN_C", "TEMP_MAX_C", "TEMP_MIN_C"]

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def classify_risk(value, q50, q75):
    if value < q50:
        return "Low"
    elif value < q75:
        return "Moderate"
    return "High"

# -------------------------
# Sidebar
# -------------------------
st.title("Nueva Vizcaya Dengue AI Dashboard (2015–2022)")
st.write("This dashboard visualizes observed dengue + climate (NASA POWER) and compares LSTM predictions on the test period.")

with st.sidebar:
    st.header("Data Source")

    use_upload = st.checkbox("Upload files instead of using repo data", value=False)

    if use_upload:
        uploaded_xlsx = st.file_uploader("Upload Excel (2015–2022 integrated dataset)", type=["xlsx"])
        uploaded_pred = st.file_uploader("Upload LSTM Predictions CSV", type=["csv"])
        sheet_name = st.text_input("Excel sheet name", value=DEFAULT_SHEET)
    else:
        xlsx_path = st.text_input("Excel path", value=DEFAULT_XLSX)
        pred_path = st.text_input("Predictions CSV path", value=DEFAULT_PRED)
        sheet_name = st.text_input("Excel sheet name", value=DEFAULT_SHEET)

    st.header("Charts")
    show_climate = st.checkbox("Show climate charts", value=True)

# -------------------------
# Load data
# -------------------------
try:
    if use_upload:
        if uploaded_xlsx is None or uploaded_pred is None:
            st.info("Upload both the Excel file and the Predictions CSV to proceed.")
            st.stop()
        df = pd.read_excel(uploaded_xlsx, sheet_name=sheet_name)
        pred = pd.read_csv(uploaded_pred)
    else:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        pred = pd.read_csv(pred_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Standardize columns
if "DATE" not in df.columns:
    st.error("Excel sheet must contain a DATE column.")
    st.stop()

df["DATE"] = pd.to_datetime(df["DATE"])

if "DENGUE_CASES" not in df.columns:
    st.error("Excel sheet must contain a DENGUE_CASES column.")
    st.stop()

missing_climate = [c for c in CLIMATE_COLS if c not in df.columns]
if missing_climate:
    st.error(f"Missing climate columns in Excel: {missing_climate}")
    st.stop()

# Predictions CSV columns expected: DATE_TARGET, ACTUAL_CASES, PRED_CASES
needed_pred_cols = ["DATE_TARGET", "ACTUAL_CASES", "PRED_CASES"]
for c in needed_pred_cols:
    if c not in pred.columns:
        st.error(f"Predictions CSV must contain column: {c}")
        st.stop()

pred["DATE_TARGET"] = pd.to_datetime(pred["DATE_TARGET"])

# Merge predictions onto main table (for plotting)
df_plot = df.copy()
df_plot = df_plot.merge(
    pred.rename(columns={"DATE_TARGET": "DATE"}),
    on="DATE",
    how="left"
)

# Compute metrics on available rows (test months)
test_rows = df_plot.dropna(subset=["ACTUAL_CASES", "PRED_CASES"]).copy()
test_mae = mae(test_rows["ACTUAL_CASES"], test_rows["PRED_CASES"])
test_rmse = rmse(test_rows["ACTUAL_CASES"], test_rows["PRED_CASES"])

# Risk thresholds from observed distribution
q50 = float(df["DENGUE_CASES"].quantile(0.50))
q75 = float(df["DENGUE_CASES"].quantile(0.75))

# -------------------------
# Layout
# -------------------------
colA, colB, colC = st.columns([1.2, 1, 1])

with colA:
    st.subheader("Dataset overview")
    st.write(f"Rows: **{len(df)}** months")
    st.write(f"Range: **{df['DATE'].min().strftime('%Y-%m')}** to **{df['DATE'].max().strftime('%Y-%m')}**")
    st.write(f"Test months with predictions: **{len(test_rows)}**")

with colB:
    st.subheader("Test performance (Observed only)")
    st.metric("MAE (cases)", f"{test_mae:.2f}")
    st.metric("RMSE (cases)", f"{test_rmse:.2f}")

with colC:
    st.subheader("Risk thresholds (from 2015–2022)")
    st.write(f"Median (P50): **{q50:.0f}**")
    st.write(f"P75: **{q75:.0f}**")

st.divider()

# -------------------------
# Plot: full series + predictions overlay
# -------------------------
st.subheader("Dengue cases: Observed vs LSTM Predicted (Test Months)")

fig = plt.figure()
plt.plot(df_plot["DATE"], df_plot["DENGUE_CASES"], marker="o", label="Observed dengue (2015–2022)")
plt.plot(df_plot["DATE"], df_plot["PRED_CASES"], marker="o", label="LSTM predicted (test months only)")

# Shade the test period
if len(test_rows) > 0:
    start_test = test_rows["DATE"].min()
    end_test = test_rows["DATE"].max()
    plt.axvspan(start_test, end_test, alpha=0.15, label="Test period")

plt.xlabel("Date")
plt.ylabel("Dengue cases")
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# -------------------------
# Plot: Actual vs Predicted (test)
# -------------------------
st.subheader("Actual vs Predicted (Test Scatter)")
fig2 = plt.figure()
plt.scatter(test_rows["ACTUAL_CASES"], test_rows["PRED_CASES"])
plt.xlabel("Actual dengue cases")
plt.ylabel("Predicted dengue cases")
plt.xticks(rotation=0)
st.pyplot(fig2)

# -------------------------
# Table + risk labels
# -------------------------
st.subheader("Test table (with risk labels)")
test_table = test_rows[["DATE", "ACTUAL_CASES", "PRED_CASES"]].copy()
test_table["ACTUAL_RISK"] = test_table["ACTUAL_CASES"].apply(lambda v: classify_risk(v, q50, q75))
test_table["PRED_RISK"] = test_table["PRED_CASES"].apply(lambda v: classify_risk(v, q50, q75))
st.dataframe(test_table)

st.download_button(
    "Download test table CSV",
    data=test_table.to_csv(index=False).encode("utf-8"),
    file_name="NV_LSTM_Test_Table_2015_2022.csv",
    mime="text/csv"
)

# -------------------------
# Climate charts
# -------------------------
if show_climate:
    st.subheader("Climate variables (NASA POWER)")

    fig3 = plt.figure()
    for c in CLIMATE_COLS:
        plt.plot(df["DATE"], df[c], label=c)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig3)

# -------------------------
# Method / Chapter 3 notes
# -------------------------
with st.expander("How this matches Chapter 3 (methods summary)"):
    st.markdown("""
**Data integration (monthly):** Climate + dengue cases are aligned by month (DATE = first day of month).  
**Feature engineering:** LSTM uses lagged inputs (12 months back) + seasonality (Month sin/cos) during training.  
**Model validation:** The predictions shown here are from the **observed-only test period** (2015–2022), which is appropriate for evaluation.  
**Evaluation metrics:** MAE and RMSE are computed on the test months included in the predictions CSV.
""")
