# =============================================================================
# Natural Gas Volatility Dashboard
# Save as:  C:\Users\Elaine\dukascopy_data\app.py
# Run with: streamlit run app.py
# =============================================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NG Volatility Dashboard",
    page_icon="🔥",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PKL = "dashboard_models.pkl"

# ── Load saved models ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)

try:
    saved       = load_models()
    rf_model    = saved["rf_model"]
    scaler      = saved["scaler"]
    features    = saved["features"]
    rv_history  = saved["rv_history"]
    low_thresh  = saved["low_thresh"]
    high_thresh = saved["high_thresh"]
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# ── Helpers ───────────────────────────────────────────────────────────────────
def price_to_rv(today_price, yesterday_price):
    if yesterday_price <= 0 or today_price <= 0:
        return 0.0
    return abs(float(np.log(today_price / yesterday_price)))

def classify_regime(rv_value, low_t, high_t):
    if rv_value <= low_t:
        return "LOW", "#2ecc71", "🟢"
    elif rv_value >= high_t:
        return "HIGH", "#e74c3c", "🔴"
    else:
        return "MEDIUM", "#f39c12", "🟡"

# ── Feature group definitions ─────────────────────────────────────────────────
HAR_FEATURES      = ["NG_daily", "NG_weekly", "NG_monthly"]
FINANCIAL_FEATURES = ["BRENT", "SPX", "GOLD", "EURUSD"]

ASSET_LABELS = {
    "BRENT" : "🛢️ Brent Crude",
    "SPX"   : "📈 S&P 500",
    "GOLD"  : "🥇 Gold",
    "EURUSD": "💱 EUR/USD"
}

HAR_LABELS = {
    "NG_daily"  : "NG Daily Lag (t-1)",
    "NG_weekly" : "NG Weekly Lag (avg 5d)",
    "NG_monthly": "NG Monthly Lag (avg 22d)"
}

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔥 Natural Gas Volatility Dashboard")
st.markdown(
    "Enter **today's and yesterday's closing price** for all 5 assets. "
    "The app computes realized volatility, classifies the market regime, "
    "and identifies which financial assets are driving NG volatility today."
)
st.markdown("---")

if not models_loaded:
    st.error(
        "⚠️ **Model file not found.**\n\n"
        "Run your project code with the save block at the end first. "
        "Once `dashboard_models.pkl` appears in your data folder, reload this page."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT — 5 assets × 2 prices
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📥 Enter Closing Prices")

h0, h1, h2 = st.columns([2, 1, 1])
h0.markdown("**Asset**")
h1.markdown("**Today's Close**")
h2.markdown("**Yesterday's Close**")

assets = [
    {"key": "ng",     "label": "🔥 Natural Gas (NGAS/USD)", "today": 3.10,   "yest": 3.05},
    {"key": "brent",  "label": "🛢️ Brent Crude (UKOIL/USD)", "today": 85.00,  "yest": 84.50},
    {"key": "spx",    "label": "📈 S&P 500 (US500)",         "today": 4500.0, "yest": 4480.0},
    {"key": "gold",   "label": "🥇 Gold (XAU/USD)",          "today": 1980.0, "yest": 1975.0},
    {"key": "eurusd", "label": "💱 EUR/USD",                  "today": 1.0850, "yest": 1.0830},
]

price_inputs = {}
for asset in assets:
    c0, c1, c2 = st.columns([2, 1, 1])
    c0.markdown(asset["label"])
    price_inputs[f"{asset['key']}_today"] = c1.number_input(
        f"Today {asset['key']}", value=float(asset["today"]),
        step=0.0001, format="%.4f",
        key=f"{asset['key']}_t", label_visibility="collapsed"
    )
    price_inputs[f"{asset['key']}_yest"] = c2.number_input(
        f"Yest {asset['key']}", value=float(asset["yest"]),
        step=0.0001, format="%.4f",
        key=f"{asset['key']}_y", label_visibility="collapsed"
    )

st.markdown("---")

# ── Live RV preview ───────────────────────────────────────────────────────────
rv_ng     = price_to_rv(price_inputs["ng_today"],     price_inputs["ng_yest"])
rv_brent  = price_to_rv(price_inputs["brent_today"],  price_inputs["brent_yest"])
rv_spx    = price_to_rv(price_inputs["spx_today"],    price_inputs["spx_yest"])
rv_gold   = price_to_rv(price_inputs["gold_today"],   price_inputs["gold_yest"])
rv_eurusd = price_to_rv(price_inputs["eurusd_today"], price_inputs["eurusd_yest"])

st.markdown("#### 📐 Computed RV (auto-calculated from your prices)")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("NG RV",     f"{rv_ng:.6f}")
r2.metric("BRENT RV",  f"{rv_brent:.6f}")
r3.metric("SPX RV",    f"{rv_spx:.6f}")
r4.metric("GOLD RV",   f"{rv_gold:.6f}")
r5.metric("EURUSD RV", f"{rv_eurusd:.6f}")
st.caption(
    "ℹ️ Single log return approximation. "
    "True intraday RV uses all 5-min returns — this is a good proxy "
    "when intraday data is unavailable."
)

# ── Optional intraday RV override ────────────────────────────────────────────
with st.expander("🔧 Have intraday RV values? Override here (optional)"):
    st.caption("Enter actual intraday RV if you have it. Leave at 0.0 to use price-computed values.")
    ov_ng     = st.number_input("NG RV",     min_value=0.0, value=0.0, format="%.6f", key="ov_ng")
    ov_brent  = st.number_input("BRENT RV",  min_value=0.0, value=0.0, format="%.6f", key="ov_b")
    ov_spx    = st.number_input("SPX RV",    min_value=0.0, value=0.0, format="%.6f", key="ov_s")
    ov_gold   = st.number_input("GOLD RV",   min_value=0.0, value=0.0, format="%.6f", key="ov_g")
    ov_eurusd = st.number_input("EURUSD RV", min_value=0.0, value=0.0, format="%.6f", key="ov_e")

final_ng     = ov_ng     if ov_ng     > 0 else rv_ng
final_brent  = ov_brent  if ov_brent  > 0 else rv_brent
final_spx    = ov_spx    if ov_spx    > 0 else rv_spx
final_gold   = ov_gold   if ov_gold   > 0 else rv_gold
final_eurusd = ov_eurusd if ov_eurusd > 0 else rv_eurusd

st.markdown("---")
run_button = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if run_button:

    # ── HAR lags from saved history ───────────────────────────────────────────
    ng_series  = rv_history["NG"].copy()
    ng_daily   = final_ng
    ng_weekly  = float(ng_series.iloc[-5:].mean())  if len(ng_series) >= 5  else float(ng_series.mean())
    ng_monthly = float(ng_series.iloc[-22:].mean()) if len(ng_series) >= 22 else float(ng_series.mean())

    # ── Feature vector ────────────────────────────────────────────────────────
    feature_values = np.array([[
        ng_daily, ng_weekly, ng_monthly,
        final_brent, final_spx, final_gold, final_eurusd
    ]])

    # ── Scale + predict ───────────────────────────────────────────────────────
    X_scaled    = scaler.transform(feature_values)
    rf_forecast = float(rf_model.predict(X_scaled)[0])
    rf_forecast = max(rf_forecast, 1e-6)

    # ── Regime ────────────────────────────────────────────────────────────────
    regime_label, regime_color, regime_icon = classify_regime(
        final_ng, low_thresh, high_thresh
    )

    # ── SHAP values ───────────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled)[0]

    # Build SHAP dict keyed by feature name
    shap_dict = dict(zip(features, shap_values))

    # Separate into HAR group and Financial Assets group
    har_shap = {HAR_LABELS[f]: shap_dict[f] for f in HAR_FEATURES}
    fin_shap = {ASSET_LABELS[f]: shap_dict[f] for f in FINANCIAL_FEATURES}

    # Combined HAR effect (sum) — single number for persistence contribution
    har_total = sum(har_shap.values())

    # Which financial asset is the biggest driver today
    top_fin_asset = max(fin_shap, key=lambda k: abs(fin_shap[k]))
    top_fin_val   = fin_shap[top_fin_asset]

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 Results")

    # ── Top metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "NG RV (today)",
        f"{final_ng:.6f}",
        delta=f"{final_ng - float(ng_series.mean()):.6f} vs hist. mean",
        delta_color="inverse"
    )
    m2.metric(
        "RF Forecast (today)",
        f"{rf_forecast:.6f}",
        delta=f"{rf_forecast - final_ng:.6f} vs yesterday",
        delta_color="inverse"
    )
    m3.metric("Regime", f"{regime_icon} {regime_label}")
    m4.metric(
        "Biggest External Driver",
        top_fin_asset.split(" ", 1)[-1],   # strip emoji for cleaner metric
        delta=f"SHAP {top_fin_val:+.5f}",
        delta_color="inverse"
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # TWO SHAP CHARTS SIDE BY SIDE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔍 What Is Driving NG Volatility Today?")
    st.markdown(
        "The forecast is driven by two forces: "
        "**NG's own persistence** (how volatile it has been recently) "
        "and **external financial asset shocks** (which market moved most)."
    )

    chart_left, chart_right = st.columns(2)

    # ── LEFT: Financial Asset Drivers (the actionable chart) ─────────────────
    with chart_left:
        st.markdown("#### 🌍 External Financial Asset Drivers")
        st.caption(
            "Which financial market is spilling over into NG volatility today? "
            "This is the actionable signal for trading and policy decisions."
        )

        fin_df = pd.DataFrame({
            "Asset": list(fin_shap.keys()),
            "SHAP":  list(fin_shap.values())
        }).sort_values("SHAP", key=abs, ascending=True)

        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        colors1 = ["#e74c3c" if v > 0 else "#2ecc71" for v in fin_df["SHAP"]]
        bars1   = ax1.barh(fin_df["Asset"], fin_df["SHAP"],
                           color=colors1, edgecolor="white", height=0.55)

        for bar, val in zip(bars1, fin_df["SHAP"]):
            x_pos = bar.get_width() + (0.000005 if val >= 0 else -0.000005)
            ax1.text(x_pos, bar.get_y() + bar.get_height() / 2,
                     f"{val:+.5f}", va="center",
                     ha="left" if val >= 0 else "right",
                     fontsize=9, color="#333333", fontweight="bold")

        ax1.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax1.set_xlabel("SHAP Value (contribution to NG forecast)", fontsize=9)
        ax1.set_title("Financial Asset Spillovers", fontsize=11,
                      fontweight="bold", pad=8)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.legend(handles=[
            mpatches.Patch(color="#e74c3c", label="↑ Increases NG volatility"),
            mpatches.Patch(color="#2ecc71", label="↓ Reduces NG volatility")
        ], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # Highlight top driver
        direction_text = "increasing" if top_fin_val > 0 else "dampening"
        st.success(
            f"**Today's biggest external driver: {top_fin_asset}**\n\n"
            f"It is {direction_text} NG volatility "
            f"(SHAP = {top_fin_val:+.5f})"
        )

    # ── RIGHT: NG Own-History Persistence ────────────────────────────────────
    with chart_right:
        st.markdown("#### 📉 NG Own-History Persistence Effect")
        st.caption(
            "How much of today's forecast is explained by NG's own recent volatility? "
            "This reflects the long-memory property of natural gas markets."
        )

        har_df = pd.DataFrame({
            "Lag":  list(har_shap.keys()),
            "SHAP": list(har_shap.values())
        }).sort_values("SHAP", key=abs, ascending=True)

        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        colors2 = ["#2e86c1" if v > 0 else "#85c1e9" for v in har_df["SHAP"]]
        bars2   = ax2.barh(har_df["Lag"], har_df["SHAP"],
                           color=colors2, edgecolor="white", height=0.55)

        for bar, val in zip(bars2, har_df["SHAP"]):
            x_pos = bar.get_width() + (0.000005 if val >= 0 else -0.000005)
            ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                     f"{val:+.5f}", va="center",
                     ha="left" if val >= 0 else "right",
                     fontsize=9, color="#333333", fontweight="bold")

        ax2.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("SHAP Value (contribution to NG forecast)", fontsize=9)
        ax2.set_title("NG Own Persistence (HAR Lags)", fontsize=11,
                      fontweight="bold", pad=8)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend(handles=[
            mpatches.Patch(color="#2e86c1", label="↑ Persistence increases forecast"),
            mpatches.Patch(color="#85c1e9", label="↓ Mean-reversion effect")
        ], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Combined persistence summary
        persist_direction = "amplifying" if har_total > 0 else "dampening"
        st.info(
            f"**Combined persistence effect: {har_total:+.5f}**\n\n"
            f"NG's own recent volatility history is {persist_direction} "
            f"today's forecast. This is the long-memory effect from your "
            f"ADF/KPSS diagnostic results."
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # REGIME PANEL
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🚦 Market Regime")

    regime_info = {
        "LOW": {
            "desc": "Calm market conditions. NG volatility is in the bottom third of the OOS distribution.",
            "best": "LightGBM / XGBoost",
            "tip":  "Gradient boosting dominates in low-volatility regimes (GW test validated).",
            "action": "Lower hedging costs. EUR/USD spillovers typically dominate in this regime.",
            "bg":   "#d5f5e3"
        },
        "MEDIUM": {
            "desc": "Normal market conditions. NG volatility is in the middle third.",
            "best": "Extra Trees",
            "tip":  "No single model dominates — forecasts carry more uncertainty.",
            "action": "Standard risk management. Monitor all four financial assets for regime shift signals.",
            "bg":   "#fef9e7"
        },
        "HIGH": {
            "desc": "Turbulent market conditions. NG volatility is in the top third of the OOS distribution.",
            "best": "Random Forest",
            "tip":  "RF statistically dominates all linear models in this regime (GW F-stat up to 70.9, p<0.001).",
            "action": "Elevated hedging required. S&P 500 spillovers amplify during high-volatility periods.",
            "bg":   "#fdedec"
        }
    }

    rd = regime_info[regime_label]
    reg_left, reg_right = st.columns([3, 2])

    with reg_left:
        st.markdown(
            f"""<div style="background:{rd['bg']};padding:24px;border-radius:10px;
                            border-left:6px solid {regime_color};">
            <h2 style="color:{regime_color};margin:0">{regime_icon} {regime_label} VOLATILITY REGIME</h2>
            <p style="margin-top:12px;font-size:1.05em">{rd['desc']}</p>
            <hr style="border-color:{regime_color};opacity:0.3"/>
            <table width="100%">
              <tr>
                <td><b>Best forecasting model:</b></td>
                <td style="color:{regime_color};font-weight:bold;font-size:1.1em">{rd['best']}</td>
              </tr>
              <tr>
                <td><b>Policy/Trading action:</b></td>
                <td>{rd['action']}</td>
              </tr>
              <tr>
                <td><b>Statistical basis:</b></td>
                <td>{rd['tip']}</td>
              </tr>
            </table>
            </div>""",
            unsafe_allow_html=True
        )

    with reg_right:
        st.markdown("##### Regime Boundaries (from your OOS data)")
        st.markdown(f"- 🟢 **Low**:    RV ≤ **{low_thresh:.4f}**")
        st.markdown(f"- 🟡 **Medium**: **{low_thresh:.4f}** < RV < **{high_thresh:.4f}**")
        st.markdown(f"- 🔴 **High**:   RV ≥ **{high_thresh:.4f}**")
        st.markdown(f"- 📍 **Today's NG RV**: **{final_ng:.6f}**")

        st.markdown("---")
        st.markdown("##### HAR Lag Values Used (auto-computed)")
        st.markdown(f"- Daily (t-1):   **{ng_daily:.6f}**")
        st.markdown(f"- Weekly (5d avg): **{ng_weekly:.6f}**")
        st.markdown(f"- Monthly (22d avg): **{ng_monthly:.6f}**")
        st.caption("These are computed automatically from your saved RV history.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # PLAIN ENGLISH SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 💬 Plain English Summary")

    fin_direction   = "increasing" if top_fin_val > 0 else "reducing"
    har_direction   = "amplifying" if har_total > 0 else "dampening"
    regime_action   = rd["action"]

    st.info(
        f"**Today's NG volatility forecast: {rf_forecast:.6f}** "
        f"— placing the market in the **{regime_label} regime**.\n\n"
        f"**External driver:** {top_fin_asset} is the biggest financial market "
        f"spillover today ({fin_direction} NG volatility, SHAP = {top_fin_val:+.5f}).\n\n"
        f"**Persistence effect:** NG's own recent history is {har_direction} the "
        f"forecast (combined HAR SHAP = {har_total:+.5f}).\n\n"
        f"**Recommended action:** {regime_action}"
    )

    # ── Full feature table (collapsed) ───────────────────────────────────────
    with st.expander("📋 See all feature values and SHAP impacts"):
        st.dataframe(pd.DataFrame({
            "Feature":      features,
            "Value Used":   [f"{v:.6f}" for v in feature_values[0]],
            "SHAP Impact":  [f"{s:+.6f}" for s in shap_values],
            "Direction":    ["↑ Higher" if s > 0 else "↓ Lower" for s in shap_values],
            "Group":        ["NG Persistence" if f in HAR_FEATURES
                             else "Financial Asset" for f in features]
        }), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Based on: Forecasting Natural Gas Realized Volatility — 11-Model ML Comparison | "
    f"Training: Sep 2012–Jan 2020 | OOS: Feb 2020–Jan 2022 | "
    f"Regime thresholds: Low ≤ {low_thresh:.4f}, High ≥ {high_thresh:.4f}"
    if models_loaded else "Models not yet loaded."
)
