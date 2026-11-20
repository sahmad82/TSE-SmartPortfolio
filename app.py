import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
import streamlit as st

from src.backtester import run_walk_forward_backtest
from src.data_loader import compute_returns, fetch_tse_prices
from src.optimizers import hierarchical_risk_parity
from src.risk_metrics import compute_drawdown_series

st.set_page_config(
    page_title="TSE Smart Portfolio | Quant Asset Allocation",
    page_icon="📈",
    layout="wide",
)

# دیکشنری نام انگلیسی نمادها برای نمایش تمیز در نمودارهای Plotly
TICKER_LABELS = {
    "فولاد": "FOOLAD (Steel)",
    "فملی": "FEMELI (Copper)",
    "فارس": "FARS (Petrochem)",
    "شپنا": "SHAPNA (Refinery)",
    "شستا": "SHASTA (Holding)",
    "خودرو": "KHODRO (Auto)",
    "خساپا": "KHESAPA (Auto)",
    "وبملت": "VBMELAT (Bank)",
    "وتجارت": "VTEJARAT (Bank)",
    "دعبید": "DABID (Pharma)",
    "غکورش": "GHKOUROSH (Food)",
    "عیار": "AYAR (Gold ETF)",
}


@st.cache_data
def load_market_data() -> pd.DataFrame:
    prices = fetch_tse_prices()
    return prices.rename(columns=TICKER_LABELS)


def build_dendrogram_fig(linkage_mat: np.ndarray, labels: list) -> go.Figure:
    """رسم درخت خوشه‌بندی سلسله‌مراتبی HRP با Plotly."""
    dendro = dendrogram(linkage_mat, labels=labels, no_plot=True)
    fig = go.Figure()

    for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    tick_vals = [5 + 10 * i for i in range(len(dendro["ivl"]))]
    fig.update_layout(
        title="Hierarchical Clustering Dendrogram (Asset Similarity Tree)",
        xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=dendro["ivl"], tickangle=-35),
        yaxis=dict(title="Correlation Distance"),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=90),
    )
    return fig


st.title("📈 TSE-SmartPortfolio: Quantitative Asset Allocation & Risk Engine")
st.markdown(
    "Out-of-sample Walk-Forward comparison of **Equal-Weight (1/N)**, **Markowitz (MVO)**, "
    "**Ledoit-Wolf Shrinkage**, and **Hierarchical Risk Parity (HRP)** on Tehran Stock Exchange data."
)

# --- تنظیمات سایدبار ---
st.sidebar.header("⚙️ Backtest Parameters")
lookback = st.sidebar.slider("Lookback Window (Trading Days)", min_value=120, max_value=360, value=242, step=20)
rebalance_freq = st.sidebar.slider("Rebalance Frequency (Days)", min_value=20, max_value=120, value=60, step=10)
max_weight = st.sidebar.slider("Max Single Asset Weight (MVO / LW)", min_value=0.15, max_value=1.0, value=0.35, step=0.05)
round_trip_fee = st.sidebar.number_input("Round-Trip Trading Fee (%)", min_value=0.0, max_value=3.0, value=1.25, step=0.05) / 100.0
rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", min_value=10.0, max_value=50.0, value=30.0, step=1.0) / 100.0

prices_df = load_market_data()
returns_df = compute_returns(prices_df)

port_rets, summary_df, weights_dfs = run_walk_forward_backtest(
    returns_df=returns_df,
    lookback=lookback,
    rebalance_freq=rebalance_freq,
    round_trip_fee=round_trip_fee,
    max_weight=max_weight,
    rf_annual=rf_annual,
)

# --- کارت‌های خلاصه عملکرد ---
best_sharpe_model = summary_df["Sharpe Ratio"].idxmax()
best_cagr_model = summary_df["CAGR (%)"].idxmax()
lowest_dd_model = summary_df["Max Drawdown (%)"].idxmax()  # کمترین منفی = بهترین

col1, col2, col3, col4 = st.columns(4)
col1.metric("Out-of-Sample Trading Days", f"{len(port_rets)} days")
col2.metric("Highest Sharpe Ratio", f"{summary_df.loc[best_sharpe_model, 'Sharpe Ratio']:.2f}", best_sharpe_model)
col3.metric("Highest Annual Return (CAGR)", f"{summary_df.loc[best_cagr_model, 'CAGR (%)']:.1f}%", best_cagr_model)
col4.metric("Lowest Max Drawdown", f"{summary_df.loc[lowest_dd_model, 'Max Drawdown (%)']:.1f}%", lowest_dd_model)

# --- جدول مقایسه شاخص‌ها ---
st.subheader("📊 Out-of-Sample Performance & Risk Summary (Net of Fees)")
st.dataframe(
    summary_df.style.highlight_max(
        subset=["Total Return (%)", "CAGR (%)", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
        color="#d4edda",
    ).highlight_min(
        subset=["Ann. Volatility (%)", "Avg Rebalance Turnover (%)"],
        color="#d4edda",
    ),
    use_container_width=True,
)

# --- نمودار رشد سرمایه و افت سرمایه ---
tab1, tab2, tab3 = st.tabs(["💰 Cumulative Wealth & Drawdown", "🌳 HRP Clustering & Correlation", "⚖️ Portfolio Weights Over Time"])

with tab1:
    wealth_df = (1.0 + port_rets).cumprod() * 100.0
    fig_wealth = px.line(
        wealth_df,
        title="Cumulative Wealth Growth (Initial Capital = 100, Net of Transaction Costs)",
        labels={"value": "Portfolio Value", "index": "Jalali Date", "variable": "Strategy"},
        template="plotly_white",
    )
    fig_wealth.update_layout(height=430, hovermode="x unified")
    st.plotly_chart(fig_wealth, use_container_width=True)

    dd_df = port_rets.apply(compute_drawdown_series) * 100.0
    fig_dd = px.area(
        dd_df,
        title="Underwater Drawdown Curve (%)",
        labels={"value": "Drawdown (%)", "index": "Jalali Date", "variable": "Strategy"},
        template="plotly_white",
    )
    fig_dd.update_layout(height=360, hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

with tab2:
    _, hrp_artifacts = hierarchical_risk_parity(returns_df.iloc[-lookback:])
    c_left, c_right = st.columns(2)

    with c_left:
        fig_dendro = build_dendrogram_fig(hrp_artifacts["linkage"], list(returns_df.columns))
        st.plotly_chart(fig_dendro, use_container_width=True)

    with c_right:
        sort_ix = hrp_artifacts["sort_ix"]
        sorted_cols = [returns_df.columns[i] for i in sort_ix]
        sorted_corr = returns_df.iloc[-lookback:][sorted_cols].corr()

        fig_corr = px.imshow(
            sorted_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Quasi-Diagonalized Correlation Matrix (HRP Ordered)",
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    selected_strategy = st.selectbox("Select Strategy to Inspect Historical Weights:", list(weights_dfs.keys()), index=3)
    w_df = weights_dfs[selected_strategy] * 100.0
    fig_weights = px.bar(
        w_df,
        title=f"Rebalance Allocation Weights (%) — {selected_strategy}",
        labels={"value": "Weight (%)", "index": "Rebalance Date", "variable": "Asset"},
        template="plotly_white",
    )
    fig_weights.update_layout(barmode="stack", height=440)
    st.plotly_chart(fig_weights, use_container_width=True)