"""
Portfolio Analyzer & Optimizer
==============================
Bloomberg-style real-time portfolio monitoring terminal.
Live market data · Risk analytics · SEBI compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

NSE_UNIVERSE = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT", "ICICIBANK.NS": "Banking", "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG", "SBIN.NS": "Banking", "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking", "LT.NS": "Infra", "HCLTECH.NS": "IT",
    "AXISBANK.NS": "Banking", "ASIANPAINT.NS": "Consumer",
    "MARUTI.NS": "Auto", "SUNPHARMA.NS": "Pharma", "TITAN.NS": "Consumer",
    "BAJFINANCE.NS": "Finance", "WIPRO.NS": "IT", "ULTRACEMCO.NS": "Cement",
    "NESTLEIND.NS": "FMCG", "TECHM.NS": "IT", "M&M.NS": "Auto",
    "POWERGRID.NS": "Utilities", "NTPC.NS": "Utilities",
    "JSWSTEEL.NS": "Metals", "TATAMOTORS.NS": "Auto", "TATASTEEL.NS": "Metals",
    "ADANIENT.NS": "Infra", "ONGC.NS": "Energy",
    "BAJAJFINSV.NS": "Finance", "ADANIPORTS.NS": "Infra",
    "COALINDIA.NS": "Mining", "GRASIM.NS": "Cement", "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma", "EICHERMOT.NS": "Auto",
    "HEROMOTOCO.NS": "Auto", "BPCL.NS": "Energy", "BRITANNIA.NS": "FMCG",
    "APOLLOHOSP.NS": "Healthcare", "TATACONSUM.NS": "FMCG",
    "INDUSINDBK.NS": "Banking", "SBILIFE.NS": "Insurance",
    "HDFCLIFE.NS": "Insurance", "DABUR.NS": "FMCG", "GODREJCP.NS": "FMCG",
    "PIDILITIND.NS": "Chemicals", "MARICO.NS": "FMCG",
    "SIEMENS.NS": "Cap Goods", "HAVELLS.NS": "Consumer",
    "AMBUJACEM.NS": "Cement", "ACC.NS": "Cement",
    "BERGEPAINT.NS": "Consumer", "COLPAL.NS": "FMCG",
    "MCDOWELL-N.NS": "Consumer", "DLF.NS": "Real Estate",
    "TRENT.NS": "Retail", "ZOMATO.NS": "Technology",
    "BANKBARODA.NS": "Banking", "AUBANK.NS": "Banking",
    "FEDERALBNK.NS": "Banking", "IDFCFIRSTB.NS": "Banking",
    "BANDHANBNK.NS": "Banking", "PNB.NS": "Banking",
    "MPHASIS.NS": "IT", "LTIM.NS": "IT", "COFORGE.NS": "IT",
    "PERSISTENT.NS": "IT", "LTTS.NS": "IT", "NAUKRI.NS": "Technology",
    "BIOCON.NS": "Pharma", "AUROPHARMA.NS": "Pharma", "LUPIN.NS": "Pharma",
    "TORNTPHARM.NS": "Pharma", "ALKEM.NS": "Pharma",
    "MAXHEALTH.NS": "Healthcare", "FORTIS.NS": "Healthcare",
    "ADANIGREEN.NS": "Energy", "TATAPOWER.NS": "Power",
    "ADANIPOWER.NS": "Power", "NHPC.NS": "Power", "IOC.NS": "Energy",
    "GAIL.NS": "Energy", "PETRONET.NS": "Energy",
    "HINDALCO.NS": "Metals", "VEDL.NS": "Metals", "NMDC.NS": "Mining",
    "JINDALSTEL.NS": "Metals", "SAIL.NS": "Metals",
    "BAJAJ-AUTO.NS": "Auto", "TVSMOTOR.NS": "Auto", "ASHOKLEY.NS": "Auto",
    "MOTHERSON.NS": "Auto Parts", "BOSCHLTD.NS": "Auto Parts",
    "BAJAJHLDNG.NS": "Finance", "CHOLAFIN.NS": "Finance",
    "MUTHOOTFIN.NS": "Finance", "MANAPPURAM.NS": "Finance",
    "SHRIRAMFIN.NS": "Finance", "PEL.NS": "Finance",
    "GODREJPROP.NS": "Real Estate", "OBEROIRLTY.NS": "Real Estate",
    "PRESTIGE.NS": "Real Estate", "BRIGADE.NS": "Real Estate",
    "SRF.NS": "Chemicals", "AARTI.NS": "Chemicals",
    "DEEPAKNTR.NS": "Chemicals", "NAVINFLUOR.NS": "Chemicals",
    "EMAMILTD.NS": "FMCG", "VBL.NS": "FMCG",
    "JUBLFOOD.NS": "Consumer", "PAGEIND.NS": "Consumer",
    "^NSEI": "Index", "^NSEBANK": "Index",
}
ALL_TICKERS_SORTED = sorted(NSE_UNIVERSE.keys())

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Analyzer & Optimizer",
    layout="wide", page_icon="◈", initial_sidebar_state="expanded",
)

# ── AGGRESSIVE CSS — Bloomberg-terminal aesthetic ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

    /* =========================================================
       ROOT — kill every white surface Streamlit creates
       ========================================================= */
    :root {
        --bg:       #060A10;
        --bg2:      #0C1017;
        --panel:    #111620;
        --card:     #161C28;
        --border:   #1E2636;
        --border2:  #2A3346;
        --blue:     #3B82F6;
        --blue-dim: rgba(59,130,246,0.12);
        --green:    #22C55E;
        --green-dim:rgba(34,197,94,0.12);
        --red:      #EF4444;
        --red-dim:  rgba(239,68,68,0.12);
        --amber:    #F59E0B;
        --t1:       #F1F5F9;   /* primary text */
        --t2:       #94A3B8;   /* secondary text */
        --t3:       #64748B;   /* muted text */
        --mono:     'JetBrains Mono', 'Consolas', monospace;
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--t1) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* Kill Streamlit top bar / header chrome */
    header[data-testid="stHeader"] {
        background: var(--bg) !important;
        border-bottom: 1px solid var(--border) !important;
    }
    /* Hide deploy button area for cleaner look */
    .stDeployButton { display: none !important; }

    /* Kill all default padding bloat */
    .block-container {
        padding: 1.5rem 2rem 1rem 2rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ───────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg2) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--t2) !important;
    }
    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: var(--t1) !important;
    }

    /* Sidebar inputs — dark styled */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: var(--card) !important;
        color: var(--t1) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] .stNumberInput > div > div > input {
        background: var(--card) !important;
        color: var(--t1) !important;
        border: 1px solid var(--border2) !important;
    }

    /* ── ALL text visible ──────────────────────── */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown h5, .stMarkdown h6, .stMarkdown strong, .stMarkdown b,
    .stCaption, p, span, label, li {
        color: var(--t1) !important;
    }
    .stMarkdown h4, .stMarkdown h5 {
        color: var(--t2) !important;
    }
    .stCaption, small {
        color: var(--t3) !important;
    }

    /* ── Metric cards — terminal style ─────────── */
    div[data-testid="stMetric"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--t3) !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        font-family: var(--mono) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: var(--t1) !important;
        font-family: var(--mono) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-family: var(--mono) !important;
        font-weight: 600 !important;
    }

    /* ── Tabs — tight, terminal bar ────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        padding: 2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 3px !important;
        padding: 6px 16px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        color: var(--t3) !important;
        font-family: var(--mono) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--blue) !important;
        color: #fff !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Dataframe / Table — FORCE DARK ────────── */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stDataFrame"] iframe {
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    /* Glide data grid overrides */
    .dvn-scroller {
        background: var(--panel) !important;
    }
    canvas + div {
        background: var(--panel) !important;
    }

    /* ── Buttons ───────────────────────────────── */
    .stButton > button {
        background: var(--card) !important;
        color: var(--t1) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    .stButton > button:hover {
        background: var(--border) !important;
        border-color: var(--blue) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--blue) !important;
        border: 1px solid var(--blue) !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2563EB !important;
    }

    /* ── Alerts — dark ─────────────────────────── */
    .stAlert {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--t1) !important;
    }
    .stAlert p, .stAlert span {
        color: var(--t1) !important;
    }

    /* Success / warning / error alerts */
    div[data-testid="stAlert"][data-baseweb] {
        background: var(--card) !important;
    }

    /* ── Expander ──────────────────────────────── */
    details {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 5px !important;
    }
    details summary {
        color: var(--t2) !important;
    }
    details summary span {
        color: var(--t2) !important;
    }

    /* ── Selectbox / Dropdown ──────────────────── */
    div[data-baseweb="select"] > div {
        background: var(--card) !important;
        border-color: var(--border2) !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: var(--t1) !important;
    }
    div[data-baseweb="popover"] {
        background: var(--panel) !important;
        border: 1px solid var(--border2) !important;
    }
    div[data-baseweb="popover"] li {
        color: var(--t1) !important;
    }
    div[data-baseweb="popover"] li:hover {
        background: var(--card) !important;
    }

    /* ── Toggle ────────────────────────────────── */
    div[data-testid="stToggle"] label span {
        color: var(--t2) !important;
    }

    /* ── Slider ────────────────────────────────── */
    .stSlider label {
        color: var(--t2) !important;
    }
    .stSlider div[data-testid="stTickBarMin"],
    .stSlider div[data-testid="stTickBarMax"] {
        color: var(--t3) !important;
    }

    /* ── Remove excess whitespace/margins ──────── */
    .element-container { margin-bottom: 0 !important; }
    .stMarkdown { margin-bottom: 0 !important; }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem !important;
    }

    /* ── Footer hide ──────────────────────────── */
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }

    /* ── Number input dark ─────────────────────── */
    .stNumberInput div[data-baseweb="input"] {
        background: var(--card) !important;
        border-color: var(--border2) !important;
    }
    .stNumberInput input {
        color: var(--t1) !important;
    }
    .stNumberInput button {
        color: var(--t2) !important;
        background: var(--panel) !important;
        border-color: var(--border2) !important;
    }

    /* ── Text input dark ──────────────────────── */
    .stTextInput div[data-baseweb="input"] {
        background: var(--card) !important;
        border-color: var(--border2) !important;
    }
    .stTextInput input {
        color: var(--t1) !important;
    }
    .stTextInput input::placeholder {
        color: var(--t3) !important;
    }

    /* ── Section divider ──────────────────────── */
    .sec-div {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 12px 0;
        padding: 0;
    }
    .sec-div .sec-label {
        font-family: var(--mono);
        font-size: 11px;
        font-weight: 700;
        color: var(--t3);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .sec-div .sec-line {
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ── Custom scrollbar ─────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "portfolio_assets" not in st.session_state:
    st.session_state.portfolio_assets = [
        {"ticker": "RELIANCE.NS", "weight": 0.20},
        {"ticker": "TCS.NS", "weight": 0.15},
        {"ticker": "HDFCBANK.NS", "weight": 0.20},
        {"ticker": "INFY.NS", "weight": 0.15},
        {"ticker": "ITC.NS", "weight": 0.10},
        {"ticker": "ICICIBANK.NS", "weight": 0.10},
        {"ticker": "SBIN.NS", "weight": 0.10},
    ]
if "price_data" not in st.session_state:
    st.session_state.price_data = None
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# ── Helpers ───────────────────────────────────────────────────────────────────
def sec_header(title):
    st.markdown(f'<div class="sec-div"><span class="sec-label">{title}</span><span class="sec-line"></span></div>', unsafe_allow_html=True)

def tlabel(t):
    return t.replace(".NS","").replace(".BO","")

def get_sector(t):
    return NSE_UNIVERSE.get(t, "Other")

CHART = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(6,10,16,0.5)",
    font=dict(family="JetBrains Mono, Consolas, monospace", color="#94A3B8", size=11),
    margin=dict(l=48, r=12, t=36, b=36),
    xaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False, showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False, showline=False),
    hoverlabel=dict(bgcolor="#161C28", bordercolor="#1E2636", font_size=12),
)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_prices(tickers, lookback=252):
    import yfinance as yf
    df = yf.download(list(tickers), start=datetime.now()-timedelta(days=lookback),
                     end=datetime.now(), auto_adjust=True, progress=False, threads=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    elif "Close" in df.columns and len(tickers) == 1:
        df = df[["Close"]]; df.columns = list(tickers)
    return df.dropna(how="all")

@st.cache_data(ttl=120, show_spinner=False)
def validate_ticker(ticker):
    import yfinance as yf
    try:
        h = yf.Ticker(ticker).history(period="5d")
        if h.empty: return {"valid": False}
        return {"valid": True, "name": yf.Ticker(ticker).info.get("shortName", ticker),
                "price": round(float(h["Close"].iloc[-1]), 2)}
    except Exception:
        return {"valid": False}

def compute_var(returns, weights, pv, conf=0.99):
    pr = returns.values @ weights
    hv = np.percentile(pr, (1-conf)*100)
    he = pr[pr <= hv].mean()
    from scipy.stats import norm
    mu, sig = pr.mean(), pr.std()
    pv_ = mu + sig * norm.ppf(1-conf)
    pe = mu - sig * norm.pdf(norm.ppf(1-conf)) / (1-conf)
    np.random.seed(42)
    cov = np.cov(returns.values.T)
    L = np.linalg.cholesky(cov + np.eye(len(weights))*1e-10)
    z = np.random.standard_normal((10000, len(weights)))
    sp = (z @ L.T + returns.mean().values) @ weights
    mv = np.percentile(sp, (1-conf)*100)
    me = sp[sp <= mv].mean()
    safe = lambda x: round(abs(x)*100, 4) if not np.isnan(x) else 0
    safe_v = lambda x: round(abs(x)*pv, 0) if not np.isnan(x) else 0
    return {
        "hist": {"pct": safe(hv), "inr": safe_v(hv), "es_pct": safe(he), "es_inr": safe_v(he)},
        "param": {"pct": safe(pv_), "inr": safe_v(pv_), "es_pct": safe(pe), "es_inr": safe_v(pe)},
        "mc": {"pct": safe(mv), "inr": safe_v(mv), "es_pct": safe(me), "es_inr": safe_v(me)},
        "vol": round(sig*np.sqrt(252)*100, 2),
        "ret": round(mu*252*100, 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**◈ PORTFOLIO BUILDER**")
    st.markdown("---")

    ptf_value = st.number_input("Portfolio Value (₹)", value=10_000_000,
                                 step=10_000, min_value=1000, format="%d")
    lookback = st.slider("Lookback (days)", 60, 756, 252, step=21)
    st.markdown("---")

    # Add from NSE
    existing = {a["ticker"] for a in st.session_state.portfolio_assets}
    avail = [t for t in ALL_TICKERS_SORTED if t not in existing]
    cs1, cs2 = st.columns([5, 1])
    with cs1:
        nt = st.selectbox("NSE", [""] + avail, 0, label_visibility="collapsed",
                          placeholder="Search NSE…")
    with cs2:
        if st.button("＋", key="a1"):
            if nt and nt not in existing:
                n = len(st.session_state.portfolio_assets) + 1
                st.session_state.portfolio_assets.append({"ticker": nt, "weight": round(1/n, 4)})
                st.rerun()

    # Custom ticker
    ct = st.text_input("Custom ticker", placeholder="AAPL, BTC-USD, GC=F",
                       label_visibility="collapsed")
    cv, ca = st.columns(2)
    with cv:
        if st.button("Validate", use_container_width=True):
            if ct:
                with st.spinner("…"):
                    r = validate_ticker(ct.upper().strip())
                    if r["valid"]: st.success(f"✓ {r['name']} — ₹{r['price']:,.2f}")
                    else: st.error("✗ Not found")
    with ca:
        if st.button("Add", key="ac", use_container_width=True):
            c = ct.upper().strip()
            if c and c not in existing:
                n = len(st.session_state.portfolio_assets) + 1
                st.session_state.portfolio_assets.append({"ticker": c, "weight": round(1/n, 4)})
                st.rerun()

    with st.expander("Quick Presets"):
        p1, p2 = st.columns(2)
        with p1:
            if st.button("NIFTY 10", use_container_width=True):
                st.session_state.portfolio_assets = [{"ticker": t, "weight": .10} for t in
                    ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                     "HINDUNILVR.NS","BHARTIARTL.NS","ITC.NS","SBIN.NS","KOTAKBANK.NS"]]
                st.session_state.price_data = None; st.rerun()
            if st.button("Pharma", use_container_width=True):
                st.session_state.portfolio_assets = [{"ticker": t, "weight": round(1/6,4)} for t in
                    ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","BIOCON.NS","LUPIN.NS"]]
                st.session_state.price_data = None; st.rerun()
        with p2:
            if st.button("Banks", use_container_width=True):
                st.session_state.portfolio_assets = [{"ticker": t, "weight": round(1/8,4)} for t in
                    ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
                     "BANKBARODA.NS","INDUSINDBK.NS","PNB.NS"]]
                st.session_state.price_data = None; st.rerun()
            if st.button("Global", use_container_width=True):
                st.session_state.portfolio_assets = [
                    {"ticker":"RELIANCE.NS","weight":.15},{"ticker":"TCS.NS","weight":.15},
                    {"ticker":"AAPL","weight":.15},{"ticker":"MSFT","weight":.15},
                    {"ticker":"GOOGL","weight":.10},{"ticker":"AMZN","weight":.10},
                    {"ticker":"BTC-USD","weight":.10},{"ticker":"GC=F","weight":.10}]
                st.session_state.price_data = None; st.rerun()

    st.markdown("---")
    st.markdown(f"**HOLDINGS** · {len(st.session_state.portfolio_assets)}")

    removals = []
    for i, a in enumerate(st.session_state.portfolio_assets):
        ct_, cw_, cr_ = st.columns([3, 3, 1])
        with ct_:
            st.markdown(f"**{tlabel(a['ticker'])}**")
            sec = NSE_UNIVERSE.get(a["ticker"], "")
            if sec: st.caption(sec)
        with cw_:
            nw = st.number_input(f"w{i}", value=a["weight"], min_value=0.0, max_value=1.0,
                                  step=0.01, format="%.4f", label_visibility="collapsed", key=f"w_{i}")
            st.session_state.portfolio_assets[i]["weight"] = nw
        with cr_:
            if st.button("✕", key=f"r_{i}"): removals.append(i)
    if removals:
        for idx in sorted(removals, reverse=True):
            st.session_state.portfolio_assets.pop(idx)
        st.rerun()

    tw = sum(a["weight"] for a in st.session_state.portfolio_assets)
    if abs(tw - 1.0) > 0.01:
        st.warning(f"Σw = {tw:.4f} ≠ 1.0")
        n1, n2 = st.columns(2)
        with n1:
            if st.button("Normalize", use_container_width=True):
                if tw > 0:
                    for a in st.session_state.portfolio_assets: a["weight"] = round(a["weight"]/tw, 4)
                    st.rerun()
        with n2:
            if st.button("Equal", use_container_width=True):
                n = len(st.session_state.portfolio_assets)
                for a in st.session_state.portfolio_assets: a["weight"] = round(1/n, 4)
                st.rerun()
    else:
        st.success(f"Σw = {tw:.4f}")

    st.markdown("---")
    st.toggle("Auto-refresh (2m)", key="_ar", value=st.session_state.auto_refresh,
              on_change=lambda: st.session_state.update(auto_refresh=not st.session_state.auto_refresh))

    if st.button("▶ RUN ANALYSIS", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
        st.session_state.price_data = None
        fetch_prices.clear()

    if st.button("Clear Portfolio", use_container_width=True):
        st.session_state.portfolio_assets = []
        st.session_state.price_data = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# Title bar
tc1, tc2 = st.columns([6, 1])
with tc1:
    st.markdown("## Portfolio Analyzer & Optimizer")
with tc2:
    if st.session_state.last_refresh:
        e = (datetime.now() - st.session_state.last_refresh).seconds
        st.markdown(f"<span style='color:#22C55E;font:700 12px var(--mono)'>● LIVE · {e}s</span>",
                    unsafe_allow_html=True)

if not st.session_state.portfolio_assets:
    st.info("Add assets in the sidebar to begin."); st.stop()

tickers = [a["ticker"] for a in st.session_state.portfolio_assets]
weights = np.array([a["weight"] for a in st.session_state.portfolio_assets])

if st.session_state.auto_refresh and st.session_state.price_data is not None:
    if st.session_state.last_refresh and (datetime.now()-st.session_state.last_refresh).seconds >= 120:
        st.session_state.run_analysis = True; st.session_state.price_data = None; fetch_prices.clear()

if st.session_state.run_analysis or st.session_state.price_data is not None:
    with st.spinner("Fetching live prices…"):
        pdf = fetch_prices(tuple(tickers), lookback)
        if pdf.empty: st.error("No data. Check tickers."); st.stop()
        at = [t for t in tickers if t in pdf.columns]
        miss = [t for t in tickers if t not in pdf.columns]
        if miss: st.warning(f"No data: {', '.join(miss)}")
        if not at: st.error("No valid data."); st.stop()
        pdf = pdf[at].dropna()
        aw = np.array([a["weight"] for a in st.session_state.portfolio_assets if a["ticker"] in at])
        ws = aw.sum()
        if ws > 0: aw = aw / ws
        st.session_state.price_data = pdf
        st.session_state.run_analysis = False
        st.session_state.last_refresh = datetime.now()

    # ── MARKET SNAPSHOT ───────────────────────────────────────────────────────
    sec_header("MARKET SNAPSHOT")
    latest = pdf.iloc[-1]
    prev = pdf.iloc[-2] if len(pdf) > 1 else latest
    chg = ((latest - prev) / prev * 100)
    nc = 4
    for ri in range((len(at) + nc - 1) // nc):
        cols = st.columns(nc)
        for j in range(nc):
            idx = ri * nc + j
            if idx >= len(at):
                with cols[j]: st.empty()
                continue
            t = at[idx]
            with cols[j]:
                st.metric(f"{tlabel(t)} · {aw[idx]*100:.0f}%",
                          f"₹{latest[t]:,.2f}", f"{chg[t]:+.2f}%")

    # ── PERFORMANCE ───────────────────────────────────────────────────────────
    rets = pdf.pct_change().dropna()
    if len(rets) < 30: st.error("Need ≥30 days."); st.stop()
    pr_ = rets[at].values @ aw
    pcum = (1 + pd.Series(pr_, index=rets.index)).cumprod()
    pval = pcum * ptf_value

    sec_header("PERFORMANCE")
    t1, t2, t3, t4 = st.tabs(["Portfolio", "Assets", "Drawdown", "Sharpe"])

    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pval.index, y=pval.values, mode="lines",
                                  line=dict(color="#3B82F6", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"))
        fig.update_layout(**CHART, height=360, yaxis_title="₹")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        acum = (1 + rets[at]).cumprod()
        fig = go.Figure()
        pal = px.colors.qualitative.Set2
        for i, t in enumerate(at):
            fig.add_trace(go.Scatter(x=acum.index, y=acum[t].values, mode="lines",
                                      name=tlabel(t), line=dict(color=pal[i%len(pal)], width=1.2)))
        fig.update_layout(**CHART, height=360, yaxis_title="Return")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        dd = (pcum / pcum.cummax() - 1) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
                                  line=dict(color="#EF4444", width=1.2),
                                  fill="tozeroy", fillcolor="rgba(239,68,68,0.06)"))
        fig.update_layout(**CHART, height=360, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        wn = 63
        if len(pr_) > wn:
            ps = pd.Series(pr_, index=rets.index)
            rs = ((ps.rolling(wn).mean()*252 - 0.065) / (ps.rolling(wn).std()*np.sqrt(252))).dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines",
                                      line=dict(color="#818CF8", width=1.2)))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.1)")
            fig.update_layout(**CHART, height=360, yaxis_title="Sharpe")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Need ≥{wn} days.")

    # ── RISK ──────────────────────────────────────────────────────────────────
    sec_header("RISK")
    with st.spinner("Computing VaR…"):
        vr = compute_var(rets[at], aw, ptf_value)
    dd_s = (pcum / pcum.cummax() - 1)
    mdd = dd_s.min()
    av = vr["vol"] / 100
    ar = vr["ret"] / 100
    sh = (ar - 0.065) / av if av > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("VOLATILITY", f"{vr['vol']:.2f}%")
    with k2: st.metric("RETURN (ANN)", f"{vr['ret']:.2f}%")
    with k3: st.metric("SHARPE", f"{sh:.3f}")
    with k4: st.metric("MAX DRAWDOWN", f"{mdd*100:.2f}%")

    # VaR table — rendered as custom HTML to stay dark
    sec_header("99% VALUE-AT-RISK")
    var_html = f"""
    <table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px;margin-bottom:16px">
    <thead>
    <tr style="border-bottom:1px solid #1E2636">
        <th style="text-align:left;padding:10px 12px;color:#94A3B8;font-weight:600">Method</th>
        <th style="text-align:right;padding:10px 12px;color:#94A3B8;font-weight:600">VaR (%)</th>
        <th style="text-align:right;padding:10px 12px;color:#94A3B8;font-weight:600">VaR (₹)</th>
        <th style="text-align:right;padding:10px 12px;color:#94A3B8;font-weight:600">ES (%)</th>
        <th style="text-align:right;padding:10px 12px;color:#94A3B8;font-weight:600">ES (₹)</th>
    </tr>
    </thead>
    <tbody>
    <tr style="border-bottom:1px solid #1E2636">
        <td style="padding:10px 12px;color:#F1F5F9">Historical</td>
        <td style="text-align:right;padding:10px 12px;color:#EF4444;font-weight:600">{vr['hist']['pct']:.4f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F1F5F9">₹{vr['hist']['inr']:,.0f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F59E0B;font-weight:600">{vr['hist']['es_pct']:.4f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F1F5F9">₹{vr['hist']['es_inr']:,.0f}</td>
    </tr>
    <tr style="border-bottom:1px solid #1E2636">
        <td style="padding:10px 12px;color:#F1F5F9">Parametric</td>
        <td style="text-align:right;padding:10px 12px;color:#EF4444;font-weight:600">{vr['param']['pct']:.4f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F1F5F9">₹{vr['param']['inr']:,.0f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F59E0B;font-weight:600">{vr['param']['es_pct']:.4f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F1F5F9">₹{vr['param']['es_inr']:,.0f}</td>
    </tr>
    <tr>
        <td style="padding:10px 12px;color:#F1F5F9">Monte Carlo (10k)</td>
        <td style="text-align:right;padding:10px 12px;color:#EF4444;font-weight:600">{vr['mc']['pct']:.4f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F1F5F9">₹{vr['mc']['inr']:,.0f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F59E0B;font-weight:600">{vr['mc']['es_pct']:.4f}</td>
        <td style="text-align:right;padding:10px 12px;color:#F1F5F9">₹{vr['mc']['es_inr']:,.0f}</td>
    </tr>
    </tbody>
    </table>
    """
    st.markdown(var_html, unsafe_allow_html=True)

    # ── ANALYTICS ─────────────────────────────────────────────────────────────
    sec_header("ANALYTICS")
    cc, ca = st.columns(2)
    with cc:
        corr = rets[at].corr()
        lb = [tlabel(t) for t in at]
        fig = px.imshow(corr.values, x=lb, y=lb, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, text_auto=".2f")
        fig.update_layout(**CHART, height=max(320, len(at)*38))
        st.plotly_chart(fig, use_container_width=True)

    with ca:
        fig = go.Figure(data=[go.Pie(
            labels=[tlabel(t) for t in at], values=aw, hole=.5,
            textinfo="label+percent", textfont=dict(size=11, color="#F1F5F9"),
            marker=dict(colors=px.colors.qualitative.Set2[:len(at)],
                        line=dict(color="#060A10", width=2)))])
        fig.update_layout(**CHART, height=max(320, len(at)*38), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Sector bar
    sd = {t: get_sector(t) for t in at}
    sw = {}
    for t, w in zip(at, aw): sw[sd[t]] = sw.get(sd[t], 0) + w
    sdf = pd.DataFrame([{"Sector": s, "Wt": round(w*100, 1)} for s, w in sorted(sw.items(), key=lambda x: -x[1])])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sdf["Sector"], y=sdf["Wt"],
                          marker_color=["#EF4444" if w > 35 else "#F59E0B" if w > 25 else "#3B82F6" for w in sdf["Wt"]],
                          text=[f"{w:.1f}%" for w in sdf["Wt"]], textposition="auto",
                          textfont=dict(color="#F1F5F9", size=12)))
    fig.add_hline(y=35, line_dash="dash", line_color="#EF4444",
                  annotation_text="SEBI 35%", annotation_font_color="#EF4444")
    fig.update_layout(**CHART, height=240, yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)

    # Return distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pr_*100, nbinsx=80, marker_color="#3B82F6", opacity=.8))
    fig.add_vline(x=-vr["hist"]["pct"], line_dash="dash", line_color="#EF4444",
                  annotation_text=f"VaR: -{vr['hist']['pct']:.2f}%", annotation_font_color="#EF4444")
    fig.update_layout(**CHART, height=240, xaxis_title="Daily Return (%)", yaxis_title="Freq")
    st.plotly_chart(fig, use_container_width=True)

    # ── COMPLIANCE ────────────────────────────────────────────────────────────
    sec_header("COMPLIANCE")
    wd = {a["ticker"]: a["weight"] for a in st.session_state.portfolio_assets if a["ticker"] in at}
    bd = {}
    try:
        import yfinance as yf
        ni = yf.download("^NSEI", start=pdf.index[0], end=pdf.index[-1],
                          auto_adjust=True, progress=False, multi_level_index=False)
        if not ni.empty:
            nr = ni["Close"].pct_change().dropna()
            for t in at:
                if t in rets.columns:
                    ci = rets[t].dropna().index.intersection(nr.index)
                    if len(ci) > 30:
                        cv = np.cov(rets[t].loc[ci], nr.loc[ci])
                        bd[t] = round(float(cv[0,1]/cv[1,1]), 3)
    except Exception: pass
    if not bd: bd = {t: 1.0 for t in at}
    pb = sum(bd.get(t, 1.0)*w for t, w in zip(at, aw))

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("PORTFOLIO BETA", f"{pb:.3f}")
        for sec, sw_ in sorted(sw.items(), key=lambda x: -x[1]):
            pct = sw_ * 100
            ic = "🔴" if pct > 35 else "🟡" if pct > 25 else "🟢"
            st.markdown(f"{ic} **{sec}** {pct:.1f}%")
    with c2:
        try:
            res = requests.post(f"{API_URL}/portfolio/compliance",
                                json={"weights": wd, "sectors": sd, "betas": bd,
                                      "fno_tickers": [], "cash_pct": 5., "leverage": 1.,
                                      "portfolio_value": ptf_value}, timeout=5)
            if res.status_code == 200:
                cd = res.json()
                if cd.get("passed"): st.success(f"SEBI: PASSED ({cd.get('checks_run',0)} rules)")
                else: st.error(f"SEBI: FAILED ({len(cd.get('errors',[]))} violations)")
                for e in cd.get("errors",[]): st.markdown(f"🔴 **{e['rule']}** {e['message']}")
                for w in cd.get("warnings",[]): st.markdown(f"🟡 **{w['rule']}** {w['message']}")
                if cd.get("suggested_fixes"):
                    with st.expander("Fixes"):
                        for f in cd["suggested_fixes"]: st.markdown(f"· {f}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            st.info("Backend offline — local checks")
            mx = max(aw); ok = True
            if mx > .15: st.warning(f"Single stock {mx*100:.1f}% > 15%"); ok = False
            for sec_, sw__ in sw.items():
                if sw__ > .35: st.warning(f"{sec_} {sw__*100:.1f}% > 35%"); ok = False
            if pb > 1: st.warning(f"Beta {pb:.3f} > 1.0"); ok = False
            if ok: st.success("Basic checks passed (offline)")

    # ── ASSET BREAKDOWN ───────────────────────────────────────────────────────
    sec_header("ASSET BREAKDOWN")
    rows = []
    for i, t in enumerate(at):
        r = rets[t]; vol = r.std()*np.sqrt(252)*100; rtn = r.mean()*252*100
        sh_ = (rtn/100-.065)/(vol/100) if vol > 0 else 0
        mdd_ = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()*100
        rows.append({"Ticker": tlabel(t), "Wt%": round(aw[i]*100,1), "Sector": sd.get(t,""),
                      "β": bd.get(t,1.0), "Vol%": round(vol,2), "Ret%": round(rtn,2),
                      "Sharpe": round(sh_,3), "MaxDD%": round(mdd_,2),
                      "VaR99%": round(abs(np.percentile(r,1))*100,2)})

    # Custom HTML table for asset breakdown — always dark
    hdr = "".join(f'<th style="text-align:{"left" if c in ("Ticker","Sector") else "right"};padding:8px 10px;color:#94A3B8;font-weight:600;border-bottom:1px solid #1E2636">{c}</th>' for c in rows[0].keys())
    body = ""
    for row in rows:
        cells = ""
        for k, v in row.items():
            align = "left" if k in ("Ticker","Sector") else "right"
            color = "#F1F5F9"
            if k == "Ret%" and isinstance(v, (int,float)): color = "#22C55E" if v >= 0 else "#EF4444"
            if k == "Sharpe" and isinstance(v, (int,float)): color = "#22C55E" if v >= 0 else "#EF4444"
            if k == "MaxDD%": color = "#EF4444"
            cells += f'<td style="text-align:{align};padding:8px 10px;color:{color};border-bottom:1px solid #1E2636">{v}</td>'
        body += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style="border:1px solid #1E2636;border-radius:6px;overflow-x:auto;margin-bottom:12px">
    <table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px">
    <thead><tr>{hdr}</tr></thead>
    <tbody>{body}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;color:#64748B;font:500 11px var(--mono);padding:20px 0 8px 0;border-top:1px solid #1E2636;margin-top:20px">
    PORTFOLIO ANALYZER & OPTIMIZER · DATA: YAHOO FINANCE (LIVE) · {datetime.now().strftime('%H:%M:%S')} · {len(at)} ASSETS
    </div>""", unsafe_allow_html=True)

    if st.session_state.auto_refresh: time.sleep(0.1); st.rerun()

else:
    # ── LANDING ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#111620;border:1px solid #1E2636;border-radius:8px;padding:40px;text-align:center;margin-top:12px">
    <div style="font:800 40px 'JetBrains Mono';color:#3B82F6;margin-bottom:8px">◈</div>
    <div style="font:700 18px Inter;color:#F1F5F9;margin-bottom:8px">Configure Portfolio</div>
    <div style="font:400 13px Inter;color:#94A3B8;max-width:420px;margin:0 auto;line-height:1.7">
    Add assets from the sidebar — NSE, BSE, US equities, crypto, commodities, ETFs.<br>
    Adjust weights and press <b style="color:#F1F5F9">▶ RUN ANALYSIS</b>.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.portfolio_assets:
        st.markdown(f"**{len(st.session_state.portfolio_assets)} assets configured**")
        cfg = pd.DataFrame(st.session_state.portfolio_assets)
        # Custom HTML table
        hdr = '<th style="text-align:left;padding:8px 10px;color:#94A3B8;font-weight:600;border-bottom:1px solid #1E2636">Ticker</th>'
        hdr += '<th style="text-align:left;padding:8px 10px;color:#94A3B8;font-weight:600;border-bottom:1px solid #1E2636">Sector</th>'
        hdr += '<th style="text-align:right;padding:8px 10px;color:#94A3B8;font-weight:600;border-bottom:1px solid #1E2636">Weight</th>'
        body = ""
        for _, row in cfg.iterrows():
            tk = tlabel(row["ticker"])
            sec = NSE_UNIVERSE.get(row["ticker"], "Custom")
            wt = f'{row["weight"]*100:.1f}%'
            body += f'<tr><td style="padding:8px 10px;color:#F1F5F9;border-bottom:1px solid #1E2636;font-weight:600">{tk}</td>'
            body += f'<td style="padding:8px 10px;color:#94A3B8;border-bottom:1px solid #1E2636">{sec}</td>'
            body += f'<td style="text-align:right;padding:8px 10px;color:#3B82F6;border-bottom:1px solid #1E2636;font-weight:600">{wt}</td></tr>'
        st.markdown(f"""
        <div style="border:1px solid #1E2636;border-radius:6px;overflow:hidden;margin-top:12px">
        <table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px">
        <thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table></div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.info("Press **▶ RUN ANALYSIS** in the sidebar.")
