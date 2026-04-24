"""
Nestle Retail Analytics — Streamlit Dashboard
=============================================
A comprehensive showcase dashboard for the multi-table retail dataset.
Designed for professionals and students building resume projects.

Requirements:
    pip install streamlit pandas plotly numpy

Run:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nestlé Retail Analytics",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f0f14;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #16161f !important;
    border-right: 1px solid #2a2a3a;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e8 !important;
}

/* Hero title */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f8c842 0%, #ff8c42 50%, #e84393 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8888aa;
    margin-top: 6px;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* Section headings */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0f0fa;
    border-left: 4px solid #f8c842;
    padding-left: 12px;
    margin-bottom: 16px;
}

/* KPI cards */
.kpi-grid {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
}
.kpi-card {
    background: #1a1a26;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 20px 24px;
    flex: 1;
    min-width: 140px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #f8c842);
}
.kpi-card:hover { border-color: #f8c842; }
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #66668a;
    font-weight: 600;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f0f0fa;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.8rem;
    color: #4caf86;
    margin-top: 4px;
}

/* Dataset info pill */
.pill {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #2e2e42;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #9999bb;
    margin: 3px;
    font-weight: 500;
}

/* Use-case cards */
.usecase-card {
    background: #1a1a26;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}
.usecase-tag {
    display: inline-block;
    background: #f8c84222;
    color: #f8c842;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.usecase-title {
    font-weight: 600;
    color: #e8e8f8;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.usecase-desc {
    font-size: 0.82rem;
    color: #7777a0;
    line-height: 1.5;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #22223a;
    margin: 28px 0;
}

/* Plotly container */
.plot-container > div {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────────────────────
ACCENT   = "#f8c842"
ACCENT2  = "#ff8c42"
ACCENT3  = "#e84393"
PALETTE  = ["#f8c842", "#ff8c42", "#e84393", "#4cbbe8", "#8c5cf8", "#4caf86", "#f87060"]
BG       = "#0f0f14"
BG2      = "#1a1a26"
GRID_CLR = "#22223a"
TEXT_CLR = "#d0d0e8"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=TEXT_CLR),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=16, r=16, t=48, b=16),
)

# Default axis styling
AXIS_STYLE = dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, color=TEXT_CLR)


def plot_layout(title=None, **kwargs):
    """Safe layout builder — merges base, axis defaults, and per-chart overrides.
    Pass title='My Title' to render a styled Playfair Display heading inside the chart."""
    layout = dict(**PLOTLY_LAYOUT)
    layout["xaxis"] = dict(**AXIS_STYLE)
    layout["yaxis"] = dict(**AXIS_STYLE)
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(family="Playfair Display", size=15, color="#f0f0fa"),
            x=0, xanchor="left", pad=dict(l=4),
        )
    for k, v in kwargs.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            layout[k] = {**AXIS_STYLE, **v}
        else:
            layout[k] = v
    return layout

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    base = Path(__file__).parent

    order       = pd.read_csv(base / "synthetic_data" / "ORDER.csv")
    customer    = pd.read_csv(base / "synthetic_data" / "CUSTOMER.csv")
    product     = pd.read_csv(base / "synthetic_data" / "PRODUCT.csv")
    category    = pd.read_csv(base / "synthetic_data" / "CATEGORY.csv")
    store       = pd.read_csv(base / "synthetic_data" / "STORE.csv")
    channel     = pd.read_csv(base / "synthetic_data" / "CHANNEL.csv")
    promotion   = pd.read_csv(base / "synthetic_data" / "PROMOTION.csv")
    price_chg   = pd.read_csv(base / "synthetic_data" / "PRICE_CHANGE.csv")
    support     = pd.read_csv(base / "synthetic_data" / "SUPPORT_TICKET.csv")
    campaign    = pd.read_csv(base / "synthetic_data" / "CAMPAIGN_TOUCH.csv")
    ext_factor  = pd.read_csv(base / "synthetic_data" / "EXTERNAL_FACTOR.csv")
    mkt_channel = pd.read_csv(base / "synthetic_data" / "MARKETING_CHANNEL.csv")

    order["order_date"] = pd.to_datetime(order["order_date"], dayfirst=True)
    customer["signup_date"] = pd.to_datetime(customer["signup_date"])
    support["created_date"] = pd.to_datetime(support["created_date"])
    campaign["touch_date"]  = pd.to_datetime(campaign["touch_date"])

    return dict(
        order=order, customer=customer, product=product,
        category=category, store=store, channel=channel,
        promotion=promotion, price_chg=price_chg, support=support,
        campaign=campaign, ext_factor=ext_factor, mkt_channel=mkt_channel,
    )

data = load_data()
order    = data["order"]
customer = data["customer"]
product  = data["product"]
category = data["category"]
store    = data["store"]
channel  = data["channel"]
support  = data["support"]
campaign = data["campaign"]
price_chg = data["price_chg"]
ext_factor = data["ext_factor"]
mkt_channel = data["mkt_channel"]

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 20px'>
      <div style='font-family:Playfair Display,serif;font-size:1.4rem;font-weight:900;
                  background:linear-gradient(135deg,#f8c842,#ff8c42);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;'>☕ Nestlé Analytics</div>
      <div style='font-size:0.72rem;color:#55557a;margin-top:2px;'>Retail Intelligence Suite</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🛍️ Sales & Orders", "👥 Customers", "📣 Marketing", "🎧 Support", "🌐 External Factors", "🔍 Price Audit", "📚 Dataset Guide"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#22223a;margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#44446a;'>Filters</div>", unsafe_allow_html=True)
    year_filter = st.multiselect("Year", sorted(order["year"].unique()), default=sorted(order["year"].unique()))

    st.markdown("<hr style='border-color:#22223a;margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.68rem;color:#44446a;line-height:1.7;'>
      Built for resume projects &amp; portfolio showcases.<br>
      Data sourced from Kaggle.
    </div>
    """, unsafe_allow_html=True)

# Apply year filter globally
order_f = order[order["year"].isin(year_filter)]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown('<p class="hero-title">Retail Intelligence<br>Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">A full-spectrum FMCG dataset — orders, customers, marketing, support & external signals.</p>', unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # KPI row
    total_orders   = len(order_f)
    total_customers = customer["customer_id"].nunique()
    active_cust    = customer[customer["status"] == "active"]["customer_id"].nunique()
    total_products = product["product_id"].nunique()
    total_stores   = store["store_id"].nunique()
    total_campaigns = campaign["campaign_name"].nunique()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col, label, value, accent="#f8c842", delta=None):
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{accent}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {"<div class='kpi-delta'>"+delta+"</div>" if delta else ""}
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "Total Orders",    f"{total_orders:,}",    "#f8c842")
    kpi(k2, "Customers",       f"{total_customers:,}", "#ff8c42")
    kpi(k3, "Active Customers",f"{active_cust:,}",     "#4caf86")
    kpi(k4, "SKUs",            f"{total_products:,}",  "#4cbbe8")
    kpi(k5, "Stores",          f"{total_stores:,}",    "#8c5cf8")
    kpi(k6, "Campaigns",       f"{total_campaigns:,}", "#e84393")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Dataset at a glance + ER overview
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown('<p class="section-title">Dataset at a Glance</p>', unsafe_allow_html=True)
        tables = [
            ("ORDER",          f"{len(order):,} rows",       "Core transactional fact table"),
            ("CUSTOMER",       f"{len(customer):,} rows",    "Demographics, segment & status"),
            ("CAMPAIGN_TOUCH", f"{len(campaign):,} rows",    "280 K marketing touch-points"),
            ("EXTERNAL_FACTOR",f"{len(ext_factor):,} rows",  "Weather, CPI, holidays per store"),
            ("SUPPORT_TICKET", f"{len(support):,} rows",     "Issue type, CSAT & resolution time"),
            ("PRICE_CHANGE",   f"{len(price_chg):,} rows",   "Weekly product-level pricing"),
            ("PRODUCT",        f"{len(product):,} rows",     "SKU catalogue with brand & category"),
            ("STORE",          f"{len(store):,} rows",       "140 stores across Indian states"),
            ("PROMOTION",      f"{len(data['promotion']):,} rows", "Discount mechanics & windows"),
        ]
        for name, rows, desc in tables:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:14px;padding:10px 0;
                        border-bottom:1px solid #1e1e2e;'>
              <div style='font-family:monospace;font-size:0.78rem;color:#f8c842;
                          background:#1e1e28;padding:3px 10px;border-radius:6px;
                          min-width:160px;'>{name}</div>
              <div style='font-size:0.78rem;color:#6666aa;min-width:90px;'>{rows}</div>
              <div style='font-size:0.82rem;color:#8888b0;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="section-title">Orders Over Time</p>', unsafe_allow_html=True)
        monthly = order_f.copy()
        monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
        monthly_cnt = monthly.groupby("month").size().reset_index(name="orders")
        fig = px.area(monthly_cnt, x="month", y="orders",
                      color_discrete_sequence=[ACCENT])
        fig.update_traces(fill="tozeroy", line_width=2,
                          fillcolor="rgba(248,200,66,0.12)")
        fig.update_layout(**plot_layout(
                          title="Monthly Order Volume",
                          height=220,
                          xaxis_title=None, yaxis_title=None,
                          xaxis=dict(tickangle=-35, gridcolor=GRID_CLR),
                          yaxis=dict(gridcolor=GRID_CLR)))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-title" style="margin-top:8px">Channel Split</p>', unsafe_allow_html=True)
        ch_cnt = order_f.merge(channel, on="channel_id")["channel_name"].value_counts().reset_index()
        ch_cnt.columns = ["channel", "orders"]
        fig2 = px.pie(ch_cnt, names="channel", values="orders",
                      color_discrete_sequence=PALETTE, hole=0.55)
        fig2.update_traces(textposition="outside", textinfo="percent+label",
                           textfont_color=TEXT_CLR)
        fig2.update_layout(**plot_layout(
                            title="Orders by Sales Channel",
                            height=250,
                            showlegend=False, margin=dict(l=10,r=10,t=10,b=10)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Explore the Tabs Above for Deep Dives ↗</p>', unsafe_allow_html=True)
    st.info("Use the **sidebar** to navigate between analysis modules. Apply **Year filters** to scope the data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SALES & ORDERS  (category / brand / sub-category deep-dive)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛍️ Sales & Orders":
    st.markdown('<p class="section-title">Sales & Order Analytics</p>', unsafe_allow_html=True)

    # ── Build enriched catalogue (product → category) + order week counts ───
    pc_rich = (price_chg
               .merge(product, on="product_id")
               .merge(category, on="category_id"))

    order_wk = (order_f
                .groupby(["year", "week"])
                .size()
                .reset_index(name="orders"))

    # Each product in PRICE_CHANGE represents a "present" SKU that week;
    # distribute order volume proportionally to active SKUs per category per week
    cat_wk = (pc_rich
              .groupby(["year", "week", "category_name"])["product_id"]
              .count()
              .reset_index(name="sku_count"))

    cat_wk = cat_wk.merge(order_wk, on=["year", "week"])
    total_sku_per_week = cat_wk.groupby(["year","week"])["sku_count"].transform("sum")
    cat_wk["cat_orders"] = (cat_wk["sku_count"] / total_sku_per_week * cat_wk["orders"]).round(0)

    brand_wk = (pc_rich
                .groupby(["year", "week", "category_name", "brand"])["product_id"]
                .count()
                .reset_index(name="sku_count"))
    brand_wk = brand_wk.merge(order_wk, on=["year", "week"])
    total_sku_bw = brand_wk.groupby(["year","week"])["sku_count"].transform("sum")
    brand_wk["brand_orders"] = (brand_wk["sku_count"] / total_sku_bw * brand_wk["orders"]).round(0)

    order_store = order_f.merge(store, on="store_id")

    # ── Sub-page selector ────────────────────────────────────────────────────
    sub = st.radio(
        "View",
        ["📦 Category Overview", "🏷️ Brand Deep-Dive", "🗺️ Store & Channel", "💰 Pricing Intelligence"],
        horizontal=True,
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    if sub == "📦 Category Overview":
    # ════════════════════════════════════════════════════════════════

        # Row 1 — category share (donut) + category order volume (bar)
        col1, col2 = st.columns([1, 1.6])

        with col1:
            cat_total = cat_wk.groupby("category_name")["cat_orders"].sum().reset_index()
            cat_total.columns = ["category", "orders"]
            cat_total = cat_total.sort_values("orders", ascending=False)

            fig = px.pie(
                cat_total, names="category", values="orders",
                color_discrete_sequence=PALETTE, hole=0.58,
            )
            fig.update_traces(
                textposition="outside", textinfo="percent+label",
                textfont=dict(color=TEXT_CLR, size=11),
                pull=[0.03]*len(cat_total),
            )
            fig.update_layout(**plot_layout(
                title="Category Share of Order Volume",
                height=380, showlegend=False,
                margin=dict(l=10, r=10, t=52, b=10),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cat_bar = cat_total.sort_values("orders")
            fig = px.bar(
                cat_bar, y="category", x="orders", orientation="h",
                color="orders",
                color_continuous_scale=[[0,"#1e1e2e"],[0.45,ACCENT2],[1,ACCENT]],
                text=cat_bar["orders"].apply(lambda v: f"{v/1e3:.0f}K"),
            )
            fig.update_traces(textposition="outside", textfont_color=TEXT_CLR)
            fig.update_layout(**plot_layout(
                title="Total Order Volume by Category",
                height=380, coloraxis_showscale=False,
                xaxis_title="Estimated Orders", yaxis_title=None,
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Row 2 — Category trend over time (stacked area)
        cat_month = cat_wk.copy()
        # Build a sortable month key from year + week
        cat_month["month_num"] = cat_month["year"] * 100 + (cat_month["week"] * 7 // 30 + 1).clip(1, 12)
        cat_month["period"] = (
            cat_month["year"].astype(str) + "-W" +
            cat_month["week"].astype(str).str.zfill(2)
        )
        cat_trend = (cat_month
                     .groupby(["period", "category_name"])["cat_orders"]
                     .sum()
                     .reset_index())
        # Keep only every 4th week label to avoid clutter; still plot all points
        all_periods = sorted(cat_trend["period"].unique())
        tick_periods = all_periods[::4]

        fig = px.area(
            cat_trend, x="period", y="cat_orders", color="category_name",
            color_discrete_sequence=PALETTE,
            groupnorm="percent",
        )
        fig.update_traces(line_width=0.6)
        fig.update_layout(**plot_layout(
            title="Category Mix Over Time  (% stacked area)",
            height=300,
            xaxis=dict(
                tickvals=tick_periods, ticktext=tick_periods,
                tickangle=-40, gridcolor=GRID_CLR,
            ),
            yaxis=dict(ticksuffix="%", gridcolor=GRID_CLR),
            xaxis_title=None, yaxis_title="Share of Orders",
            legend=dict(orientation="h", y=-0.28, bgcolor="rgba(0,0,0,0)"),
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Row 3 — SKUs per category (treemap) + avg price by category (radar)
        col3, col4 = st.columns(2)

        with col3:
            sku_cat = pc_rich.groupby(["category_name","brand"])["product_id"].nunique().reset_index()
            sku_cat.columns = ["category","brand","skus"]
            fig = px.treemap(
                sku_cat, path=["category","brand"], values="skus",
                color="skus",
                color_continuous_scale=[[0,"#1a1a30"],[0.4,ACCENT2],[1,ACCENT]],
            )
            fig.update_traces(
                textinfo="label+value",
                textfont=dict(family="DM Sans", size=12),
                marker_line_width=1,
                marker_line_color="#0f0f14",
            )
            fig.update_layout(**plot_layout(
                title="SKU Catalogue — Category → Brand (Treemap)",
                height=340, coloraxis_showscale=False,
                margin=dict(l=4, r=4, t=52, b=4),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            avg_price = pc_rich.groupby("category_name")["unit_price"].mean().reset_index()
            avg_price.columns = ["category","avg_price"]
            # Radar / polar bar
            fig = go.Figure(go.Barpolar(
                r=avg_price["avg_price"],
                theta=avg_price["category"],
                marker_color=PALETTE[:len(avg_price)],
                marker_line_color="#0f0f14",
                marker_line_width=1,
                opacity=0.88,
            ))
            fig.update_layout(**plot_layout(
                title="Avg Unit Price by Category (₹)",
                height=340,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(
                        tickfont=dict(size=9, color=TEXT_CLR),
                        gridcolor=GRID_CLR,
                    ),
                    radialaxis=dict(
                        tickfont=dict(size=8, color=TEXT_CLR),
                        gridcolor=GRID_CLR,
                        tickprefix="₹",
                    ),
                ),
                showlegend=False,
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Row 4 — YoY category price change heatmap
        pivot_price = (pc_rich.groupby(["category_name","year"])["unit_price"]
                       .mean().unstack().round(1))
        fig = go.Figure(go.Heatmap(
            z=pivot_price.values,
            x=[str(c) for c in pivot_price.columns],
            y=pivot_price.index.tolist(),
            colorscale=[[0,"#1a1a30"],[0.5,ACCENT2],[1,ACCENT]],
            text=pivot_price.values.round(0),
            texttemplate="₹%{text}",
            textfont=dict(size=11, color="#f0f0fa"),
            showscale=True,
        ))
        fig.update_layout(**plot_layout(
            title="Average Unit Price by Category × Year (₹)",
            height=300,
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            xaxis_title=None, yaxis_title=None,
            margin=dict(l=16, r=80, t=52, b=16),
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    elif sub == "🏷️ Brand Deep-Dive":
    # ════════════════════════════════════════════════════════════════

        # Filter by category
        all_cats = sorted(category["category_name"].unique())
        sel_cat = st.selectbox("Filter by Category", ["All Categories"] + all_cats)

        if sel_cat == "All Categories":
            pc_sel = pc_rich.copy()
            bw_sel = brand_wk.copy()
        else:
            pc_sel = pc_rich[pc_rich["category_name"] == sel_cat].copy()
            bw_sel = brand_wk[brand_wk["category_name"] == sel_cat].copy()

        col1, col2 = st.columns(2)

        with col1:
            brand_vol = (bw_sel.groupby("brand")["brand_orders"]
                         .sum().reset_index()
                         .sort_values("brand_orders", ascending=False)
                         .head(15))
            fig = px.bar(
                brand_vol, y="brand", x="brand_orders", orientation="h",
                color="brand_orders",
                color_continuous_scale=[[0,"#1e1e2e"],[0.5,"#8c5cf8"],[1,"#4cbbe8"]],
                text=brand_vol["brand_orders"].apply(lambda v: f"{v/1e3:.0f}K"),
            )
            fig.update_traces(textposition="outside", textfont_color=TEXT_CLR)
            fig.update_layout(**plot_layout(
                title="Top 15 Brands by Estimated Order Volume",
                height=380, coloraxis_showscale=False,
                xaxis_title="Estimated Orders", yaxis_title=None,
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            brand_price = (pc_sel.groupby("brand")["unit_price"]
                           .agg(["mean","min","max"])
                           .reset_index()
                           .sort_values("mean", ascending=False)
                           .head(15))
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=brand_price["brand"], x=brand_price["mean"],
                orientation="h", name="Avg Price",
                marker_color=ACCENT,
            ))
            fig.add_trace(go.Scatter(
                y=brand_price["brand"],
                x=brand_price["max"],
                mode="markers", name="Max",
                marker=dict(color=ACCENT3, size=8, symbol="diamond"),
            ))
            fig.add_trace(go.Scatter(
                y=brand_price["brand"],
                x=brand_price["min"],
                mode="markers", name="Min",
                marker=dict(color="#4caf86", size=8, symbol="diamond"),
            ))
            fig.update_layout(**plot_layout(
                title="Brand Price Range — Avg / Min / Max (₹)",
                height=380,
                xaxis_title="Unit Price (₹)", yaxis_title=None,
                legend=dict(orientation="h", y=1.08),
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Brand × category bubble chart (SKU count vs avg price vs orders)
        brand_bubble = (pc_rich
                        .groupby(["brand","category_name"])
                        .agg(skus=("product_id","nunique"), avg_price=("unit_price","mean"))
                        .reset_index())
        brand_orders_tot = (brand_wk
                            .groupby("brand")["brand_orders"]
                            .sum().reset_index())
        brand_bubble = brand_bubble.merge(brand_orders_tot, on="brand")

        fig = px.scatter(
            brand_bubble, x="avg_price", y="brand_orders",
            size="skus", color="category_name",
            hover_name="brand",
            color_discrete_sequence=PALETTE,
            size_max=40,
            log_x=True,
        )
        fig.update_layout(**plot_layout(
            title="Brand Landscape — Avg Price vs Order Volume (bubble = SKU count)",
            height=380,
            xaxis_title="Avg Unit Price ₹ (log)", yaxis_title="Estimated Orders",
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Brand price trajectory over time (top 8 brands)
        top8_brands = (brand_wk.groupby("brand")["brand_orders"]
                       .sum().nlargest(8).index.tolist())
        pc_top = pc_rich[pc_rich["brand"].isin(top8_brands)].copy()
        pc_top["period"] = (pc_top["year"].astype(str) + "-W" +
                            pc_top["week"].astype(str).str.zfill(2))
        pc_trend = pc_top.groupby(["period","brand"])["unit_price"].mean().reset_index()
        all_periods = sorted(pc_trend["period"].unique())
        tick_periods = all_periods[::4]

        fig = px.line(
            pc_trend, x="period", y="unit_price", color="brand",
            color_discrete_sequence=PALETTE, line_shape="spline",
        )
        fig.update_traces(line_width=2)
        fig.update_layout(**plot_layout(
            title="Price Trajectory — Top 8 Brands Over Time (₹)",
            height=300,
            xaxis=dict(tickvals=tick_periods, ticktext=tick_periods,
                       tickangle=-40, gridcolor=GRID_CLR),
            xaxis_title=None, yaxis_title="Avg Unit Price (₹)",
            legend=dict(orientation="h", y=-0.30, bgcolor="rgba(0,0,0,0)"),
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    elif sub == "🗺️ Store & Channel":
    # ════════════════════════════════════════════════════════════════

        col1, col2 = st.columns(2)

        with col1:
            state_cnt = (order_store.groupby("store_state")
                         .size().reset_index(name="orders")
                         .sort_values("orders", ascending=True).tail(15))
            fig = px.bar(
                state_cnt, y="store_state", x="orders", orientation="h",
                color="orders",
                color_continuous_scale=[[0,"#1e1e2e"],[0.5,ACCENT2],[1,ACCENT]],
                text=state_cnt["orders"].apply(lambda v: f"{v/1e3:.0f}K"),
            )
            fig.update_traces(textposition="outside", textfont_color=TEXT_CLR)
            fig.update_layout(**plot_layout(
                title="Top 15 States by Order Volume",
                height=380, coloraxis_showscale=False,
                xaxis_title="Orders", yaxis_title=None,
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            stype_ch = (order_store
                        .merge(channel, on="channel_id")
                        .groupby(["store_type","channel_name"])
                        .size().reset_index(name="orders"))
            fig = px.bar(
                stype_ch, x="store_type", y="orders",
                color="channel_name", barmode="group",
                color_discrete_sequence=PALETTE,
                text_auto=False,
            )
            fig.update_layout(**plot_layout(
                title="Orders by Store Type × Sales Channel",
                height=380,
                xaxis_title=None, yaxis_title="Orders",
                legend=dict(orientation="h", y=1.06),
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Weekly heatmap
        wk_year = order_f.groupby(["year","week"]).size().reset_index(name="orders")
        pivot = wk_year.pivot(index="year", columns="week", values="orders").fillna(0)
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(w) for w in pivot.columns],
            y=[str(y) for y in pivot.index],
            colorscale=[[0,"#0f0f14"],[0.4,ACCENT2],[1,ACCENT]],
            showscale=True,
        ))
        fig.update_layout(**plot_layout(
            title="Weekly Order Volume Heatmap (Year × Week)",
            height=200,
            xaxis=dict(tickvals=list(range(0,52,4)), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            xaxis_title="Week of Year", yaxis_title=None,
            margin=dict(l=16, r=60, t=52, b=16),
        ))
        st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            pay = order_f["payment_type"].value_counts().reset_index()
            pay.columns = ["payment_type","count"]
            fig = px.pie(
                pay, names="payment_type", values="count",
                color_discrete_sequence=PALETTE, hole=0.55,
            )
            fig.update_traces(
                textposition="outside", textinfo="percent+label",
                textfont_color=TEXT_CLR,
            )
            fig.update_layout(**plot_layout(
                title="Payment Type Distribution",
                height=280, showlegend=False,
                margin=dict(l=10, r=10, t=52, b=10),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            ch_stype = (order_store
                        .merge(channel, on="channel_id")
                        .groupby("channel_name")
                        .size().reset_index(name="orders"))
            fig = px.funnel(
                ch_stype.sort_values("orders", ascending=False),
                x="orders", y="channel_name",
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(**plot_layout(
                title="Order Funnel by Sales Channel",
                height=280, showlegend=False,
                xaxis_title="Orders", yaxis_title=None,
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    elif sub == "💰 Pricing Intelligence":
    # ════════════════════════════════════════════════════════════════

        col1, col2 = st.columns(2)

        with col1:
            # Price volatility (std dev) per category
            price_vol = (pc_rich.groupby("category_name")["unit_price"]
                         .std().reset_index()
                         .rename(columns={"unit_price":"price_std"})
                         .sort_values("price_std", ascending=True))
            fig = px.bar(
                price_vol, y="category_name", x="price_std", orientation="h",
                color="price_std",
                color_continuous_scale=[[0,"#1e1e2e"],[0.5,ACCENT3],[1,ACCENT]],
            )
            fig.update_layout(**plot_layout(
                title="Price Volatility by Category (Std Dev ₹)",
                height=320, coloraxis_showscale=False,
                xaxis_title="Std Dev of Unit Price (₹)", yaxis_title=None,
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot of prices per category
            fig = px.box(
                pc_rich, x="category_name", y="unit_price",
                color="category_name",
                color_discrete_sequence=PALETTE,
                points=False,
            )
            fig.update_layout(**plot_layout(
                title="Unit Price Distribution by Category (₹)",
                height=320, showlegend=False,
                xaxis=dict(tickangle=-30, gridcolor=GRID_CLR),
                xaxis_title=None, yaxis_title="Unit Price (₹)",
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Price trend per category over time (line)
        cat_price_trend = (pc_rich
                           .groupby(["year","week","category_name"])["unit_price"]
                           .mean().reset_index())
        cat_price_trend["period"] = (cat_price_trend["year"].astype(str) + "-W" +
                                     cat_price_trend["week"].astype(str).str.zfill(2))
        all_periods = sorted(cat_price_trend["period"].unique())
        tick_periods = all_periods[::4]

        fig = px.line(
            cat_price_trend, x="period", y="unit_price",
            color="category_name", line_shape="spline",
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(line_width=1.8)
        fig.update_layout(**plot_layout(
            title="Average Weekly Unit Price Trend by Category (₹)",
            height=320,
            xaxis=dict(tickvals=tick_periods, ticktext=tick_periods,
                       tickangle=-40, gridcolor=GRID_CLR),
            xaxis_title=None, yaxis_title="Avg Unit Price (₹)",
            legend=dict(orientation="h", y=-0.30, bgcolor="rgba(0,0,0,0)"),
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Promotion discount types
        promo = data["promotion"]
        col3, col4 = st.columns(2)
        with col3:
            promo_type = promo["promo_type"].value_counts().reset_index()
            promo_type.columns = ["promo_type","count"]
            fig = px.bar(
                promo_type, x="promo_type", y="count",
                color="promo_type", color_discrete_sequence=PALETTE,
            )
            fig.update_layout(**plot_layout(
                title="Promotions by Type",
                height=260, showlegend=False,
                xaxis_title=None, yaxis_title="# Promotions",
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.box(
                promo, x="promo_type", y="discount_value",
                color="promo_type", color_discrete_sequence=PALETTE,
                points="all",
            )
            fig.update_layout(**plot_layout(
                title="Discount Value Distribution by Promo Type",
                height=260, showlegend=False,
                xaxis_title=None, yaxis_title="Discount Value",
            ))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customers":
    st.markdown('<p class="section-title">Customer Analytics</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    seg_counts = customer["segment"].value_counts()
    status_counts = customer["status"].value_counts()

    def mini_kpi(col, label, value, color):
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    mini_kpi(c1, "Premium Seg.",   f"{seg_counts.get('Premium',0):,}",    ACCENT)
    mini_kpi(c2, "Mainstream Seg.",f"{seg_counts.get('Mainstream',0):,}", ACCENT2)
    mini_kpi(c3, "Value Seg.",     f"{seg_counts.get('Value',0):,}",      "#4cbbe8")
    mini_kpi(c4, "Churned",        f"{status_counts.get('churned',0):,}", ACCENT3)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        cross = customer.groupby(["segment","status"]).size().reset_index(name="count")
        fig = px.bar(cross, x="segment", y="count", color="status",
                     barmode="group", color_discrete_sequence=PALETTE)
        fig.update_layout(**plot_layout(
                title="Customer Segment × Status", height=300,
                          xaxis_title=None, yaxis_title="Customers"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        customer["month"] = customer["signup_date"].dt.to_period("M").astype(str)
        monthly_signup = customer.groupby("month").size().reset_index(name="signups")
        fig = px.line(monthly_signup, x="month", y="signups",
                      color_discrete_sequence=[ACCENT2])
        fig.update_traces(line_width=2)
        fig.update_layout(**plot_layout(
                title="New Customer Signups Over Time", height=300,
                          xaxis_title=None, yaxis_title="New Signups",
                          xaxis=dict(tickangle=-35, gridcolor=GRID_CLR)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        cust_state = customer.groupby("state").size().reset_index(name="customers").sort_values("customers", ascending=False).head(15)
        fig = px.bar(cust_state, x="customers", y="state", orientation="h",
                     color="customers",
                     color_continuous_scale=[[0,"#1e1e2e"],[0.5,"#8c5cf8"],[1,"#4cbbe8"]])
        fig.update_layout(**plot_layout(
                title="Top 15 Customer States by Order Volume", height=350,
                          coloraxis_showscale=False, xaxis_title="Customers", yaxis_title=None))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        order_cust = order_f.merge(customer[["customer_id","segment"]], on="customer_id")
        seg_orders = order_cust.groupby("segment").size().reset_index(name="orders")
        fig = px.pie(seg_orders, names="segment", values="orders",
                     color_discrete_sequence=PALETTE, hole=0.55)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont_color=TEXT_CLR)
        fig.update_layout(**plot_layout(
                title="Orders per Customer Segment", height=350, showlegend=False,
                          margin=dict(l=10,r=10,t=10,b=10)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    cust_freq = order_f.groupby("customer_id").size().reset_index(name="order_count")
    cust_freq["bucket"] = pd.cut(cust_freq["order_count"],
                                  bins=[0,1,3,5,10,20,100],
                                  labels=["1","2–3","4–5","6–10","11–20","20+"])
    bucket_cnt = cust_freq["bucket"].value_counts().sort_index().reset_index()
    bucket_cnt.columns = ["orders_placed", "customers"]
    fig = px.bar(bucket_cnt, x="orders_placed", y="customers",
                 color="customers",
                 color_continuous_scale=[[0,"#1e1e2e"],[0.4,ACCENT2],[1,ACCENT]])
    fig.update_layout(**plot_layout(
                title="Customer Order Frequency Distribution", height=250, coloraxis_showscale=False,
                      xaxis_title="# Orders Placed", yaxis_title="# Customers"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MARKETING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📣 Marketing":
    st.markdown('<p class="section-title">Marketing & Campaign Analytics</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        ch_touch = campaign["channel"].value_counts().reset_index()
        ch_touch.columns = ["channel","touches"]
        fig = px.bar(ch_touch, x="channel", y="touches",
                     color="channel", color_discrete_sequence=PALETTE)
        fig.update_layout(**plot_layout(
                title="Campaign Touch-Points by Channel", height=300, showlegend=False,
                          xaxis_title=None, yaxis_title="Touch-Points"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        out = campaign["outcome"].value_counts().reset_index()
        out.columns = ["outcome","count"]
        fig = px.pie(out, names="outcome", values="count",
                     color_discrete_sequence=PALETTE, hole=0.5)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont_color=TEXT_CLR)
        fig.update_layout(**plot_layout(
                title="Campaign Outcome Distribution", height=300, showlegend=False,
                          margin=dict(l=10,r=10,t=10,b=10)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        camp_out = campaign[campaign["outcome"].isin(["Clicked","Converted"])]
        top_camps = camp_out["campaign_name"].value_counts().head(10).reset_index()
        top_camps.columns = ["campaign","engagements"]
        fig = px.bar(top_camps, y="campaign", x="engagements",
                     orientation="h", color="engagements",
                     color_continuous_scale=[[0,"#1e1e2e"],[0.5,ACCENT3],[1,ACCENT]])
        fig.update_layout(**plot_layout(
                title="Top 10 Campaigns by Engagement", height=320, coloraxis_showscale=False,
                          xaxis_title="Engagements", yaxis_title=None))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        conv = campaign.groupby("channel").apply(
            lambda x: (x["outcome"]=="Converted").sum() / len(x) * 100
        ).reset_index(name="conversion_rate")
        fig = px.bar(conv, x="channel", y="conversion_rate",
                     color="conversion_rate",
                     color_continuous_scale=[[0,"#1e1e2e"],[0.5,"#4cbbe8"],[1,"#4caf86"]])
        fig.update_layout(**plot_layout(
                title="Conversion Rate by Channel (%)", height=320, coloraxis_showscale=False,
                          xaxis_title=None, yaxis_title="Conversion Rate (%)"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    campaign["month"] = campaign["touch_date"].dt.to_period("M").astype(str)
    monthly_camp = campaign.groupby(["month","channel"]).size().reset_index(name="touches")
    fig = px.line(monthly_camp, x="month", y="touches", color="channel",
                  color_discrete_sequence=PALETTE)
    fig.update_layout(**plot_layout(
                title="Monthly Campaign Touch-Point Volume", height=280,
                      xaxis_title=None, yaxis_title="Touch-Points",
                      xaxis=dict(tickangle=-35, gridcolor=GRID_CLR)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    cost_tbl = data["mkt_channel"][["channel_name","channel_type","cost_tier"]].copy()
    st.dataframe(cost_tbl.style.set_properties(**{
        "background-color": "#1a1a26",
        "color": "#d0d0e8",
        "border": "1px solid #2a2a3a",
    }), use_container_width=True, height=250)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SUPPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎧 Support":
    st.markdown('<p class="section-title">Customer Support Analytics</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    def mini_kpi(col, label, value, color):
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    mini_kpi(c1, "Total Tickets",   f"{len(support):,}",                                           ACCENT)
    mini_kpi(c2, "Avg CSAT",        f"{support['csat_score'].mean():.2f}",                         "#4caf86")
    mini_kpi(c3, "Avg Resolution",  f"{support['resolution_time_hr'].mean():.0f}h",                ACCENT2)
    mini_kpi(c4, "Open Tickets",    f"{(support['status']=='Open').sum():,}",                      ACCENT3)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        issue = support["issue_type"].value_counts().reset_index()
        issue.columns = ["issue_type","count"]
        fig = px.bar(issue, y="issue_type", x="count", orientation="h",
                     color="count",
                     color_continuous_scale=[[0,"#1e1e2e"],[0.5,ACCENT2],[1,ACCENT]])
        fig.update_layout(**plot_layout(
                title="Tickets by Issue Type", height=300, coloraxis_showscale=False,
                          xaxis_title="Tickets", yaxis_title=None))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        csat_cnt = support["csat_score"].value_counts().sort_index().reset_index()
        csat_cnt.columns = ["csat_score","count"]
        fig = px.bar(csat_cnt, x="csat_score", y="count",
                     color="csat_score",
                     color_continuous_scale=[[0,ACCENT3],[0.5,ACCENT2],[1,"#4caf86"]])
        fig.update_layout(**plot_layout(
                title="CSAT Score Distribution (1–5)", height=300, coloraxis_showscale=False,
                          xaxis_title="CSAT Score (1–5)", yaxis_title="Tickets"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(support, x="issue_type", y="resolution_time_hr",
                     color="issue_type", color_discrete_sequence=PALETTE,
                     points=False)
        fig.update_layout(**plot_layout(
                title="Resolution Time by Issue Type", height=300, showlegend=False,
                          xaxis_title=None, yaxis_title="Resolution Time (hrs)"))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        sample = support.sample(min(600, len(support)), random_state=42)
        fig = px.scatter(sample, x="resolution_time_hr", y="csat_score",
                         color="issue_type", opacity=0.6,
                         color_discrete_sequence=PALETTE)
        fig.update_layout(**plot_layout(
                title="CSAT Score vs Resolution Time", height=300,
                          xaxis_title="Resolution Time (hrs)", yaxis_title="CSAT Score"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    support["month"] = support["created_date"].dt.to_period("M").astype(str)
    monthly_tickets = support.groupby(["month","status"]).size().reset_index(name="tickets")
    fig = px.bar(monthly_tickets, x="month", y="tickets", color="status",
                 color_discrete_sequence=PALETTE, barmode="stack")
    fig.update_layout(**plot_layout(
                title="Ticket Volume Over Time", height=280,
                      xaxis_title=None, yaxis_title="Tickets",
                      xaxis=dict(tickangle=-35, gridcolor=GRID_CLR)))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EXTERNAL FACTORS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 External Factors":
    st.markdown('<p class="section-title">External Factor Analytics</p>', unsafe_allow_html=True)
    st.caption("Weather, CPI, trend index and holiday flags — linked per store per week.")

    # Sample for speed
    ef = ext_factor.sample(min(3000, len(ext_factor)), random_state=42)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(ef, x="temp_c", y="trend_index",
                         color="is_holiday",
                         color_discrete_map={True: ACCENT, False: "#3a3a55"},
                         opacity=0.55, size_max=6)
        fig.update_layout(**plot_layout(
                title="Temperature vs Trend Index", height=300,
                          xaxis_title="Temperature (°C)", yaxis_title="Trend Index"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(ext_factor, x="year", y="cpi_index",
                        color="year", color_discrete_sequence=PALETTE,
                        box=True, points=False)
        fig.update_layout(**plot_layout(
                title="CPI Index Distribution by Year", height=300, showlegend=False,
                          xaxis_title="Year", yaxis_title="CPI Index"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(ef, x="rainfall_mm", nbins=40,
                           color_discrete_sequence=[ACCENT2])
        fig.update_layout(**plot_layout(
                title="Rainfall Distribution (mm)", height=260,
                          xaxis_title="Rainfall (mm)", yaxis_title="Frequency"))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        ef_order = ext_factor.merge(
            order_f[["store_id","year","week"]].groupby(["store_id","year","week"]).size().reset_index(name="orders"),
            on=["store_id","year","week"], how="left"
        ).dropna(subset=["orders"])
        hol_grp = ef_order.groupby("is_holiday")["orders"].mean().reset_index()
        hol_grp["is_holiday"] = hol_grp["is_holiday"].map({True:"Holiday",False:"Non-Holiday"})
        fig = px.bar(hol_grp, x="is_holiday", y="orders",
                     color="is_holiday",
                     color_discrete_map={"Holiday": ACCENT, "Non-Holiday": "#3a3a55"})
        fig.update_layout(**plot_layout(
                title="Avg Orders — Holiday vs Non-Holiday Weeks", height=260, showlegend=False,
                          xaxis_title=None, yaxis_title="Avg Orders / Week"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    cpi_wk = ext_factor.groupby(["year","week"])["cpi_index"].mean().reset_index()
    fig = px.line(cpi_wk, x="week", y="cpi_index", color="year",
                  color_discrete_sequence=PALETTE)
    fig.update_layout(**plot_layout(
                title="Weekly CPI Trend Across Years", height=260,
                      xaxis_title="Week of Year", yaxis_title="Avg CPI Index"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DATASET GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Dataset Guide":
    st.markdown('<p class="section-title">Dataset Guide & Project Ideas</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#8888b0;font-size:0.92rem;margin-bottom:20px;">
    A structured guide to help professionals and students pick the right analysis
    for their portfolio or resume project.
    </p>
    """, unsafe_allow_html=True)

    USE_CASES = [
        ("📈 SALES FORECASTING",    "Time-Series / ML",
         "Use ORDER + PRICE_CHANGE + EXTERNAL_FACTOR to build weekly demand forecasting models. Incorporate weather, CPI, and holiday flags as exogenous regressors in SARIMA, Prophet, or LSTM models."),
        ("🛒 CUSTOMER SEGMENTATION", "Clustering / RFM",
         "Build an RFM (Recency, Frequency, Monetary) model using ORDER data. Validate against the pre-labelled 'segment' field in CUSTOMER. Try K-Means, DBSCAN, or Gaussian Mixture Models."),
        ("📣 MARKETING MIX MODEL",   "Regression / Attribution",
         "Combine CAMPAIGN_TOUCH (impressions, clicks, conversions) with ORDER volume to compute channel-level ROI. Build a multi-touch attribution model or Bayesian MMM."),
        ("😊 CHURN PREDICTION",      "Classification / ML",
         "Use CUSTOMER status (active/churned), ORDER frequency, SUPPORT_TICKET CSAT scores and CAMPAIGN_TOUCH outcomes as features. Train XGBoost or Logistic Regression churn classifiers."),
        ("💬 SUPPORT ANALYSIS",      "NLP / Operations",
         "Analyse resolution_time_hr and csat_score by issue_type and status. Build an SLA breach predictor or CSAT regression model. Explore ticket seasonality."),
        ("💰 PRICE ELASTICITY",      "Econometrics",
         "Use PRICE_CHANGE weekly unit_price alongside ORDER volumes to estimate price elasticity by category/brand. Build OLS or log-log demand curves."),
        ("🗺️ GEO ANALYTICS",         "Spatial / BI",
         "Map orders by store_state/store_city using STORE + ORDER. Identify white-space markets. Overlay EXTERNAL_FACTOR weather data to analyse regional demand patterns."),
        ("🏷️ PROMO EFFECTIVENESS",   "A/B Testing / Stats",
         "Join PROMOTION with ORDER_ITEM to measure lift from PCT_OFF, BOGO, and FLAT promotions. Run difference-in-differences or uplift modelling."),
        ("🔄 MARKET BASKET ANALYSIS","Association Rules",
         "Use ORDER_ITEM to identify frequently co-purchased SKUs. Run Apriori or FP-Growth to generate association rules. Visualise product affinity networks."),
        ("⭐ CLV MODELLING",          "Probabilistic / ML",
         "Use ORDER history and CUSTOMER signup_date to estimate Customer Lifetime Value with BG/NBD + Gamma-Gamma models or survival analysis."),
    ]

    cols = st.columns(2)
    for i, (tag, method, desc) in enumerate(USE_CASES):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="usecase-card">
              <div class="usecase-tag">{method}</div>
              <div class="usecase-title">{tag}</div>
              <div class="usecase-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Table Relationships Quick Reference</p>', unsafe_allow_html=True)

    rel_df = pd.DataFrame([
        ["ORDER",          "CUSTOMER",       "customer_id",     "Many orders per customer"],
        ["ORDER",          "STORE",          "store_id",        "Many orders per store"],
        ["ORDER",          "CHANNEL",        "channel_id",      "Sales channel attribution"],
        ["ORDER",          "PRICE_CHANGE",   "year + week",     "Weekly price at time of order"],
        ["PRICE_CHANGE",   "PRODUCT",        "product_id",      "Price per SKU per week"],
        ["PRODUCT",        "CATEGORY",       "category_id",     "Product categorisation"],
        ["CUSTOMER",       "SUPPORT_TICKET", "customer_id",     "Support history per customer"],
        ["CUSTOMER",       "CAMPAIGN_TOUCH", "customer_id",     "Marketing touchpoints"],
        ["STORE",          "EXTERNAL_FACTOR","store_id",        "Weather/CPI per store per week"],
    ], columns=["From Table", "To Table", "Join Key", "Relationship"])

    st.dataframe(rel_df.style.set_properties(**{
        "background-color": "#1a1a26",
        "color": "#d0d0e8",
        "border": "1px solid #2a2a3a",
    }), use_container_width=True, height=300)

    st.markdown("---")
    st.markdown('<p class="section-title">Recommended Stack</p>', unsafe_allow_html=True)

    stack_cols = st.columns(4)
    stacks = [
        ("🐍 Python", ["pandas", "scikit-learn", "prophet", "lifetimes", "mlxtend"]),
        ("📊 Visualisation", ["plotly", "seaborn", "matplotlib", "folium"]),
        ("🤖 ML / Stats", ["xgboost", "statsmodels", "pymc", "shap"]),
        ("🏗️ Dashboard", ["streamlit", "dash", "gradio", "power bi"]),
    ]
    for col, (title, items) in zip(stack_cols, stacks):
        col.markdown(f"**{title}**")
        for item in items:
            col.markdown(f'<span class="pill">{item}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("📦 **Dataset on Kaggle** — Download all CSV files and this dashboard from the Kaggle dataset page. Place all CSVs in the same folder as `dashboard.py` and run `streamlit run dashboard.py`.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PRICE AUDIT  (dataset vs real world)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Price Audit":
    st.markdown('<p class="hero-title" style="font-size:2.4rem">Price Reality Check</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="hero-sub">
      How does the dataset's synthetic pricing compare to real Indian retail shelves?
      A systematic audit across brands, pack sizes, and price dynamics.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Build enriched price table ────────────────────────────────────────────
    pc_rich = (price_chg
               .merge(product, on="product_id")
               .merge(category, on="category_id"))

    # Week-over-week % change per product
    pc_sorted = pc_rich.sort_values(["product_id","year","week"]).copy()
    pc_sorted["prev_price"] = pc_sorted.groupby("product_id")["unit_price"].shift(1)
    pc_sorted["pct_chg"] = ((pc_sorted["unit_price"] - pc_sorted["prev_price"])
                             / pc_sorted["prev_price"] * 100)
    pc_sorted = pc_sorted.dropna(subset=["pct_chg"])

    # ── Section 1: Verdict summary cards ──────────────────────────────────────
    st.markdown('<p class="section-title">Verdict at a Glance</p>', unsafe_allow_html=True)

    verdicts = [
        ("🟢 Structurally Sound",  "Pack-size bulk discounts are internally consistent (10-pack ~8.2× single). Price recording frequency (~14 weeks avg) matches real quarterly FMCG review cycles.", "#4caf86"),
        ("🟡 Unit Ambiguity",      "Nescafe, Milo, Maggi 'Single' SKUs are priced as if they are mid-size jars/tins, not the Rs 2–14 sachets/bars sold at kirana stores. The 'unit' is undefined.", "#f8c842"),
        ("🔴 Magnitude Errors",    "Boost avg ₹4,069 vs real ~₹750 (5× high). NAN PRO ₹1,966 vs real ~₹830 (2.4× high). Munch single ₹35 vs real ₹5–20 (1.8–7×). These are synthetic scale artifacts.", "#e84393"),
        ("🔴 Nonsensical SKU Names","'Masala KitKat', 'Vanilla Pure Life Water', 'Chocolate Maggi' — flavour attributes are randomised from a shared pool. Real products don't have these variants.", "#ff8c42"),
        ("🟡 No Brand Correlation", "Prices of MILO, NESCAFE, NESTLE within Beverages have near-zero correlation (r ≈ 0.1). Real FMCG brands move together when input costs (coffee, cocoa) change.", "#8c5cf8"),
        ("🟡 Random Walk Volatility","Week-on-week swings of ±4% std dev per category. Real FMCG prices are sticky — a 5% change makes headlines. Dataset prices drift randomly rather than step-change.", "#4cbbe8"),
    ]

    col1, col2, col3 = st.columns(3)
    cols_cycle = [col1, col2, col3, col1, col2, col3]
    for (col, (title, desc, color)) in zip(cols_cycle, verdicts):
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{color};min-height:130px;margin-bottom:14px;">
          <div class="kpi-label" style="color:{color};font-size:0.78rem;">{title}</div>
          <div style="font-size:0.8rem;color:#9090b8;line-height:1.55;margin-top:6px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 2: Side-by-side comparison table ──────────────────────────────
    st.markdown('<p class="section-title">Brand-by-Brand: Dataset vs Real Shelf Price (₹)</p>', unsafe_allow_html=True)

    comparison_data = pd.DataFrame([
        # brand, sku_desc, real_low, real_high, ds_price, note
        ("KitKat",   "Single finger bar (13g–37g)",        30,   50,   35.2, "✅ Reasonable"),
        ("Munch",    "Smallest bar (5.5g–10g)",              5,   10,   35.4, "⚠️ 3–7× high — likely priced as large bar"),
        ("Munch",    "Large bar (35g)",                     20,   30,   35.4, "✅ Acceptable if large SKU"),
        ("Bar-One",  "Single bar (55g)",                    30,   45,   35.1, "✅ Close"),
        ("Maggi",    "Single 70g noodle pack",              14,   20,   44.8, "⚠️ 2–3× high"),
        ("Maggi",    "12-pack multipack",                  160,  200,  374.0, "⚠️ ~2× high for 10-pack equivalent"),
        ("Nescafe",  "Sachet 1.5g (kirana)",                 2,    5,  108.7, "❌ 22–54× high — wrong unit assumed"),
        ("Nescafe",  "Classic 50g jar",                    155,  175,  108.7, "✅ OK if 50g jar is the unit"),
        ("Nescafe",  "Classic 200g jar",                   490,  580,  108.7, "❌ 5× too low if 200g jar"),
        ("Milo",     "Single 400g tin",                    300,  360,  110.4, "❌ 3× too low"),
        ("Milo",     "Sachet 25g",                          20,   30,  110.4, "❌ 4× too high"),
        ("Pure Life","500ml bottle",                        15,   20,   24.9, "✅ OK (slight premium)"),
        ("Lactogen", "Stage 1 — 400g tin",                470,  520,  383.5, "✅ Reasonable"),
        ("Cerelac",  "Wheat 300g pouch",                   260,  310,  378.9, "✅ Slightly high but OK"),
        ("NAN PRO",  "Stage 1 — 400g tin",                800,  900, 1966.1, "⚠️ 2.4× high — possibly pricing 900g"),
        ("Peptamen", "400g clinical tin",                 1900, 2300, 1769.4, "✅ Within range"),
        ("Boost",    "1kg jar",                            700,  850, 4069.0, "❌ 5× high — likely mis-scaled"),
        ("Fitnesse", "Box of 5 sachets",                   150,  200,  510.9, "⚠️ 2.5× high"),
        ("Felix",    "Cat food pouch/tin",                 100,  180, 1196.3, "⚠️ Possibly 12-pack pricing"),
        ("Purina",   "Dog food 1.2kg bag",                 480,  600,  530.3, "✅ Plausible"),
    ], columns=["Brand","SKU Description","Real Low ₹","Real High ₹","Dataset ₹","Verdict"])

    comparison_data["Ratio (DS/Real Mid)"] = (
        comparison_data["Dataset ₹"] /
        ((comparison_data["Real Low ₹"] + comparison_data["Real High ₹"]) / 2)
    ).round(2)

    def color_verdict(val):
        if "✅" in val: return "color:#4caf86;font-weight:600"
        if "⚠️" in val: return "color:#f8c842;font-weight:600"
        if "❌" in val: return "color:#e84393;font-weight:600"
        return ""

    def color_ratio(val):
        if val < 0.6:  return "color:#e84393"
        if val > 2.0:  return "color:#e84393"
        if val > 1.4:  return "color:#f8c842"
        return "color:#4caf86"

    styled = (comparison_data.style
              .map(color_verdict, subset=["Verdict"])
              .map(color_ratio, subset=["Ratio (DS/Real Mid)"])
              .set_properties(**{"background-color":"#1a1a26","color":"#d0d0e8","border":"1px solid #2a2a3a"})
              .format({"Real Low ₹":"₹{:,.0f}","Real High ₹":"₹{:,.0f}",
                       "Dataset ₹":"₹{:,.1f}","Ratio (DS/Real Mid)":"{:.2f}×"}))
    st.dataframe(styled, use_container_width=True, height=560)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 3: Ratio waterfall chart ──────────────────────────────────────
    st.markdown('<p class="section-title">Price Deviation Ratio — Dataset ÷ Real World Midpoint</p>', unsafe_allow_html=True)
    st.caption("Green band = 0.6× to 1.4× (acceptable). Outside = synthetic artifact.")

    ratio_plot = comparison_data.drop_duplicates("Brand").copy()
    ratio_plot = ratio_plot.sort_values("Ratio (DS/Real Mid)", ascending=True)

    bar_colors = [
        "#4caf86" if 0.6 <= r <= 1.4
        else "#f8c842" if 1.4 < r <= 2.5
        else "#e84393"
        for r in ratio_plot["Ratio (DS/Real Mid)"]
    ]

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.6, x1=1.4,
                  y0=-0.5, y1=len(ratio_plot)-0.5,
                  fillcolor="rgba(76,175,134,0.08)",
                  line=dict(color="rgba(76,175,134,0.3)", width=1))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#4caf86", line_width=1.5)
    fig.add_trace(go.Bar(
        y=ratio_plot["Brand"],
        x=ratio_plot["Ratio (DS/Real Mid)"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{r:.2f}×" for r in ratio_plot["Ratio (DS/Real Mid)"]],
        textposition="outside",
        textfont=dict(color=TEXT_CLR, size=11),
    ))
    fig.update_layout(**plot_layout(
        title="Price Deviation: Dataset ÷ Real Mid-price (1.0× = perfect match)",
        height=340,
        xaxis=dict(ticksuffix="×", gridcolor=GRID_CLR, range=[0, 6.5]),
        xaxis_title="Ratio", yaxis_title=None,
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 4: Price volatility analysis ──────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-title">Weekly Price Volatility by Category</p>', unsafe_allow_html=True)
        st.caption("Real FMCG: ±0–1% per week. Dataset: ±3–5% std dev — random walk behavior.")

        vol_cat = (pc_sorted.groupby("category_name")["pct_chg"]
                   .agg(["std","mean"]).reset_index()
                   .rename(columns={"std":"volatility_pct","mean":"avg_drift_pct"})
                   .sort_values("volatility_pct", ascending=True))

        fig = go.Figure()
        # Real-world benchmark bar (invisible, for reference line)
        fig.add_vline(x=1.0, line_dash="dot",
                      line_color="#4caf86", line_width=1.5,
                      annotation_text="Real FMCG max ~1%",
                      annotation_font_color="#4caf86",
                      annotation_position="top right")
        fig.add_trace(go.Bar(
            y=vol_cat["category_name"],
            x=vol_cat["volatility_pct"],
            orientation="h",
            marker_color=ACCENT3,
            opacity=0.85,
            text=[f"{v:.1f}%" for v in vol_cat["volatility_pct"]],
            textposition="outside",
            textfont_color=TEXT_CLR,
        ))
        fig.update_layout(**plot_layout(
            title="Weekly Price Std Dev % by Category",
            height=340,
            xaxis=dict(ticksuffix="%", gridcolor=GRID_CLR),
            xaxis_title="Std Dev of Weekly % Change", yaxis_title=None,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Price Change Distribution</p>', unsafe_allow_html=True)
        st.caption("Real FMCG prices step-change sharply and rarely. Dataset prices drift continuously — a random walk signature.")

        fig = px.histogram(
            pc_sorted, x="pct_chg", nbins=60,
            color_discrete_sequence=[ACCENT2],
            marginal="box",
        )
        fig.add_vline(x=0, line_dash="dash", line_color=TEXT_CLR, line_width=1)
        fig.add_vrect(x0=-2, x1=2,
                      fillcolor="rgba(76,175,134,0.08)",
                      line_width=0,
                      annotation_text="Real-world zone",
                      annotation_font_color="#4caf86",
                      annotation_position="top left")
        fig.update_layout(**plot_layout(
            title="Distribution of All Week-on-Week Price Changes (%)",
            height=340,
            xaxis_title="Price Change (%)", yaxis_title="Frequency",
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 5: Intra-category brand correlation ───────────────────────────
    st.markdown('<p class="section-title">Brand Price Correlation Within Categories</p>', unsafe_allow_html=True)
    st.caption("In real markets, brands in the same category share input costs (coffee, milk, cocoa) → prices move together. Dataset shows near-zero correlations — prices are independently randomised.")

    corr_cats = ["Beverages (Coffee/Malt)", "Confectionery", "Dairy & Nutrition",
                 "Infant Nutrition", "Culinary (Soups/Seasonings)"]
    corr_cols = st.columns(len(corr_cats))

    for col_i, cat_name in zip(corr_cols, corr_cats):
        cat_df = pc_rich[pc_rich["category_name"] == cat_name]
        pivot = cat_df.pivot_table(
            index=["year","week"], columns="brand", values="unit_price", aggfunc="mean"
        )
        if pivot.shape[1] < 2:
            col_i.caption(f"{cat_name}: only 1 brand")
            continue
        corr = pivot.corr().round(2)
        fig = px.imshow(
            corr,
            color_continuous_scale=[[0,ACCENT3],[0.5,"#1a1a26"],[1,ACCENT]],
            zmin=-1, zmax=1,
            text_auto=True,
        )
        fig.update_coloraxes(showscale=False)
        short_name = cat_name.split("(")[0].strip().replace(" & "," &\n")
        fig.update_layout(**plot_layout(
            title=short_name,
            height=220,
            margin=dict(l=4, r=4, t=52, b=4),
            xaxis=dict(tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
        ))
        col_i.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 6: SKU naming absurdities ─────────────────────────────────────
    st.markdown('<p class="section-title">The SKU Name Problem</p>', unsafe_allow_html=True)

    absurd = pd.DataFrame([
        ("KitKat",    "KITKAT Confectionery - Masala (Large, Single)",
         "KitKat Masala doesn't exist. Real variants: Original, Dark, Ruby, Chunky."),
        ("Pure Life",  "PURE LIFE Bottled Water - Vanilla (Small, Pack of 4)",
         "Nestle Pure Life is unflavoured water. 'Vanilla water' is not a product."),
        ("Maggi",     "MAGGI Culinary - Chocolate (Family, Pack of 4)",
         "Maggi makes noodles, masalas, ketchup. A 'Chocolate' Maggi doesn't exist."),
        ("Munch",     "MUNCH Confectionery - Masala (Regular, Pack of 6)",
         "Munch is a chocolate wafer bar. 'Masala Munch' is not a product variant."),
        ("Nescafe",   "NESCAFE Beverages - Lite (Large, Pack of 10)",
         "Nescafe has Lite/Gold variants but 'Large' packs of 10 aren't standard SKUs."),
        ("Lactogen",  "LACTOGEN Infant Nutrition - Masala (Small, Pack of 10)",
         "Baby formula with 'Masala' flavour — completely fabricated product variant."),
    ])
    absurd.columns = ["Brand","Dataset SKU Name","What's Wrong"]

    st.dataframe(
        absurd.style.set_properties(**{
            "background-color":"#1a1a26","color":"#d0d0e8","border":"1px solid #2a2a3a"
        }),
        use_container_width=True, height=240
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 7: Summary + guidance ─────────────────────────────────────────
    st.markdown('<p class="section-title">What This Means For Your Project</p>', unsafe_allow_html=True)

    guidance = [
        ("✅ Use Price Data For",
         ["Relative price comparisons within a category", "Bulk discount & pack-size elasticity analysis",
          "Price change frequency & seasonality patterns", "Pricing tier segmentation (budget vs premium)"],
         "#4caf86"),
        ("⚠️ Use With Caution",
         ["Absolute price levels (off by 2–5× for several brands)",
          "Cross-category price comparisons (scale is inconsistent)",
          "Any analysis requiring real-world ₹ benchmarks"],
         "#f8c842"),
        ("❌ Avoid or Acknowledge",
         ["SKU-level product descriptions ('Masala KitKat' etc.)",
          "Revenue estimation using dataset prices × volume",
          "Claiming price-based insights reflect true Indian retail",
          "Brand correlation analysis (independently randomised)"],
         "#e84393"),
    ]

    g1, g2, g3 = st.columns(3)
    for gcol, (title, points, color) in zip([g1, g2, g3], guidance):
        gcol.markdown(f"""
        <div class="kpi-card" style="--accent:{color};padding:20px;">
          <div class="kpi-label" style="color:{color};font-size:0.8rem;margin-bottom:12px;">{title}</div>
          {"".join(f'<div style="font-size:0.8rem;color:#9090b8;padding:4px 0;border-bottom:1px solid #22223a;">{p}</div>' for p in points)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("""
    💡 **Bottom line:** The pricing data is **synthetic — generated, not scraped**.
    Absolute prices were set using a base-price random draw per SKU with a pack-size multiplier and week-to-week random drift (±4% std dev).
    The result is *structurally consistent* (bulk packs cost more, prices change gradually) but *not calibrated* to real Indian retail.
    For resume projects, add a disclaimer: *"Prices are synthetic and scaled for analytical demonstration only."*
    """)