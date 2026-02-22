"""
SYNTHETIC NESTLÉ-LIKE FMCG DATA GENERATOR (PK/FK-consistent)
===========================================================

Generates (per your config):
- customers: 10,000
- weeks: 156 (3 years)
- orders: ~350,000 (we calibrate to hit exactly unless you disable)
- order_items: ~1,100,000 (we calibrate to hit exactly)
- promo_rate: 0.45 (share of order_items with promo_id)
- churn_inactivity_weeks: 8
- overall_churn_3y: 0.28
- prechurn_drift_share: 0.70

Outputs CSVs:
- CATEGORY.csv
- PRODUCT.csv
- PROMOTION.csv
- STORE.csv
- CHANNEL.csv
- CUSTOMER.csv
- ORDER.csv
- ORDER_ITEM.csv

Notes:
- Uses an internal 10-category map aligned to your mix.
- Builds 200 products distributed by your mix (same as earlier).
- Calibrates order volume via a simple scaling loop.

Requirements:
pip install pandas numpy pyyaml
"""

import os
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None


# ----------------------------
# CONFIG LOADING
# ----------------------------
DEFAULT_CONFIG = {
    "customers": 10_000,
    "weeks": 156,
    "orders": 350_000,
    "order_items": 1_100_000,
    "promo_rate": 0.45,
    "churn_inactivity_weeks": 8,
    "overall_churn_3y": 0.28,
    "prechurn_drift_share": 0.70,
}

CONFIG_PATH = "/mnt/data/config.yaml"  # user uploaded
SEED = 42
OUTPUT_DIR = "synthetic_data"

TODAY = date(2026, 2, 22)  # per system date
# We'll generate the 156-week window ending at TODAY (inclusive-ish).
END_DATE = TODAY
START_DATE = END_DATE - timedelta(weeks=DEFAULT_CONFIG["weeks"])  # overridden after config load


def load_config(path: str) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path) and yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update({k: user_cfg[k] for k in user_cfg if k in cfg})
    return cfg


# ----------------------------
# CATEGORY / PRODUCT PLAN (your mix)
# ----------------------------
CATEGORY_ROWS = [
    ("CAT001", "Beverages (Coffee/Malt)", 0.18, 36),
    ("CAT002", "Dairy & Nutrition", 0.14, 28),
    ("CAT003", "Infant Nutrition", 0.06, 12),
    ("CAT004", "Confectionery", 0.12, 24),
    ("CAT005", "Culinary (Soups/Seasonings)", 0.10, 20),
    ("CAT006", "Breakfast Cereals", 0.08, 16),
    ("CAT007", "Petcare", 0.06, 12),
    ("CAT008", "Bottled Water", 0.10, 20),
    ("CAT009", "Health Science & Nutrition Add-ons", 0.06, 12),
    ("CAT010", "Others", 0.10, 20),
]
assert sum(r[3] for r in CATEGORY_ROWS) == 200

# Brand pools (Nestlé-like, not claiming exact regional availability)
BRANDS_BY_CAT = {
    "CAT001": ["NESCAFE", "MILO", "NESTLE"],
    "CAT002": ["MILKMAID", "NESPRAY", "NESTLE"],
    "CAT003": ["CERELAC", "NAN", "LACTOGEN"],
    "CAT004": ["KITKAT", "MUNCH", "BAR-ONE"],
    "CAT005": ["MAGGI"],
    "CAT006": ["KOKO KRUNCH", "NESQUIK", "CHEERIOS", "NESTUM", "FITNESSE", "HONEY STARS"],
    "CAT007": ["PURINA", "PRO PLAN", "FRISKIES", "FELIX"],
    "CAT008": ["PURE LIFE"],
    "CAT009": ["RESOURCE", "OPTIFAST", "PEPTAMEN", "BOOST"],
    "CAT010": ["NESTLE"],
}

# Category-based unit price ranges (INR), rough realism for synthetic purposes
PRICE_RANGES_INR = {
    "CAT001": (80, 650),
    "CAT002": (90, 800),
    "CAT003": (220, 1800),
    "CAT004": (10, 450),
    "CAT005": (12, 280),
    "CAT006": (120, 650),
    "CAT007": (180, 2500),
    "CAT008": (10, 300),
    "CAT009": (300, 2800),
    "CAT010": (50, 900),
}

# ----------------------------
# CUSTOMER (India-based)
# ----------------------------
STATE_CITY = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Delhi": ["New Delhi", "Dwarka", "Rohini", "Saket"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri", "Durgapur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Uttar Pradesh": ["Lucknow", "Noida", "Kanpur", "Varanasi"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur"],
    "Punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar"],
}

SEGMENTS = ["Value", "Mainstream", "Premium", "Health-focused"]
SEGMENT_WEIGHTS = [0.40, 0.35, 0.15, 0.10]
SEGMENT_MULTIPLIER = {
    "Value": 0.85,
    "Mainstream": 1.00,
    "Premium": 1.15,
    "Health-focused": 1.05,
}


# ----------------------------
# STORE / CHANNEL
# ----------------------------
STORE_TYPES = ["General Trade", "Modern Trade", "Supermarket", "Hypermarket", "Warehouse"]
CHANNELS = [("CH001", "Online"), ("CH002", "Mobile App"), ("CH003", "Modern Trade"), ("CH004", "General Trade")]

PAYMENT_TYPES = ["UPI", "Credit Card", "Debit Card", "Wallet", "Cash on Delivery"]


# ----------------------------
# PROMOTIONS
# ----------------------------
PROMO_TYPES = ["PCT_OFF", "FLAT_OFF", "BOGO"]
PROMO_CHANNEL_BIAS = {
    "Online": 1.10,
    "Mobile App": 1.15,
    "Modern Trade": 0.95,
    "General Trade": 0.90,
}


# ----------------------------
# HELPERS
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def daterange_days(start: date, end: date):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)


def random_date_between(d1: date, d2: date) -> date:
    # inclusive
    delta = (d2 - d1).days
    return d1 + timedelta(days=random.randint(0, max(delta, 0)))


def week_start_for(d: date, start: date) -> int:
    """Return integer week index since start_date (0..weeks-1)."""
    return (d - start).days // 7


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# GENERATORS
# ----------------------------
def gen_category() -> pd.DataFrame:
    return pd.DataFrame([{"category_id": cid, "category_name": name} for cid, name, _, _ in CATEGORY_ROWS])


def gen_product() -> pd.DataFrame:
    rows = []
    pid = 1
    for cid, cname, share, n in CATEGORY_ROWS:
        brands = BRANDS_BY_CAT[cid]
        lo, hi = PRICE_RANGES_INR[cid]
        for i in range(n):
            brand = random.choice(brands)
            # Simple SKU naming pattern (synthetic)
            size_tag = random.choice(["Small", "Regular", "Large", "Family"])
            variant = random.choice(["Classic", "Masala", "Vanilla", "Chocolate", "Original", "Lite", "Pro", "Plus"])
            pack = random.choice(["Single", "Pack of 2", "Pack of 4", "Pack of 6", "Pack of 10"])
            sku_name = f"{brand} {cname.split('(')[0].strip()} - {variant} ({size_tag}, {pack})"
            base_price = int(np.round(np.random.uniform(lo, hi)))
            rows.append(
                {
                    "product_id": f"PROD{pid:05d}",
                    "category_id": cid,
                    "brand": brand,
                    "sku_name": sku_name,
                    "base_price_inr": base_price,  # not in your schema; useful for generation
                }
            )
            pid += 1
    df = pd.DataFrame(rows)
    # Keep only schema fields for PRODUCT table; retain price map separately
    return df


def gen_store(n_stores: int = 120) -> pd.DataFrame:
    states = list(STATE_CITY.keys())
    rows = []
    for i in range(1, n_stores + 1):
        st = random.choice(states)
        city = random.choice(STATE_CITY[st])
        stype = random.choice(STORE_TYPES)
        rows.append(
            {
                "store_id": f"ST{i:04d}",
                "store_city": city,
                "store_state": st,
                "store_type": stype,
            }
        )
    return pd.DataFrame(rows)


def gen_channel() -> pd.DataFrame:
    return pd.DataFrame([{"channel_id": cid, "channel_name": name} for cid, name in CHANNELS])


def gen_promotion(start_date: date, end_date: date, n_promos: int = 260) -> pd.DataFrame:
    """
    Make promos active across the 3-year window.
    We'll create overlapping promos with varying durations.
    """
    rows = []
    for i in range(1, n_promos + 1):
        ptype = random.choices(PROMO_TYPES, weights=[0.70, 0.25, 0.05], k=1)[0]
        dur_days = random.choice([7, 10, 14, 21, 28, 35, 42])
        s = random_date_between(start_date, end_date - timedelta(days=dur_days))
        e = s + timedelta(days=dur_days)

        # discount_value meaning depends on type:
        # PCT_OFF -> percent (5-40)
        # FLAT_OFF -> INR (5-200)
        # BOGO -> treat as pct-ish equivalent later (e.g., 50% effective cap)
        if ptype == "PCT_OFF":
            disc = float(random.choice([5, 10, 12, 15, 18, 20, 25, 30, 35, 40]))
        elif ptype == "FLAT_OFF":
            disc = float(random.choice([5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]))
        else:
            disc = 50.0

        rows.append(
            {
                "promo_id": f"PR{i:05d}",
                "promo_type": ptype,
                "start_date": s.isoformat(),
                "end_date": e.isoformat(),
                "discount_value": disc,
            }
        )
    return pd.DataFrame(rows)


def gen_customer(n_customers: int, start_date: date, end_date: date) -> pd.DataFrame:
    states = list(STATE_CITY.keys())
    rows = []
    for i in range(1, n_customers + 1):
        st = random.choice(states)
        city = random.choice(STATE_CITY[st])
        seg = random.choices(SEGMENTS, weights=SEGMENT_WEIGHTS, k=1)[0]
        signup = random_date_between(start_date, end_date)
        rows.append(
            {
                "customer_id": f"CUST{i:06d}",
                "signup_date": signup.isoformat(),
                "city": city,
                "state": st,
                "segment": seg,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ChurnPlan:
    is_churner: bool
    churn_week: Optional[int]          # last active week index (exclusive-ish)
    drift_start_week: Optional[int]    # week index when drift begins
    drift_multiplier: float            # rate multiplier during drift


def build_churn_plans(
    customers: pd.DataFrame,
    weeks: int,
    churn_rate: float,
    churn_inactivity_weeks: int,
    prechurn_drift_share: float,
) -> Dict[str, ChurnPlan]:
    """
    Define churn week for churners so that they have >= churn_inactivity_weeks inactivity at end.
    drift_start_week applies to some churners (prechurn_drift_share).
    """
    n = len(customers)
    n_churn = int(round(n * churn_rate))

    ids = customers["customer_id"].tolist()
    churners = set(random.sample(ids, n_churn))

    plans = {}
    for cid in ids:
        if cid not in churners:
            plans[cid] = ChurnPlan(False, None, None, 1.0)
            continue

        # churn week must allow inactivity window at end: churn_week <= weeks - churn_inactivity_weeks
        churn_week = random.randint(0, max(0, weeks - churn_inactivity_weeks))
        has_drift = random.random() < prechurn_drift_share

        if has_drift:
            drift_len = random.choice([4, 6, 8, 10, 12])  # weeks before churn where behavior drifts
            drift_start = max(0, churn_week - drift_len)
            drift_mult = random.choice([0.35, 0.45, 0.55, 0.65])
            plans[cid] = ChurnPlan(True, churn_week, drift_start, drift_mult)
        else:
            plans[cid] = ChurnPlan(True, churn_week, None, 1.0)

    return plans


def expected_weekly_rate_for_customer(segment: str) -> float:
    """
    Base orders/week.
    Mean over population should roughly match 350k orders / (10k*156) ~ 0.224 orders/week.
    We'll draw from lognormal then scale globally.
    """
    # lognormal parameters chosen for long-tail behavior
    base = np.random.lognormal(mean=-1.6, sigma=0.8)  # ~0.20-ish mean pre-scale
    return float(base * SEGMENT_MULTIPLIER.get(segment, 1.0))


def calibrate_rates_to_target_orders(
    customers: pd.DataFrame,
    churn_plans: Dict[str, ChurnPlan],
    weeks: int,
    target_orders: int,
    max_iters: int = 12,
) -> Tuple[np.ndarray, float]:
    """
    For each customer, create weekly rate vector sum => expected orders.
    Then scale all rates by a multiplier so total sampled orders ≈ target_orders.
    Returns:
      weekly_rates_matrix: shape (n_customers, weeks)
      final_scale
    """
    n = len(customers)
    weekly = np.zeros((n, weeks), dtype=np.float32)

    # base rate per customer
    base_rates = np.array([expected_weekly_rate_for_customer(s) for s in customers["segment"]], dtype=np.float32)

    for idx, row in enumerate(customers.itertuples(index=False)):
        cid = row.customer_id
        plan = churn_plans[cid]
        for w in range(weeks):
            # Before signup -> no activity
            # (optional) could enforce signup week, but customers already have signup_date. We'll enforce.
            # We'll later compute signup_week and zero out prior weeks.
            weekly[idx, w] = base_rates[idx]

        # Apply churn/drift modifiers
        if plan.is_churner:
            # after churn_week -> no activity
            cw = plan.churn_week
            if cw is not None:
                weekly[idx, cw:] = 0.0
            # drift window
            if plan.drift_start_week is not None and plan.churn_week is not None:
                weekly[idx, plan.drift_start_week:plan.churn_week] *= plan.drift_multiplier

    # Enforce signup week: no orders before signup
    signup_weeks = np.array(
        [clamp(week_start_for(date.fromisoformat(d), START_DATE), 0, weeks - 1) for d in customers["signup_date"]],
        dtype=int,
    )
    for i in range(n):
        weekly[i, : signup_weeks[i]] = 0.0

    # Scale loop: sample orders and adjust
    scale = 1.0
    for _ in range(max_iters):
        lam = weekly * scale
        expected_total = float(lam.sum())
        if expected_total <= 1e-6:
            scale = 1.0
            break
        # scale toward target in expectation
        scale *= (target_orders / expected_total)
        # stop if close in expectation
        if abs(expected_total - target_orders) / target_orders < 0.01:
            break

    return weekly, float(scale)


def sample_orders_per_customer(weekly: np.ndarray, scale: float, target_orders: int) -> np.ndarray:
    """
    Sample total orders per customer from Poisson with mean = sum(weekly*scale).
    Then adjust to hit target_orders exactly by adding/subtracting orders to customers
    proportional to their means.
    """
    means = (weekly * scale).sum(axis=1)
    orders = np.random.poisson(lam=means).astype(int)

    diff = int(target_orders - orders.sum())
    if diff == 0:
        return orders

    # Probability weights for adjustment
    w = means + 1e-9
    w = w / w.sum()

    if diff > 0:
        # add orders: sample customers with replacement
        add_idx = np.random.choice(len(orders), size=diff, replace=True, p=w)
        np.add.at(orders, add_idx, 1)
    else:
        # remove orders: pick customers with orders>0, biased by w
        diff = -diff
        for _ in range(diff):
            eligible = np.where(orders > 0)[0]
            if len(eligible) == 0:
                break
            p = w[eligible]
            p = p / p.sum()
            i = np.random.choice(eligible, p=p)
            orders[i] -= 1

    return orders


def allocate_order_weeks_for_customer(week_rates: np.ndarray, n_orders: int) -> np.ndarray:
    """
    Allocate n_orders into weeks by sampling with probability proportional to week_rates.
    Returns array of week indices.
    """
    if n_orders <= 0:
        return np.array([], dtype=int)
    r = week_rates.copy()
    s = r.sum()
    if s <= 0:
        # fallback: uniform weeks
        return np.random.randint(0, len(r), size=n_orders)
    p = r / s
    return np.random.choice(len(r), size=n_orders, replace=True, p=p)


def gen_orders_and_items(
    customers: pd.DataFrame,
    stores: pd.DataFrame,
    channels: pd.DataFrame,
    products: pd.DataFrame,
    promos: pd.DataFrame,
    weeks: int,
    target_orders: int,
    target_items: int,
    promo_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build ORDER and ORDER_ITEM tables with:
    - PKs
    - FKs to CUSTOMER, STORE, CHANNEL, PRODUCT, PROMOTION
    - Computed totals
    """
    # Build churn plans
    churn_plans = build_churn_plans(
        customers=customers,
        weeks=weeks,
        churn_rate=cfg["overall_churn_3y"],
        churn_inactivity_weeks=cfg["churn_inactivity_weeks"],
        prechurn_drift_share=cfg["prechurn_drift_share"],
    )

    # Calibrate weekly rates
    weekly, scale = calibrate_rates_to_target_orders(customers, churn_plans, weeks, target_orders)
    orders_per_cust = sample_orders_per_customer(weekly, scale, target_orders)

    # Maps
    store_ids = stores["store_id"].tolist()
    channel_ids = channels["channel_id"].tolist()
    channel_map = dict(zip(channels["channel_id"], channels["channel_name"]))

    product_ids = products["product_id"].tolist()
    prod_price_map = dict(zip(products["product_id"], products["base_price_inr"]))  # synthetic helper

    # Promo lookup: which promos are active on a given date
    promos2 = promos.copy()
    promos2["start_date"] = pd.to_datetime(promos2["start_date"]).dt.date
    promos2["end_date"] = pd.to_datetime(promos2["end_date"]).dt.date
    promo_rows = promos2.to_dict("records")

    def active_promos(on_date: date) -> List[Dict]:
        # simple scan (OK for a few hundred promos)
        return [p for p in promo_rows if p["start_date"] <= on_date <= p["end_date"]]

    # Precompute week start dates
    week_start_dates = [START_DATE + timedelta(weeks=w) for w in range(weeks)]

    # Build ORDER rows
    order_rows = []
    order_item_rows = []

    # Items per order distribution (avg ~ 3.14)
    # We'll sample using shifted Poisson with cap, then adjust to target_items exactly.
    def sample_items_per_order(n_orders: int) -> np.ndarray:
        # shift+cap for realism
        x = np.random.poisson(lam=2.2, size=n_orders) + 1  # mean ~3.2
        x = np.clip(x, 1, 10)
        return x.astype(int)

    items_per_order = sample_items_per_order(target_orders)
    diff_items = int(target_items - items_per_order.sum())
    if diff_items != 0:
        # Adjust by adding/removing 1 item at a time, keeping bounds 1..10
        idxs = np.arange(target_orders)
        if diff_items > 0:
            while diff_items > 0:
                i = np.random.choice(idxs)
                if items_per_order[i] < 10:
                    items_per_order[i] += 1
                    diff_items -= 1
        else:
            diff_items = -diff_items
            while diff_items > 0:
                i = np.random.choice(idxs)
                if items_per_order[i] > 1:
                    items_per_order[i] -= 1
                    diff_items -= 1

    assert items_per_order.sum() == target_items, "Failed to match target order_items."

    # For faster customer iteration
    cust_ids = customers["customer_id"].tolist()
    signup_dates = pd.to_datetime(customers["signup_date"]).dt.date.values

    order_id_counter = 1
    order_item_id_counter = 1

    # Allocate order indices per customer
    order_customer_index = np.repeat(np.arange(len(customers)), orders_per_cust)
    assert len(order_customer_index) == target_orders

    # Randomize order sequence so later adjustments don’t bias early customers
    perm = np.random.permutation(target_orders)
    order_customer_index = order_customer_index[perm]
    items_per_order = items_per_order[perm]

    for o_idx in range(target_orders):
        c_index = int(order_customer_index[o_idx])
        cid = cust_ids[c_index]
        seg = customers.iloc[c_index]["segment"]

        # Sample which week this customer's order occurs in
        week_rates = weekly[c_index] * scale
        w = int(allocate_order_weeks_for_customer(week_rates, 1)[0])

        # Pick a random day inside that week, but not before signup date
        base_day = week_start_dates[w]
        order_day = base_day + timedelta(days=int(np.random.randint(0, 7)))
        if order_day < signup_dates[c_index]:
            order_day = signup_dates[c_index]

        # Channel & store selection (slight bias: online/app more for premium/health-focused)
        if seg in ("Premium", "Health-focused"):
            ch = random.choices(channel_ids, weights=[0.35, 0.35, 0.15, 0.15], k=1)[0]
        else:
            ch = random.choices(channel_ids, weights=[0.22, 0.20, 0.25, 0.33], k=1)[0]

        st = random.choice(store_ids)
        payment = random.choices(PAYMENT_TYPES, weights=[0.42, 0.18, 0.14, 0.16, 0.10], k=1)[0]

        # Build items
        n_items = int(items_per_order[o_idx])
        order_items = []
        active = active_promos(order_day)

        # Promo application at item level
        for _ in range(n_items):
            pid = random.choice(product_ids)
            unit_price = float(prod_price_map[pid])

            qty = int(np.random.choice([1, 1, 1, 2, 2, 3], p=[0.45, 0.15, 0.10, 0.18, 0.08, 0.04]))
            promo_id = None
            promo_type = None
            promo_val = 0.0

            # channel bias impacts promo probability slightly
            ch_name = channel_map[ch]
            pr = clamp(promo_rate * PROMO_CHANNEL_BIAS.get(ch_name, 1.0), 0.05, 0.85)

            if active and (random.random() < pr):
                p = random.choice(active)
                promo_id = p["promo_id"]
                promo_type = p["promo_type"]
                promo_val = float(p["discount_value"])

            gross = unit_price * qty
            item_discount = 0.0

            if promo_id is not None:
                if promo_type == "PCT_OFF":
                    item_discount = gross * (promo_val / 100.0)
                elif promo_type == "FLAT_OFF":
                    item_discount = min(gross, promo_val)  # flat discount per line, capped
                else:  # BOGO
                    # simple effective discount: ~50% on one unit if qty>=2 else 0
                    if qty >= 2:
                        item_discount = unit_price * 1.0  # one unit free
                    else:
                        item_discount = 0.0

            # extra small random markdown noise (returns, rounding, etc.)
            item_discount = float(max(0.0, item_discount + np.random.normal(0, gross * 0.01)))
            item_discount = float(min(gross, item_discount))

            line_total = gross - item_discount

            order_items.append(
                {
                    "order_item_id": f"OI{order_item_id_counter:08d}",
                    "order_id": f"ORD{order_id_counter:08d}",
                    "product_id": pid,
                    "promo_id": promo_id,
                    "quantity": qty,
                    "unit_price": round(unit_price, 2),
                    "item_discount": round(item_discount, 2),
                    "line_total": round(line_total, 2),
                }
            )
            order_item_id_counter += 1

        order_total = float(sum(x["line_total"] for x in order_items))
        discount_total = float(sum(x["item_discount"] for x in order_items))

        order_rows.append(
            {
                "order_id": f"ORD{order_id_counter:08d}",
                "customer_id": cid,
                "order_date": order_day.isoformat(),
                "store_id": st,
                "channel_id": ch,
                "order_total": round(order_total, 2),
                "discount_total": round(discount_total, 2),
                "payment_type": payment,
            }
        )
        order_item_rows.extend(order_items)
        order_id_counter += 1

    orders_df = pd.DataFrame(order_rows)
    items_df = pd.DataFrame(order_item_rows)

    # sanity checks
    assert len(orders_df) == target_orders
    assert len(items_df) == target_items
    assert orders_df["order_id"].is_unique
    assert items_df["order_item_id"].is_unique

    return orders_df, items_df

"""
ADD-ON TABLE GENERATOR (SESSION, SUPPORT_TICKET, CAMPAIGN_TOUCH, EXTERNAL_FACTOR)
================================================================================
This extends the earlier generator and produces these additional tables:

1) SESSION
   - Linked to CUSTOMER (FK customer_id)
   - Created primarily around orders + some non-purchase browsing
   - Realistic device/referrer, duration/pages behavior
   - Session_start is datetime

2) SUPPORT_TICKET
   - Linked to CUSTOMER (FK customer_id)
   - Ticket probability increases with: recent heavy purchases, high discounts, and churn drift window
   - Includes issue_type, status, resolution_time_hr, csat_score

3) CAMPAIGN_TOUCH
   - Linked to CUSTOMER (FK customer_id)
   - Generated weekly-ish; higher frequency for high value + near-churn reactivation
   - Includes channel, campaign_name, outcome

4) EXTERNAL_FACTOR
   - Linked to STORE (FK store_id)
   - One record per store per week across full time range
   - Includes holiday flag, temp, rainfall, trend_index, cpi_index

Outputs CSVs:
- SESSION.csv
- SUPPORT_TICKET.csv
- CAMPAIGN_TOUCH.csv
- EXTERNAL_FACTOR.csv

How to use:
- Paste these functions into your existing generate_data.py
- In main(), call generate_addon_tables(...) after ORDER + ORDER_ITEM are created,
  then write to CSV.

Optional tuning via addon_config.yaml keys (if present):
  sessions_per_order_mean: 3.5
  non_purchase_sessions_per_customer_mean: 18  # across full 156 weeks
  support_ticket_rate_per_customer_3y: 0.18    # expected tickets/customer in 3 years
  campaign_touches_per_customer_3y: 30
  external_factor_granularity: "weekly"        # only weekly supported here
"""

import random
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# ADDON DEFAULTS (override via cfg if provided)
# -----------------------------
ADDON_DEFAULTS = {
    "sessions_per_order_mean": 3.2,                 # ~3 sessions per order (browse -> cart -> pay)
    "sessions_per_order_cap": 8,
    "non_purchase_sessions_per_customer_mean": 14,  # extra browsing across 3 years (per customer)
    "support_ticket_rate_per_customer_3y": 0.16,     # expected tickets per customer in 3 years
    "campaign_touches_per_customer_3y": 28,          # expected touches per customer in 3 years
}


# -----------------------------
# Helpers
# -----------------------------
def _dt_at_day(d: date) -> datetime:
    # random time in day
    hh = int(np.random.randint(7, 23))
    mm = int(np.random.randint(0, 60))
    ss = int(np.random.randint(0, 60))
    return datetime(d.year, d.month, d.day, hh, mm, ss)

def _choice_weighted(items, weights):
    return items[int(np.random.choice(len(items), p=np.array(weights) / np.sum(weights)))]

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# -----------------------------
# 1) SESSION
# -----------------------------
def gen_sessions(
    cfg: Dict,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    churn_inactivity_weeks: int,
    end_date: date,
) -> pd.DataFrame:
    """
    Strategy:
    - For each order, generate k sessions around order_date (same day ± 2 days)
    - Add extra non-purchase sessions per customer across the full period
    """
    s_mean = float(cfg.get("sessions_per_order_mean", ADDON_DEFAULTS["sessions_per_order_mean"]))
    s_cap = int(cfg.get("sessions_per_order_cap", ADDON_DEFAULTS["sessions_per_order_cap"]))
    extra_mean = float(cfg.get("non_purchase_sessions_per_customer_mean", ADDON_DEFAULTS["non_purchase_sessions_per_customer_mean"]))

    # Device/referrer distributions
    DEVICES = ["Mobile", "Desktop", "Tablet"]
    DEVICE_W = [0.72, 0.24, 0.04]

    REFERRERS = ["Direct", "Google Search", "Instagram", "YouTube", "Email", "Push", "Affiliate", "Marketplace"]
    REF_W = [0.20, 0.30, 0.10, 0.07, 0.10, 0.12, 0.06, 0.05]

    # Precompute churn window start date for "inactive" tag effect (optional)
    inactive_cutoff = end_date - timedelta(weeks=int(churn_inactivity_weeks))

    orders2 = orders.copy()
    orders2["order_date"] = pd.to_datetime(orders2["order_date"]).dt.date

    # --- sessions from orders ---
    n_orders = len(orders2)
    # sessions per order ~ 1 + Poisson(s_mean-1)
    lam = max(0.1, s_mean - 1.0)
    k = (np.random.poisson(lam=lam, size=n_orders) + 1).astype(int)
    k = np.clip(k, 1, s_cap)

    # Expand indices
    order_idx = np.repeat(np.arange(n_orders), k)
    base = orders2.iloc[order_idx][["customer_id", "order_date", "channel_id"]].reset_index(drop=True)

    # jitter day within +/-2 days around order
    jitter = np.random.randint(-2, 3, size=len(base))
    sess_dates = base["order_date"].values + pd.to_timedelta(jitter, unit="D")

    # Build columns
    device = np.random.choice(DEVICES, size=len(base), p=np.array(DEVICE_W))
    ref = np.random.choice(REFERRERS, size=len(base), p=np.array(REF_W))

    # Duration/pages: longer for desktop, shorter for mobile; add mild uplift for search/affiliate
    base_dur = np.random.lognormal(mean=5.1, sigma=0.55, size=len(base))  # seconds scale ~ 150-350 sec typical
    dur_mult = np.where(device == "Desktop", 1.15, np.where(device == "Tablet", 1.05, 1.0))
    dur_mult *= np.where((ref == "Google Search") | (ref == "Affiliate"), 1.10, 1.0)
    duration_sec = np.clip((base_dur * dur_mult).astype(int), 20, 3600)

    pages = np.random.poisson(lam=5.2, size=len(base)) + 1
    pages = np.clip(pages + np.where(device == "Desktop", 2, 0), 1, 60).astype(int)

    # session_start datetime
    session_start = [ _dt_at_day(d.date()) for d in pd.to_datetime(sess_dates) ]

    sessions_order = pd.DataFrame({
        "session_id": [f"SE{idx:09d}" for idx in range(1, len(base) + 1)],
        "customer_id": base["customer_id"].values,
        "session_start": session_start,
        "session_duration_sec": duration_sec,
        "pages_viewed": pages,
        "device": device,
        "referrer": ref,
    })

    # --- extra non-purchase sessions ---
    cust_ids = customers["customer_id"].values
    # total extra sessions
    extra_counts = np.random.poisson(lam=max(0.1, extra_mean), size=len(cust_ids)).astype(int)
    extra_total = int(extra_counts.sum())

    if extra_total > 0:
        extra_cust = np.repeat(cust_ids, extra_counts)

        # pick random dates across full window for extra sessions
        # use orders date range for realism
        min_day = orders2["order_date"].min()
        max_day = orders2["order_date"].max()
        day_span = (max_day - min_day).days
        rand_days = np.random.randint(0, max(1, day_span + 1), size=extra_total)
        extra_dates = np.array([min_day + timedelta(days=int(x)) for x in rand_days], dtype=object)

        # slightly fewer sessions near "inactive" tail window (simulate churners)
        # If date > inactive_cutoff, downweight by dropping some
        keep = np.ones(extra_total, dtype=bool)
        late_mask = extra_dates >= inactive_cutoff
        drop_prob = 0.12
        keep[late_mask] = (np.random.rand(late_mask.sum()) > drop_prob)
        extra_cust = extra_cust[keep]
        extra_dates = extra_dates[keep]
        extra_total = len(extra_cust)

        device2 = np.random.choice(DEVICES, size=extra_total, p=np.array(DEVICE_W))
        ref2 = np.random.choice(REFERRERS, size=extra_total, p=np.array(REF_W))

        base_dur2 = np.random.lognormal(mean=4.9, sigma=0.60, size=extra_total)
        duration2 = np.clip((base_dur2 * np.where(device2 == "Desktop", 1.12, 1.0)).astype(int), 15, 3000)
        pages2 = np.clip((np.random.poisson(lam=4.5, size=extra_total) + 1).astype(int), 1, 50)

        session_start2 = [ _dt_at_day(d) for d in extra_dates ]

        sessions_extra = pd.DataFrame({
            "session_id": [f"SE{idx:09d}" for idx in range(len(sessions_order) + 1, len(sessions_order) + extra_total + 1)],
            "customer_id": extra_cust,
            "session_start": session_start2,
            "session_duration_sec": duration2,
            "pages_viewed": pages2,
            "device": device2,
            "referrer": ref2,
        })

        sessions = pd.concat([sessions_order, sessions_extra], ignore_index=True)
    else:
        sessions = sessions_order

    return sessions


# -----------------------------
# 2) SUPPORT_TICKET
# -----------------------------
def gen_support_tickets(
    cfg: Dict,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    churn_inactivity_weeks: int,
    end_date: date,
) -> pd.DataFrame:
    """
    Tickets are sparse; tie likelihood to:
    - higher discount usage
    - recent purchase frequency
    - inactivity/churn tail (reactivation/complaints)
    """
    rate = float(cfg.get("support_ticket_rate_per_customer_3y", ADDON_DEFAULTS["support_ticket_rate_per_customer_3y"]))
    target_tickets = int(round(len(customers) * rate))

    ISSUE_TYPES = ["Delivery Delay", "Damaged Item", "Wrong Item", "Refund/Return", "Payment Issue", "Coupon/Promo Issue", "Quality Concern"]
    ISSUE_W = [0.22, 0.12, 0.08, 0.20, 0.12, 0.16, 0.10]

    orders2 = orders.copy()
    orders2["order_date"] = pd.to_datetime(orders2["order_date"]).dt.date

    # Customer weights based on discount rate + order frequency
    cust_order_stats = orders2.groupby("customer_id").agg(
        orders=("order_id", "count"),
        avg_discount=("discount_total", "mean"),
        avg_total=("order_total", "mean"),
        last_order=("order_date", "max"),
    ).reset_index()

    cust_order_stats["disc_ratio"] = cust_order_stats["avg_discount"] / (cust_order_stats["avg_total"] + 1e-6)
    cust_order_stats["recency_days"] = (end_date - cust_order_stats["last_order"]).apply(lambda x: x.days)
    cust_order_stats["near_inactive"] = (cust_order_stats["recency_days"] >= (churn_inactivity_weeks * 7)).astype(int)

    # Weight: more orders + more discount problems + near-inactive
    w = (
        0.50 * np.log1p(cust_order_stats["orders"].values) +
        0.35 * np.sqrt(np.clip(cust_order_stats["disc_ratio"].values, 0, 1)) +
        0.25 * cust_order_stats["near_inactive"].values
    )
    w = np.maximum(w, 1e-6)
    w = w / w.sum()

    ticket_customers = np.random.choice(cust_order_stats["customer_id"].values, size=target_tickets, replace=True, p=w)

    # Ticket date: around last order date or random
    last_map = dict(zip(cust_order_stats["customer_id"], cust_order_stats["last_order"]))
    created_dates = []
    for cid in ticket_customers:
        base = last_map.get(cid, orders2["order_date"].min())
        # ticket around base date ± 10 days; sometimes later
        jitter = int(np.random.randint(-10, 21))
        d = base + timedelta(days=jitter)
        # clamp to overall range
        if d < orders2["order_date"].min():
            d = orders2["order_date"].min()
        if d > end_date:
            d = end_date
        created_dates.append(d)

    # Resolution time depends on issue
    # faster for coupon/payment; slower for refund/damaged
    res_time = []
    status = []
    csat = []
    for i in range(target_tickets):
        itype = _choice_weighted(ISSUE_TYPES, ISSUE_W)
        if itype in ("Payment Issue", "Coupon/Promo Issue"):
            rt = int(np.clip(np.random.lognormal(2.4, 0.6), 1, 72))   # 1-72h
        elif itype in ("Refund/Return", "Damaged Item"):
            rt = int(np.clip(np.random.lognormal(2.9, 0.7), 4, 240))  # 4-240h
        else:
            rt = int(np.clip(np.random.lognormal(2.6, 0.65), 2, 168)) # 2-168h
        res_time.append(rt)

        # Mostly resolved
        st = _choice_weighted(["Resolved", "Open", "Escalated"], [0.86, 0.10, 0.04])
        status.append(st)

        # CSAT inversely related to resolution time; open tickets have NA/low
        if st == "Open":
            cs = int(np.random.choice([1, 2, 3], p=[0.45, 0.35, 0.20]))
        else:
            score = 5 - (rt / 80.0) + np.random.normal(0, 0.6)
            cs = int(np.clip(round(score), 1, 5))
        csat.append(cs)

    # assign issue types now (vector)
    issue_types = np.random.choice(ISSUE_TYPES, size=target_tickets, p=np.array(ISSUE_W)/np.sum(ISSUE_W))

    tickets = pd.DataFrame({
        "ticket_id": [f"T{idx:08d}" for idx in range(1, target_tickets + 1)],
        "customer_id": ticket_customers,
        "created_date": [d.isoformat() for d in created_dates],
        "issue_type": issue_types,
        "status": status,
        "resolution_time_hr": res_time,
        "csat_score": csat,
    })
    return tickets


# -----------------------------
# 3) CAMPAIGN_TOUCH
# -----------------------------
def gen_campaign_touches(
    cfg: Dict,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    churn_inactivity_weeks: int,
    end_date: date,
) -> pd.DataFrame:
    """
    Touches:
    - Base touches per customer across 3y
    - Higher touches for Premium/Health-focused
    - Reactivation touches for near-inactive customers
    """
    touches_per_customer = float(cfg.get("campaign_touches_per_customer_3y", ADDON_DEFAULTS["campaign_touches_per_customer_3y"]))
    expected_total = int(round(len(customers) * touches_per_customer))

    TOUCH_CHANNELS = ["Email", "SMS", "Push", "Social Ads", "Search Ads", "WhatsApp"]
    CH_W = [0.28, 0.18, 0.22, 0.15, 0.10, 0.07]

    CAMPAIGNS = [
        "New Launch Promo", "Weekend Deal", "Cart Abandon Reminder", "Loyalty Bonus",
        "Health Bundle Offer", "Reactivation Offer", "Seasonal Sale", "Subscribe & Save"
    ]
    # outcomes depend on channel/segment
    OUTCOMES = ["Ignored", "Opened", "Clicked", "Converted"]
    base_out_w = np.array([0.55, 0.25, 0.14, 0.06])

    orders2 = orders.copy()
    orders2["order_date"] = pd.to_datetime(orders2["order_date"]).dt.date

    # customer recency for reactivation targeting
    last_order = orders2.groupby("customer_id")["order_date"].max().reset_index()
    last_map = dict(zip(last_order["customer_id"], last_order["order_date"]))

    min_day = orders2["order_date"].min()
    max_day = orders2["order_date"].max()

    # allocate touches per customer using Poisson around mean
    seg = customers["segment"].values
    seg_mult = np.where(seg == "Premium", 1.25, np.where(seg == "Health-focused", 1.18, 1.0))
    lam = touches_per_customer * seg_mult
    touches_c = np.random.poisson(lam=lam).astype(int)

    # Add extra reactivation touches to near-inactive customers
    inactive_cutoff = end_date - timedelta(weeks=int(churn_inactivity_weeks))
    cust_ids = customers["customer_id"].values
    last = np.array([last_map.get(cid, min_day) for cid in cust_ids], dtype=object)
    near_inactive = (last < inactive_cutoff)
    touches_c = touches_c + near_inactive.astype(int) * np.random.poisson(lam=2.0, size=len(cust_ids)).astype(int)

    # Now rescale total to expected_total (approx) by global multiplier
    current_total = touches_c.sum()
    if current_total > 0:
        scale = expected_total / current_total
        touches_c = np.maximum(0, np.round(touches_c * scale).astype(int))

    total = int(touches_c.sum())
    if total == 0:
        return pd.DataFrame(columns=["touch_id","customer_id","touch_date","channel","campaign_name","outcome"])

    touch_customers = np.repeat(cust_ids, touches_c)

    # Touch dates: uniform across range with a small bias to sales seasons (synthetic)
    # We'll bias toward weeks 10-14 and 45-49 (festival-ish), and 0-4 (new year)
    span = (max_day - min_day).days
    u = np.random.rand(total)
    # mixture sampling
    touch_dates = []
    for i in range(total):
        r = u[i]
        if r < 0.20:
            # festival window A
            d = min_day + timedelta(days=int(np.random.randint(10*7, 14*7)))
        elif r < 0.35:
            # festival window B
            d = min_day + timedelta(days=int(np.random.randint(45*7, 49*7)))
        elif r < 0.45:
            # early window
            d = min_day + timedelta(days=int(np.random.randint(0, 4*7)))
        else:
            d = min_day + timedelta(days=int(np.random.randint(0, max(1, span+1))))
        if d > end_date:
            d = end_date
        touch_dates.append(d)

    ch = np.random.choice(TOUCH_CHANNELS, size=total, p=np.array(CH_W)/np.sum(CH_W))
    camp = np.random.choice(CAMPAIGNS, size=total)

    # outcome weights: reactivation + push -> higher click/convert
    out = []
    for i in range(total):
        w = base_out_w.copy()
        if ch[i] in ("Push", "WhatsApp"):
            w *= np.array([0.92, 1.02, 1.15, 1.30])
        if camp[i] == "Reactivation Offer":
            w *= np.array([0.88, 1.05, 1.15, 1.35])
        if camp[i] == "Cart Abandon Reminder":
            w *= np.array([0.85, 1.10, 1.20, 1.30])
        w = w / w.sum()
        out.append(np.random.choice(OUTCOMES, p=w))

    touches = pd.DataFrame({
        "touch_id": [f"CT{idx:09d}" for idx in range(1, total + 1)],
        "customer_id": touch_customers,
        "touch_date": [d.isoformat() for d in touch_dates],
        "channel": ch,
        "campaign_name": camp,
        "outcome": out,
    })
    return touches


# -----------------------------
# 4) EXTERNAL_FACTOR (weekly per store)
# -----------------------------
def gen_external_factors(
    stores: pd.DataFrame,
    start_date: date,
    weeks: int,
) -> pd.DataFrame:
    """
    One row per store per week:
      - is_holiday: synthetic holiday flag (festival-ish peaks)
      - temp_c: seasonal sinusoid + regional baseline + noise
      - rainfall_mm: monsoon-ish seasonality + noise
      - trend_index: slow trend with noise
      - cpi_index: slow inflation drift with noise
    """
    store_ids = stores["store_id"].values
    n_stores = len(store_ids)

    # Week start dates
    week_dates = np.array([start_date + timedelta(weeks=w) for w in range(weeks)], dtype=object)

    # Regional baselines by state (rough)
    state = stores["store_state"].values
    temp_base = []
    rain_base = []
    for st in state:
        if st in ("Delhi", "Punjab", "Haryana", "Rajasthan", "Uttar Pradesh", "Bihar"):
            temp_base.append(27.0); rain_base.append(3.0)
        elif st in ("Kerala", "Tamil Nadu", "Karnataka", "Telangana"):
            temp_base.append(28.5); rain_base.append(6.0)
        elif st in ("Maharashtra", "Gujarat", "Madhya Pradesh"):
            temp_base.append(28.0); rain_base.append(4.0)
        elif st in ("West Bengal", "Assam", "Odisha"):
            temp_base.append(27.5); rain_base.append(7.0)
        else:
            temp_base.append(27.5); rain_base.append(4.5)
    temp_base = np.array(temp_base, dtype=float)
    rain_base = np.array(rain_base, dtype=float)

    # Create grid: store x week
    store_rep = np.repeat(store_ids, weeks)
    week_rep = np.tile(week_dates, n_stores)

    # seasonal temp: peak ~May/Jun, trough ~Dec/Jan
    t = np.tile(np.arange(weeks), n_stores)
    temp_season = 6.0 * np.sin(2*np.pi*(t/52.0) - 1.0)  # shift phase
    # rainfall season: monsoon-ish peak mid-year
    rain_season = 12.0 * np.maximum(0, np.sin(2*np.pi*(t/52.0) - 2.2))

    # Holiday: higher probability around certain weeks
    # (synthetic: weeks 8-12, 40-46, 50-52)
    holiday_prob = np.full_like(t, 0.06, dtype=float)
    holiday_prob += ((t % 52 >= 8) & (t % 52 <= 12)) * 0.10
    holiday_prob += ((t % 52 >= 40) & (t % 52 <= 46)) * 0.14
    holiday_prob += ((t % 52 >= 50) | (t % 52 <= 1)) * 0.10
    is_holiday = (np.random.rand(len(t)) < holiday_prob).astype(bool)

    # Apply store baselines
    temp = np.repeat(temp_base, weeks) + temp_season + np.random.normal(0, 1.5, size=len(t))
    temp = np.clip(temp, 8.0, 44.0)

    rain = np.repeat(rain_base, weeks) + rain_season + np.random.gamma(shape=1.3, scale=2.0, size=len(t))
    rain = np.clip(rain, 0.0, 120.0)

    # trend_index: slow trend upward with noise, slight holiday uplift
    trend = 100 + (t * (2.2/52.0)) + np.random.normal(0, 1.2, size=len(t)) + is_holiday.astype(int) * 1.5

    # cpi_index: slow inflation drift upward
    cpi = 100 + (t * (5.0/52.0)) + np.random.normal(0, 0.8, size=len(t))

    factors = pd.DataFrame({
        "factor_id": [f"F{idx:09d}" for idx in range(1, len(t) + 1)],
        "store_id": store_rep,
        "factor_date": [d.isoformat() for d in week_rep],
        "is_holiday": is_holiday,
        "temp_c": np.round(temp, 2),
        "rainfall_mm": np.round(rain, 2),
        "trend_index": np.round(trend, 2),
        "cpi_index": np.round(cpi, 2),
    })
    return factors


# -----------------------------
# MASTER: Generate all addon tables
# -----------------------------
def generate_addon_tables(
    cfg: Dict,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    stores: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    churn_inact = int(cfg["churn_inactivity_weeks"])
    weeks = int(cfg["weeks"])

    session_df = gen_sessions(cfg, customers, orders, churn_inact, end_date)
    ticket_df = gen_support_tickets(cfg, customers, orders, churn_inact, end_date)
    touch_df = gen_campaign_touches(cfg, customers, orders, churn_inact, end_date)
    factor_df = gen_external_factors(stores, start_date, weeks)

    return session_df, ticket_df, touch_df, factor_df


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    cfg = load_config(CONFIG_PATH)

    # Dates
    START_DATE = END_DATE - timedelta(weeks=int(cfg["weeks"]))
    # Customer signup window: 104–156 weeks ago relative to END_DATE, per your earlier rule
    signup_start = END_DATE - timedelta(weeks=156)
    signup_end = END_DATE - timedelta(weeks=104)

    ensure_dir(OUTPUT_DIR)

    # Build base dimensions
    category_df = gen_category()
    product_raw = gen_product()  # includes helper base_price_inr
    product_df = product_raw[["product_id", "category_id", "brand", "sku_name"]].copy()

    store_df = gen_store(n_stores=140)
    channel_df = gen_channel()
    promo_df = gen_promotion(START_DATE, END_DATE, n_promos=280)

    # Customers
    customer_df = gen_customer(int(cfg["customers"]), signup_start, signup_end)

    # Fact tables
    orders_df, order_items_df = gen_orders_and_items(
        customers=customer_df,
        stores=store_df,
        channels=channel_df,
        products=product_raw,   # include base_price_inr for generation
        promos=promo_df,
        weeks=int(cfg["weeks"]),
        target_orders=int(cfg["orders"]),
        target_items=int(cfg["order_items"]),
        promo_rate=float(cfg["promo_rate"]),
    )
    # After you have: customer_df, orders_df, store_df, start_date, END_DATE, cfg

    # Merge addon defaults (only for missing keys)
    for k,v in ADDON_DEFAULTS.items():
        cfg.setdefault(k, v)

    session_df, ticket_df, touch_df, factor_df = generate_addon_tables(
    cfg=cfg,
    customers=customer_df,
    orders=orders_df,
    stores=store_df,
    start_date=START_DATE,
    end_date=END_DATE,
    )
    # Save
    category_df.to_csv(os.path.join(OUTPUT_DIR, "CATEGORY.csv"), index=False)
    product_df.to_csv(os.path.join(OUTPUT_DIR, "PRODUCT.csv"), index=False)
    promo_df.to_csv(os.path.join(OUTPUT_DIR, "PROMOTION.csv"), index=False)
    store_df.to_csv(os.path.join(OUTPUT_DIR, "STORE.csv"), index=False)
    channel_df.to_csv(os.path.join(OUTPUT_DIR, "CHANNEL.csv"), index=False)
    customer_df.to_csv(os.path.join(OUTPUT_DIR, "CUSTOMER.csv"), index=False)
    orders_df.to_csv(os.path.join(OUTPUT_DIR, "ORDER.csv"), index=False)
    order_items_df.to_csv(os.path.join(OUTPUT_DIR, "ORDER_ITEM.csv"), index=False)
    session_df.to_csv(os.path.join(OUTPUT_DIR, "SESSION.csv"), index=False)
    ticket_df.to_csv(os.path.join(OUTPUT_DIR, "SUPPORT_TICKET.csv"), index=False)
    touch_df.to_csv(os.path.join(OUTPUT_DIR, "CAMPAIGN_TOUCH.csv"), index=False)
    factor_df.to_csv(os.path.join(OUTPUT_DIR, "EXTERNAL_FACTOR.csv"), index=False)
    # Quick report
    print("✅ Done. Files written to:", OUTPUT_DIR)
    print("Rows:")
    print("  CATEGORY     :", len(category_df))
    print("  PRODUCT      :", len(product_df))
    print("  PROMOTION    :", len(promo_df))
    print("  STORE        :", len(store_df))
    print("  CHANNEL      :", len(channel_df))
    print("  CUSTOMER     :", len(customer_df))
    print("  ORDER        :", len(orders_df))
    print("  ORDER_ITEM   :", len(order_items_df))

    # Promo realized rate
    realized_promo = order_items_df["promo_id"].notna().mean()
    print("Realized promo_rate (order_item-level):", round(realized_promo, 4))

    # Churn proxy check: customers with no orders in last churn_inactivity_weeks
    last_week_start = END_DATE - timedelta(weeks=int(cfg["churn_inactivity_weeks"]))
    last_week_start = pd.to_datetime(last_week_start).date()

    last_order = (
        orders_df.assign(order_date=pd.to_datetime(orders_df["order_date"]).dt.date)
        .groupby("customer_id")["order_date"]
        .max()
        .reset_index()
    )
    last_order["inactive_8w"] = last_order["order_date"] < last_week_start
    print("Inactive (>= churn_inactivity_weeks) share:", round(last_order["inactive_8w"].mean(), 4))
    print("ADDON ROWS:",
      "\\n  SESSION        ", len(session_df),
      "\\n  SUPPORT_TICKET ", len(ticket_df),
      "\\n  CAMPAIGN_TOUCH ", len(touch_df),
      "\\n  EXTERNAL_FACTOR", len(factor_df))

    # Quick FK sanity checks
    assert session_df["customer_id"].isin(customer_df["customer_id"]).all()
    assert ticket_df["customer_id"].isin(customer_df["customer_id"]).all()
    assert touch_df["customer_id"].isin(customer_df["customer_id"]).all()
    assert factor_df["store_id"].isin(store_df["store_id"]).all()