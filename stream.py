import math

import pandas as pd
import streamlit as st
import yfinance as yf

NAN = float("nan")

# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="Penman Stock Screener", page_icon="📊", layout="wide")
st.session_state.setdefault("screen_results", None)
st.session_state.setdefault("screen_assumptions", None)

st.title("📊 Residual-Income + DCF Stock Screener")
st.caption(
    "Educational research tool, not investment advice. Yahoo Finance accounting data can be "
    "missing or inconsistent. Valuations are estimates, not precise targets; inspect warnings, "
    "payout assumptions, model disagreement and terminal-value dependence before relying on a result."
)

# ============================================================
# HELPERS
# ============================================================


def safe_float(v):
    """Convert scalar financial data only when it is finite and numeric."""
    try:
        if v is None or pd.isna(v):
            return None
        value = float(v)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def clamp(v, lo, hi):
    return None if v is None else max(lo, min(hi, v))


def pick(df, names, col=0):
    if df is None or df.empty or col >= df.shape[1]:
        return None
    for n in names:
        if n in df.index:
            v = df.loc[n].iloc[col]
            if pd.notna(v):
                return safe_float(v)
    return None


def row_values(df, names):
    if df is None or df.empty:
        return []
    for n in names:
        if n in df.index:
            return [safe_float(v) for v in df.loc[n].values]
    return []


def total_debt_for_period(balance, col=0):
    total = pick(balance, ["Total Debt"], col)
    if total is not None:
        return total
    long_term = pick(balance, ["Long Term Debt And Capital Lease Obligation", "Long Term Debt"], col)
    current = pick(balance, ["Current Debt And Capital Lease Obligation", "Current Debt"], col)
    # Never silently omit current debt when reconstructing total debt.
    return long_term + current if long_term is not None and current is not None else None


def equity_row_source(df, col=0):
    if df is None or df.empty or col >= df.shape[1]:
        return None
    for name in EQUITY:
        if name in df.index and pd.notna(df.loc[name].iloc[col]):
            return name
    return None


def spend(v):
    v = safe_float(v)
    return None if v is None else abs(v)


def payout_ratio_from_cashflows(net_income, dividends_paid, buybacks_paid):
    """Estimate shareholder payout only when income, dividends and repurchases are all known."""
    ni = safe_float(net_income)
    dividends = spend(dividends_paid)
    buybacks = spend(buybacks_paid)
    if ni is None or ni <= 0 or dividends is None or buybacks is None:
        return None
    return (dividends + buybacks) / ni


def ratio(a, b):
    return None if a is None or b is None or b <= 0 else a / b


def fmt_pct(v):
    return "—" if v is None or pd.isna(v) else f"{v:.1%}"


def fmt_num(v):
    return "—" if v is None or pd.isna(v) else f"{v:.2f}"


# Current yfinance names plus legacy Yahoo row labels used by older responses.
EQUITY = [
    "Stockholders Equity", "Common Stock Equity", "Total Stockholder Equity",
    "Total Stockholders Equity", "Total Equity", "Total Equity Gross Minority Interest",
]
EBIT_NAMES = ["Operating Income", "EBIT"]


def _period_years(columns, newer_idx, older_idx):
    """Elapsed calendar years between two statement columns (Yahoo usually newest-first)."""
    try:
        newer = pd.Timestamp(columns[newer_idx])
        older = pd.Timestamp(columns[older_idx])
        span = abs((newer - older).days) / 365.2425
        return span if span > 0 else None
    except Exception:
        span = abs(older_idx - newer_idx)
        return float(span) if span > 0 else None


def _cagr_between(values, columns, positive_only=True):
    valid = [(i, safe_float(v)) for i, v in enumerate(values) if safe_float(v) is not None]
    if len(valid) < 2:
        return None
    # Sort by fiscal-period date when available; Yahoo usually returns newest-first,
    # but relying on column order alone can reverse the CAGR for other providers.
    try:
        valid.sort(key=lambda item: pd.Timestamp(columns[item[0]]), reverse=True)
    except Exception:
        valid.sort(key=lambda item: item[0])
    newer_i, newer_v = valid[0]
    older_i, older_v = valid[-1]
    # Do not skip a reported loss/zero at either endpoint to manufacture a positive CAGR.
    if positive_only and (newer_v <= 0 or older_v <= 0):
        return None
    if newer_v <= 0 or older_v <= 0:
        return None
    years = _period_years(columns, newer_i, older_i)
    if years is None:
        return None
    try:
        return (newer_v / older_v) ** (1 / years) - 1
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def compute_history(income, balance, cash_flow, price, shares):
    out = {"roe_hist": None, "earnings_cagr": None, "fcf_yield": None}
    earn = row_values(income, ["Net Income Common Stockholders", "Net Income", "Net Income From Continuing Operation Net Minority Interest", "Net Income From Continuing Operations"])
    if earn and balance is not None and not balance.empty:
        roes = []
        for i in range(min(5, len(earn), balance.shape[1] - 1)):
            end_equity = pick(balance, EQUITY, i)
            prior_equity = pick(balance, EQUITY, i + 1)
            if earn[i] is not None and end_equity is not None and prior_equity is not None:
                avg_equity = (end_equity + prior_equity) / 2
                if avg_equity > 0:
                    roes.append(earn[i] / avg_equity)
        if roes:
            out["roe_hist"] = float(pd.Series(roes).median())
        out["earnings_cagr"] = _cagr_between(earn, income.columns, positive_only=True)
    cfo = pick(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)
    capex = spend(pick(cash_flow, ["Capital Expenditure", "Capital Expenditures"], 0))
    if cfo is not None and capex is not None and price and shares and price > 0 and shares > 0:
        out["fcf_yield"] = (cfo - capex) / (price * shares)
    return out


# ============================================================
# DATA
# ============================================================


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol):
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        bs = t.balance_sheet
    except Exception:
        bs = pd.DataFrame()
    try:
        inc = t.financials
    except Exception:
        inc = pd.DataFrame()
    try:
        cf = t.cashflow
    except Exception:
        cf = pd.DataFrame()

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    price_source = "Yahoo current/regular market price" if price is not None else None
    if price is None:
        try:
            h = t.history(period="5d")
            if not h.empty and not h["Close"].dropna().empty:
                price = float(h["Close"].dropna().iloc[-1])
                price_source = "last available close (5-day history fallback)"
        except Exception:
            pass

    shares = safe_float(info.get("sharesOutstanding"))
    shares_source = "Yahoo current shares outstanding" if shares is not None and shares > 0 else None
    if shares is None or shares <= 0:
        shares = pick(bs, ["Ordinary Shares Number", "Share Issued"], 0)
        if shares is not None and shares > 0:
            shares_source = "balance-sheet share count fallback"
    if shares is None or shares <= 0:
        shares = pick(inc, ["Diluted Average Shares", "Weighted Average Shares Diluted", "Basic Average Shares"], 0)
        if shares is not None and shares > 0:
            shares_source = "historical average shares fallback"

    if price is None or bs is None or bs.empty or inc is None or inc.empty:
        raise ValueError("Yahoo data unavailable (possibly rate-limited); try again in a few minutes")
    if safe_float(price) is None or safe_float(price) <= 0:
        raise ValueError("invalid or non-positive market price")

    fin_cur, px_cur = info.get("financialCurrency"), info.get("currency")
    if fin_cur and px_cur and fin_cur != px_cur:
        raise ValueError(f"currency mismatch (financials {fin_cur}, price {px_cur}); skipped")

    years = []
    for col in (0, 1):
        years.append({
            "assets": pick(bs, ["Total Assets"], col),
            "liab": pick(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"], col),
            "debt": total_debt_for_period(bs, col),
            "cash": pick(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial"], col),
            "equity": pick(bs, EQUITY, col),
            "equity_source": equity_row_source(bs, col),
            "cur_assets": pick(bs, ["Total Current Assets", "Current Assets"], col),
            "cur_liab": pick(bs, ["Total Current Liabilities", "Current Liabilities"], col),
            "cur_debt": pick(bs, ["Current Debt And Capital Lease Obligation", "Current Debt"], col),
            "minority": pick(bs, ["Minority Interest"], col),
            "goodwill": pick(bs, ["Goodwill And Other Intangible Assets", "Goodwill"], col),
        })

    def op_nwc(y):
        # operating working capital: (current assets - cash) - (current liabilities - short-term debt)
        if any(y[k] is None for k in ("cur_assets", "cur_liab", "cash", "cur_debt")):
            return None
        return (y["cur_assets"] - y["cash"]) - (y["cur_liab"] - y["cur_debt"])

    revs = row_values(inc, ["Total Revenue", "Revenue"])
    ebits = row_values(inc, EBIT_NAMES)
    revenue = pick(inc, ["Total Revenue", "Revenue"], 0)
    ebit = pick(inc, EBIT_NAMES, 0)
    pretax = pick(inc, ["Pretax Income"], 0)
    tax = pick(inc, ["Tax Provision", "Income Tax Expense"], 0)

    margins = [e / r for e, r in zip(ebits, revs) if e is not None and r and r > 0][:3]
    avg_margin = sum(margins) / len(margins) if margins else None
    rev_cagr = _cagr_between(revs, inc.columns, positive_only=True)

    etr = None
    if pretax is not None and pretax > 0 and tax is not None:
        etr = max(0.0, min(0.45, tax / pretax))

    capex = spend(pick(cf, ["Capital Expenditure", "Capital Expenditures"], 0))
    dep = pick(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation", "Depreciation, Depletion And Amortization"], 0)
    hist = compute_history(inc, bs, cf, price, shares)
    yahoo_roe = safe_float(info.get("returnOnEquity"))
    roe_hist = hist["roe_hist"] if hist["roe_hist"] is not None else yahoo_roe
    roe_source = "historical average equity" if hist["roe_hist"] is not None else (
        "Yahoo-reported ROE fallback" if yahoo_roe is not None else None
    )

    ni_latest = pick(inc, ["Net Income Common Stockholders", "Net Income", "Net Income Including Noncontrolling Interests", "Net Income From Continuing Operation Net Minority Interest", "Net Income From Continuing Operations"], 0)
    dividends_latest = spend(pick(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"], 0))
    buybacks_latest = spend(pick(cf, ["Repurchase Of Capital Stock", "Common Stock Payments"], 0))
    payout_ratio = payout_ratio_from_cashflows(ni_latest, dividends_latest, buybacks_latest)

    return {
        "ticker": symbol.upper(),
        "price": safe_float(price),
        "price_source": price_source or "unavailable",
        "currency": px_cur or fin_cur or "unknown",
        "shares": safe_float(shares),
        "shares_source": shares_source or "unavailable",
        "sector": info.get("sector", "") or "",
        "industry": info.get("industry", "") or "",
        "quote_type": info.get("quoteType", "") or "",
        "beta": safe_float(info.get("beta")),
        "beta_source": "Yahoo beta" if safe_float(info.get("beta")) is not None else "default beta (1.0)",
        "years": years,
        "revenue": revenue,
        "rev_prev": pick(inc, ["Total Revenue", "Revenue"], 1),
        "ebit": ebit,
        "pretax": pretax,
        "tax": tax,
        "interest": pick(inc, ["Interest Expense", "Interest Expense Non Operating"], 0),
        "ni": pick(inc, ["Net Income Common Stockholders", "Net Income", "Net Income Including Noncontrolling Interests", "Net Income From Continuing Operation Net Minority Interest", "Net Income From Continuing Operations"], 0),
        "cfo": pick(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0),
        "div": pick(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"], 0),
        "buyback": pick(cf, ["Repurchase Of Capital Stock", "Common Stock Payments"], 0),
        "payout_ratio": payout_ratio,
        "rd": row_values(inc, ["Research And Development", "Research Development"]),
        "diluted_shares": pick(inc, ["Diluted Average Shares", "Weighted Average Shares Diluted"], 0),
        "effective_tax_rate": etr,
        "rev_cagr": rev_cagr,
        "roe_hist": roe_hist,
        "roe_source": roe_source,
        "earnings_cagr": hist["earnings_cagr"],
        "fcf_yield": hist["fcf_yield"],
        "company_ratios": {
            "margin": avg_margin,
            "tax": etr,
            "capex_ratio": ratio(capex, revenue),
            "dep_ratio": ratio(dep, revenue),
            "nwc_ratio": (op_nwc(years[0]) / revenue) if op_nwc(years[0]) is not None and revenue and revenue > 0 else None,
        },
    }


# ============================================================
# PENMAN OPERATING ANALYSIS
# ============================================================


def operating_analysis(data):
    y0, y1 = data["years"]

    def noa(y):
        if any(y[k] is None for k in ("assets", "cash", "liab", "debt")):
            return None
        return (y["assets"] - y["cash"]) - (y["liab"] - y["debt"])

    def nfo(y):
        if y["debt"] is None or y["cash"] is None:
            return None
        return y["debt"] - y["cash"]

    noa0, noa1, nfo0, nfo1 = noa(y0), noa(y1), nfo(y0), nfo(y1)
    avg_noa = (noa0 + noa1) / 2 if noa0 is not None and noa1 is not None else None
    avg_nfo = (nfo0 + nfo1) / 2 if nfo0 is not None and nfo1 is not None else None

    if data["pretax"] and data["pretax"] > 0 and data["tax"] is not None:
        tax_rate = max(0.0, min(0.40, data["tax"] / data["pretax"]))
    else:
        tax_rate = 0.21

    oi_after_tax = data["ebit"] * (1 - tax_rate) if data.get("ebit") is not None else None
    nfe = abs(data["interest"]) * (1 - tax_rate) if data.get("interest") is not None else None

    rnoa = oi_after_tax / avg_noa if avg_noa is not None and avg_noa > 0 and oi_after_tax is not None else None
    nbc = nfe / avg_nfo if avg_nfo is not None and avg_nfo > 0 and nfe is not None else None
    common_equity_available = "Gross Minority Interest" not in str(y0.get("equity_source", ""))
    flev = nfo0 / y0["equity"] if common_equity_available and nfo0 is not None and y0["equity"] and y0["equity"] > 0 else None
    spread = rnoa - nbc if rnoa is not None and nbc is not None else None

    avg_assets = (y0["assets"] + y1["assets"]) / 2 if y0["assets"] is not None and y1["assets"] is not None else None
    accruals = None
    if data["cfo"] is not None and data["ni"] is not None and avg_assets is not None and avg_assets > 0:
        accruals = (data["ni"] - data["cfo"]) / avg_assets

    return {"RNOA": rnoa, "NBC": nbc, "FLEV": flev, "spread": spread, "accruals": accruals}


# ============================================================
# RATES AND GROWTH
# ============================================================


def discount_rates(data, p):
    """Return (cost of equity, WACC)."""
    if not p["use_capm"]:
        return p["flat_ke"], p["flat_wacc"]
    beta = data.get("beta")
    if beta is None or not (0.0 <= beta <= 3.0):
        beta = 1.0
    ke = p["rf"] + beta * p["erp"]
    if not (0.0 < ke < 0.50):
        raise ValueError(f"implausible CAPM cost of equity ({ke:.1%}); review beta and rate assumptions")
    debt = max(data["years"][0]["debt"] or 0.0, 0.0)
    mcap = data["price"] * data["shares"]
    w_d = debt / (debt + mcap) if (debt + mcap) > 0 else 0.0
    t = data.get("effective_tax_rate")
    t = 0.21 if t is None else t
    wacc = (1 - w_d) * ke + w_d * p["kd"] * (1 - t)
    if not (0.0 < wacc < 0.50):
        raise ValueError(f"implausible WACC ({wacc:.1%}); review leverage and rate assumptions")
    return ke, wacc


def growth_path(start, terminal, years):
    """Revenue growth fading linearly from `start` (year 1) to `terminal` (year N)."""
    if years <= 1:
        return [terminal]
    return [start + (terminal - start) * i / (years - 1) for i in range(years)]


def start_growth(data, p):
    if p["use_company_growth"] and data.get("rev_cagr") is not None:
        return clamp(data["rev_cagr"], -0.10, p["max_start_growth"])
    return p["start_growth"]


# ============================================================
# RESIDUAL INCOME (equity-based, growth-consistent)
# ============================================================


def residual_income_valuation(data, ke, omega, g_path, g_term):
    """Residual-income valuation with explicit clean-surplus bridge and a stable terminal state."""
    y0, y1 = data["years"]
    bv0, bv_prev = y0["equity"], y1["equity"]
    ni, shares, price = data["ni"], data["shares"], data["price"]

    if "Gross Minority Interest" in str(y0.get("equity_source", "")):
        raise ValueError("only total equity including minority interest is available; common shareholders' book equity is unavailable")
    if bv0 is None or bv0 <= 0:
        raise ValueError("non-positive or missing book equity")
    if bv_prev is None or bv_prev <= 0:
        raise ValueError("non-positive or missing prior book equity")
    if ni is None:
        raise ValueError("missing net income")
    if not shares or shares <= 0 or not price or price <= 0:
        raise ValueError("missing shares or price")
    if not (0 < ke < 0.50) or ke <= g_term:
        raise ValueError("cost of equity must be positive, plausible, and exceed terminal growth")
    if not (0 <= omega <= 1) or not g_path:
        raise ValueError("invalid persistence or empty forecast path")

    hist_roe = data.get("roe_hist")
    if hist_roe is None:
        raise ValueError("historical ROE unavailable; RI cannot be validated without a defensible ROE input")
    roe_base = clamp(hist_roe, -0.10, 0.40)

    payout_ratio = data.get("payout_ratio")
    if payout_ratio is None:
        raise ValueError("dividend/buyback payout data unavailable")
    if not (0 <= payout_ratio <= 1):
        raise ValueError(f"observed payout ratio {payout_ratio:.1%} is outside 0–100%; RI bridge is unreliable")

    bv = bv0
    pv_ri = 0.0
    forecast = []
    for t, g in enumerate(g_path, 1):
        roe_t = ke + (roe_base - ke) * (omega ** t)
        beg = bv
        earnings = roe_t * beg
        ri = earnings - ke * beg
        payout = max(0.0, earnings) * payout_ratio
        end = beg + earnings - payout
        if not pd.notna(end) or end <= 0:
            raise ValueError("forecast book equity becomes non-positive under the earnings/payout bridge")
        pv = ri / (1 + ke) ** t
        pv_ri += pv
        forecast.append({"Year": t, "Beginning BV": beg, "ROE": roe_t, "Earnings": earnings,
                         "Net Payout": payout, "Residual Income": ri, "Ending BV": end, "PV Residual Income": pv})
        bv = end

    n = len(g_path)
    terminal_roe = ke + (roe_base - ke) * (omega ** (n + 1))
    # In steady state, clean surplus requires g = retention × ROE.
    # The terminal payout is therefore set to 1 - g/ROE, rather than carrying
    # forward the historical payout if that payout implies a different book growth rate.
    if terminal_roe <= g_term or terminal_roe <= 0:
        raise ValueError("terminal ROE must be positive and exceed terminal growth for a sustainable clean-surplus steady state")
    terminal_payout = 1.0 - (g_term / terminal_roe)
    terminal_ri = (terminal_roe - ke) * bv
    tv = terminal_ri / (ke - g_term)
    pv_tv = tv / (1 + ke) ** n
    equity = bv0 + pv_ri + pv_tv
    vps = equity / shares
    numeric_outputs = (pv_ri, terminal_roe, terminal_payout, terminal_ri, tv, pv_tv, equity, vps)
    if any(not math.isfinite(float(v)) for v in numeric_outputs):
        raise ValueError("residual-income calculation produced a non-finite value")
    return {
        "historical_roe": hist_roe,
        "roe_base_used": roe_base,
        "roe_was_clipped": abs(roe_base - hist_roe) > 1e-12,
        "terminal_roe": terminal_roe,
        "terminal_payout_ratio": terminal_payout,
        "payout_ratio": payout_ratio,
        "value_per_share": vps,
        "V/P": vps / price,
        "P/B": price * shares / bv0,
        "terminal_pct": pv_tv / equity if equity > 0 else None,
        "forecast": forecast,
    }


def calculate_implied_omega(data, ke, g_path, g_term, target=1.0):
    """Solve implied omega only when sampled valuation is finite and monotonic."""
    def vp(w):
        try:
            v = residual_income_valuation(data, ke, w, g_path, g_term)["V/P"]
            return None if not pd.notna(v) or not (-1e6 < v < 1e6) else float(v)
        except Exception:
            return None

    grid = [i / 20 for i in range(21)]
    vals = [vp(w) for w in grid]
    if any(v is None for v in vals):
        return None
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    tol = 1e-8
    increasing = all(d >= -tol for d in diffs)
    decreasing = all(d <= tol for d in diffs)
    if not (increasing or decreasing):
        return None
    if abs(vals[0] - target) <= tol:
        return 0.0
    if abs(vals[-1] - target) <= tol:
        return 1.0
    if (vals[0] - target) * (vals[-1] - target) > 0:
        return None

    lo, hi = 0.0, 1.0
    f_lo = vals[0] - target
    for _ in range(60):
        mid = (lo + hi) / 2
        v = vp(mid)
        if v is None:
            return None
        if abs(v - target) < 1e-9:
            return mid
        if (v - target) * f_lo > 0:
            lo, f_lo = mid, v - target
        else:
            hi = mid
    return (lo + hi) / 2


# ============================================================
# DCF (FCFF, value-driver terminal value)
# ============================================================


def dcf_valuation(data, wacc, g_path, g_term, margin, tax, capex_ratio, dep_ratio, nwc_ratio, term_roic, margin_end=None):
    if wacc <= g_term:
        raise ValueError("WACC must exceed terminal growth")
    if term_roic is None or term_roic <= g_term or term_roic <= 0:
        raise ValueError("terminal ROIC must be positive and exceed terminal growth")
    if any(v is None or not pd.notna(v) for v in (margin, tax, capex_ratio, dep_ratio, nwc_ratio)):
        raise ValueError("one or more DCF operating assumptions are missing or non-finite")
    dcf_ratios = {"margin": margin, "tax": tax, "capex_ratio": capex_ratio,
                  "dep_ratio": dep_ratio, "nwc_ratio": nwc_ratio}
    for key, value in dcf_ratios.items():
        lo, hi = DCF_HARD_LIMITS[key]
        if not (lo <= value <= hi):
            raise ValueError(f"{DCF_LABELS[key]} {value:.1%} is outside hard model bounds ({lo:.1%} to {hi:.1%})")
    if margin_end is not None and (not math.isfinite(float(margin_end)) or not (DCF_HARD_LIMITS["margin"][0] <= margin_end <= DCF_HARD_LIMITS["margin"][1])):
        raise ValueError("terminal forecast EBIT margin is outside hard model bounds")
    if not g_path or any(g is None or not pd.notna(g) or g <= -1 for g in g_path):
        raise ValueError("growth path is empty or contains invalid growth rates")
    y0 = data["years"][0]
    base_rev, shares, price = data["revenue"], data["shares"], data["price"]
    if not base_rev or base_rev <= 0:
        raise ValueError("missing revenue")
    if not shares or shares <= 0 or not price or price <= 0:
        raise ValueError("missing shares or price")
    if y0["debt"] is None or y0["cash"] is None:
        raise ValueError("total debt or cash unavailable; net debt cannot be estimated safely")
    if not (0 <= tax <= 0.60):
        raise ValueError("tax rate must be between 0% and 60%")
    if not (DCF_HARD_LIMITS["margin"][0] <= margin <= DCF_HARD_LIMITS["margin"][1]):
        raise ValueError("EBIT margin is outside hard model bounds")
    net_debt = y0["debt"] - y0["cash"] + (y0["minority"] if y0["minority"] is not None else 0.0)

    rev = base_rev
    nwc_prev = nwc_ratio * base_rev  # same ratio as forecast, so no artificial year-1 jump
    pv_fcff, forecast, nopat = 0.0, [], 0.0
    for t, g in enumerate(g_path, 1):
        rev *= 1 + g
        # Anchor year 1 at the observed/normalized starting margin and the final
        # explicit year at the terminal margin; avoid shifting the entire path by 1/N.
        fade = (t - 1) / (len(g_path) - 1) if len(g_path) > 1 else 1.0
        end_margin = margin if margin_end is None else margin_end
        m_t = margin + (end_margin - margin) * fade
        ebit = rev * m_t
        nopat = ebit * (1 - tax)
        dep, nwc = rev * dep_ratio, rev * nwc_ratio
        # Transition from observed CapEx intensity to a modest terminal maintenance
        # level; the terminal FCFF separately ties total reinvestment to g / ROIC.
        cap_target = dep_ratio * 1.1
        cap_r = capex_ratio + (cap_target - capex_ratio) * fade
        capex = rev * cap_r
        d_nwc = nwc - nwc_prev
        net_inv = capex - dep + d_nwc
        fcff = nopat - net_inv
        pv = fcff / (1 + wacc) ** t
        pv_fcff += pv
        forecast.append({"Year": t, "Growth": g, "Revenue": rev, "EBIT": ebit, "NOPAT": nopat, "D&A": dep,
                         "CapEx": capex, "ΔNWC": d_nwc, "Net Investment": net_inv, "FCFF": fcff, "PV FCFF": pv})
        nwc_prev = nwc

    n = len(g_path)
    # Terminal: FCFF = NOPAT x (1 - g / ROIC). Reinvestment is tied to growth and returns.
    term_fcff = nopat * (1 + g_term) * (1 - g_term / term_roic)
    tv = term_fcff / (wacc - g_term)
    pv_tv = tv / (1 + wacc) ** n
    ev = pv_fcff + pv_tv
    equity = ev - net_debt
    vps = equity / shares
    numeric_outputs = (net_debt, pv_fcff, term_fcff, tv, pv_tv, ev, equity, vps)
    if any(not math.isfinite(float(v)) for v in numeric_outputs):
        raise ValueError("DCF calculation produced a non-finite value")
    return {
        "net_debt": net_debt, "enterprise_value": ev, "equity_value": equity,
        "value_per_share": vps, "V/P": vps / price, "terminal_pct": pv_tv / ev if ev > 0 else None,
        "term_roic": term_roic, "forecast": forecast,
        "pv_explicit": pv_fcff, "pv_terminal": pv_tv, "term_fcff": term_fcff,
    }


def implied_start_growth(data, wacc, g_term, years, dcf_inputs):
    """Reverse DCF; solve only when price/value is monotonic over the supported growth range."""
    def vp(g0):
        return dcf_valuation(data, wacc, growth_path(g0, g_term, years), g_term, **dcf_inputs)["V/P"]

    try:
        grid = [-0.10 + i * 0.03 for i in range(21)]  # -10% through 50%
        vals = [vp(g) for g in grid]
        if any(not pd.notna(v) or not (-1e6 < v < 1e6) for v in vals):
            return None
        diffs = [b - a for a, b in zip(vals, vals[1:])]
        if not all(d >= -1e-8 for d in diffs):
            return None
        if vals[0] >= 1.0:
            return -0.10  # price is supported even at the lowest tested growth
        if vals[-1] < 1.0:
            return float("inf")  # price requires more than the supported 50% growth
        lo, hi = -0.10, 0.50
        for _ in range(60):
            mid = (lo + hi) / 2
            v = vp(mid)
            if not pd.notna(v):
                return None
            if abs(v - 1.0) < 1e-9:
                return mid
            lo, hi = (mid, hi) if v < 1.0 else (lo, mid)
        return (lo + hi) / 2
    except Exception:
        return None


# Soft guardrails are diagnostic ranges, not clipping limits. Real companies can
# legitimately sit outside them (especially CapEx and operating working capital).
# Hard bounds reject only extreme values likely to reflect a data/definition problem.
DCF_LIMITS = {"margin": (-0.20, 0.60), "tax": (0.10, 0.30), "capex_ratio": (0.0, 0.50),
              "dep_ratio": (0.0, 0.20), "nwc_ratio": (-0.25, 0.50)}
DCF_HARD_LIMITS = {"margin": (-1.0, 1.0), "tax": (0.0, 0.60), "capex_ratio": (0.0, 2.0),
                   "dep_ratio": (0.0, 0.50), "nwc_ratio": (-1.0, 1.0)}
DCF_LABELS = {"margin": "EBIT margin (3y avg)", "tax": "tax rate", "capex_ratio": "CapEx / revenue",
              "dep_ratio": "D&A / revenue", "nwc_ratio": "operating NWC / revenue"}
FALLBACK_KEYS = {"margin": "fb_margin", "tax": "fb_tax", "capex_ratio": "fb_capex",
                 "dep_ratio": "fb_dep", "nwc_ratio": "fb_nwc"}


def resolve_dcf_inputs(data, p):
    """Use observed company ratios without clipping; fallback only for missing/extreme data."""
    values, sources = {}, {}
    for key, fb in FALLBACK_KEYS.items():
        cv = safe_float(data.get("company_ratios", {}).get(key))
        fallback = safe_float(p.get(fb))
        if fallback is None or not (DCF_HARD_LIMITS[key][0] <= fallback <= DCF_HARD_LIMITS[key][1]):
            raise ValueError(f"invalid sidebar fallback for {DCF_LABELS[key]}")
        if p["override"]:
            values[key], sources[key] = fallback, "sidebar override"
        elif cv is None:
            values[key], sources[key] = fallback, "sidebar fallback (company data missing)"
        elif not (DCF_HARD_LIMITS[key][0] <= cv <= DCF_HARD_LIMITS[key][1]):
            values[key], sources[key] = fallback, "sidebar fallback (company ratio outside hard bounds)"
        else:
            values[key] = cv
            outside_soft = not (DCF_LIMITS[key][0] <= cv <= DCF_LIMITS[key][1])
            sources[key] = "company (outside usual range)" if outside_soft else "company"
    return values, sources


# ============================================================
# COMPANY ANALYSIS
# ============================================================


def analyze_company(data, p):
    y0, y1 = data["years"]
    required = {
        "assets": y0["assets"], "liabilities": y0["liab"], "debt": y0["debt"], "cash": y0["cash"],
        "prior assets": y1["assets"], "prior liabilities": y1["liab"], "prior debt": y1["debt"],
        "prior cash": y1["cash"], "operating income": data["ebit"], "revenue": data["revenue"],
        "price": data["price"], "shares": data["shares"],
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError("missing data: " + ", ".join(missing))

    operating = operating_analysis(data)
    ke, wacc = discount_rates(data, p)
    years, g_term = p["years"], p["terminal_growth"]
    if ke <= g_term + 0.01 or wacc <= g_term:
        raise ValueError("discount rate too close to terminal growth")
    g0 = start_growth(data, p)
    if p["use_company_growth"] and data.get("rev_cagr") is not None and not (-0.10 <= data["rev_cagr"] <= p["max_start_growth"]):
        warnings_note = (f"Historical revenue CAGR {data['rev_cagr']:.1%} is outside the supported start-growth range; "
                         f"model uses {g0:.1%} after clipping.")
    else:
        warnings_note = None
    is_growth = g0 >= p["growth_threshold"]  # growth mode: longer fade, latest margin, optional margin ramp
    if is_growth:
        years = p["growth_years"]
    g_path = growth_path(g0, g_term, years)

    warnings = []
    if warnings_note:
        warnings.append(warnings_note)
    if data.get("price_source") != "Yahoo current/regular market price":
        warnings.append(f"Current quote unavailable; using {data.get('price_source', 'an unverified price fallback')}. Verify the quote timestamp before comparing valuation with market price.")
    if data.get("shares_source") != "Yahoo current shares outstanding":
        warnings.append(f"Current shares outstanding unavailable; using {data.get('shares_source', 'an unverified fallback')}. Per-share values and market capitalization may be unreliable.")
    if data.get("beta_source") != "Yahoo beta" or data.get("beta") is None or not (0 <= data.get("beta", 1.0) <= 3.0):
        warnings.append("Yahoo beta is missing or outside the supported 0–3 range; CAPM uses a beta of 1.0. Review cost of equity.")
    elif p["use_capm"] and data.get("beta") < 0.60:
        warnings.append(f"Low Yahoo beta ({data['beta']:.2f}) produces a relatively low CAPM cost of equity; compare with a normalized/industry beta as a sensitivity case.")
    if p["use_capm"] and ke < 0.07:
        warnings.append(f"Cost of equity is low at {ke:.1%}; valuation is sensitive to beta and the equity-risk premium. Review higher discount-rate scenarios before relying on the estimate.")
    if data.get("roe_source") == "Yahoo-reported ROE fallback":
        warnings.append("Historical ROE could not be reconstructed from reported balance-sheet history; using Yahoo-reported ROE as a fallback. Verify the equity denominator before relying on RI.")
    if data.get("payout_ratio") is None:
        warnings.append("Dividend/buyback payout data unavailable; RI is disabled rather than assuming zero payout.")
    elif not (0 <= data["payout_ratio"] <= 1):
        warnings.append(f"Observed payout ratio is {data['payout_ratio']:.1%}, outside 0–100%; RI is disabled pending review.")
    reported_eq0, reported_eq1 = y0["equity"], y1["equity"]
    eq0, eq1, ni_adj = reported_eq0, reported_eq1, data["ni"]
    adjusted = False
    if p["tangible_book"]:
        gw0, gw1 = y0["goodwill"], y1["goodwill"]
        if gw0 is not None and gw1 is not None and eq0 is not None and eq1 is not None:
            tangible0, tangible1 = eq0 - gw0, eq1 - gw1
            if tangible0 > 0 and tangible1 > 0:
                eq0, eq1, adjusted = tangible0, tangible1, True
                warnings.append("RI uses tangible book (goodwill and intangibles removed from equity).")
            else:
                warnings.append("Tangible book is non-positive for the current or prior period; RI automatically falls back to reported book equity.")
        else:
            warnings.append("Goodwill/intangibles not reported; RI uses reported book equity.")
    rd = [v for v in (data.get("rd") or []) if v is not None and v > 0]
    if p["capitalize_rd"] and len(rd) >= 2 and eq0 and eq1 and ni_adj is not None:
        life, tx = 5, data["effective_tax_rate"] if data["effective_tax_rate"] is not None else 0.21
        # Yahoo annual statements are normally newest-first. Treat current-year R&D as
        # an asset at year-end, amortize older vintages straight-line, and construct the
        # prior-year asset without including current-year spend. With only 3-4 years of
        # Yahoo history this is necessarily a partial-capitalization approximation.
        current_asset = sum(v * max(0.0, 1 - k / life) for k, v in enumerate(rd[:life]))
        prior_vintages = rd[1:life + 1]
        prior_asset = sum(v * max(0.0, 1 - k / life) for k, v in enumerate(prior_vintages))
        amort0 = sum(rd[k] / life for k in range(1, min(len(rd), life + 1)))
        a0, a1 = current_asset * (1 - tx), prior_asset * (1 - tx)
        eq0, eq1 = eq0 + a0, eq1 + a1
        ni_adj, adjusted = ni_adj + (rd[0] - amort0) * (1 - tx), True
        warnings.append("RI capitalizes R&D (5-year life, after tax) into book equity and earnings; approximate, based on the R&D years Yahoo reports.")
    ri_data = data
    if adjusted:
        avg_adjusted_equity = (eq0 + eq1) / 2 if eq0 is not None and eq1 is not None else None
        ri_data = {**data, "years": [{**y0, "equity": eq0}, {**y1, "equity": eq1}], "ni": ni_adj,
                   "roe_hist": (ni_adj / avg_adjusted_equity) if ni_adj is not None and avg_adjusted_equity is not None and avg_adjusted_equity > 0 else None}
    base = cons = implied_omega = None
    try:
        base = residual_income_valuation(ri_data, ke, p["omega"], g_path, g_term)
        # Conservative case: faster fade if ROE > ke, slower recovery if ROE < ke
        cons_omega = p["omega"] * 0.75 if base["historical_roe"] >= ke else min(1.0, p["omega"] * 1.25)
        cons = residual_income_valuation(ri_data, ke + 0.01, cons_omega, g_path, g_term)
        implied_omega = calculate_implied_omega(ri_data, ke, g_path, g_term)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        base = cons = implied_omega = None
        warnings.append(f"Residual income unavailable: {e}."
                        + (" Adjusted book is negative or missing; untick the tangible-book option to use reported book." if ri_data is not data else ""))

    ri_valid = base is not None
    if data.get("roe_source") == "Yahoo-reported ROE fallback" and base is not None:
        ri_valid = False
        implied_omega = None
        warnings.append("RI disabled: historical ROE could not be reconstructed from balance-sheet history. Yahoo-reported ROE is shown for context but is not sufficient to validate the RI forecast.")
    if base and base.get("roe_was_clipped"):
        ri_valid = False
        implied_omega = None
        warnings.append(
            f"RI disabled: ROE input {base['historical_roe']:.1%} exceeds the model's supported range and was clipped to "
            f"{base['roe_base_used']:.1%}. Review adjusted book equity / buyback effects before using RI."
        )
    eq_rep = y0["equity"]
    common_equity_available = "Gross Minority Interest" not in str(y0.get("equity_source", ""))
    pb_rep = data["price"] * data["shares"] / eq_rep if common_equity_available and eq_rep and eq_rep > 0 else NAN
    ri_eq0 = ri_data["years"][0].get("equity") if ri_data.get("years") else None
    ri_pb = data["price"] * data["shares"] / ri_eq0 if common_equity_available and ri_eq0 and ri_eq0 > 0 else NAN
    if not common_equity_available:
        warnings.append("Only total equity including minority interest is available; P/B and FLEV are suppressed and RI is disabled because common-equity data is not comparable.")
    roe_rep = ri_data.get("roe_hist") if ri_data.get("roe_hist") is not None else (base["historical_roe"] if base else NAN)
    if ri_valid and (not pd.notna(ri_pb) or ri_pb > p["max_pb"] or not pd.notna(roe_rep) or roe_rep > p["max_roe"]):
        ri_valid, implied_omega = False, None
        warnings.append(
            f"RI not meaningful: adjusted RI P/B is {ri_pb:.1f}x and RI ROE is {roe_rep:.0%}; "
            f"the configured limits are P/B ≤ {p['max_pb']:.1f}x and ROE ≤ {p['max_roe']:.0%}. "
            "High P/B or extreme ROE can make book-value-based RI unstable; verdict uses DCF only."
        )

    dcf_inputs, dcf_sources = resolve_dcf_inputs(data, p)
    lo = wacc + 0.01
    rnoa = operating["RNOA"]
    high_return = rnoa is not None and rnoa >= 0.25
    hi = max(p["quality_roic_cap"] if high_return else p["roic_cap"], lo)
    latest_margin = (data["ebit"] / data["revenue"] if data.get("ebit") is not None and data.get("revenue") else None)
    # In growth mode, start at the available multi-year average and fade toward the latest
    # margin; do not assume the latest peak/trough margin persists unchanged for 15 years.
    if is_growth and not p["override"] and dcf_sources.get("margin", "").startswith("sidebar fallback") and latest_margin is not None:
        # If the 3-year average is missing, use latest reported margin only if within hard bounds.
        if DCF_HARD_LIMITS["margin"][0] <= latest_margin <= DCF_HARD_LIMITS["margin"][1]:
            dcf_inputs["margin"] = latest_margin
            dcf_sources["margin"] = "latest company margin (fallback)"
    margin_end_base = latest_margin if is_growth and not p["override"] and latest_margin is not None else dcf_inputs["margin"]
    proposed_margin_end = margin_end_base + (p["margin_expansion"] if is_growth else 0.0)
    if not (DCF_HARD_LIMITS["margin"][0] <= proposed_margin_end <= DCF_HARD_LIMITS["margin"][1]):
        warnings.append("Forecast endpoint EBIT margin would exceed hard bounds; endpoint falls back to the starting DCF margin. Review the latest margin and margin-expansion assumption.")
        proposed_margin_end = dcf_inputs["margin"]
    dcf_inputs["margin_end"] = proposed_margin_end
    dcf_inputs["term_roic"] = clamp(rnoa if rnoa is not None else lo + 0.02, lo, hi)
    dcf = dcf_valuation(data, wacc, g_path, g_term, **dcf_inputs)
    implied_g = implied_start_growth(data, wacc, g_term, years, dcf_inputs)
    dcf_ok = dcf["value_per_share"] > 0

    def scenario_vp(g_mult, g_add, margin_add, roic_add):
        try:
            gs = clamp((g0 * g_mult if g0 > 0 else g0) + g_add, -0.10, 0.50)
            inp = dict(dcf_inputs)
            inp["margin_end"] = clamp(inp["margin_end"] + margin_add, *DCF_HARD_LIMITS["margin"])
            inp["term_roic"] = clamp(inp["term_roic"] + roic_add, lo, max(hi, lo + 0.15))
            v = dcf_valuation(data, wacc, growth_path(gs, g_term, years), g_term, **inp)["V/P"]
            return v if v > 0 else NAN
        except Exception:
            return NAN

    bear_vp = scenario_vp(0.6, -0.01, -0.03, -0.05)
    bull_vp = scenario_vp(1.3, 0.02, 0.04, 0.05)
    if is_growth:
        warnings.append(
            f"Growth mode: starting growth {g0:.1%} is at or above {p['growth_threshold']:.0%}, so the fade runs {years} years "
            f"and EBIT margin fades from {dcf_inputs['margin']:.1%} toward {dcf_inputs['margin_end']:.1%}.")
    if high_return:
        warnings.append(f"High-return business (RNOA {rnoa:.0%}): terminal ROIC cap raised to {p['quality_roic_cap']:.0%}.")
    selected_roic_cap = p["quality_roic_cap"] if high_return else p["roic_cap"]
    if lo > selected_roic_cap:
        warnings.append(f"WACC + 1% ({lo:.1%}) exceeds the selected terminal ROIC cap ({selected_roic_cap:.1%}); terminal ROIC is forced above the configured cap to remain above WACC.")
    if not dcf_ok:
        implied_g = None
        warnings.append("DCF equity value is negative (net debt exceeds enterprise value); DCF shown as n/m.")
    rp = data.get("rev_prev")
    if rp and rp > 0 and abs(data["revenue"] / rp - 1) > 0.30:
        warnings.append("Revenue changed >30% year over year: possible merger or spin-off; financials may not match the current price.")
    if data.get("fcf_yield") is not None and data["fcf_yield"] > 0.20:
        warnings.append("FCF yield above 20%: check for a spin-off, one-off cash flow or data mismatch.")
    m_avg = data["company_ratios"].get("margin")
    if m_avg and m_avg > 0 and abs(data["ebit"] / data["revenue"] / m_avg - 1) > 0.40:
        warnings.append("Latest EBIT margin differs >40% from the 3-year average: earnings may be cyclical or volatile. DCF starts from the average and fades toward the latest margin in growth mode; review the margin path.")

    if data.get("effective_tax_rate") is None:
        warnings.append("Effective tax rate unavailable: DCF uses its configured tax fallback, while WACC's after-tax debt shield uses a separate 21% fallback. Review both assumptions.")
    if y0.get("minority") is None:
        warnings.append("Minority interest was not reported; DCF assumes zero in the enterprise-to-equity bridge. Verify this assumption.")
    if not p["override"]:
        for k, source in dcf_sources.items():
            if source == "sidebar":
                warnings.append(f"Company {DCF_LABELS[k]} unavailable; DCF uses the sidebar fallback.")
            elif source == "company (outside usual range)":
                warnings.append(f"Company {DCF_LABELS[k]} is {dcf_inputs[k]:.1%}, outside the usual diagnostic range {DCF_LIMITS[k][0]:.1%} to {DCF_LIMITS[k][1]:.1%}; the observed ratio is used without clipping. Verify the statement line items and consider a sensitivity case.")
            elif source == "sidebar fallback (company ratio outside hard bounds)":
                warnings.append(f"Company {DCF_LABELS[k]} was outside hard safety bounds; DCF uses the sidebar fallback instead of the reported ratio. Investigate the underlying data before relying on the estimate.")
            elif source == "sidebar fallback (company data missing)":
                warnings.append(f"Company {DCF_LABELS[k]} unavailable; DCF uses the sidebar fallback.")
            elif source == "latest company margin (fallback)":
                warnings.append("Three-year average EBIT margin is unavailable; DCF uses the latest reported EBIT margin as a fallback.")
    if p["use_capm"]:
        warnings.append("WACC uses reported/book debt as a proxy for market-value debt and the configured cost of debt; market-value debt and credit spreads are not available from these Yahoo summary fields.")
    if dcf_inputs["margin"] < 0.03:
        warnings.append("Thin EBIT margin: DCF is highly sensitive to small margin changes.")
    if any(row["Net Investment"] < 0 for row in dcf["forecast"]):
        warnings.append("At least one forecast year has negative net investment (CapEx − D&A + change in operating NWC), boosting FCFF. Verify whether the cash release or low reinvestment is sustainable.")
    if any(row["FCFF"] < 0 for row in dcf["forecast"]):
        warnings.append("At least one forecast year has negative FCFF. The DCF may still be mathematically valid, but check the transition to positive terminal cash flow.")
    if dcf["term_roic"] <= wacc:
        warnings.append(f"Terminal ROIC ({dcf['term_roic']:.1%}) is at or below WACC ({wacc:.1%}); terminal growth creates little or negative economic value. Review the terminal return assumption.")
    if dcf["terminal_pct"] is not None and dcf["terminal_pct"] > 0.65:
        warnings.append(
            f"Present value of terminal value is {dcf['terminal_pct']:.0%} of enterprise value; "
            "the estimate is materially sensitive to terminal assumptions. Stress-test terminal growth/ROIC and consider a longer explicit forecast."
        )
    dil = data.get("diluted_shares")
    if dil and data["shares"] and abs(dil - data["shares"]) / data["shares"] > 0.05:
        warnings.append("Current shares outstanding differs >5% from the historical diluted average share count; check dilution, buybacks and share-count timing.")
    if ri_valid and dcf_ok and base is not None and base["value_per_share"] > 0 and dcf["value_per_share"] > 0:
        gap = abs(base["value_per_share"] - dcf["value_per_share"]) / max(base["value_per_share"], dcf["value_per_share"])
        if gap >= 0.50:
            warnings.append(f"Model disagreement: DCF and RI values differ by {gap:.0%} of the higher estimate. Review margins, reinvestment, book equity, payout and terminal assumptions; values are not averaged.")

    # Keep DCF estimates visible for exploration, but do not let a positive number qualify a
    # candidate when a core input is substituted/clipped or the equity bridge is incomplete.
    beta_is_usable = (not p["use_capm"]) or (
        data.get("beta_source") == "Yahoo beta" and data.get("beta") is not None
        and 0 <= data.get("beta") <= 3
    )
    shares_are_current = data.get("shares_source") == "Yahoo current shares outstanding"
    price_is_current = data.get("price_source") == "Yahoo current/regular market price"
    minority_is_known = y0.get("minority") is not None
    growth_is_unclipped = warnings_note is None and not (p["use_company_growth"] and data.get("rev_cagr") is None)
    dcf_inputs_are_company_reported = all(source == "company" for source in dcf_sources.values())
    dcf_reliable = bool(
        dcf_ok and beta_is_usable and shares_are_current and price_is_current and minority_is_known
        and growth_is_unclipped and dcf_inputs_are_company_reported
    )
    if dcf_ok and not dcf_reliable:
        warnings.append("DCF is provisional, not screening-grade: at least one core input is clipped, substituted, unusual, or unavailable. It is displayed for analysis but cannot qualify the stock as a candidate; review the input warnings.")

    def ri(res, key):
        return res[key] if (ri_valid and res is not None) else NAN

    # A valuation can pass the numeric thresholds and still be too assumption-heavy
    # for an automatic candidate label. Keep this gate explicit and visible.
    quality_flags = []
    if not dcf_reliable:
        quality_flags.append("DCF inputs or equity bridge are not screening-grade")
    if not (DCF_LIMITS["margin"][0] <= dcf_inputs["margin_end"] <= DCF_LIMITS["margin"][1]):
        quality_flags.append("forecast endpoint margin is outside the usual range")
    if data.get("price_source") != "Yahoo current/regular market price":
        quality_flags.append("fallback market quote")
    if not data.get("currency") or data.get("currency") == "unknown":
        quality_flags.append("security currency unavailable")
    if data.get("shares_source") != "Yahoo current shares outstanding":
        quality_flags.append("fallback share count")
    if data.get("roe_source") == "Yahoo-reported ROE fallback":
        quality_flags.append("fallback ROE")
    if not ri_valid:
        quality_flags.append("RI valuation unavailable or invalid")
    if any(str(src).startswith("sidebar fallback") for src in dcf_sources.values()):
        quality_flags.append("one or more DCF inputs use a fallback")
    if any(src == "company (outside usual range)" for src in dcf_sources.values()):
        quality_flags.append("one or more company ratios are outside usual ranges")
    if dcf["terminal_pct"] is not None and dcf["terminal_pct"] > 0.65:
        quality_flags.append("DCF terminal value exceeds 65% of enterprise value")
    if any(row["Net Investment"] < 0 or row["FCFF"] < 0 for row in dcf["forecast"]):
        quality_flags.append("forecast contains negative net investment or FCFF")
    if any("Revenue changed >30%" in w or "Latest EBIT margin differs >40%" in w for w in warnings):
        quality_flags.append("materially unusual operating history")
    if any("Model disagreement" in w for w in warnings):
        quality_flags.append("DCF and RI materially disagree")
    if dcf_ok and not dcf_reliable:
        quality_flags.append("DCF core inputs are provisional")
    if p["use_capm"] and ke < 0.07:
        quality_flags.append("low cost-of-equity assumption")
    valuation_quality = "REVIEW" if quality_flags else "OK"
    if quality_flags:
        warnings.append("Valuation quality gate: automatic candidate status is withheld pending review of " + ", ".join(quality_flags) + ".")

    eq0 = y0["equity"]
    return {
        "Ticker": data["ticker"],
        "Currency": data.get("currency", "unknown"),
        "Price": data["price"],
        "P/B": data["price"] * data["shares"] / eq0 if common_equity_available and eq0 and eq0 > 0 else NAN,
        "RI P/B": ri_pb,
        "ROE": data.get("roe_hist") if data.get("roe_hist") is not None else (base["historical_roe"] if base else NAN),
        "RI ROE Input": base["historical_roe"] if base else NAN,
        "RI ROE Applied": base["roe_base_used"] if base else NAN,
        "ROE Source": data.get("roe_source") or "unavailable",
        "Cost of Equity": ke,
        "WACC": wacc,
        "Start Growth": g0,
        "Market-Implied Growth": implied_g,
        "Earnings CAGR": data.get("earnings_cagr"),
        "FCF Yield": data.get("fcf_yield"),
        "Implied Omega": implied_omega,
        "RNOA": operating["RNOA"],
        "NBC": operating["NBC"],
        "FLEV": operating["FLEV"],
        "Spread": operating["spread"],
        "Accruals": operating["accruals"],
        "RI Valid": ri_valid,
        "DCF Reliable": dcf_reliable,
        "Valuation Quality": valuation_quality,
        "RI Value": ri(base, "value_per_share"),
        "RI V/P": ri(base, "V/P"),
        "RI Conservative Value": ri(cons, "value_per_share"),
        "RI Conservative V/P": ri(cons, "V/P"),
        "DCF Value": dcf["value_per_share"] if dcf_ok else NAN,
        "DCF V/P": dcf["V/P"] if dcf_ok else NAN,
        "DCF Bear V/P": bear_vp,
        "DCF Bull V/P": bull_vp,
        "Net Debt (B)": dcf["net_debt"] / 1e9,
        "Shares Outstanding": data["shares"],
        "RI Terminal Value Share": base["terminal_pct"] if ri_valid and base else NAN,
        "RI Terminal ROE": base["terminal_roe"] if ri_valid and base else NAN,
        "RI Terminal Payout": base["terminal_payout_ratio"] if ri_valid and base else NAN,
        "DCF Terminal Value Share": dcf["terminal_pct"],
        "DCF Terminal Growth": g_term,
        "DCF Terminal ROIC": dcf["term_roic"],
        "warnings": warnings,
        "_currency": data.get("currency", "unknown"), "_price_source": data.get("price_source", "unavailable"),
        "_base": base if ri_valid else None, "_dcf": dcf, "_dcf_inputs": dcf_inputs, "_dcf_sources": dcf_sources,
        "_data": ri_data, "_years": years, "_g0": g0, "_ke": ke, "_wacc": wacc,
    }


def should_skip_company(data):
    """Exclude financials and non-common-equity instruments from this operating-company model."""
    sector = str(data.get("sector", "")).lower()
    industry = str(data.get("industry", "")).lower()
    quote_type = str(data.get("quote_type", "")).upper()
    financial_terms = (
        "financial", "bank", "insurance", "insurer", "capital markets", "asset management",
        "credit services", "mortgage finance", "consumer finance", "investment banking",
    )
    if "financial" in sector or any(term in industry for term in financial_terms):
        return True, "financial companies require a different accounting/valuation reformulation"
    if quote_type and quote_type not in {"EQUITY"}:
        return True, f"Yahoo quote type {quote_type} is not a standard common equity"
    return False, ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Discount rates")
    use_capm = st.checkbox("Per-company CAPM rates", value=True,
                           help="Cost of equity = risk-free + beta x ERP; WACC blends it with after-tax cost of debt.")
    rf = st.number_input("Risk-free rate", 0.00, 0.10, 0.043, 0.001, format="%.3f")
    erp = st.number_input("Equity risk premium", 0.02, 0.09, 0.045, 0.005, format="%.3f")
    kd = st.number_input("Pre-tax cost of debt", 0.01, 0.15, 0.055, 0.005, format="%.3f")
    flat_ke = st.number_input("Flat cost of equity (if CAPM off)", 0.05, 0.30, 0.09, 0.005, format="%.3f")
    flat_wacc = st.number_input("Flat WACC (if CAPM off)", 0.05, 0.30, 0.08, 0.005, format="%.3f")

    st.divider()
    st.header("Growth")
    terminal_growth = st.number_input("Terminal growth (both models)", -0.02, 0.04, 0.025, 0.005, format="%.3f")
    forecast_years = st.slider("Forecast years (growth fade)", 5, 15, 10)
    use_company_growth = st.checkbox("Start from company revenue CAGR", value=True)
    max_start_growth = st.number_input("Max starting growth", 0.00, 0.40, 0.20, 0.01, format="%.2f")
    start_growth_fb = st.number_input("Starting growth (fallback / manual)", -0.10, 0.40, 0.06, 0.005, format="%.3f")
    growth_threshold = st.number_input("Growth mode if starting growth ≥", 0.05, 0.40, 0.10, 0.01, format="%.2f",
                                       help="Growth names get a longer fade and the latest margin instead of the 3-year average.")
    growth_years = st.number_input("Growth mode: fade years", 10, 25, 15, 1)

    st.divider()
    st.header("Residual income")
    omega = st.slider("ROE persistence (ω)", 0.0, 1.0, 0.80, 0.05)
    max_pb = st.number_input("RI valid only if P/B ≤", 1.0, 100.0, 10.0, 1.0)
    max_roe = st.number_input("RI valid only if ROE ≤", 0.10, 3.00, 0.50, 0.05)
    capitalize_rd = st.checkbox("RI: capitalize R&D into book equity and earnings", value=True,
                                help="Treats R&D as investment (5-year life). Helps tech and pharma, where book value omits R&D.")
    tangible_book = st.checkbox("RI: use tangible book (subtract goodwill & intangibles)", value=True)

    st.divider()
    st.header("DCF")
    dcf_override = st.checkbox("Use sidebar ratios for all companies", value=False,
                               help="Off: each company's own margin, tax, CapEx, D&A and NWC ratios; sidebar values are fallbacks.")
    fb_margin = st.number_input("EBIT margin", -0.50, 0.80, 0.15, 0.005, format="%.3f")
    fb_tax = st.number_input("Tax rate", 0.00, 0.60, 0.21, 0.005, format="%.3f")
    fb_capex = st.number_input("CapEx / revenue", 0.00, 1.00, 0.04, 0.005, format="%.3f")
    fb_dep = st.number_input("D&A / revenue", 0.00, 0.50, 0.035, 0.005, format="%.3f")
    fb_nwc = st.number_input("NWC / revenue", -0.50, 0.80, 0.05, 0.005, format="%.3f")
    roic_cap = st.number_input("Terminal ROIC cap", 0.05, 0.50, 0.15, 0.01, format="%.2f",
                               help="Terminal reinvestment = g / ROIC. ROIC is the company's RNOA, floored at WACC+1% and capped here.")

    quality_roic_cap = st.number_input("Terminal ROIC cap, high-return firms (RNOA ≥ 25%)", 0.10, 0.60, 0.25, 0.01, format="%.2f",
                                       help="Firms with a durable moat can keep higher returns on new investment.")
    margin_expansion = st.number_input("Growth mode: extra margin expansion by end of fade", 0.00, 0.20, 0.00, 0.01, format="%.2f",
                                       help="Added to the latest EBIT margin, ramped linearly. 0 = hold the latest margin.")

    st.divider()
    st.header("Screening rules")
    min_ri_vp = st.number_input("Minimum RI V/P", 0.5, 5.0, 1.20, 0.05)
    min_dcf_vp = st.number_input("Minimum DCF V/P", 0.5, 5.0, 1.20, 0.05)
    min_cons_vp = st.number_input("Minimum conservative RI V/P", 0.5, 3.0, 1.00, 0.05)
    strong_ri_vp = st.number_input("Strong: minimum RI V/P", 0.5, 5.0, 1.50, 0.05)
    strong_max_omega = st.number_input("Strong: maximum implied ω", 0.0, 1.0, 0.80, 0.05)
    max_accruals = st.number_input("Maximum accruals", 0.0, 0.5, 0.10, 0.01, format="%.2f")
    max_flev = st.number_input("Maximum FLEV", 0.0, 10.0, 2.0, 0.25)

P = {
    "use_capm": use_capm, "rf": rf, "erp": erp, "kd": kd, "flat_ke": flat_ke, "flat_wacc": flat_wacc,
    "terminal_growth": terminal_growth, "years": forecast_years, "use_company_growth": use_company_growth,
    "max_start_growth": max_start_growth, "start_growth": start_growth_fb, "omega": omega,
    "max_pb": max_pb, "max_roe": max_roe, "tangible_book": tangible_book, "capitalize_rd": capitalize_rd, "override": dcf_override, "fb_margin": fb_margin,
    "fb_tax": fb_tax, "fb_capex": fb_capex, "fb_dep": fb_dep, "fb_nwc": fb_nwc, "roic_cap": roic_cap,
    "growth_threshold": growth_threshold, "growth_years": growth_years, "quality_roic_cap": quality_roic_cap,
    "margin_expansion": margin_expansion,
}

# ============================================================
# INPUT
# ============================================================

st.subheader("Enter stocks")
ticker_text = st.text_area("Ticker symbols", value="MSFT COR VRSK KO AAPL GOOGL META AMZN", height=100,
                           help="Separate tickers using spaces, commas, or new lines.")
run_screener = st.button("🚀 Run Screener", type="primary", width="stretch")

# ============================================================
# RESULTS
# ============================================================


def style_table(df, fmt):
    return df.style.format(fmt, na_rep="—")


def sensitivity_table(rows, row_key, col_key, val_key):
    if not rows:
        return None
    t = pd.DataFrame(rows).pivot_table(index=row_key, columns=col_key, values=val_key, aggfunc="first")
    t.index = [f"{x:.1%}" for x in t.index]
    t.columns = [f"{x:.2f}" if col_key == "ω" else f"{x:.1%}" for x in t.columns]
    return t.style.format("{:,.2f}", na_rep="—")


def classify_verdict(row, min_ri_vp, min_dcf_vp, min_cons_vp, strong_ri_vp, strong_max_omega, max_accruals, max_flev):
    """Conservative verdict logic: missing required quality data can never pass as a candidate."""
    ri_flag = row.get("RI Valid", False)
    ri_ok = bool(ri_flag) if pd.notna(ri_flag) else False
    dcf_vp, ri_vp, cons = row["DCF V/P"], row["RI V/P"], row["RI Conservative V/P"]
    iw = row["Implied Omega"]
    if pd.isna(dcf_vp) or pd.isna(row["Accruals"]) or pd.isna(row["FLEV"]):
        return "INSUFFICIENT DATA"
    dcf_flag = row.get("DCF Reliable", False)
    dcf_reliable = bool(dcf_flag) if pd.notna(dcf_flag) else False
    common = dcf_reliable and dcf_vp >= min_dcf_vp and row["Accruals"] <= max_accruals and row["FLEV"] <= max_flev
    passes = bool(
        ri_ok and common and pd.notna(ri_vp) and ri_vp >= min_ri_vp
        and pd.notna(cons) and cons >= min_cons_vp
        and pd.notna(row["ROE"]) and pd.notna(row["Cost of Equity"])
        and row["ROE"] > row["Cost of Equity"]
    )
    quality = row.get("Valuation Quality", "REVIEW")
    if passes and quality != "OK":
        return "WATCH"
    if passes and ri_vp >= strong_ri_vp and pd.notna(iw) and iw <= strong_max_omega:
        return "STRONG CANDIDATE"
    if passes:
        return "CANDIDATE"
    if (ri_ok and pd.notna(ri_vp) and ri_vp >= min_ri_vp) or dcf_vp >= min_dcf_vp:
        return "WATCH"
    return "NO"


def render_results(df):
    if df is None or df.empty:
        st.info("Enter tickers above and click **Run Screener**.")
        return

    df = df.copy()
    df["Verdict"] = df.apply(lambda row: classify_verdict(
        row, min_ri_vp, min_dcf_vp, min_cons_vp, strong_ri_vp,
        strong_max_omega, max_accruals, max_flev), axis=1)
    df = df.sort_values(["DCF V/P", "RI V/P"], ascending=False, na_position="last").reset_index(drop=True)
    st.session_state.screen_results = df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks analyzed", len(df))
    c2.metric("Candidates", int(df["Verdict"].isin(["STRONG CANDIDATE", "CANDIDATE"]).sum()))
    c3.metric("Highest DCF V/P", f"{df['DCF V/P'].max():.2f}x")
    c4.metric("Median DCF V/P", f"{df['DCF V/P'].median():.2f}x")

    st.subheader("📈 Screening results")
    cols = ["Ticker", "Currency", "Price", "P/B", "RI P/B", "ROE", "Cost of Equity", "WACC", "Start Growth", "Market-Implied Growth",
            "Earnings CAGR", "FCF Yield", "RNOA", "NBC", "FLEV", "Spread", "Accruals", "Implied Omega", "RI Valid", "DCF Reliable",
            "ROE Source", "Valuation Quality", "RI ROE Input", "RI ROE Applied", "RI Terminal Payout", "RI Value", "RI V/P", "RI Conservative Value", "RI Conservative V/P", "DCF Value", "DCF V/P", "DCF Bear V/P", "DCF Bull V/P",
            "Net Debt (B)", "Shares Outstanding", "RI Terminal Value Share", "RI Terminal ROE", "DCF Terminal Value Share", "DCF Terminal Growth", "DCF Terminal ROIC", "Verdict"]
    display = df[cols].copy()
    display["Market-Implied Growth"] = display["Market-Implied Growth"].replace(float("inf"), NAN)
    fmt = {
        "Price": "{:.2f}", "P/B": "{:.2f}x", "RI P/B": "{:.2f}x", "ROE": "{:.1%}", "Cost of Equity": "{:.1%}", "WACC": "{:.1%}",
        "Start Growth": "{:.1%}", "Market-Implied Growth": "{:.1%}", "Earnings CAGR": "{:.1%}", "FCF Yield": "{:.1%}",
        "RNOA": "{:.1%}", "NBC": "{:.1%}", "FLEV": "{:.2f}", "Spread": "{:.1%}", "Accruals": "{:.1%}",
        "Implied Omega": "{:.2f}", "RI Value": "{:,.2f}", "RI V/P": "{:.2f}x",
        "RI Conservative Value": "{:,.2f}", "RI Conservative V/P": "{:.2f}x", "DCF Value": "{:,.2f}",
        "DCF V/P": "{:.2f}x", "DCF Bear V/P": "{:.2f}x", "DCF Bull V/P": "{:.2f}x", "Net Debt (B)": "{:,.1f}B", "Shares Outstanding": "{:,.0f}",
        "RI ROE Input": "{:.1%}", "RI ROE Applied": "{:.1%}", "RI Terminal Payout": "{:.1%}",
        "RI Terminal Value Share": "{:.1%}", "RI Terminal ROE": "{:.1%}",
        "DCF Terminal Value Share": "{:.1%}", "DCF Terminal Growth": "{:.1%}", "DCF Terminal ROIC": "{:.1%}",
    }
    st.dataframe(style_table(display, fmt), width="stretch", hide_index=True)
    st.caption("Price and per-share values are in each security's displayed currency. Market-Implied Growth = starting revenue growth the DCF needs to equal today's price. "
               "Terminal Value Share is discounted terminal value divided by enterprise value for DCF and by equity value for RI; it is not the perpetual growth rate. "
               "RI columns are blank where RI is not meaningful (see warnings). Missing accruals or FLEV prevent a candidate verdict.")

    st.download_button("⬇️ Download results as CSV", display.to_csv(index=False).encode("utf-8"),
                       file_name="screen_results.csv", mime="text/csv")

    st.divider()
    st.subheader("🔎 Company deep dive")
    sel_t = st.selectbox("Select a company", df["Ticker"].tolist())
    sel = df.loc[df["Ticker"] == sel_t].iloc[0]
    base, dcf, data = sel["_base"], sel["_dcf"], sel["_data"]

    left, right = st.columns(2)
    with left:
        st.markdown("### Valuation")
        currency = data.get("currency", "unknown")
        st.write(f"**Market price:** {sel['Price']:.2f} {currency}")
        if pd.isna(sel["DCF V/P"]):
            st.write("**DCF value / V/P:** n/m (negative equity value)")
        else:
            st.write(f"**DCF value / V/P:** {sel['DCF Value']:.2f} {currency} / {sel['DCF V/P']:.2f}x")
        mig = sel["Market-Implied Growth"]
        mig_txt = ">50% (beyond model range)" if pd.notna(mig) and mig == float("inf") else (
            "≤ -10% (cheap even with shrinking sales)" if pd.notna(mig) and mig <= -0.10 else fmt_pct(mig))
        st.write(f"**Market-implied starting growth:** {mig_txt} (model start: {fmt_pct(sel['Start Growth'])})")
        st.write(f"**DCF range (bear / base / bull V/P):** {fmt_num(sel['DCF Bear V/P'])}x / {fmt_num(sel['DCF V/P'])}x / {fmt_num(sel['DCF Bull V/P'])}x")
        if sel["RI Valid"]:
            st.write(f"**RI value / V/P:** {sel['RI Value']:.2f} {currency} / {sel['RI V/P']:.2f}x")
            st.write(f"**Conservative RI value / V/P:** {sel['RI Conservative Value']:.2f} {currency} / {sel['RI Conservative V/P']:.2f}x")
        else:
            st.write("**RI value:** not meaningful for this company")
        st.write(f"**Reported P/B:** {fmt_num(sel['P/B'])}x")
        st.write(f"**RI adjusted P/B:** {fmt_num(sel['RI P/B'])}x")
        st.write(f"**PV of terminal value / total value (DCF / RI):** {fmt_pct(sel['DCF Terminal Value Share'])} / {fmt_pct(sel['RI Terminal Value Share'])}")
        st.write(f"**DCF terminal assumptions (growth / ROIC):** {fmt_pct(sel['DCF Terminal Growth'])} / {fmt_pct(sel['DCF Terminal ROIC'])}")
    with right:
        st.markdown("### Diagnostics")
        for label, key, f in [("ROE", "ROE", fmt_pct), ("RNOA", "RNOA", fmt_pct), ("NBC", "NBC", fmt_pct),
                              ("FLEV", "FLEV", fmt_num), ("Spread", "Spread", fmt_pct), ("Accruals", "Accruals", fmt_pct),
                              ("Cost of equity", "Cost of Equity", fmt_pct), ("WACC", "WACC", fmt_pct)]:
            st.write(f"**{label}:** {f(sel[key])}")
        st.write(f"**DCF input quality:** {'screening-grade inputs' if sel.get('DCF Reliable', False) else 'provisional — review warnings'}")
        st.write(f"**ROE source:** {sel['ROE Source']}")
        st.write(f"**Shares source:** {data.get('shares_source', 'unavailable')}")
        st.write(f"**Price source:** {data.get('price_source', 'unavailable')}")
        st.write(f"**Financial currency:** {data.get('currency', 'unknown')}")
        st.write(f"**RI ROE input / applied:** {fmt_pct(sel['RI ROE Input'])} / {fmt_pct(sel['RI ROE Applied'])}")
        st.write(f"**RI terminal payout:** {fmt_pct(sel['RI Terminal Payout'])}")
        st.write(f"**Terminal ROIC used:** {fmt_pct(dcf['term_roic'])}")
        st.write(f"**Net debt:** {sel['Net Debt (B)']:,.1f} {currency}B")

    v = sel["Verdict"]
    if v == "STRONG CANDIDATE":
        st.success("STRONG CANDIDATE: passes the stronger RI and DCF checks.")
    elif v == "CANDIDATE":
        st.success("CANDIDATE: passes the current screening rules.")
    elif v == "WATCH":
        st.info("WATCH: valuation is interesting but does not pass all rules.")
    else:
        st.warning("NO: does not pass the screening rules.")

    mig, g0 = sel["Market-Implied Growth"], sel["Start Growth"]
    if pd.isna(mig):
        read = "The reverse DCF could not be solved for this company."
    elif mig == float("inf"):
        read = "The price needs more than 50% starting revenue growth, so it is far above what this DCF can justify."
    elif mig <= -0.10:
        read = "The price is below DCF value even if revenue shrinks 10% a year."
    elif mig > g0 + 0.02:
        read = f"The market prices in about {mig:.1%} starting growth versus {g0:.1%} in the model, so the stock looks expensive on these assumptions."
    elif mig < g0 - 0.02:
        read = f"The market needs only {mig:.1%} starting growth versus {g0:.1%} in the model, so the stock looks cheap on these assumptions."
    else:
        read = f"The market prices in about {mig:.1%} starting growth, close to the {g0:.1%} in the model, so it looks roughly fairly priced."
    st.info("**Valuation read:** " + read)

    st.markdown(f"### DCF bridge ({data.get('currency', 'unknown')} billions except per share)")
    bridge = pd.DataFrame([
        ("PV of explicit-period FCFF", dcf["pv_explicit"] / 1e9),
        ("PV of terminal value", dcf["pv_terminal"] / 1e9),
        ("Enterprise value", dcf["enterprise_value"] / 1e9),
        ("Less: net debt (incl. minority interest)", -dcf["net_debt"] / 1e9),
        ("Equity value", dcf["equity_value"] / 1e9),
        (f"Value per share ({data.get('currency', 'unknown')})", dcf["value_per_share"]),
        (f"Market price ({data.get('currency', 'unknown')})", sel["Price"]),
    ], columns=["Item", "Amount"])
    st.dataframe(bridge.style.format({"Amount": "{:,.2f}"}), width="stretch", hide_index=True)

    if sel["warnings"]:
        with st.expander("⚠️ Data-quality warnings", expanded=not sel["RI Valid"]):
            for w in sel["warnings"]:
                st.warning(w)

    if base:
        st.subheader("📊 Residual-income forecast")
        st.dataframe(style_table(pd.DataFrame(base["forecast"]), {
            "Beginning BV": "{:,.0f}", "ROE": "{:.1%}", "Earnings": "{:,.0f}",
            "Net Payout": "{:,.0f}", "Residual Income": "{:,.0f}", "Ending BV": "{:,.0f}",
            "PV Residual Income": "{:,.0f}"}), width="stretch", hide_index=True)

    st.subheader("💸 FCFF forecast")
    used, src = sel["_dcf_inputs"], sel["_dcf_sources"]
    st.caption("Inputs: " + ", ".join(f"{DCF_LABELS[k]} {used[k]:.1%} ({src[k]})" for k in DCF_LABELS)
               + f", margin path {used['margin']:.1%} to {used['margin_end']:.1%}, terminal ROIC {used['term_roic']:.1%}")
    st.dataframe(style_table(pd.DataFrame(dcf["forecast"]), {
        "Growth": "{:.1%}", "Revenue": "{:,.0f}", "EBIT": "{:,.0f}", "NOPAT": "{:,.0f}", "D&A": "{:,.0f}",
        "CapEx": "{:,.0f}", "ΔNWC": "{:,.0f}", "Net Investment": "{:,.0f}", "FCFF": "{:,.0f}",
        "PV FCFF": "{:,.0f}"}), width="stretch", hide_index=True)

    with st.expander("📊 Sensitivity analysis"):
        g_term, years, g0 = P["terminal_growth"], sel["_years"], sel["_g0"]
        if base:
            st.markdown("**RI value per share: cost of equity vs ω**")
            rows = []
            for dk in (-0.02, -0.01, 0.0, 0.01, 0.02):
                for w in (0.4, 0.6, 0.8, 0.9):
                    try:
                        k = sel["_ke"] + dk
                        val = residual_income_valuation(sel["_data"], k, w, growth_path(g0, g_term, years), g_term)["value_per_share"]
                        rows.append({"Cost of equity": k, "ω": w, "v": val})
                    except Exception:
                        pass
            t = sensitivity_table(rows, "Cost of equity", "ω", "v")
            if t is not None:
                st.dataframe(t, width="stretch")
        st.markdown("**DCF value per share: WACC vs terminal growth**")
        rows = []
        for dw in (-0.01, 0.0, 0.01, 0.02):
            for g in (0.015, 0.025, 0.035):
                try:
                    w = sel["_wacc"] + dw
                    val = dcf_valuation(data, w, growth_path(g0, g, years), g, **sel["_dcf_inputs"])["value_per_share"]
                    rows.append({"WACC": w, "Terminal g": g, "v": val})
                except Exception:
                    pass
        t = sensitivity_table(rows, "WACC", "Terminal g", "v")
        if t is not None:
            st.dataframe(t, width="stretch")

    with st.expander("📚 How to read this"):
        st.markdown(
            """
**Residual income:** value = book equity + PV of forecast residual income. This is an ROE/book-equity model; it does not use the DCF revenue-growth path to forecast earnings, so the RI forecast deliberately does not display that revenue-growth path. Forecast book equity rolls forward as beginning book + earnings − modeled payouts. RI is disabled when payout data or defensible ROE is unavailable, or when payout is outside 0–100%. In the terminal steady state, payout is set to 1 − terminal growth / terminal ROE so retained earnings imply the stated book-equity growth. This remains approximate because Yahoo summaries do not provide a full clean-surplus equity roll-forward or other-comprehensive-income reconciliation.

**DCF:** FCFF = NOPAT − (CapEx − D&A + change in operating NWC). Net investment can be negative in a cash-release year; investigate whether that is sustainable rather than flooring it to zero. Revenue growth fades linearly to terminal growth; terminal FCFF = NOPAT × (1 + g) × (1 − g/ROIC), so terminal reinvestment matches growth. “DCF Terminal Value Share” is discounted terminal value divided by enterprise value, not the terminal growth rate. Missing cash or debt data disables DCF. If minority interest is not reported, the model assumes zero and warns; verify this when material.

**Input-quality gate:** a DCF estimate can be displayed for exploration when an assumption is substituted, clipped, unusual, or unavailable, but it is marked provisional and cannot qualify a stock as a candidate. RI is disabled when its book-equity denominator, ROE, payout bridge, or common-equity basis is not defensible.

**Gaps vs price:** a V/P well below 1 means the market is paying for more growth, higher returns or a lower discount rate than the model assumes. Market-Implied Growth shows how much growth that is.
"""
        )


# ============================================================
# RUN
# ============================================================

if run_screener:
    if P["terminal_growth"] < -0.10 or P["terminal_growth"] > 0.04:
        st.error("Terminal growth is outside the supported model range.")
        st.stop()

    symbols = list(dict.fromkeys(x.upper() for x in ticker_text.replace(",", " ").split() if x.strip()))
    if not symbols:
        st.warning("Enter at least one ticker.")
        st.stop()

    progress = st.progress(0)
    results, errors = [], []
    for i, sym in enumerate(symbols):
        try:
            d = fetch_data(sym)
            skip_company, reason = should_skip_company(d)
            if skip_company:
                errors.append(f"{sym}: skipped; {reason}.")
            else:
                results.append(analyze_company(d, P))
        except Exception as e:
            errors.append(f"{sym}: {e}")
        progress.progress((i + 1) / len(symbols))

    if errors:
        with st.expander(f"⚠️ {len(errors)} stock(s) skipped"):
            for e in errors:
                st.write(e)
    if not results:
        st.error("No stocks could be analyzed.")
        st.stop()

    df = pd.DataFrame(results)
    skip = {"Ticker", "Currency", "warnings", "RI Valid", "ROE Source", "Valuation Quality"}
    num_cols = [c for c in df.columns if not c.startswith("_") and c not in skip]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["RI Valid"] = df["RI Valid"].map(lambda value: bool(value) if pd.notna(value) else False)
    df["DCF Reliable"] = df["DCF Reliable"].map(lambda value: bool(value) if pd.notna(value) else False)
    st.session_state.screen_results = df
    st.session_state.screen_assumptions = dict(P)
    render_results(df)
else:
    if st.session_state.screen_results is not None:
        if st.session_state.screen_assumptions != P:
            st.warning("Assumptions changed. Click **Run Screener** to update valuations; "
                       "verdict rules apply immediately but the values are from the last run.")
        render_results(st.session_state.screen_results)
    else:
        st.info("Enter tickers above and click **Run Screener**.")
