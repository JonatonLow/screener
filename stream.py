
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
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
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


def spend(v):
    v = safe_float(v)
    return None if v is None else abs(v)


def ratio(a, b):
    return None if a is None or b is None or b <= 0 else a / b


def fmt_pct(v):
    return "—" if v is None or pd.isna(v) else f"{v:.1%}"


def fmt_num(v):
    return "—" if v is None or pd.isna(v) else f"{v:.2f}"


# Current yfinance names plus legacy Yahoo row labels used by older responses.
EQUITY = [
    "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest",
    "Total Stockholder Equity", "Total Stockholders Equity", "Total Equity",
]
EBIT_NAMES = ["Operating Income", "EBIT"]


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
        valid = [(i, v) for i, v in enumerate(earn) if v is not None and v > 0]
        if len(valid) >= 3:
            span = valid[-1][0] - valid[0][0]
            if span > 0:
                out["earnings_cagr"] = (valid[0][1] / valid[-1][1]) ** (1 / span) - 1
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
    if price is None:
        try:
            h = t.history(period="5d")
            if not h.empty:
                price = float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass

    shares = info.get("sharesOutstanding")
    if shares is None:
        shares = pick(bs, ["Ordinary Shares Number", "Share Issued"], 0)
    if shares is None:
        shares = pick(inc, ["Diluted Average Shares", "Weighted Average Shares Diluted", "Basic Average Shares"], 0)

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
            "debt": pick(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"], col),
            "cash": pick(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial"], col),
            "equity": pick(bs, EQUITY, col),
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
    pos_revs = [r for r in revs if r and r > 0][:4]
    rev_cagr = (pos_revs[0] / pos_revs[-1]) ** (1 / (len(pos_revs) - 1)) - 1 if len(pos_revs) >= 2 else None

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
    payout_ratio = None
    if ni_latest is not None and ni_latest > 0 and (dividends_latest is not None or buybacks_latest is not None):
        payout_ratio = min(1.0, max(0.0, ((dividends_latest or 0.0) + (buybacks_latest or 0.0)) / ni_latest))

    return {
        "ticker": symbol.upper(),
        "price": safe_float(price),
        "shares": safe_float(shares),
        "sector": info.get("sector", "") or "",
        "beta": safe_float(info.get("beta")),
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

    oi_after_tax = data["ebit"] * (1 - tax_rate)
    nfe = abs(data["interest"] or 0) * (1 - tax_rate)

    rnoa = oi_after_tax / avg_noa if avg_noa is not None and avg_noa > 0 and data["ebit"] is not None else None
    nbc = nfe / avg_nfo if avg_nfo is not None and avg_nfo > 0 else None
    flev = nfo0 / y0["equity"] if nfo0 is not None and y0["equity"] and y0["equity"] > 0 else None
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
    beta = 1.0 if beta is None else beta
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
        return clamp(data["rev_cagr"], -0.02, p["max_start_growth"])
    return p["start_growth"]


# ============================================================
# RESIDUAL INCOME (equity-based, growth-consistent)
# ============================================================


def residual_income_valuation(data, ke, omega, g_path, g_term):
    y0, y1 = data["years"]
    bv0, bv_prev = y0["equity"], y1["equity"]
    ni, shares, price = data["ni"], data["shares"], data["price"]

    if bv0 is None or bv0 <= 0:
        raise ValueError("non-positive book equity")
    if bv_prev is None or bv_prev <= 0:
        raise ValueError("non-positive prior book equity")
    if ni is None:
        raise ValueError("missing net income")
    if not shares or shares <= 0 or not price or price <= 0:
        raise ValueError("missing shares or price")
    if ke <= g_term:
        raise ValueError("cost of equity must exceed terminal growth")

    hist_roe = data.get("roe_hist")
    if hist_roe is None:
        hist_roe = ni / bv_prev
    roe_base = clamp(hist_roe, -0.10, 0.40)

    payout_ratio = data.get("payout_ratio")
    if payout_ratio is None:
        payout_ratio = 0.0
    payout_ratio = max(0.0, min(1.0, payout_ratio))
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
        if end <= 0:
            raise ValueError("forecast book equity becomes non-positive under the earnings/payout bridge")
        pv = ri / (1 + ke) ** t
        pv_ri += pv
        forecast.append({"Year": t, "Growth": g, "Beginning BV": beg, "ROE": roe_t, "Earnings": earnings,
                         "Net Payout": payout, "Residual Income": ri, "Ending BV": end, "PV Residual Income": pv})
        bv = end

    n = len(g_path)
    # Terminal RI uses next-period ROE after the same persistence fade, rather than copying the last explicit RI.
    # This is still a simplified perpetuity: Yahoo summary data cannot fully reconcile terminal book growth and payouts.
    terminal_roe = ke + (roe_base - ke) * (omega ** (n + 1))
    terminal_ri = (terminal_roe - ke) * bv
    tv = terminal_ri / (ke - g_term)
    pv_tv = tv / (1 + ke) ** n
    equity = bv0 + pv_ri + pv_tv
    vps = equity / shares
    return {
        "historical_roe": hist_roe,
        "terminal_roe": terminal_roe,
        "payout_ratio": payout_ratio,
        "value_per_share": vps,
        "V/P": vps / price,
        "P/B": price * shares / bv0,
        "terminal_pct": pv_tv / equity if equity > 0 else None,
        "forecast": forecast,
    }


def calculate_implied_omega(data, ke, g_path, g_term, target=1.0):
    def vp(w):
        try:
            v = residual_income_valuation(data, ke, w, g_path, g_term)["V/P"]
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    lo_v, hi_v = vp(0.0), vp(1.0)
    if lo_v is None or hi_v is None:
        return None
    if lo_v >= target:
        return 0.0
    if hi_v < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        v = vp(mid)
        if v is None:
            return None
        lo, hi = (mid, hi) if v < target else (lo, mid)
    return (lo + hi) / 2


# ============================================================
# DCF (FCFF, value-driver terminal value)
# ============================================================


def dcf_valuation(data, wacc, g_path, g_term, margin, tax, capex_ratio, dep_ratio, nwc_ratio, term_roic, margin_end=None):
    if wacc <= g_term:
        raise ValueError("WACC must exceed terminal growth")
    if term_roic is None or term_roic <= g_term or term_roic <= 0:
        raise ValueError("terminal ROIC must be positive and exceed terminal growth")
    if any(v is None for v in (margin, tax, capex_ratio, dep_ratio, nwc_ratio)):
        raise ValueError("one or more DCF operating assumptions are missing")
    y0 = data["years"][0]
    base_rev, shares, price = data["revenue"], data["shares"], data["price"]
    if not base_rev or base_rev <= 0:
        raise ValueError("missing revenue")
    if not shares or shares <= 0 or not price or price <= 0:
        raise ValueError("missing shares or price")
    if y0["debt"] is None:
        raise ValueError("total debt unavailable")
    net_debt = y0["debt"] - (y0["cash"] or 0.0) + (y0["minority"] or 0.0)

    rev = base_rev
    nwc_prev = nwc_ratio * base_rev  # same ratio as forecast, so no artificial year-1 jump
    pv_fcff, forecast, nopat = 0.0, [], 0.0
    for t, g in enumerate(g_path, 1):
        rev *= 1 + g
        m_t = margin + ((margin if margin_end is None else margin_end) - margin) * t / len(g_path)  # linear ramp
        ebit = rev * m_t
        nopat = ebit * (1 - tax)
        dep, nwc = rev * dep_ratio, rev * nwc_ratio
        # capex/revenue fades toward ~1.1x D&A by the last forecast year (matches the terminal reinvestment logic)
        cap_target = dep_ratio * 1.1
        cap_r = capex_ratio + (cap_target - capex_ratio) * (t / len(g_path)) if capex_ratio > cap_target else capex_ratio
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
    return {
        "net_debt": net_debt, "enterprise_value": ev, "equity_value": equity,
        "value_per_share": vps, "V/P": vps / price, "terminal_pct": pv_tv / ev if ev > 0 else None,
        "term_roic": term_roic, "forecast": forecast,
        "pv_explicit": pv_fcff, "pv_terminal": pv_tv, "term_fcff": term_fcff,
    }


def implied_start_growth(data, wacc, g_term, years, dcf_inputs):
    """Starting revenue growth the market price implies (reverse DCF).
    Returns inf if the price needs more than 50% growth, -0.10 if it is cheap even at -10% growth."""
    def vp(g0):
        return dcf_valuation(data, wacc, growth_path(g0, g_term, years), g_term, **dcf_inputs)["V/P"]

    try:
        lo, hi = -0.10, 0.50
        if vp(lo) >= 1.0:
            return lo
        if vp(hi) < 1.0:
            return float("inf")
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if vp(mid) < 1.0 else (lo, mid)
        return (lo + hi) / 2
    except Exception:
        return None


DCF_LIMITS = {"margin": (-0.20, 0.60), "tax": (0.10, 0.30), "capex_ratio": (0.0, 0.30),
              "dep_ratio": (0.0, 0.20), "nwc_ratio": (-0.05, 0.40)}
DCF_LABELS = {"margin": "EBIT margin (3y avg)", "tax": "tax rate", "capex_ratio": "CapEx / revenue",
              "dep_ratio": "D&A / revenue", "nwc_ratio": "operating NWC / revenue"}
FALLBACK_KEYS = {"margin": "fb_margin", "tax": "fb_tax", "capex_ratio": "fb_capex",
                 "dep_ratio": "fb_dep", "nwc_ratio": "fb_nwc"}


def resolve_dcf_inputs(data, p):
    values, sources = {}, {}
    for key, fb in FALLBACK_KEYS.items():
        cv = data["company_ratios"].get(key)
        if p["override"] or cv is None:
            values[key], sources[key] = p[fb], "sidebar"
        else:
            values[key], sources[key] = clamp(cv, *DCF_LIMITS[key]), "company"
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
    is_growth = g0 >= p["growth_threshold"]  # growth mode: longer fade, latest margin, optional margin ramp
    if is_growth:
        years = p["growth_years"]
    g_path = growth_path(g0, g_term, years)

    warnings = []
    if data.get("roe_source") == "Yahoo-reported ROE fallback":
        warnings.append("Historical ROE could not be reconstructed from reported balance-sheet history; using Yahoo-reported ROE as a fallback. Verify the equity denominator before relying on RI.")
    if data.get("payout_ratio") is None:
        warnings.append("Dividend/buyback payout data unavailable; RI book-equity bridge assumes zero payout. Treat RI as lower-confidence.")
    eq0, eq1, ni_adj = y0["equity"], y1["equity"], data["ni"]
    adjusted = False
    if p["tangible_book"]:
        gw0, gw1 = y0["goodwill"], y1["goodwill"]
        if gw0 is not None and gw1 is not None and eq0 and eq1:
            eq0, eq1, adjusted = eq0 - gw0, eq1 - gw1, True
            warnings.append("RI uses tangible book (goodwill and intangibles removed from equity).")
        else:
            warnings.append("Goodwill/intangibles not reported; RI uses reported book equity.")
    rd = [v for v in (data.get("rd") or []) if v is not None and v > 0]
    if p["capitalize_rd"] and len(rd) >= 2 and eq0 and eq1 and ni_adj is not None:
        life, tx = 5, data["effective_tax_rate"] if data["effective_tax_rate"] is not None else 0.21
        # straight-line amortization over `life` years; only the years Yahoo reports (usually 3-4) are available
        a0 = sum(v * max(0.0, 1 - (k + 0.5) / life) for k, v in enumerate(rd)) * (1 - tx)
        a1 = sum(v * max(0.0, 1 - (k - 0.5) / life) for k, v in enumerate(rd) if k >= 1) * (1 - tx)
        amort0 = sum(rd[k] / life for k in range(1, min(len(rd), life + 1)))
        eq0, eq1, ni_adj, adjusted = eq0 + a0, eq1 + a1, ni_adj + (rd[0] - amort0) * (1 - tx), True
        warnings.append("RI capitalizes R&D (5-year life, after tax) into book equity and earnings; approximate, based on the R&D years Yahoo reports.")
    ri_data = data
    if adjusted:
        ri_data = {**data, "years": [{**y0, "equity": eq0}, {**y1, "equity": eq1}], "ni": ni_adj,
                   "roe_hist": (ni_adj / eq1) if ni_adj is not None and eq1 > 0 else None}
    base = cons = implied_omega = None
    try:
        base = residual_income_valuation(ri_data, ke, p["omega"], g_path, g_term)
        # Conservative case: faster fade if ROE > ke, slower recovery if ROE < ke
        cons_omega = p["omega"] * 0.75 if base["historical_roe"] >= ke else min(1.0, p["omega"] * 1.25)
        cons = residual_income_valuation(ri_data, ke + 0.01, cons_omega, g_path, g_term)
        implied_omega = calculate_implied_omega(ri_data, ke, g_path, g_term)
    except ValueError as e:
        warnings.append(f"Residual income unavailable: {e}."
                        + (" Adjusted book is negative or missing; untick the tangible-book option to use reported book." if ri_data is not data else ""))

    ri_valid = base is not None
    eq_rep = y0["equity"]
    pb_rep = data["price"] * data["shares"] / eq_rep if eq_rep and eq_rep > 0 else NAN
    roe_rep = data["roe_hist"] if data.get("roe_hist") is not None else (base["historical_roe"] if base else NAN)
    if ri_valid and (pb_rep > p["max_pb"] or roe_rep > p["max_roe"]):
        ri_valid, implied_omega = False, None
        warnings.append(
            f"RI not meaningful: P/B {pb_rep:.1f}x and ROE {roe_rep:.0%} exceed limits. "
            "Book equity is too small relative to earnings power (buybacks, expensed intangibles); verdict uses DCF only."
        )

    dcf_inputs, dcf_sources = resolve_dcf_inputs(data, p)
    lo = wacc + 0.01
    rnoa = operating["RNOA"]
    high_return = rnoa is not None and rnoa >= 0.25
    hi = max(p["quality_roic_cap"] if high_return else p["roic_cap"], lo)
    if is_growth and not p["override"]:
        if data["ebit"] is not None and data["revenue"]:
            dcf_inputs["margin"] = clamp(data["ebit"] / data["revenue"], *DCF_LIMITS["margin"])
    dcf_inputs["margin_end"] = clamp(dcf_inputs["margin"] + (p["margin_expansion"] if is_growth else 0.0), -0.20, 0.60)
    dcf_inputs["term_roic"] = clamp(rnoa if rnoa is not None else lo + 0.02, lo, hi)
    dcf = dcf_valuation(data, wacc, g_path, g_term, **dcf_inputs)
    implied_g = implied_start_growth(data, wacc, g_term, years, dcf_inputs)
    dcf_ok = dcf["value_per_share"] > 0

    def scenario_vp(g_mult, g_add, margin_add, roic_add):
        try:
            gs = clamp((g0 * g_mult if g0 > 0 else g0) + g_add, -0.10, 0.30)
            inp = dict(dcf_inputs)
            inp["margin_end"] = clamp(inp["margin_end"] + margin_add, -0.20, 0.60)
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
            f"and the DCF uses the latest EBIT margin ({dcf_inputs['margin']:.1%}) ending at {dcf_inputs['margin_end']:.1%}.")
    if high_return:
        warnings.append(f"High-return business (RNOA {rnoa:.0%}): terminal ROIC cap raised to {p['quality_roic_cap']:.0%}.")
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
        warnings.append("Latest EBIT margin differs >40% from the 3-year average: cyclical or volatile earnings; DCF uses the average.")

    if data.get("effective_tax_rate") is None:
        warnings.append("Effective tax rate unavailable; model uses the configured DCF tax fallback. Review tax-normalization assumptions.")
    if not p["override"]:
        for k, s in dcf_sources.items():
            if s == "sidebar":
                warnings.append(f"Company {DCF_LABELS[k]} unavailable; DCF uses the sidebar fallback.")
    if dcf_inputs["margin"] < 0.03:
        warnings.append("Thin EBIT margin: DCF is highly sensitive to small margin changes.")
    if dcf["terminal_pct"] is not None and dcf["terminal_pct"] > 0.80:
        warnings.append(
            f"Present value of terminal value is {dcf['terminal_pct']:.0%} of enterprise value; "
            "the estimate is highly sensitive to terminal assumptions. Consider a longer explicit forecast."
        )
    dil = data.get("diluted_shares")
    if dil and data["shares"] and abs(dil - data["shares"]) / data["shares"] > 0.05:
        warnings.append("Share count differs >5% from diluted average; per-share values may be unreliable.")
    if ri_valid and dcf_ok and base is not None and base["value_per_share"] > 0 and dcf["value_per_share"] > 0:
        gap = abs(base["value_per_share"] - dcf["value_per_share"]) / max(base["value_per_share"], dcf["value_per_share"])
        if gap >= 0.50:
            warnings.append(f"Model disagreement: DCF and RI values differ by {gap:.0%} of the higher estimate. Review margins, reinvestment, book equity, payout and terminal assumptions; values are not averaged.")

    def ri(res, key):
        return res[key] if (ri_valid and res is not None) else NAN

    eq0 = y0["equity"]
    return {
        "Ticker": data["ticker"],
        "Price": data["price"],
        "P/B": data["price"] * data["shares"] / eq0 if eq0 and eq0 > 0 else NAN,
        "ROE": data.get("roe_hist") if data.get("roe_hist") is not None else (base["historical_roe"] if base else NAN),
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
        "RI Value": ri(base, "value_per_share"),
        "RI V/P": ri(base, "V/P"),
        "RI Conservative Value": ri(cons, "value_per_share"),
        "RI Conservative V/P": ri(cons, "V/P"),
        "DCF Value": dcf["value_per_share"] if dcf_ok else NAN,
        "DCF V/P": dcf["V/P"] if dcf_ok else NAN,
        "DCF Bear V/P": bear_vp,
        "DCF Bull V/P": bull_vp,
        "Net Debt ($B)": dcf["net_debt"] / 1e9,
        "Shares Outstanding": data["shares"],
        "RI Terminal Value Share": base["terminal_pct"] if base else NAN,
        "RI Terminal ROE": base["terminal_roe"] if base else NAN,
        "DCF Terminal Value Share": dcf["terminal_pct"],
        "DCF Terminal Growth": g_term,
        "DCF Terminal ROIC": dcf["term_roic"],
        "warnings": warnings,
        "_base": base, "_dcf": dcf, "_dcf_inputs": dcf_inputs, "_dcf_sources": dcf_sources,
        "_data": ri_data, "_years": years, "_g0": g0, "_ke": ke, "_wacc": wacc,
    }


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
    max_start_growth = st.number_input("Max starting growth", 0.00, 0.40, 0.15, 0.01, format="%.2f")
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
    fb_margin = st.number_input("EBIT margin", -0.20, 0.50, 0.15, 0.005, format="%.3f")
    fb_tax = st.number_input("Tax rate", 0.00, 0.40, 0.21, 0.005, format="%.3f")
    fb_capex = st.number_input("CapEx / revenue", 0.00, 0.30, 0.04, 0.005, format="%.3f")
    fb_dep = st.number_input("D&A / revenue", 0.00, 0.20, 0.035, 0.005, format="%.3f")
    fb_nwc = st.number_input("NWC / revenue", -0.05, 0.40, 0.05, 0.005, format="%.3f")
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


def render_results(df):
    if df is None or df.empty:
        st.info("Enter tickers above and click **Run Screener**.")
        return

    def verdict(row):
        ri_ok = bool(row["RI Valid"])
        dcf_vp, ri_vp, cons = row["DCF V/P"], row["RI V/P"], row["RI Conservative V/P"]
        iw = row["Implied Omega"]
        common = (
            dcf_vp >= min_dcf_vp
            and (pd.isna(row["Accruals"]) or row["Accruals"] <= max_accruals)
            and (pd.isna(row["FLEV"]) or row["FLEV"] <= max_flev)
        )
        if ri_ok:
            passes = (
                common and pd.notna(ri_vp) and ri_vp >= min_ri_vp
                and pd.notna(cons) and cons >= min_cons_vp
                and pd.notna(row["ROE"]) and row["ROE"] > row["Cost of Equity"]
            )
        else:
            passes = False
        if passes and ri_ok and ri_vp >= strong_ri_vp and pd.notna(iw) and iw <= strong_max_omega:
            return "STRONG CANDIDATE"
        if passes and ri_ok:  # a DCF-only pass is capped at WATCH: no second model to confirm it
            return "CANDIDATE"
        if (ri_ok and ri_vp >= min_ri_vp) or dcf_vp >= min_dcf_vp:
            return "WATCH"
        return "NO"

    df = df.copy()
    df["Verdict"] = df.apply(verdict, axis=1)
    df = df.sort_values(["DCF V/P", "RI V/P"], ascending=False, na_position="last").reset_index(drop=True)
    st.session_state.screen_results = df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks analyzed", len(df))
    c2.metric("Candidates", int(df["Verdict"].isin(["STRONG CANDIDATE", "CANDIDATE"]).sum()))
    c3.metric("Highest DCF V/P", f"{df['DCF V/P'].max():.2f}x")
    c4.metric("Median DCF V/P", f"{df['DCF V/P'].median():.2f}x")

    st.subheader("📈 Screening results")
    cols = ["Ticker", "Price", "P/B", "ROE", "Cost of Equity", "WACC", "Start Growth", "Market-Implied Growth",
            "Earnings CAGR", "FCF Yield", "RNOA", "NBC", "FLEV", "Spread", "Accruals", "Implied Omega", "RI Valid",
            "RI Value", "RI V/P", "RI Conservative Value", "RI Conservative V/P", "DCF Value", "DCF V/P", "DCF Bear V/P", "DCF Bull V/P",
            "Net Debt ($B)", "Shares Outstanding", "RI Terminal Value Share", "RI Terminal ROE", "DCF Terminal Value Share", "DCF Terminal Growth", "DCF Terminal ROIC", "Verdict"]
    display = df[cols].copy()
    display["Market-Implied Growth"] = display["Market-Implied Growth"].replace(float("inf"), NAN)
    fmt = {
        "Price": "${:.2f}", "P/B": "{:.2f}x", "ROE": "{:.1%}", "Cost of Equity": "{:.1%}", "WACC": "{:.1%}",
        "Start Growth": "{:.1%}", "Market-Implied Growth": "{:.1%}", "Earnings CAGR": "{:.1%}", "FCF Yield": "{:.1%}",
        "RNOA": "{:.1%}", "NBC": "{:.1%}", "FLEV": "{:.2f}", "Spread": "{:.1%}", "Accruals": "{:.1%}",
        "Implied Omega": "{:.2f}", "RI Value": "${:,.2f}", "RI V/P": "{:.2f}x",
        "RI Conservative Value": "${:,.2f}", "RI Conservative V/P": "{:.2f}x", "DCF Value": "${:,.2f}",
        "DCF V/P": "{:.2f}x", "DCF Bear V/P": "{:.2f}x", "DCF Bull V/P": "{:.2f}x", "Net Debt ($B)": "${:,.1f}B", "Shares Outstanding": "{:,.0f}",
        "RI Terminal Value Share": "{:.1%}", "RI Terminal ROE": "{:.1%}",
        "DCF Terminal Value Share": "{:.1%}", "DCF Terminal Growth": "{:.1%}", "DCF Terminal ROIC": "{:.1%}",
    }
    st.dataframe(style_table(display, fmt), width="stretch", hide_index=True)
    st.caption("Market-Implied Growth = starting revenue growth the DCF needs to equal today's price. "
               "Terminal Value Share is the present value of terminal value divided by total value; it is not the perpetual growth rate. "
               "RI columns are blank where RI is not meaningful (see warnings).")

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
        st.write(f"**Market price:** ${sel['Price']:.2f}")
        if pd.isna(sel["DCF V/P"]):
            st.write("**DCF value / V/P:** n/m (negative equity value)")
        else:
            st.write(f"**DCF value / V/P:** ${sel['DCF Value']:.2f} / {sel['DCF V/P']:.2f}x")
        mig = sel["Market-Implied Growth"]
        mig_txt = ">50% (beyond model range)" if pd.notna(mig) and mig == float("inf") else (
            "≤ -10% (cheap even with shrinking sales)" if pd.notna(mig) and mig <= -0.10 else fmt_pct(mig))
        st.write(f"**Market-implied starting growth:** {mig_txt} (model start: {fmt_pct(sel['Start Growth'])})")
        st.write(f"**DCF range (bear / base / bull V/P):** {fmt_num(sel['DCF Bear V/P'])}x / {fmt_num(sel['DCF V/P'])}x / {fmt_num(sel['DCF Bull V/P'])}x")
        if sel["RI Valid"]:
            st.write(f"**RI value / V/P:** ${sel['RI Value']:.2f} / {sel['RI V/P']:.2f}x")
            st.write(f"**Conservative RI value / V/P:** ${sel['RI Conservative Value']:.2f} / {sel['RI Conservative V/P']:.2f}x")
        else:
            st.write("**RI value:** not meaningful for this company")
        st.write(f"**P/B:** {fmt_num(sel['P/B'])}x")
        st.write(f"**PV of terminal value / total value (DCF / RI):** {fmt_pct(sel['DCF Terminal Value Share'])} / {fmt_pct(sel['RI Terminal Value Share'])}")
        st.write(f"**DCF terminal assumptions (growth / ROIC):** {fmt_pct(sel['DCF Terminal Growth'])} / {fmt_pct(sel['DCF Terminal ROIC'])}")
    with right:
        st.markdown("### Diagnostics")
        for label, key, f in [("ROE", "ROE", fmt_pct), ("RNOA", "RNOA", fmt_pct), ("NBC", "NBC", fmt_pct),
                              ("FLEV", "FLEV", fmt_num), ("Spread", "Spread", fmt_pct), ("Accruals", "Accruals", fmt_pct),
                              ("Cost of equity", "Cost of Equity", fmt_pct), ("WACC", "WACC", fmt_pct)]:
            st.write(f"**{label}:** {f(sel[key])}")
        st.write(f"**Terminal ROIC used:** {fmt_pct(dcf['term_roic'])}")
        st.write(f"**Net debt:** ${sel['Net Debt ($B)']:,.1f}B")

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

    st.markdown("### DCF bridge ($B except per share)")
    bridge = pd.DataFrame([
        ("PV of explicit-period FCFF", dcf["pv_explicit"] / 1e9),
        ("PV of terminal value", dcf["pv_terminal"] / 1e9),
        ("Enterprise value", dcf["enterprise_value"] / 1e9),
        ("Less: net debt (incl. minority interest)", -dcf["net_debt"] / 1e9),
        ("Equity value", dcf["equity_value"] / 1e9),
        ("Value per share ($)", dcf["value_per_share"]),
        ("Market price ($)", sel["Price"]),
    ], columns=["Item", "Amount"])
    st.dataframe(bridge.style.format({"Amount": "{:,.2f}"}), width="stretch", hide_index=True)

    if sel["warnings"]:
        with st.expander("⚠️ Data-quality warnings", expanded=not sel["RI Valid"]):
            for w in sel["warnings"]:
                st.warning(w)

    if base:
        st.subheader("📊 Residual-income forecast")
        st.dataframe(style_table(pd.DataFrame(base["forecast"]), {
            "Growth": "{:.1%}", "Beginning BV": "{:,.0f}", "ROE": "{:.1%}", "Earnings": "{:,.0f}",
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
**Residual income:** value = book equity + PV of forecast residual income. Forecast book equity rolls forward as beginning book + earnings − modeled payouts; payouts are estimated from the latest reported dividends and buybacks when available. ROE fades toward the cost of equity at rate ω. Missing payout data lowers confidence; clean-surplus reconciliation is approximate because Yahoo's summarized statements do not provide a complete equity roll-forward.

**DCF:** FCFF = NOPAT − (CapEx − D&A + change in operating NWC). Net investment can be negative in a cash-release year; investigate whether that is sustainable rather than flooring it to zero. Revenue growth fades linearly to terminal growth; terminal FCFF = NOPAT × (1 + g) × (1 − g/ROIC), so terminal reinvestment matches growth. “DCF Terminal Value Share” is discounted terminal value divided by enterprise value, not the terminal growth rate.

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
            if "financial" in d["sector"].lower():
                errors.append(f"{sym}: skipped, financial companies need a different accounting reformulation.")
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
    skip = {"Ticker", "warnings", "RI Valid"}
    num_cols = [c for c in df.columns if not c.startswith("_") and c not in skip]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["RI Valid"] = df["RI Valid"].astype(bool)
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
