import math
import sys
from types import SimpleNamespace

import pandas as pd

TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    sys.argv = [arg for arg in sys.argv if arg != "--test"]
if not TEST_MODE:
    import streamlit as st
    import yfinance as yf
else:
    # Only the cache decorator is needed while loading pure functions for tests.
    st = SimpleNamespace(cache_data=lambda **kwargs: (lambda fn: fn))

NAN = float("nan")

# ============================================================
# PAGE
# ============================================================

if not TEST_MODE:
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


def lease_for_period(balance, col=0):
    """Lease liabilities as Yahoo reports them (finance + operating), or None if absent."""
    total = pick(balance, ["Capital Lease Obligations"], col)
    if total is not None:
        return total
    long_term = pick(balance, ["Long Term Capital Lease Obligation"], col)
    current = pick(balance, ["Current Capital Lease Obligation"], col)
    if long_term is None and current is None:
        return None
    return (long_term or 0.0) + (current or 0.0)


def _lease_excluded(y, p):
    """Lease amount to strip from debt in the DCF bridge (only when Yahoo's Total Debt includes it)."""
    lease, debt = y.get("lease"), y.get("debt")
    if not p.get("exclude_leases", True) or not y.get("lease_in_debt") or lease is None or debt is None:
        return 0.0
    return min(max(lease, 0.0), max(debt, 0.0))


def total_liabilities_for_period(balance, col=0):
    """Return reported total liabilities, or derive them from the balance-sheet identity."""
    reported = pick(balance, [
        "Total Liabilities Net Minority Interest", "Total Liabilities",
        "Total Liabilities Excluding Minority Interest",
    ], col)
    if reported is not None:
        return reported, "Yahoo reported total liabilities"

    assets = pick(balance, ["Total Assets"], col)
    # Prefer total equity including non-controlling interests for the accounting identity.
    total_equity = pick(balance, [
        "Total Equity Gross Minority Interest", "Total Stockholders Equity",
        "Total Stockholder Equity", "Stockholders Equity", "Total Equity",
        "Common Stock Equity",
    ], col)
    if assets is not None and total_equity is not None and assets >= total_equity:
        return assets - total_equity, "derived: total assets minus total equity"
    return None, "unavailable"


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


def avg_payout_ratio(income, cash_flow, max_years=4):
    """Pooled (dividends + buybacks) / net income over up to `max_years` matched fiscal years."""
    if income is None or cash_flow is None or income.empty or cash_flow.empty:
        return None
    inc_map = _period_map(income)
    paid = earned = 0.0
    used = 0
    for cf_i, col in enumerate(cash_flow.columns[:max_years]):
        try:
            key = pd.Timestamp(col).normalize()
        except Exception:
            continue
        if key not in inc_map:
            continue
        ni = pick(income, ["Net Income Common Stockholders", "Net Income"], inc_map[key])
        dividends = spend(pick(cash_flow, ["Cash Dividends Paid", "Common Stock Dividend Paid"], cf_i))
        buybacks = spend(pick(cash_flow, ["Repurchase Of Capital Stock", "Common Stock Payments"], cf_i))
        if ni is None or ni <= 0 or dividends is None or buybacks is None:
            continue
        paid += dividends + buybacks
        earned += ni
        used += 1
    return paid / earned if used and earned > 0 else None


def ratio(a, b):
    return None if a is None or b is None or b <= 0 else a / b


def fmt_pct(v):
    return "—" if v is None or pd.isna(v) else f"{v:.1%}"


def fmt_num(v):
    return "—" if v is None or pd.isna(v) else f"{v:.2f}"


def parse_rate_overrides(text):
    """Parse 'SGD=0.028/0.055, HKD=0.04' into {ccy: (risk_free, erp_or_None)}."""
    out = {}
    for item in str(text or "").replace(";", ",").replace("\n", ",").split(","):
        item = item.strip()
        if "=" not in item:
            continue
        ccy, rest = item.split("=", 1)
        parts = rest.split("/")
        try:
            rf_v = float(parts[0])
            erp_v = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        except ValueError:
            continue
        if 0.0 <= rf_v <= 0.15 and (erp_v is None or 0.02 <= erp_v <= 0.12) and ccy.strip():
            out[ccy.strip().upper()] = (rf_v, erp_v)
    return out


def rates_for(data, p):
    """Risk-free rate and ERP for this company's currency (global inputs unless overridden)."""
    ccy = str(data.get("currency", "")).upper()
    rf, erp = p["rf"], p["erp"]
    override = (p.get("rate_overrides") or {}).get(ccy)
    if override:
        rf = override[0]
        erp = override[1] if override[1] is not None else erp
    return rf, erp


def parse_growth_overrides(text):
    """Parse 'MSFT=0.12, KO=4%' into {TICKER: starting_growth} (bounded to -10%..40%)."""
    out = {}
    for item in str(text or "").replace(";", ",").replace("\n", ",").split(","):
        item = item.strip()
        if "=" not in item:
            continue
        tkr, val = item.split("=", 1)
        val = val.strip()
        try:
            g = float(val[:-1]) / 100 if val.endswith("%") else float(val)
        except ValueError:
            continue
        if tkr.strip() and -0.10 <= g <= 0.40:
            out[tkr.strip().upper()] = g
    return out


# Current yfinance names plus legacy Yahoo row labels used by older responses.
EQUITY = [
    "Stockholders Equity", "Common Stock Equity", "Total Stockholder Equity",
    "Total Stockholders Equity", "Total Equity", "Total Equity Gross Minority Interest",
]
EBIT_NAMES = ["Operating Income", "EBIT"]
CURRENT_SHARE_SOURCES = {
    "Yahoo current shares outstanding",
    "Yahoo market cap / price (multi-class cross-check)",
}


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


def _period_map(df):
    """Map normalized fiscal-period end dates to column positions; never infer by position."""
    out = {}
    if df is None or df.empty:
        return out
    for i, col in enumerate(df.columns):
        try:
            key = pd.Timestamp(col).normalize()
            out[key] = i
        except Exception:
            continue
    return out


def aligned_statement_periods(income, balance):
    """Return (income date, income column, matching balance column, prior balance column)."""
    im, bm = _period_map(income), _period_map(balance)
    dates = sorted(set(im).intersection(bm), reverse=True)
    out = []
    for d in dates:
        prior = [x for x in bm if x < d]
        if not prior:
            continue
        # Require the immediately preceding annual balance sheet, not merely any older date.
        prev = max(prior)
        if 300 <= (d - prev).days <= 430:
            out.append((d, im[d], bm[d], bm[prev]))
    return out


def normalized_tax_rate(income, pretax_fallback=None, tax_fallback=None):
    """Median of up to five positive-pretax annual effective tax rates, bounded for safety."""
    rates = []
    pretax = row_values(income, ["Pretax Income", "Income Before Tax"])
    taxes = row_values(income, ["Tax Provision", "Income Tax Expense"])
    for pt, tx in zip(pretax[:5], taxes[:5]):
        if pt is not None and pt > 0 and tx is not None:
            r = tx / pt
            if 0 <= r <= 0.60:
                rates.append(r)
    if rates:
        return float(pd.Series(rates).median()), "median historical effective tax rate"
    if pretax_fallback and pretax_fallback > 0 and tax_fallback is not None:
        r = tax_fallback / pretax_fallback
        if 0 <= r <= 0.60:
            return r, "latest effective tax rate (single-year fallback)"
    return None, "unavailable; manual tax assumption required"


class FinancialDataProvider:
    """Provider boundary: valuation functions consume normalized dictionaries, not yfinance objects."""
    def get_company(self, symbol):
        return fetch_yahoo_company(symbol)


def compute_history(income, balance, cash_flow, price, shares):
    out = {"roe_hist": None, "earnings_cagr": None, "fcf_yield": None,
           "roe_periods": [], "period_alignment_ok": False}
    common_earn = row_values(income, ["Net Income Common Stockholders", "Net Income"])
    aligned = aligned_statement_periods(income, balance)
    roes = []
    for date, ii, bi, pi in aligned[:5]:
        ni = pick(income, ["Net Income Common Stockholders", "Net Income"], ii)
        end_eq = pick(balance, EQUITY, bi)
        prior_eq = pick(balance, EQUITY, pi)
        if ni is not None and end_eq is not None and prior_eq is not None:
            avg_eq = (end_eq + prior_eq) / 2
            if avg_eq > 0:
                roes.append(ni / avg_eq)
                out["roe_periods"].append(str(date.date()))
    if roes:
        out["roe_hist"] = float(pd.Series(roes).median())
        out["period_alignment_ok"] = True
    if common_earn:
        out["earnings_cagr"] = _cagr_between(common_earn, income.columns, positive_only=True)
    cfo = pick(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)
    capex = spend(pick(cash_flow, ["Capital Expenditure", "Capital Expenditures"], 0))
    if cfo is not None and capex is not None and price and shares and price > 0 and shares > 0:
        out["fcf_yield"] = (cfo - capex) / (price * shares)
    return out


# ============================================================
# DATA
# ============================================================


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_company(symbol):
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

    # Reorder income statement and balance sheet onto common fiscal dates before any col-0 picks.
    income_dates, balance_dates = _period_map(inc), _period_map(bs)
    common_dates = sorted(set(income_dates).intersection(balance_dates), reverse=True)
    if len(common_dates) < 2:
        raise ValueError("fewer than two matching annual fiscal periods across income statement and balance sheet; skipped rather than mixing periods")
    inc_cols = {pd.Timestamp(c).normalize(): c for c in inc.columns}
    bs_cols = {pd.Timestamp(c).normalize(): c for c in bs.columns}
    inc = inc.loc[:, [inc_cols[d] for d in common_dates]]
    bs = bs.loc[:, [bs_cols[d] for d in common_dates]]
    if cf is not None and not cf.empty:
        cf_dates = _period_map(cf)
        cf_cols = {pd.Timestamp(c).normalize(): c for c in cf.columns}
        shared_cf_dates = sorted(set(cf_dates).intersection(common_dates), reverse=True)
        if common_dates[0] in cf_dates and shared_cf_dates:
            cf = cf.loc[:, [cf_cols[d] for d in shared_cf_dates]]
        else:
            # Never pair current income/balance data with an older unmatched cash-flow period.
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
    # Multi-class cross-check: market cap covers all share classes while the quoted price is
    # per class, so market cap / price is robust for dual-class names (GOOGL, META, BRK).
    mcap = safe_float(info.get("marketCap"))
    px_val = safe_float(price)
    if mcap is not None and mcap > 0 and px_val is not None and px_val > 0:
        implied_shares = mcap / px_val
        if shares is None or shares <= 0 or abs(implied_shares / shares - 1) > 0.05:
            shares = implied_shares
            shares_source = "Yahoo market cap / price (multi-class cross-check)"
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
        assets = pick(bs, ["Total Assets"], col)
        equity = pick(bs, EQUITY, col)
        liab, liab_source = total_liabilities_for_period(bs, col)
        minority_reported = pick(bs, [
            "Minority Interest", "Minority Interest And Preferred Stock",
            "Noncontrolling Interest", "Non Controlling Interest",
            "Noncontrolling Interests", "Minority Interest in Subsidiaries",
        ], col)
        # Yahoo often omits a separate NCI row even when the company has no material NCI.
        # Use zero only to keep the exploratory DCF bridge calculable, and retain an explicit
        # source flag so the estimate remains provisional until the latest filing is checked.
        minority_assumed_zero = minority_reported is None
        years.append({
            "fiscal_period": str(pd.Timestamp(bs.columns[col]).date()) if col < len(bs.columns) else None,
            "assets": assets,
            "liab": liab,
            "liab_source": liab_source,
            "debt": total_debt_for_period(bs, col),
            # Yahoo's "Total Debt" includes lease obligations; only then can leases be stripped safely.
            "lease": lease_for_period(bs, col),
            "lease_in_debt": pick(bs, ["Total Debt"], col) is not None,
            "nonop_inv": pick(bs, ["Investments And Advances", "Long Term Equity Investment"], col),
            "cash": pick(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial"], col),
            "equity": equity,
            "equity_source": equity_row_source(bs, col),
            "cur_assets": pick(bs, ["Total Current Assets", "Current Assets"], col),
            "cur_liab": pick(bs, ["Total Current Liabilities", "Current Liabilities"], col),
            "cur_debt": pick(bs, ["Current Debt And Capital Lease Obligation", "Current Debt"], col),
            "minority": 0.0 if minority_assumed_zero else minority_reported,
            "minority_source": "not separately reported by Yahoo; assumed zero pending filing verification" if minority_assumed_zero else "Yahoo reported",
            "minority_assumed_zero": minority_assumed_zero,
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

    etr, tax_source = normalized_tax_rate(inc, pretax, tax)

    capex = spend(pick(cf, ["Capital Expenditure", "Capital Expenditures"], 0))
    dep = pick(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation", "Depreciation, Depletion And Amortization"], 0)
    hist = compute_history(inc, bs, cf, price, shares)
    yahoo_roe = safe_float(info.get("returnOnEquity"))
    roe_hist = hist["roe_hist"] if hist["roe_hist"] is not None else yahoo_roe
    roe_source = "historical average equity" if hist["roe_hist"] is not None else (
        "Yahoo-reported ROE fallback" if yahoo_roe is not None else None
    )

    ni_source = "Net Income Common Stockholders" if "Net Income Common Stockholders" in inc.index else ("Net Income" if "Net Income" in inc.index else None)
    ni_latest = pick(inc, ["Net Income Common Stockholders", "Net Income"], 0)
    consolidated_ni = pick(inc, ["Net Income Including Noncontrolling Interests", "Net Income From Continuing Operation Net Minority Interest", "Net Income", "Net Income From Continuing Operations", "Net Income Common Stockholders"], 0)
    dividends_latest = spend(pick(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"], 0))
    buybacks_latest = spend(pick(cf, ["Repurchase Of Capital Stock", "Common Stock Payments"], 0))
    # Pooled multi-year payout is more stable than a single year; fall back to the latest year.
    payout_avg = avg_payout_ratio(inc, cf)
    payout_ratio = payout_avg if payout_avg is not None else payout_ratio_from_cashflows(ni_latest, dividends_latest, buybacks_latest)
    avg_gross_debt = None
    if years[0]["debt"] is not None and years[1]["debt"] is not None:
        avg_gross_debt = (years[0]["debt"] + years[1]["debt"]) / 2
    implied_cost_debt = abs(pick(inc, ["Interest Expense", "Interest Expense Non Operating"], 0)) / avg_gross_debt if avg_gross_debt and avg_gross_debt > 0 and pick(inc, ["Interest Expense", "Interest Expense Non Operating"], 0) is not None else None
    if implied_cost_debt is not None and not (0.01 <= implied_cost_debt <= 0.25):
        implied_cost_debt = None
    aligned = aligned_statement_periods(inc, bs)
    clean_surplus_residual = None
    clean_surplus_ratio = None
    if aligned:
        _, _, b_end, b_prior = aligned[0]
        be = pick(bs, EQUITY, b_end)
        bp = pick(bs, EQUITY, b_prior)
        ni_cs = pick(inc, ["Net Income Common Stockholders", "Net Income"], aligned[0][1])
        if all(v is not None for v in (be, bp, ni_cs, dividends_latest, buybacks_latest)):
            # Approximate common-equity clean-surplus bridge; unexplained OCI/issuance remains visible.
            clean_surplus_residual = be - bp - ni_cs + dividends_latest + buybacks_latest
            denom = max(abs(be), abs(bp), 1.0)
            clean_surplus_ratio = abs(clean_surplus_residual) / denom

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
        "ni": ni_latest,
        "ni_source": ni_source,
        "consolidated_ni": consolidated_ni,
        "tax_source": tax_source,
        "roe_periods": hist.get("roe_periods", []),
        "period_alignment_ok": hist.get("period_alignment_ok", False),
        "cfo": pick(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0),
        "div": pick(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"], 0),
        "buyback": pick(cf, ["Repurchase Of Capital Stock", "Common Stock Payments"], 0),
        "payout_ratio": payout_ratio,
        "clean_surplus_residual": clean_surplus_residual,
        "clean_surplus_ratio": clean_surplus_ratio,
        "rd": row_values(inc, ["Research And Development", "Research Development"]),
        "diluted_shares": pick(inc, ["Diluted Average Shares", "Weighted Average Shares Diluted"], 0),
        "effective_tax_rate": etr,
        "effective_tax_source": tax_source,
        "implied_cost_of_debt": implied_cost_debt,
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


def fetch_data(symbol):
    """Stable provider facade; swap FinancialDataProvider implementation without changing valuation logic."""
    return FinancialDataProvider().get_company(symbol)


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

    tax_rate = data.get("effective_tax_rate")
    if tax_rate is None:
        tax_rate = 0.21  # explicitly surfaced fallback; prevents silent tax inference

    oi_after_tax = data["ebit"] * (1 - tax_rate) if data.get("ebit") is not None else None
    nfe = abs(data["interest"]) * (1 - tax_rate) if data.get("interest") is not None else None

    rnoa = oi_after_tax / avg_noa if avg_noa is not None and avg_noa > 0 and oi_after_tax is not None else None
    nbc = nfe / avg_nfo if avg_nfo is not None and avg_nfo > 0 and nfe is not None else None
    common_equity_available = "Gross Minority Interest" not in str(y0.get("equity_source", ""))
    flev = nfo0 / y0["equity"] if common_equity_available and nfo0 is not None and y0["equity"] and y0["equity"] > 0 else None
    spread = rnoa - nbc if rnoa is not None and nbc is not None else None

    avg_assets = (y0["assets"] + y1["assets"]) / 2 if y0["assets"] is not None and y1["assets"] is not None else None
    accruals = None
    accrual_ni = data.get("consolidated_ni", data.get("ni"))
    if data["cfo"] is not None and accrual_ni is not None and avg_assets is not None and avg_assets > 0:
        accruals = (accrual_ni - data["cfo"]) / avg_assets

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
        raise ValueError("CAPM beta is missing or outside 0–3; enter a defensible industry/analyst beta or disable CAPM and set explicit rates")
    rf, erp = rates_for(data, p)
    beta_used = 0.67 * beta + 0.33 if p.get("use_blume_beta") else beta
    ke = rf + beta_used * erp
    if not (0.0 < ke < 0.50):
        raise ValueError(f"implausible CAPM cost of equity ({ke:.1%}); review beta and rate assumptions")
    y0 = data["years"][0]
    debt = max((y0["debt"] or 0.0) - (y0.get("lease_excluded") or 0.0), 0.0)
    mcap = data["price"] * data["shares"]
    w_d = debt / (debt + mcap) if (debt + mcap) > 0 else 0.0
    t = data.get("effective_tax_rate")
    t = 0.21 if t is None else t
    kd_used = data.get("implied_cost_of_debt")
    if kd_used is None:
        kd_used = p["kd"]
    wacc = (1 - w_d) * ke + w_d * kd_used * (1 - t)
    if not (0.0 < wacc < 0.50):
        raise ValueError(f"implausible WACC ({wacc:.1%}); review leverage and rate assumptions")
    return ke, wacc


def normalized_beta_sensitivity(data, p):
    """Return a Blume-adjusted beta and CAPM rate sensitivity without changing the base case.

    This is a sensitivity reference only: unless the sidebar Blume option is ticked, the primary
    DCF continues to use the reported Yahoo beta so the app never silently substitutes a
    different beta into its headline value.
    """
    if not p.get("use_capm", True):
        return NAN, NAN, NAN
    beta = data.get("beta")
    if beta is None or not (0.0 <= beta <= 3.0):
        return NAN, NAN, NAN
    # Common Blume-style mean reversion toward market beta = 1.0.
    beta_norm = 0.67 * beta + 0.33 * 1.0
    rf, erp = rates_for(data, p)
    ke_norm = rf + beta_norm * erp
    y0 = data["years"][0]
    debt = max((y0["debt"] or 0.0) - (y0.get("lease_excluded") or 0.0), 0.0)
    mcap = data["price"] * data["shares"]
    w_d = debt / (debt + mcap) if (debt + mcap) > 0 else 0.0
    tax = data.get("effective_tax_rate")
    tax = 0.21 if tax is None else tax
    kd_used = data.get("implied_cost_of_debt")
    if kd_used is None:
        kd_used = p["kd"]
    wacc_norm = (1 - w_d) * ke_norm + w_d * kd_used * (1 - tax)
    return beta_norm, ke_norm, wacc_norm


def growth_path(start, terminal, years):
    """Revenue growth fading linearly from `start` (year 1) to `terminal` (year N)."""
    if years <= 1:
        return [terminal]
    return [start + (terminal - start) * i / (years - 1) for i in range(years)]


def start_growth(data, p):
    # A per-ticker manual override wins. Otherwise historical reported CAGR is shown as context,
    # not assumed to be organic forecast growth; the user must explicitly choose the
    # reported-CAGR option, and the default is the analyst-normalized sidebar input.
    override = (p.get("growth_overrides") or {}).get(str(data.get("ticker", "")).upper())
    if override is not None:
        return override
    if p.get("use_company_growth") and data.get("rev_cagr") is not None:
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


def dcf_valuation(data, wacc, g_path, g_term, margin, tax, capex_ratio, dep_ratio, nwc_ratio, term_roic, margin_end=None, capex_terminal_mult=1.1):
    if wacc <= g_term:
        raise ValueError("WACC must exceed terminal growth")
    if term_roic is None or term_roic <= g_term or term_roic <= 0:
        raise ValueError("terminal ROIC must be positive and exceed terminal growth")
    if capex_terminal_mult is None or not (0.0 <= capex_terminal_mult <= 5.0):
        raise ValueError("terminal CapEx / D&A multiple is outside 0–5x")
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
    if y0.get("minority") is None:
        raise ValueError("minority interest is unavailable; enterprise-to-equity bridge is incomplete")
    # Net debt = debt - (lease liabilities excluded when EBIT is already lease-burdened)
    #            - cash - (optional non-operating investments) + minority interest.
    net_debt = (y0["debt"] - (y0.get("lease_excluded") or 0.0) - y0["cash"]
                - (y0.get("nonop_assets") or 0.0) + y0["minority"])

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
        # level (capex_terminal_mult x D&A, a visible sidebar assumption); the terminal
        # FCFF separately ties total reinvestment to g / ROIC.
        cap_target = dep_ratio * capex_terminal_mult
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


def implied_irr(data, g_path, g_term, dcf_inputs):
    """Market-implied IRR: the discount rate at which the DCF value per share equals today's price.

    Solved only when V/P is finite and non-increasing in the discount rate over 3%-40%
    (approximately g_term + 0.5% to 40%); otherwise returns None.
    """
    lo, hi = g_term + 0.005, 0.40

    def vp(r):
        return dcf_valuation(data, r, g_path, g_term, **dcf_inputs)["V/P"]

    try:
        v_lo, v_hi = vp(lo), vp(hi)
        if not (math.isfinite(v_lo) and math.isfinite(v_hi)):
            return None
        if v_lo < 1.0 or v_hi > 1.0:
            return None  # price implies a return outside the supported bracket
        grid = [lo + (hi - lo) * i / 12 for i in range(13)]
        vals = [vp(r) for r in grid]
        if any(not math.isfinite(v) for v in vals) or any(b > a + 1e-8 for a, b in zip(vals, vals[1:])):
            return None
        for _ in range(60):
            mid = (lo + hi) / 2
            v = vp(mid)
            if abs(v - 1.0) < 1e-9:
                return mid
            if v > 1.0:
                lo = mid
            else:
                hi = mid
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
    # Attach DCF-bridge adjustments (lease exclusion, optional non-operating investments) to a
    # copy of the data so every downstream DCF call and sensitivity sees the same net debt.
    data = {**data, "years": [
        {**y, "lease_excluded": _lease_excluded(y, p),
         "nonop_assets": max(y.get("nonop_inv") or 0.0, 0.0) if p.get("include_lt_investments", False) else 0.0}
        for y in data["years"]]}
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
    normalized_beta, normalized_ke, normalized_wacc = normalized_beta_sensitivity(data, p)
    years, g_term = p["years"], p["terminal_growth"]
    if ke <= g_term + 0.01 or wacc <= g_term:
        raise ValueError("discount rate too close to terminal growth")
    g_override = (p.get("growth_overrides") or {}).get(str(data.get("ticker", "")).upper())
    g0 = start_growth(data, p)
    if g_override is None and p["use_company_growth"] and data.get("rev_cagr") is not None and not (-0.10 <= data["rev_cagr"] <= p["max_start_growth"]):
        warnings_note = (f"Historical revenue CAGR {data['rev_cagr']:.1%} is outside the supported start-growth range; "
                         f"model uses {g0:.1%} after clipping.")
    else:
        warnings_note = None
    is_growth = g0 >= p["growth_threshold"]  # growth mode: longer fade, latest margin, optional margin ramp
    if is_growth:
        years = p["growth_years"]
    g_path = growth_path(g0, g_term, years)

    warnings = ["Research limitation: DCF and RI forecasts are independently generated in this version; model disagreement is shown and neither valuation is averaged. A fully integrated statement forecast requires filing-level segment, working-capital and equity-bridge data not reliably supplied by Yahoo summaries."]
    if warnings_note:
        warnings.append(warnings_note)
    if g_override is not None:
        warnings.append(f"Starting growth {g0:.1%} is a manual per-ticker input.")
    elif not p["use_company_growth"]:
        warnings.append(f"Starting growth is the global sidebar default ({g0:.1%}) rather than a company-specific forecast; enter a per-ticker growth override (e.g. from consensus) before trusting cross-company rankings.")
    if y0.get("lease_excluded"):
        warnings.append(f"Lease liabilities of {y0['lease_excluded'] / 1e9:,.2f}B are excluded from DCF net debt and the WACC debt weight because Yahoo's Total Debt includes them and GAAP EBIT is already burdened by operating-lease expense. Untick the lease option for IFRS filers, where leases are financing.")
    if y0.get("nonop_assets"):
        warnings.append(f"Long-term investments of {y0['nonop_assets'] / 1e9:,.2f}B are treated as non-operating assets in DCF net debt; verify they are not operating or strategic holdings that generate EBIT.")
    industry_text = (str(data.get("industry", "")) + " " + str(data.get("sector", ""))).lower()
    if any(term in industry_text for term in ("semiconductor", "memory", "chip")):
        warnings.append("Cyclical semiconductor exposure: current margins and revenue growth may be mid-cycle or peak-cycle; use normalized mid-cycle assumptions and inspect a full cycle before screening.")
    if any(term in industry_text for term in ("software", "internet", "technology")):
        warnings.append("Intangible-intensive business: reported book equity and conventional RI may understate internally developed assets; inspect R&D capitalization sensitivity and SBC dilution.")
    if any(term in industry_text for term in ("pharmaceutical", "biotechnology", "biotech")):
        warnings.append("Pharma/biotech exposure: R&D projects have highly uncertain useful lives and failure rates; do not use a mechanical R&D asset without project-level review.")
    if data.get("price_source") != "Yahoo current/regular market price":
        warnings.append(f"Current quote unavailable; using {data.get('price_source', 'an unverified price fallback')}. Verify the quote timestamp before comparing valuation with market price.")
    if data.get("shares_source") not in CURRENT_SHARE_SOURCES:
        warnings.append(f"Current shares outstanding unavailable; using {data.get('shares_source', 'an unverified fallback')}. Per-share values and market capitalization may be unreliable.")
    elif data.get("shares_source") == "Yahoo market cap / price (multi-class cross-check)":
        warnings.append("Yahoo sharesOutstanding disagreed with market cap / price by more than 5% (typical of multi-class shares); the market-cap-based share count is used.")
    if data.get("beta_source") != "Yahoo beta" or data.get("beta") is None or not (0 <= data.get("beta", 1.0) <= 3.0):
        warnings.append("Yahoo beta is missing or outside the supported 0–3 range; CAPM valuation is not permitted until a defensible beta is supplied.")
    elif p["use_capm"] and data.get("beta") < 0.60 and not p.get("use_blume_beta"):
        warnings.append(f"Low Yahoo beta ({data['beta']:.2f}) produces a relatively low CAPM cost of equity. Blume-adjusted beta sensitivity is {normalized_beta:.2f} (cost of equity {normalized_ke:.1%}, WACC {normalized_wacc:.1%}); these are sensitivity rates only, not the rates used in the headline DCF. Compare with an industry beta if available.")
    if p["use_capm"] and p.get("use_blume_beta") and data.get("beta") is not None:
        warnings.append(f"CAPM uses the Blume-adjusted beta {normalized_beta:.2f} (raw Yahoo beta {data['beta']:.2f}).")
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
                warnings.append("Tangible book is non-positive for the current or prior period; RI uses reported book equity instead. This fallback does not override other RI validity checks.")
        else:
            warnings.append("Goodwill/intangibles not reported; RI uses reported book equity.")
    rd = [safe_float(v) for v in (data.get("rd") or [])]
    rd_life = int(p.get("rd_life", 5))
    rd_history_complete = len(rd) >= rd_life and all(v is not None and v > 0 for v in rd[:rd_life])
    if (p["capitalize_rd"] and rd_history_complete and eq0 and eq1
            and ni_adj is not None and data.get("effective_tax_rate") is not None):
        life, tx = int(p.get("rd_life", 5)), data["effective_tax_rate"]
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
        warnings.append(f"RI capitalizes R&D ({life}-year straight-line life, after tax); this remains an analyst assumption and needs sensitivity analysis.")
    if p["capitalize_rd"] and (not rd_history_complete or data.get("effective_tax_rate") is None):
        positive_years = sum(v is not None and v > 0 for v in rd[:rd_life])
        warnings.append(f"R&D capitalization requested but the first {rd_life} annual vintages are not all present and positive ({positive_years}/{rd_life} usable), or normalized tax is unavailable. R&D capitalization was not applied.")
    ri_data = data
    if adjusted:
        avg_adjusted_equity = (eq0 + eq1) / 2 if eq0 is not None and eq1 is not None else None
        ri_data = {**data, "years": [{**y0, "equity": eq0}, {**y1, "equity": eq1}], "ni": ni_adj,
                   "roe_hist": (ni_adj / avg_adjusted_equity) if ni_adj is not None and avg_adjusted_equity is not None and avg_adjusted_equity > 0 else None}
    base = cons = implied_omega = None
    ri_status = "Valid"
    try:
        base = residual_income_valuation(ri_data, ke, p["omega"], g_path, g_term)
        # Conservative case: faster fade if ROE > ke, slower recovery if ROE < ke
        cons_omega = p["omega"] * 0.75 if base["historical_roe"] >= ke else min(1.0, p["omega"] * 1.25)
        cons = residual_income_valuation(ri_data, ke + 0.01, cons_omega, g_path, g_term)
        implied_omega = calculate_implied_omega(ri_data, ke, g_path, g_term)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        base = cons = implied_omega = None
        ri_status = str(e)
        warnings.append(f"Residual income unavailable: {e}."
                        + (" Adjusted book is negative or missing; untick the tangible-book option to use reported book." if ri_data is not data else ""))

    ri_valid = base is not None
    cs_resid = data.get("clean_surplus_residual")
    cs_ratio = data.get("clean_surplus_ratio")
    if "clean_surplus_residual" in data and cs_resid is None:
        warnings.append("Clean-surplus reconciliation unavailable from Yahoo fields; RI remains provisional and cannot be treated as fully validated.")
        ri_valid = False
        ri_status = "Historical clean-surplus reconciliation unavailable"
        implied_omega = None
    elif cs_ratio is not None and cs_ratio > 0.10:
        warnings.append(f"Clean-surplus reconciliation residual is {cs_resid:,.0f} ({cs_ratio:.1%} of book equity); OCI, issuance, SBC or other equity movements are material. RI disabled pending reconciliation.")
        ri_valid = False
        ri_status = "Material clean-surplus reconciliation residual"
        implied_omega = None
    if data.get("ni_source") not in (None, "Net Income Common Stockholders") and base is not None:
        ri_valid = False
        implied_omega = None
        ri_status = "Common-shareholder net income unavailable; RI earnings basis is not confirmed"
        warnings.append("RI disabled: Yahoo did not provide a distinct Net Income Common Stockholders line; reported Net Income may include non-controlling interests and is not assumed comparable to common book equity.")
    if data.get("roe_source") == "Yahoo-reported ROE fallback" and base is not None:
        ri_valid = False
        implied_omega = None
        ri_status = "Historical ROE is only a Yahoo-reported fallback"
        warnings.append("RI disabled: historical ROE could not be reconstructed from balance-sheet history. Yahoo-reported ROE is shown for context but is not sufficient to validate the RI forecast.")
    if base and base.get("roe_was_clipped"):
        ri_valid = False
        implied_omega = None
        ri_status = f"ROE {base['historical_roe']:.1%} exceeds supported range; clipped to {base['roe_base_used']:.1%}"
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
        ri_valid, implied_omega = False, None
        ri_status = "Common shareholders' book equity unavailable; only total equity including minority interest was reported"
        warnings.append("Only total equity including minority interest is available; P/B and FLEV are suppressed and RI is disabled because common-equity data is not comparable.")
    roe_rep = ri_data.get("roe_hist") if ri_data.get("roe_hist") is not None else (base["historical_roe"] if base else NAN)
    ri_limit_breaches = []
    if not pd.notna(ri_pb):
        ri_limit_breaches.append("adjusted RI P/B is unavailable")
    elif ri_pb > p["max_pb"]:
        ri_limit_breaches.append(f"adjusted RI P/B {ri_pb:.2f}x exceeds the {p['max_pb']:.1f}x limit")
    if not pd.notna(roe_rep):
        ri_limit_breaches.append("RI ROE is unavailable")
    elif roe_rep > p["max_roe"]:
        ri_limit_breaches.append(f"RI ROE {roe_rep:.1%} exceeds the {p['max_roe']:.1%} limit")
    if ri_limit_breaches:
        if ri_valid:
            ri_valid, implied_omega = False, None
            ri_status = "; ".join(ri_limit_breaches)
        else:
            # Preserve the first failure (e.g. clipped ROE) while exposing additional
            # independent reasons the RI result must not be used.
            ri_status = f"{ri_status}; additionally: " + "; ".join(ri_limit_breaches)
            warnings.append("Additional RI validity issue: " + "; ".join(ri_limit_breaches) + ".")
        warnings.append(
            "RI is not screening-grade: " + "; ".join(ri_limit_breaches) + ". "
            "High P/B or extreme ROE can make book-value-based RI unstable; verdict uses DCF only."
        )

    dcf_inputs, dcf_sources = resolve_dcf_inputs(data, p)
    roic_floor = g_term + 0.005
    rnoa = operating["RNOA"]
    high_return = rnoa is not None and rnoa >= 0.25
    hi = p["quality_roic_cap"] if high_return else p["roic_cap"]
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
    dcf_inputs["capex_terminal_mult"] = p.get("capex_terminal_mult", 1.1)
    if dcf_inputs["dep_ratio"] > 0 and dcf_inputs["capex_ratio"] / dcf_inputs["dep_ratio"] > 1.75:
        warnings.append(
            f"CapEx intensity ({dcf_inputs['capex_ratio']:.1%} of revenue) is more than 1.75x D&A ({dcf_inputs['dep_ratio']:.1%}); "
            f"the forecast fades CapEx to {dcf_inputs['capex_terminal_mult']:.2f}x D&A. This is a major swing assumption: "
            "stress-test it (sidebar) for companies in an investment cycle.")
    industry_roic = p.get("industry_terminal_roic", 0.10)
    persistence = p.get("roic_persistence", 0.50)
    if rnoa is None:
        terminal_roic_raw = industry_roic
        warnings.append("Historical RNOA unavailable; terminal ROIC uses the explicit industry-normalized sidebar assumption.")
    else:
        # Historical RNOA is only a proxy for incremental ROIC; mean-revert it toward an explicit industry assumption.
        terminal_roic_raw = persistence * rnoa + (1 - persistence) * industry_roic
        warnings.append("Terminal ROIC is a scenario assumption blending historical RNOA with the analyst's industry-normalized ROIC; historical RNOA is not the same as incremental ROIC.")
    dcf_inputs["term_roic"] = clamp(terminal_roic_raw, roic_floor, hi)
    dcf = dcf_valuation(data, wacc, g_path, g_term, **dcf_inputs)
    implied_g = implied_start_growth(data, wacc, g_term, years, dcf_inputs)
    # Explicit valuation sensitivity using the Blume-adjusted WACC; this does not
    # replace the headline DCF or silently alter the base-case discount rate.
    dcf_normalized_beta_vp = NAN
    if math.isfinite(normalized_wacc) and normalized_wacc > g_term:
        try:
            norm_dcf = dcf_valuation(data, normalized_wacc, g_path, g_term, **dcf_inputs)
            dcf_normalized_beta_vp = norm_dcf["V/P"] if norm_dcf["value_per_share"] > 0 else NAN
        except Exception:
            pass
    dcf_ok = dcf["value_per_share"] > 0
    irr_value = implied_irr(data, g_path, g_term, dcf_inputs) if dcf_ok else None

    def scenario_vp(g_mult, g_add, margin_add, roic_add):
        try:
            gs = clamp((g0 * g_mult if g0 > 0 else g0) + g_add, -0.10, 0.50)
            inp = dict(dcf_inputs)
            inp["margin_end"] = clamp(inp["margin_end"] + margin_add, *DCF_HARD_LIMITS["margin"])
            inp["term_roic"] = clamp(inp["term_roic"] + roic_add, g_term + 0.005, max(hi, g_term + 0.15))
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
    if not dcf_ok:
        implied_g = None
        warnings.append("DCF equity value is negative (net debt exceeds enterprise value); DCF shown as n/m.")
    elif irr_value is None:
        warnings.append("Implied IRR could not be solved inside the supported 3%–40% range (price implies a return outside it, or V/P is not monotonic in the discount rate).")
    rp = data.get("rev_prev")
    if rp and rp > 0 and abs(data["revenue"] / rp - 1) > 0.30:
        warnings.append("Revenue changed >30% year over year: possible merger or spin-off; financials may not match the current price.")
    if data.get("fcf_yield") is not None and data["fcf_yield"] > 0.20:
        warnings.append("FCF yield above 20%: check for a spin-off, one-off cash flow or data mismatch.")
    m_avg = data["company_ratios"].get("margin")
    if m_avg and m_avg > 0 and abs(data["ebit"] / data["revenue"] / m_avg - 1) > 0.40:
        warnings.append("Latest EBIT margin differs >40% from the 3-year average: earnings may be cyclical or volatile. DCF starts from the average and fades toward the latest in growth mode; review the margin path.")

    if data.get("effective_tax_rate") is None:
        warnings.append("Effective tax rate unavailable: DCF uses its configured tax fallback, while WACC's after-tax debt shield uses a separate 21% fallback. Review both assumptions.")
    if y0.get("minority_assumed_zero") or str(y0.get("minority_source", "")).startswith("not separately reported"):
        warnings.append("Minority interest is not separately reported by Yahoo; zero is used only to display an exploratory DCF. Verify the latest filing for non-controlling interests or preferred claims before relying on the enterprise-to-equity bridge.")
    if not p["override"]:
        for k, source in dcf_sources.items():
            if source == "company (outside usual range)":
                warnings.append(f"Company {DCF_LABELS[k]} is {dcf_inputs[k]:.1%}, outside the usual diagnostic range {DCF_LIMITS[k][0]:.1%} to {DCF_LIMITS[k][1]:.1%}; the observed ratio is used without clipping. Verify the statement line items and consider a sensitivity case.")
            elif source == "sidebar fallback (company ratio outside hard bounds)":
                warnings.append(f"Company {DCF_LABELS[k]} was outside hard safety bounds; DCF uses the sidebar fallback instead of the reported ratio. Investigate the underlying data before relying on the estimate.")
            elif source == "sidebar fallback (company data missing)":
                warnings.append(f"Company {DCF_LABELS[k]} unavailable; DCF uses the sidebar fallback.")
            elif source == "latest company margin (fallback)":
                warnings.append("Three-year average EBIT margin is unavailable; DCF uses the latest reported EBIT margin as a fallback.")
    if p["use_capm"]:
        kd_source = "company-implied interest / average debt" if data.get("implied_cost_of_debt") is not None else "sidebar cost-of-debt assumption"
        warnings.append(f"WACC uses {kd_source}; debt is book debt as a proxy for market-value debt. A single historical implied borrowing rate is not a forward credit spread. Verify the risk-free rate and ERP match the reporting currency ({data.get('currency', 'unknown')}).")
        ccy_upper = str(data.get("currency", "unknown")).upper()
        if ccy_upper not in ("USD", "UNKNOWN") and ccy_upper not in (p.get("rate_overrides") or {}):
            warnings.append(f"No currency-specific risk-free rate/ERP supplied for {ccy_upper}; the global (USD-calibrated) inputs are used. Add one in the sidebar (e.g. {ccy_upper}=0.03/0.055).")
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
    # candidate when a core input is substituted or the equity bridge is incomplete.
    beta_is_usable = (not p["use_capm"]) or (
        data.get("beta_source") == "Yahoo beta" and data.get("beta") is not None
        and 0 <= data.get("beta") <= 3
    )
    shares_are_current = data.get("shares_source") in CURRENT_SHARE_SOURCES
    price_is_current = data.get("price_source") == "Yahoo current/regular market price"
    minority_is_known = y0.get("minority") is not None and not y0.get("minority_assumed_zero", False)
    growth_is_unclipped = warnings_note is None and (g_override is not None or not (p["use_company_growth"] and data.get("rev_cagr") is None))
    # Real company ratios that merely sit outside the usual diagnostic range are still real data
    # (they are warned about, not clipped); only fallbacks/substitutions make the DCF provisional.
    dcf_inputs_are_company_reported = all(source in ("company", "company (outside usual range)") for source in dcf_sources.values())
    terminal_dependence_ok = dcf["terminal_pct"] is not None and dcf["terminal_pct"] <= 0.65
    discount_rate_ok = (not p["use_capm"]) or ke >= 0.07
    dcf_reliable = bool(
        dcf_ok and beta_is_usable and shares_are_current and price_is_current and minority_is_known
        and growth_is_unclipped and dcf_inputs_are_company_reported
        and terminal_dependence_ok and discount_rate_ok
        and data.get("period_alignment_ok", True)
        and data.get("effective_tax_rate") is not None
    )
    if dcf_ok and dcf["terminal_pct"] is not None and dcf["terminal_pct"] > 0.65:
        warnings.append("Reliability gate: terminal value exceeds 65% of enterprise value; DCF is provisional until terminal assumptions are stress-tested.")
    if dcf_ok and p["use_capm"] and ke < 0.07:
        warnings.append("Reliability gate: CAPM cost of equity is below 7%; DCF is provisional until a normalized discount-rate sensitivity is reviewed.")
    if dcf_ok and not dcf_reliable:
        warnings.append("DCF is provisional, not screening-grade: at least one core input is substituted, unavailable, or fails a reliability gate. It is displayed for analysis but cannot qualify the stock as a candidate; review the input warnings.")

    def ri(res, key):
        return res[key] if (ri_valid and res is not None) else NAN

    # Hard data-integrity gates only. Informational items (negative net investment, unusual-but-real
    # ratios, thresholds not backtested, moat/management review) are surfaced as warnings and in the
    # caption, not as gates; otherwise no stock could ever reach a CANDIDATE label.
    quality_flags = []
    if not dcf_reliable:
        quality_flags.append("DCF inputs or equity bridge are not screening-grade")
    if not (DCF_LIMITS["margin"][0] <= dcf_inputs["margin_end"] <= DCF_LIMITS["margin"][1]):
        quality_flags.append("forecast endpoint margin is outside the usual range")
    if data.get("price_source") != "Yahoo current/regular market price":
        quality_flags.append("fallback market quote")
    if not data.get("currency") or data.get("currency") == "unknown":
        quality_flags.append("security currency unavailable")
    if data.get("shares_source") not in CURRENT_SHARE_SOURCES:
        quality_flags.append("fallback share count")
    if data.get("roe_source") == "Yahoo-reported ROE fallback":
        quality_flags.append("fallback ROE")
    if data.get("ni_source") not in (None, "Net Income Common Stockholders"):
        quality_flags.append("common-shareholder earnings basis not confirmed")
    if not data.get("period_alignment_ok", False):
        quality_flags.append("fiscal-period alignment not established")
    if "clean_surplus_residual" in data and data.get("clean_surplus_residual") is None:
        quality_flags.append("clean-surplus bridge unavailable")
    if data.get("clean_surplus_ratio") is not None and data.get("clean_surplus_ratio") > 0.10:
        quality_flags.append("material unexplained clean-surplus residual")
    if not ri_valid and p.get("ri_required", True):
        quality_flags.append("RI valuation unavailable or invalid")
    if any(str(src).startswith("sidebar fallback") for src in dcf_sources.values()):
        quality_flags.append("one or more DCF inputs use a fallback")
    if dcf["terminal_pct"] is not None and dcf["terminal_pct"] > 0.65:
        quality_flags.append("DCF terminal value exceeds 65% of enterprise value")
    if any("Revenue changed >30%" in w or "Latest EBIT margin differs >40%" in w for w in warnings):
        quality_flags.append("materially unusual operating history")
    if any("Model disagreement" in w for w in warnings):
        quality_flags.append("DCF and RI materially disagree")
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
        "Normalized Beta Sensitivity": normalized_beta,
        "Normalized Cost of Equity": normalized_ke,
        "Normalized WACC": normalized_wacc,
        "DCF V/P at Normalized WACC": dcf_normalized_beta_vp,
        "Start Growth": g0,
        "Market-Implied Growth": implied_g,
        "Implied IRR": irr_value,
        "Earnings CAGR": data.get("earnings_cagr"),
        "FCF Yield": data.get("fcf_yield"),
        "Implied Omega": implied_omega,
        "RNOA": operating["RNOA"],
        "NBC": operating["NBC"],
        "FLEV": operating["FLEV"],
        "Spread": operating["spread"],
        "Accruals": operating["accruals"],
        "RI Valid": ri_valid,
        "RI Status": "Valid" if ri_valid else ri_status,
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
    if "reit" in industry or "real estate investment trust" in industry:
        return True, "REITs require an FFO/AFFO-based valuation framework; conventional FCFF/RI is not appropriate"
    if quote_type and quote_type not in {"EQUITY"}:
        return True, f"Yahoo quote type {quote_type} is not a standard common equity"
    return False, ""


# ============================================================
# SIDEBAR
# ============================================================

if not TEST_MODE:
    with st.sidebar:
        st.header("Discount rates")
        use_capm = st.checkbox("Per-company CAPM rates", value=True,
                               help="Cost of equity = risk-free + beta x ERP; WACC blends it with after-tax cost of debt.")
        rf = st.number_input("Risk-free rate", 0.00, 0.10, 0.043, 0.001, format="%.3f")
        erp = st.number_input("Equity risk premium", 0.02, 0.09, 0.045, 0.005, format="%.3f")
        rate_text = st.text_input("Per-currency rates (optional)", value="",
                                  help="CCY=riskfree/ERP, comma-separated, e.g. SGD=0.028/0.055, HKD=0.04. Currencies not listed use the global inputs above.")
        rate_overrides = parse_rate_overrides(rate_text)
        use_blume_beta = st.checkbox("Use Blume-adjusted beta (0.67 x raw + 0.33)", value=False,
                                     help="Off: the headline DCF uses raw Yahoo beta and shows the Blume-adjusted rates as a sensitivity.")
        kd = st.number_input("Pre-tax cost of debt", 0.01, 0.15, 0.055, 0.005, format="%.3f")
        flat_ke = st.number_input("Flat cost of equity (if CAPM off)", 0.05, 0.30, 0.09, 0.005, format="%.3f")
        flat_wacc = st.number_input("Flat WACC (if CAPM off)", 0.05, 0.30, 0.08, 0.005, format="%.3f")

        st.divider()
        st.header("Growth")
        terminal_growth = st.number_input("Terminal growth (both models)", -0.02, 0.04, 0.025, 0.005, format="%.3f")
        forecast_years = st.slider("Forecast years (growth fade)", 5, 15, 10)
        use_company_growth = st.checkbox("Use reported historical revenue CAGR (not organic growth)", value=False,
                                          help="Off by default: acquisitions, FX and cyclicality can distort reported CAGR. Enter a normalized forecast below.")
        max_start_growth = st.number_input("Max starting growth", 0.00, 0.40, 0.20, 0.01, format="%.2f")
        start_growth_fb = st.number_input("Starting growth (fallback / manual)", -0.10, 0.40, 0.06, 0.005, format="%.3f")
        growth_text = st.text_area("Per-ticker starting growth (optional)", value="", height=70,
                                   help="TICKER=growth, comma-separated, e.g. MSFT=0.12, KO=4%. Overrides the global value and the reported CAGR for those tickers (range -10% to 40%).")
        growth_overrides = parse_growth_overrides(growth_text)
        growth_threshold = st.number_input("Growth mode if starting growth ≥", 0.05, 0.40, 0.10, 0.01, format="%.2f",
                                           help="Growth names get a longer fade and the latest margin instead of the 3-year average.")
        growth_years = st.number_input("Growth mode: fade years", 10, 25, 15, 1)

        st.divider()
        st.header("Residual income")
        omega = st.slider("ROE persistence (ω)", 0.0, 1.0, 0.80, 0.05)
        max_pb = st.number_input("RI valid only if P/B ≤", 1.0, 100.0, 10.0, 1.0)
        max_roe = st.number_input("RI valid only if ROE ≤", 0.10, 3.00, 0.50, 0.05)
        capitalize_rd = st.checkbox("RI: capitalize R&D (requires complete vintage history)", value=False,
                                    help="Optional analyst adjustment; only applied when the full selected useful-life history is available.")
        rd_life = st.selectbox("R&D useful life (years)", [3, 4, 5, 7, 10], index=2)
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
        capex_terminal_mult = st.number_input("Terminal CapEx / D&A multiple", 0.50, 3.00, 1.10, 0.05, format="%.2f",
                                              help="Explicit-period CapEx fades from today's intensity to this multiple of D&A. A low value inflates FCFF for heavy investors (e.g. AI-capex hyperscalers); stress-test it.")
        exclude_leases = st.checkbox("Exclude lease liabilities from DCF net debt", value=True,
                                     help="US GAAP operating-lease expense is already inside EBIT, so counting lease liabilities as debt double-counts. Untick for IFRS filers. Applied only when Yahoo's Total Debt row is present.")
        include_lt_investments = st.checkbox("Treat long-term investments as non-operating assets", value=False,
                                             help="Adds Yahoo's long-term investment line to DCF net cash. Check that these are not operating or strategic holdings.")
        industry_terminal_roic = st.number_input("Normalized industry terminal ROIC", 0.03, 0.40, 0.10, 0.01, format="%.2f",
                                                  help="Analyst assumption for returns on future incremental investment; do not equate it with historical RNOA.")
        roic_persistence = st.slider("Historical RNOA persistence in terminal ROIC", 0.0, 1.0, 0.50, 0.05)
        roic_cap = st.number_input("Terminal ROIC cap", 0.05, 0.50, 0.15, 0.01, format="%.2f",
                                   help="Cap only; terminal ROIC may be below WACC if reinvestment destroys value. It must exceed terminal growth.")

        quality_roic_cap = st.number_input("Terminal ROIC cap, high-return firms (RNOA ≥ 25%)", 0.10, 0.60, 0.25, 0.01, format="%.2f",
                                           help="Firms with a durable moat can keep higher returns on new investment.")
        margin_expansion = st.number_input("Growth mode: extra margin expansion by end of fade", 0.00, 0.20, 0.00, 0.01, format="%.2f",
                                           help="Added to the latest EBIT margin, ramped linearly. 0 = hold the latest margin.")

        st.divider()
        st.header("Screening rules (heuristic — not backtested)")
        st.caption("Thresholds are analyst-set filters, not empirically validated predictors. CANDIDATE labels are automatic only when every data-integrity gate passes; they still need point-in-time backtesting and moat/management review before any capital is committed.")
        ri_required = st.checkbox("Require valid RI for CANDIDATE", value=False,
                                  help="Off: the DCF leads and RI is a cross-check, so names where book-value RI is meaningless (large buybacks, high P/B) can still be flagged 'CANDIDATE (DCF-led)'. On: RI must also pass.")
        min_ri_vp = st.number_input("Minimum RI V/P", 0.5, 5.0, 1.20, 0.05)
        min_dcf_vp = st.number_input("Minimum DCF V/P", 0.5, 5.0, 1.20, 0.05)
        min_cons_vp = st.number_input("Minimum conservative RI V/P", 0.5, 3.0, 1.00, 0.05)
        strong_ri_vp = st.number_input("Strong: minimum RI V/P", 0.5, 5.0, 1.50, 0.05)
        strong_max_omega = st.number_input("Strong: maximum implied ω", 0.0, 1.0, 0.80, 0.05)
        max_accruals = st.number_input("Maximum accruals", 0.0, 0.5, 0.10, 0.01, format="%.2f")
        max_flev = st.number_input("Maximum FLEV", 0.0, 10.0, 2.0, 0.25)

    P = {
        "use_capm": use_capm, "rf": rf, "erp": erp, "kd": kd, "flat_ke": flat_ke, "flat_wacc": flat_wacc,
        "rate_overrides": rate_overrides, "use_blume_beta": use_blume_beta,
        "terminal_growth": terminal_growth, "years": forecast_years, "use_company_growth": use_company_growth,
        "max_start_growth": max_start_growth, "start_growth": start_growth_fb, "growth_overrides": growth_overrides, "omega": omega,
        "max_pb": max_pb, "max_roe": max_roe, "tangible_book": tangible_book, "capitalize_rd": capitalize_rd, "rd_life": rd_life,
        "industry_terminal_roic": industry_terminal_roic, "roic_persistence": roic_persistence, "override": dcf_override, "fb_margin": fb_margin,
        "fb_tax": fb_tax, "fb_capex": fb_capex, "fb_dep": fb_dep, "fb_nwc": fb_nwc, "roic_cap": roic_cap,
        "capex_terminal_mult": capex_terminal_mult, "exclude_leases": exclude_leases, "include_lt_investments": include_lt_investments,
        "growth_threshold": growth_threshold, "growth_years": growth_years, "quality_roic_cap": quality_roic_cap,
        "margin_expansion": margin_expansion, "ri_required": ri_required,
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


    def classify_verdict(row, min_ri_vp, min_dcf_vp, min_cons_vp, strong_ri_vp, strong_max_omega, max_accruals, max_flev, ri_required=True):
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
        roe_ok = bool(pd.notna(row["ROE"]) and pd.notna(row["Cost of Equity"]) and row["ROE"] > row["Cost of Equity"])
        passes = bool(
            ri_ok and common and pd.notna(ri_vp) and ri_vp >= min_ri_vp
            and pd.notna(cons) and cons >= min_cons_vp and roe_ok
        )
        # DCF-led path: RI is invalid (or unusable) for this company and the user has not required it.
        passes_dcf_led = bool((not ri_required) and (not ri_ok) and common and roe_ok)
        quality = row.get("Valuation Quality", "REVIEW")
        if (passes or passes_dcf_led) and quality != "OK":
            return "WATCH"
        if passes and ri_vp >= strong_ri_vp and pd.notna(iw) and iw <= strong_max_omega:
            return "STRONG CANDIDATE"
        if passes:
            return "CANDIDATE"
        if passes_dcf_led:
            return "CANDIDATE (DCF-led)"
        if (ri_ok and pd.notna(ri_vp) and ri_vp >= min_ri_vp) or dcf_vp >= min_dcf_vp:
            return "WATCH"
        return "NO"


    def render_results(df, allow_verdict=True):
        if df is None or df.empty:
            st.info("Enter tickers above and click **Run Screener**.")
            return

        df = df.copy()
        if allow_verdict:
            df["Verdict"] = df.apply(lambda row: classify_verdict(
                row, min_ri_vp, min_dcf_vp, min_cons_vp, strong_ri_vp,
                strong_max_omega, max_accruals, max_flev, ri_required=ri_required), axis=1)
        else:
            df["Verdict"] = "STALE — RERUN REQUIRED"
        df = df.sort_values(["DCF V/P", "RI V/P"], ascending=False, na_position="last").reset_index(drop=True)
        st.session_state.screen_results = df

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks analyzed", len(df))
        c2.metric("Candidates", int(df["Verdict"].isin(["STRONG CANDIDATE", "CANDIDATE", "CANDIDATE (DCF-led)"]).sum()))
        c3.metric("Highest DCF V/P", f"{df['DCF V/P'].max():.2f}x")
        c4.metric("Median DCF V/P", f"{df['DCF V/P'].median():.2f}x")

        st.subheader("📈 Screening results")
        cols = ["Ticker", "Currency", "Price", "P/B", "RI P/B", "ROE", "Cost of Equity", "WACC",
                "Normalized Beta Sensitivity", "Normalized Cost of Equity", "Normalized WACC",
                "Start Growth", "Market-Implied Growth", "Implied IRR",
                "Earnings CAGR", "FCF Yield", "RNOA", "NBC", "FLEV", "Spread", "Accruals", "Implied Omega", "RI Valid", "RI Status", "DCF Reliable",
                "ROE Source", "Valuation Quality", "RI ROE Input", "RI ROE Applied", "RI Terminal Payout", "RI Value", "RI V/P", "RI Conservative Value", "RI Conservative V/P", "DCF Value", "DCF V/P", "DCF Bear V/P", "DCF Bull V/P",
                "Net Debt (B)", "Shares Outstanding", "RI Terminal Value Share", "RI Terminal ROE", "DCF Terminal Value Share", "DCF Terminal Growth", "DCF Terminal ROIC", "Verdict"]
        display = df[cols].copy()
        display["Market-Implied Growth"] = display["Market-Implied Growth"].replace(float("inf"), NAN)
        fmt = {
            "Price": "{:.2f}", "P/B": "{:.2f}x", "RI P/B": "{:.2f}x", "ROE": "{:.1%}", "Cost of Equity": "{:.1%}", "WACC": "{:.1%}",
            "Normalized Beta Sensitivity": "{:.2f}", "Normalized Cost of Equity": "{:.1%}", "Normalized WACC": "{:.1%}",
            "Start Growth": "{:.1%}", "Market-Implied Growth": "{:.1%}", "Implied IRR": "{:.1%}", "Earnings CAGR": "{:.1%}", "FCF Yield": "{:.1%}",
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
                   "Implied IRR = discount rate at which the DCF equals today's price (the return implied if your forecast is right). "
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
            mig = sel["Market-Implied Growth"]
            mig_txt = ">50% (beyond model range)" if pd.notna(mig) and mig == float("inf") else (
                "≤ -10% (cheap even with shrinking sales)" if pd.notna(mig) and mig <= -0.10 else fmt_pct(mig))
            st.write(f"**Market-implied starting growth:** {mig_txt} (model start: {fmt_pct(sel['Start Growth'])})")
            st.write(f"**Implied IRR (DCF at today's price):** {fmt_pct(sel['Implied IRR'])} vs WACC {fmt_pct(sel['WACC'])}")
            if pd.isna(sel["DCF V/P"]):
                st.write("**DCF value / V/P:** n/m (negative equity value)")
            else:
                st.write(f"**DCF value / V/P:** {sel['DCF Value']:.2f} {currency} / {sel['DCF V/P']:.2f}x")
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
        elif v == "CANDIDATE (DCF-led)":
            st.success("CANDIDATE (DCF-led): passes the DCF and quality rules; RI is not valid for this company and is not required.")
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
                   + f", margin path {used['margin']:.1%} to {used['margin_end']:.1%}, terminal ROIC {used['term_roic']:.1%}"
                   + f", terminal CapEx {used['capex_terminal_mult']:.2f}x D&A")
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

            st.markdown("**DCF value per share: terminal ROIC stress test**")
            rows = []
            base_roic = sel["_dcf_inputs"]["term_roic"]
            for roic_delta in (-0.05, -0.03, 0.0, 0.03, 0.05):
                try:
                    stress_inputs = dict(sel["_dcf_inputs"])
                    stress_inputs["term_roic"] = max(g_term + 0.005, base_roic + roic_delta)
                    val = dcf_valuation(data, sel["_wacc"], growth_path(g0, g_term, years), g_term, **stress_inputs)["value_per_share"]
                    rows.append({"Terminal ROIC": stress_inputs["term_roic"], "v": val})
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Terminal ROIC").style.format("{:,.2f}", na_rep="—"), width="stretch")

            st.markdown("**DCF value per share: terminal CapEx / D&A multiple stress test**")
            rows = []
            for mult in (1.0, 1.1, 1.5, 2.0, 2.5):
                try:
                    stress_inputs = dict(sel["_dcf_inputs"])
                    stress_inputs["capex_terminal_mult"] = mult
                    val = dcf_valuation(data, sel["_wacc"], growth_path(g0, g_term, years), g_term, **stress_inputs)["value_per_share"]
                    rows.append({"Terminal CapEx / D&A": mult, "v": val})
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Terminal CapEx / D&A").style.format("{:,.2f}", na_rep="—"), width="stretch")

        with st.expander("📚 How to read this"):
            st.markdown(
                """
    **Residual income:** value = book equity + PV of forecast residual income. This is an ROE/book-equity model; it does not use the DCF revenue-growth path to forecast earnings, so the RI forecast deliberately does not display that revenue-growth path. Forecast book equity rolls forward as beginning book + earnings − modeled payouts, using a payout ratio pooled over up to four matched fiscal years. RI is disabled when payout data or defensible ROE is unavailable, or when payout is outside 0–100%. In the terminal steady state, payout is set to 1 − terminal growth / terminal ROE so retained earnings imply the stated book-equity growth. This remains approximate because Yahoo summaries do not provide a full clean-surplus equity roll-forward or other-comprehensive-income reconciliation.

    **DCF:** FCFF = NOPAT − (CapEx − D&A + change in operating NWC). Net investment can be negative in a cash-release year; investigate whether that is sustainable rather than flooring it to zero. Revenue growth fades linearly to terminal growth; terminal FCFF = NOPAT × (1 + g) × (1 − g/ROIC), so terminal reinvestment matches growth. “DCF Terminal Value Share” is discounted terminal value divided by enterprise value, not the terminal growth rate. Missing cash or debt disables DCF. If Yahoo omits minority interest, zero is used only for an exploratory DCF and the equity bridge is explicitly marked provisional until checked against the latest filing. Lease liabilities are removed from net debt by default (US GAAP EBIT is already lease-burdened); long-term investments are only treated as non-operating if you tick that option. Operating NWC remains a Yahoo summary-line approximation and should be manually checked against detailed filings.

    **Implied IRR:** the discount rate at which the DCF value per share equals today's price, solved between roughly 3% and 40%. It is the return implied if your forecast is right, not a probability-weighted expected return; compare it with WACC and with your own hurdle rate. It is blank when the price implies a return outside that range or the DCF is not monotonic in the discount rate.

    **Input-quality gate:** a DCF estimate can be displayed for exploration when an assumption is substituted, clipped, unusual, or unavailable, but it is marked provisional and cannot qualify a stock as a candidate. Real company ratios outside the usual diagnostic range are used unclipped and warned about; only fallbacks and missing data make the DCF provisional. RI is disabled when its book-equity denominator, ROE, payout bridge, fiscal-period alignment or common-equity basis is not defensible. With “Require valid RI” off, a company with invalid RI can still be labeled CANDIDATE (DCF-led) if the DCF and quality gates pass. Yahoo does not provide a complete OCI/share-issuance/SBC bridge; RI clean-surplus checks are therefore provisional, not filing-level assurance.

    **Important limitations:** DCF and RI still use separate forecast engines; disagreement is surfaced, not hidden or averaged. Historical Yahoo RNOA is only a proxy for incremental ROIC. The industry-normalized terminal ROIC and persistence controls are analyst assumptions. Starting growth defaults to one global sidebar value unless you enter per-ticker growth. Screening thresholds have not been backtested. Business moat, management quality and competitive position require analyst review; this app does not manufacture numerical moat scores. Annual statements can be up to a year stale.

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
        skip = {"Ticker", "Currency", "warnings", "RI Valid", "RI Status", "ROE Source", "Valuation Quality"}
        num_cols = [c for c in df.columns if not c.startswith("_") and c not in skip]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        df["RI Valid"] = df["RI Valid"].map(lambda value: bool(value) if pd.notna(value) else False)
        df["DCF Reliable"] = df["DCF Reliable"].map(lambda value: bool(value) if pd.notna(value) else False)
        st.session_state.screen_results = df
        st.session_state.screen_assumptions = dict(P)
        render_results(df)
    else:
        if st.session_state.screen_results is not None:
            assumptions_match = st.session_state.screen_assumptions == P
            if not assumptions_match:
                st.warning("Assumptions changed. All valuation outputs and verdicts are stale. Candidate labels are disabled until you click **Run Screener** again.")
            render_results(st.session_state.screen_results, allow_verdict=assumptions_match)
        else:
            st.info("Enter tickers above and click **Run Screener**.")


# ============================================================
# OPTIONAL EMBEDDED SMOKE TESTS
# These are included in this single-file bundle. The Streamlit app does not
# run them: the source parse / exec below only happens with `--test`.
# ============================================================

"""Dependency-light smoke tests for stream.py's valuation and screening functions.

Run with: python stream.py --test
This loads pure functions from this file without executing its Streamlit UI or Yahoo calls.
"""
import ast
import math
import unittest
from pathlib import Path

import pandas as pd

FUNCTIONS = {
    "safe_float", "clamp", "pick", "row_values", "spend", "payout_ratio_from_cashflows", "avg_payout_ratio", "ratio", "fmt_pct", "fmt_num",
    "total_debt_for_period", "_lease_excluded", "equity_row_source", "_period_years", "_cagr_between",
    "compute_history", "aligned_statement_periods", "_period_map", "normalized_tax_rate", "total_liabilities_for_period", "operating_analysis", "discount_rates", "normalized_beta_sensitivity", "growth_path", "start_growth",
    "parse_rate_overrides", "rates_for", "parse_growth_overrides",
    "residual_income_valuation", "calculate_implied_omega", "dcf_valuation",
    "implied_start_growth", "implied_irr", "resolve_dcf_inputs", "analyze_company", "should_skip_company",
    "classify_verdict",
}
CONSTANTS = {"NAN", "EQUITY", "EBIT_NAMES", "CURRENT_SHARE_SOURCES", "DCF_LIMITS", "DCF_HARD_LIMITS", "DCF_LABELS", "FALLBACK_KEYS"}

if TEST_MODE:
    SOURCE = Path(__file__)
    TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
    BODY = [n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    BODY += [n for n in ast.walk(TREE) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in CONSTANTS for t in n.targets
    )]
    NS = {"pd": pd, "math": math, "NAN": float("nan")}
    exec(compile(ast.Module(body=BODY, type_ignores=[]), str(SOURCE), "exec"), NS)
else:
    NS = {}


class ScreenerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.path = NS["growth_path"](0.08, 0.025, 10)
        self.data = {
            "years": [
                {"equity": 3000.0, "equity_source": "Stockholders Equity", "debt": 800.0,
                 "cash": 500.0, "minority": 0.0, "assets": 5000.0, "liab": 2000.0,
                 "cur_assets": 1000.0, "cur_liab": 600.0, "cur_debt": 100.0, "goodwill": 100.0},
                {"equity": 2800.0, "equity_source": "Stockholders Equity", "debt": 750.0,
                 "cash": 450.0, "minority": 0.0, "assets": 4700.0, "liab": 1900.0,
                 "cur_assets": 950.0, "cur_liab": 580.0, "cur_debt": 90.0, "goodwill": 90.0},
            ],
            "ni": 500.0, "shares": 100.0, "price": 50.0, "roe_hist": 0.17,
            "payout_ratio": 0.26, "revenue": 4000.0,
        }

    def analysis_params(self):
        return {"use_capm": True, "rf": 0.043, "erp": 0.045, "kd": 0.055,
                "flat_ke": 0.09, "flat_wacc": 0.08, "terminal_growth": 0.025,
                "years": 10, "use_company_growth": True, "max_start_growth": 0.20,
                "start_growth": 0.06, "omega": 0.80, "max_pb": 10.0, "max_roe": 0.50,
                "tangible_book": True, "capitalize_rd": False, "override": False,
                "fb_margin": 0.15, "fb_tax": 0.21, "fb_capex": 0.04, "fb_dep": 0.035,
                "fb_nwc": 0.05, "roic_cap": 0.15, "growth_threshold": 0.10,
                "growth_years": 15, "quality_roic_cap": 0.25, "margin_expansion": 0.0}

    def company_case(self, ticker="MSFT"):
        return {"ticker": ticker, "price": 539.70, "price_source": "Yahoo current/regular market price",
                "currency": "USD", "shares": 7.4e9, "shares_source": "Yahoo current shares outstanding",
                "sector": "Technology", "industry": "Software", "quote_type": "EQUITY",
                "beta": 1.1, "beta_source": "Yahoo beta",
                "years": [
                    {"assets": 500e9, "liab": 200e9, "debt": 80e9, "cash": 100e9,
                     "equity": 500e9, "equity_source": "Stockholders Equity", "minority": 0.0,
                     "goodwill": 0.0, "cur_assets": 200e9, "cur_liab": 100e9, "cur_debt": 10e9},
                    {"assets": 470e9, "liab": 190e9, "debt": 75e9, "cash": 90e9,
                     "equity": 450e9, "equity_source": "Stockholders Equity", "minority": 0.0,
                     "goodwill": 0.0, "cur_assets": 190e9, "cur_liab": 95e9, "cur_debt": 9e9}],
                "revenue": 280e9, "rev_prev": 245e9, "rev_cagr": 0.161, "ebit": 130e9,
                "pretax": 120e9, "tax": 20e9, "interest": 3e9, "ni": 100e9,
                "cfo": 120e9, "div": 20e9, "buyback": 30e9, "payout_ratio": 0.50,
                "rd": [30e9, 25e9, 20e9, 15e9], "diluted_shares": 7.5e9,
                "effective_tax_rate": 0.167, "roe_hist": 0.478,
                "roe_source": "historical average equity", "earnings_cagr": 0.22, "fcf_yield": 0.02,
                "company_ratios": {"margin": 0.44, "tax": 0.167, "capex_ratio": 0.12,
                                   "dep_ratio": 0.04, "nwc_ratio": -0.10}}

    def test_msft_high_roe_clipping_disables_ri_and_keeps_dcf(self):
        data = self.company_case()
        p = self.analysis_params()
        p["tangible_book"] = False
        result = NS["analyze_company"](data, p)
        self.assertFalse(result["RI Valid"])
        self.assertTrue(math.isnan(result["RI V/P"]))
        self.assertTrue(math.isfinite(result["DCF V/P"]))
        self.assertIsNone(result["_base"])
        self.assertTrue(math.isnan(result["RI Terminal Value Share"]))
        self.assertTrue(any("ROE input" in warning and "clipped" in warning for warning in result["warnings"]))

    def test_ko_negative_tangible_book_falls_back_and_high_pb_disables_ri(self):
        data = self.company_case("KO")
        data.update({"price": 90.66, "shares": 4.3e9, "revenue": 47e9, "rev_prev": 45e9,
                     "rev_cagr": 0.04, "ebit": 12e9, "ni": 4e9, "roe_hist": 0.16,
                     "payout_ratio": 0.70, "rd": [],
                     "company_ratios": {"margin": 0.25, "tax": 0.20, "capex_ratio": 0.06,
                                         "dep_ratio": 0.05, "nwc_ratio": -0.10}})
        data["years"][0].update({"equity": 25e9, "goodwill": 35e9, "debt": 40e9, "cash": 8e9})
        data["years"][1].update({"equity": 24e9, "goodwill": 34e9, "debt": 38e9, "cash": 7e9})
        p = self.analysis_params()
        result = NS["analyze_company"](data, p)
        self.assertFalse(result["RI Valid"])
        self.assertGreater(result["RI P/B"], 10.0)
        self.assertTrue(any("uses reported book equity instead" in warning for warning in result["warnings"]))
        self.assertTrue(any("adjusted RI P/B" in warning for warning in result["warnings"]))

    def test_unreliable_dcf_cannot_pass_candidate_thresholds(self):
        row = {"RI Valid": True, "DCF Reliable": False, "Valuation Quality": "OK",
               "DCF V/P": 1.8, "RI V/P": 1.8, "RI Conservative V/P": 1.2,
               "Implied Omega": 0.5, "Accruals": 0.01, "FLEV": 0.5,
               "ROE": 0.2, "Cost of Equity": 0.1}
        verdict = NS["classify_verdict"](row, 1.2, 1.2, 1.0, 1.5, 0.8, 0.1, 2.0)
        self.assertEqual(verdict, "WATCH")

    def test_nan_dcf_reliability_cannot_be_treated_as_true(self):
        row = {"RI Valid": True, "DCF Reliable": float("nan"), "Valuation Quality": "OK",
               "DCF V/P": 1.8, "RI V/P": 1.8, "RI Conservative V/P": 1.2,
               "Implied Omega": 0.5, "Accruals": 0.01, "FLEV": 0.5,
               "ROE": 0.2, "Cost of Equity": 0.1}
        verdict = NS["classify_verdict"](row, 1.2, 1.2, 1.0, 1.5, 0.8, 0.1, 2.0)
        self.assertEqual(verdict, "WATCH")

    def test_missing_payout_component_is_not_silently_assumed_zero(self):
        self.assertIsNone(NS["payout_ratio_from_cashflows"](100.0, 20.0, None))
        self.assertIsNone(NS["payout_ratio_from_cashflows"](100.0, None, 10.0))
        self.assertAlmostEqual(NS["payout_ratio_from_cashflows"](100.0, -20.0, -10.0), 0.30)

    def test_avg_payout_ratio_pools_multiple_years(self):
        d = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-31")]
        inc = pd.DataFrame({d[0]: [100.0], d[1]: [100.0], d[2]: [100.0]},
                           index=["Net Income Common Stockholders"])
        cf = pd.DataFrame({d[0]: [-20.0, -10.0], d[1]: [-30.0, -30.0], d[2]: [-10.0, -20.0]},
                          index=["Cash Dividends Paid", "Repurchase Of Capital Stock"])
        self.assertAlmostEqual(NS["avg_payout_ratio"](inc, cf), 120.0 / 300.0)
        self.assertIsNone(NS["avg_payout_ratio"](inc, pd.DataFrame()))

    def test_nonfinite_numeric_data_is_rejected(self):
        self.assertIsNone(NS["safe_float"](float("inf")))
        self.assertIsNone(NS["safe_float"](float("nan")))

    def test_nan_ri_valid_flag_cannot_pass_verdict(self):
        row = {"RI Valid": float("nan"), "DCF V/P": 1.5, "RI V/P": 1.8, "RI Conservative V/P": 1.2,
               "Implied Omega": 0.5, "Accruals": 0.01, "FLEV": 0.5,
               "ROE": 0.2, "Cost of Equity": 0.1}
        verdict = NS["classify_verdict"](row, 1.2, 1.2, 1.0, 1.5, 0.8, 0.1, 2.0)
        self.assertNotIn(verdict, ("CANDIDATE", "STRONG CANDIDATE"))

    def test_dcf_led_candidate_only_when_ri_not_required(self):
        row = {"RI Valid": False, "DCF Reliable": True, "Valuation Quality": "OK",
               "DCF V/P": 1.5, "RI V/P": float("nan"), "RI Conservative V/P": float("nan"),
               "Implied Omega": float("nan"), "Accruals": 0.0, "FLEV": 0.5,
               "ROE": 0.2, "Cost of Equity": 0.1}
        args = (1.2, 1.2, 1.0, 1.5, 0.8, 0.1, 2.0)
        self.assertEqual(NS["classify_verdict"](row, *args, ri_required=False), "CANDIDATE (DCF-led)")
        self.assertEqual(NS["classify_verdict"](row, *args), "WATCH")
        row["Valuation Quality"] = "REVIEW"
        self.assertEqual(NS["classify_verdict"](row, *args, ri_required=False), "WATCH")

    def test_ri_terminal_steady_state_payout_matches_growth(self):
        out = NS["residual_income_valuation"](self.data, 0.09, 0.8, self.path, 0.025)
        self.assertAlmostEqual(out["terminal_payout_ratio"], 1 - 0.025 / out["terminal_roe"])
        self.assertEqual(len(out["forecast"]), 10)
        self.assertNotIn("Growth", out["forecast"][0])

    def test_ri_rejects_missing_or_implausible_payout(self):
        for payout in (None, -0.01, 1.01):
            with self.subTest(payout=payout), self.assertRaises(ValueError):
                NS["residual_income_valuation"]({**self.data, "payout_ratio": payout}, 0.09, 0.8, self.path, 0.025)

    def test_ri_rejects_book_equity_including_minority_interest(self):
        bad = {**self.data, "years": [{**self.data["years"][0], "equity_source": "Total Equity Gross Minority Interest"}, self.data["years"][1]]}
        with self.assertRaisesRegex(ValueError, "minority interest"):
            NS["residual_income_valuation"](bad, 0.09, 0.8, self.path, 0.025)

    def test_omega_solver_returns_finite_solution_when_bracketed(self):
        val = NS["calculate_implied_omega"](self.data, 0.09, self.path, 0.025)
        self.assertTrue(val is None or 0 <= val <= 1)

    def test_dcf_margin_fade_starts_and_ends_at_configured_margins(self):
        dcf_inputs = {"margin": 0.15, "tax": 0.21, "capex_ratio": 0.04,
                      "dep_ratio": 0.03, "nwc_ratio": 0.05, "term_roic": 0.12,
                      "margin_end": 0.30}
        dcf_data = {**self.data, "revenue": 4000.0,
                    "years": [{"debt": 100.0, "cash": 50.0, "minority": 0.0}]}
        out = NS["dcf_valuation"](dcf_data, 0.08, NS["growth_path"](0.08, 0.025, 10), 0.025, **dcf_inputs)
        first_margin = out["forecast"][0]["EBIT"] / out["forecast"][0]["Revenue"]
        last_margin = out["forecast"][-1]["EBIT"] / out["forecast"][-1]["Revenue"]
        self.assertAlmostEqual(first_margin, 0.15)
        self.assertAlmostEqual(last_margin, 0.30)

    def test_dcf_sensitivity_behaves_directionally(self):
        dcf_inputs = {"margin": 0.20, "tax": 0.21, "capex_ratio": 0.05,
                      "dep_ratio": 0.03, "nwc_ratio": 0.05, "term_roic": 0.15}
        data = {**self.data, "revenue": 4000.0, "shares": 100.0, "price": 50.0,
                "years": [{"debt": 800.0, "cash": 500.0, "minority": 0.0}]}
        g = NS["growth_path"](0.08, 0.025, 10)
        base = NS["dcf_valuation"](data, 0.08, g, 0.025, **dcf_inputs)["value_per_share"]
        higher_wacc = NS["dcf_valuation"](data, 0.09, g, 0.025, **dcf_inputs)["value_per_share"]
        higher_terminal_growth = NS["dcf_valuation"](data, 0.08, g, 0.035, **dcf_inputs)["value_per_share"]
        self.assertLess(higher_wacc, base)
        self.assertGreater(higher_terminal_growth, base)

    def test_implied_irr_reprices_dcf_to_market(self):
        data = {**self.data, "revenue": 4000.0, "shares": 100.0, "price": 50.0,
                "years": [{"debt": 800.0, "cash": 500.0, "minority": 0.0}]}
        inp = {"margin": 0.20, "tax": 0.21, "capex_ratio": 0.05,
               "dep_ratio": 0.03, "nwc_ratio": 0.05, "term_roic": 0.15}
        g = NS["growth_path"](0.08, 0.025, 10)
        irr = NS["implied_irr"](data, g, 0.025, inp)
        self.assertIsNotNone(irr)
        self.assertAlmostEqual(NS["dcf_valuation"](data, irr, g, 0.025, **inp)["V/P"], 1.0, places=4)

    def test_terminal_capex_multiple_is_explicit_and_lowers_value_when_raised(self):
        data = {**self.data, "revenue": 4000.0, "shares": 100.0, "price": 50.0,
                "years": [{"debt": 800.0, "cash": 500.0, "minority": 0.0}]}
        inp = {"margin": 0.20, "tax": 0.21, "capex_ratio": 0.10,
               "dep_ratio": 0.03, "nwc_ratio": 0.05, "term_roic": 0.15}
        g = NS["growth_path"](0.08, 0.025, 10)
        low = NS["dcf_valuation"](data, 0.09, g, 0.025, capex_terminal_mult=1.1, **inp)["value_per_share"]
        high = NS["dcf_valuation"](data, 0.09, g, 0.025, capex_terminal_mult=2.5, **inp)["value_per_share"]
        self.assertLess(high, low)
        with self.assertRaisesRegex(ValueError, "CapEx / D&A multiple"):
            NS["dcf_valuation"](data, 0.09, g, 0.025, capex_terminal_mult=9.0, **inp)

    def test_dcf_requires_cash_and_valid_terminal_assumptions(self):
        dcf_inputs = {"margin": 0.15, "tax": 0.21, "capex_ratio": 0.04,
                      "dep_ratio": 0.03, "nwc_ratio": 0.05, "term_roic": 0.12}
        dcf_data = {**self.data, "years": [{"debt": 100.0, "cash": 50.0, "minority": 0.0}]}
        out = NS["dcf_valuation"](dcf_data, 0.08, self.path, 0.025, **dcf_inputs)
        self.assertTrue(math.isfinite(out["value_per_share"]))
        with self.assertRaisesRegex(ValueError, "cash unavailable"):
            NS["dcf_valuation"]({**dcf_data, "years": [{"debt": 100.0, "cash": None, "minority": 0.0}]},
                                 0.08, self.path, 0.025, **dcf_inputs)

    def test_cagr_uses_calendar_span_and_rejects_loss_endpoint(self):
        cols = pd.to_datetime(["2025-12-31", "2022-12-31", "2019-12-31"])
        cagr = NS["_cagr_between"]([133.1, None, 100.0], cols)
        self.assertAlmostEqual(cagr, (133.1 / 100.0) ** (1 / 6.0015) - 1, places=4)
        self.assertIsNone(NS["_cagr_between"]([-5.0, 100.0], pd.to_datetime(["2025-12-31", "2020-12-31"])))

    def test_total_debt_does_not_omit_current_debt(self):
        balance = pd.DataFrame({pd.Timestamp("2025-12-31"): [300.0, 100.0]},
                               index=["Long Term Debt", "Current Debt"])
        self.assertEqual(NS["total_debt_for_period"](balance, 0), 400.0)
        incomplete = pd.DataFrame({pd.Timestamp("2025-12-31"): [300.0]}, index=["Long Term Debt"])
        self.assertIsNone(NS["total_debt_for_period"](incomplete, 0))

    def test_financial_sector_exclusion(self):
        skip, _ = NS["should_skip_company"]({"sector": "Financial Services", "industry": "", "quote_type": "EQUITY"})
        self.assertTrue(skip)
        skip, _ = NS["should_skip_company"]({"sector": "Technology", "industry": "Software", "quote_type": "EQUITY"})
        self.assertFalse(skip)


    def test_dcf_guardrails_allow_negative_working_capital_and_higher_capex_intensity(self):
        limits = NS["DCF_LIMITS"]
        self.assertEqual(limits["nwc_ratio"][0], -0.25)
        self.assertEqual(limits["capex_ratio"][1], 0.50)
        values, sources = NS["resolve_dcf_inputs"](
            {"company_ratios": {"margin": 0.20, "tax": 0.21, "capex_ratio": 0.35,
                                 "dep_ratio": 0.05, "nwc_ratio": -0.12}},
            {"override": False, "fb_margin": 0.15, "fb_tax": 0.21, "fb_capex": 0.04,
             "fb_dep": 0.035, "fb_nwc": 0.05})
        self.assertAlmostEqual(values["capex_ratio"], 0.35)
        self.assertEqual(sources["capex_ratio"], "company")
        self.assertAlmostEqual(values["nwc_ratio"], -0.12)
        self.assertEqual(sources["nwc_ratio"], "company")

    def test_dcf_observed_ratios_outside_soft_range_are_not_clipped(self):
        values, sources = NS["resolve_dcf_inputs"](
            {"company_ratios": {"margin": 0.20, "tax": 0.21, "capex_ratio": 0.72,
                                 "dep_ratio": 0.05, "nwc_ratio": -0.40}},
            {"override": False, "fb_margin": 0.15, "fb_tax": 0.21, "fb_capex": 0.04,
             "fb_dep": 0.035, "fb_nwc": 0.05})
        self.assertAlmostEqual(values["capex_ratio"], 0.72)
        self.assertEqual(sources["capex_ratio"], "company (outside usual range)")
        self.assertAlmostEqual(values["nwc_ratio"], -0.40)
        self.assertEqual(sources["nwc_ratio"], "company (outside usual range)")

    def test_dcf_extreme_ratio_uses_visible_fallback_instead_of_clipping(self):
        values, sources = NS["resolve_dcf_inputs"](
            {"company_ratios": {"margin": 0.20, "tax": 0.21, "capex_ratio": 3.2,
                                 "dep_ratio": 0.05, "nwc_ratio": 0.05}},
            {"override": False, "fb_margin": 0.15, "fb_tax": 0.21, "fb_capex": 0.04,
             "fb_dep": 0.035, "fb_nwc": 0.05})
        self.assertAlmostEqual(values["capex_ratio"], 0.04)
        self.assertEqual(sources["capex_ratio"], "sidebar fallback (company ratio outside hard bounds)")


    def synthetic_company(self, **overrides):
        data = {
            "ticker": "TEST", "price": 50.0, "price_source": "Yahoo current/regular market price",
            "shares": 100.0, "shares_source": "Yahoo current shares outstanding",
            "sector": "Technology", "industry": "Software", "quote_type": "EQUITY",
            "beta": 1.0, "beta_source": "Yahoo beta",
            "years": [
                {"assets": 5000.0, "liab": 2000.0, "debt": 800.0, "cash": 500.0,
                 "equity": 3000.0, "equity_source": "Stockholders Equity", "cur_assets": 1000.0,
                 "cur_liab": 600.0, "cur_debt": 100.0, "goodwill": 100.0, "minority": 0.0},
                {"assets": 4700.0, "liab": 1900.0, "debt": 750.0, "cash": 450.0,
                 "equity": 2800.0, "equity_source": "Stockholders Equity", "cur_assets": 950.0,
                 "cur_liab": 580.0, "cur_debt": 90.0, "goodwill": 90.0, "minority": 0.0},
            ],
            "revenue": 4000.0, "rev_prev": 3700.0, "rev_cagr": 0.08, "ebit": 800.0,
            "pretax": 600.0, "tax": 120.0, "interest": 50.0, "ni": 500.0,
            "cfo": 600.0, "div": 100.0, "buyback": 30.0, "payout_ratio": 0.26,
            "rd": [], "diluted_shares": 100.0, "effective_tax_rate": 0.20,
            "roe_hist": 0.17, "roe_source": "historical average equity", "earnings_cagr": 0.10,
            "fcf_yield": 0.05,
            "company_ratios": {"margin": 0.20, "tax": 0.20, "capex_ratio": 0.05,
                                "dep_ratio": 0.03, "nwc_ratio": 0.10},
        }
        data.update(overrides)
        return data

    def model_params(self):
        return {
            "use_capm": True, "rf": 0.043, "erp": 0.045, "kd": 0.055,
            "flat_ke": 0.09, "flat_wacc": 0.08, "terminal_growth": 0.025,
            "years": 10, "use_company_growth": True, "max_start_growth": 0.20,
            "start_growth": 0.06, "omega": 0.80, "max_pb": 10.0, "max_roe": 0.50,
            "tangible_book": True, "capitalize_rd": True, "override": False,
            "fb_margin": 0.15, "fb_tax": 0.21, "fb_capex": 0.04, "fb_dep": 0.035,
            "fb_nwc": 0.05, "roic_cap": 0.15, "growth_threshold": 0.10,
            "growth_years": 15, "quality_roic_cap": 0.25, "margin_expansion": 0.0,
        }

    def test_lease_exclusion_reduces_dcf_net_debt_only_when_total_debt_includes_it(self):
        data = self.synthetic_company()
        data["years"][0].update({"debt": 1000.0, "lease": 300.0, "lease_in_debt": True})
        data["years"][1].update({"debt": 950.0, "lease": 280.0, "lease_in_debt": True})
        on = NS["analyze_company"](data, {**self.model_params(), "exclude_leases": True})
        off = NS["analyze_company"](data, {**self.model_params(), "exclude_leases": False})
        self.assertAlmostEqual((off["Net Debt (B)"] - on["Net Debt (B)"]) * 1e9, 300.0, places=3)
        # If Yahoo's Total Debt row was absent (debt reconstructed), no lease is stripped.
        data["years"][0]["lease_in_debt"] = False
        reconstructed = NS["analyze_company"](data, {**self.model_params(), "exclude_leases": True})
        self.assertAlmostEqual(reconstructed["Net Debt (B)"], off["Net Debt (B)"])

    def test_rate_and_growth_override_parsers(self):
        r = NS["parse_rate_overrides"]("sgd=0.028/0.055, HKD=0.04, bad, USD=abc")
        self.assertEqual(r["SGD"], (0.028, 0.055))
        self.assertEqual(r["HKD"], (0.04, None))
        self.assertNotIn("USD", r)
        base = {"rf": 0.043, "erp": 0.045, "rate_overrides": r}
        self.assertEqual(NS["rates_for"]({"currency": "SGD"}, base), (0.028, 0.055))
        self.assertEqual(NS["rates_for"]({"currency": "HKD"}, base), (0.04, 0.045))
        self.assertEqual(NS["rates_for"]({"currency": "USD"}, base), (0.043, 0.045))
        g = NS["parse_growth_overrides"]("msft=0.12, KO = 0.04; X=9, Y=5%")
        self.assertAlmostEqual(g["MSFT"], 0.12)
        self.assertAlmostEqual(g["KO"], 0.04)
        self.assertAlmostEqual(g["Y"], 0.05)
        self.assertNotIn("X", g)

    def test_per_ticker_growth_override_sets_start_growth(self):
        data = self.synthetic_company()
        p = {**self.model_params(), "growth_overrides": {"TEST": 0.12}}
        out = NS["analyze_company"](data, p)
        self.assertAlmostEqual(out["Start Growth"], 0.12)

    def test_dcf_reliability_flag_requires_reported_core_inputs(self):
        out = NS["analyze_company"](self.synthetic_company(), self.model_params())
        self.assertTrue(out["DCF Reliable"])
        fallback_data = self.synthetic_company(company_ratios={"margin": 0.20, "tax": 0.20,
            "capex_ratio": None, "dep_ratio": 0.03, "nwc_ratio": 0.10})
        fallback_out = NS["analyze_company"](fallback_data, self.model_params())
        self.assertFalse(fallback_out["DCF Reliable"])
        self.assertTrue(any("sidebar fallback" in w for w in fallback_out["warnings"]))

    def test_low_beta_reports_normalized_beta_sensitivity_without_changing_base_rate(self):
        data = self.synthetic_company(beta=0.32)
        params = self.model_params()
        params["rf"] = 0.04
        params["erp"] = 0.05
        out = NS["analyze_company"](data, params)
        self.assertAlmostEqual(out["Normalized Beta Sensitivity"], 0.67 * 0.32 + 0.33, places=6)
        self.assertAlmostEqual(out["Cost of Equity"], 0.04 + 0.32 * 0.05, places=6)
        self.assertGreater(out["Normalized Cost of Equity"], out["Cost of Equity"])
        self.assertTrue(any("Blume-adjusted beta sensitivity" in w for w in out["warnings"]))

    def test_blume_option_moves_base_cost_of_equity(self):
        data = self.synthetic_company(beta=0.32)
        params = self.model_params()
        params["rf"], params["erp"], params["use_blume_beta"] = 0.04, 0.05, True
        out = NS["analyze_company"](data, params)
        self.assertAlmostEqual(out["Cost of Equity"], 0.04 + (0.67 * 0.32 + 0.33) * 0.05, places=6)

    def test_ri_reports_pb_limit_even_when_roe_clipping_already_invalidates_ri(self):
        data = self.synthetic_company(beta=0.32, roe_hist=0.428, price=90.76, shares=4.3e9)
        data["years"][0].update({"equity": 25e9, "goodwill": 35e9})
        data["years"][1].update({"equity": 24e9, "goodwill": 34e9})
        out = NS["analyze_company"](data, self.model_params())
        self.assertFalse(out["RI Valid"])
        self.assertIn("clipped", out["RI Status"].lower())
        self.assertIn("adjusted ri p/b", out["RI Status"].lower())
        self.assertTrue(any("Additional RI validity issue" in w for w in out["warnings"]))

    def test_low_capm_cost_of_equity_downgrades_dcf_reliability(self):
        data = self.synthetic_company(beta=0.2)
        params = self.model_params()
        params["rf"] = 0.04
        params["erp"] = 0.04
        out = NS["analyze_company"](data, params)
        self.assertLess(out["Cost of Equity"], 0.07)
        self.assertFalse(out["DCF Reliable"])
        self.assertTrue(any("below 7%" in w for w in out["warnings"]))

    def test_terminal_value_concentration_downgrades_dcf_reliability(self):
        data = self.synthetic_company()
        params = self.model_params()
        params["years"] = 5
        params["growth_years"] = 10
        params["rf"] = 0.03
        params["erp"] = 0.03
        out = NS["analyze_company"](data, params)
        if out["DCF Terminal Value Share"] > 0.65:
            self.assertFalse(out["DCF Reliable"])
            self.assertTrue(any("terminal value exceeds 65%" in w for w in out["warnings"]))

    def test_ri_failure_reason_is_exposed_in_result(self):
        data = self.synthetic_company(payout_ratio=None)
        out = NS["analyze_company"](data, self.model_params())
        self.assertFalse(out["RI Valid"])
        self.assertTrue(out["RI Status"])
        self.assertIn("payout", out["RI Status"].lower())

    def test_ri_is_disabled_when_roe_is_only_yahoo_fallback(self):
        out = NS["analyze_company"](
            self.synthetic_company(roe_source="Yahoo-reported ROE fallback"), self.model_params())
        self.assertFalse(out["RI Valid"])
        self.assertTrue(math.isnan(out["RI V/P"]))
        self.assertTrue(any("RI disabled" in w for w in out["warnings"]))

    def test_candidate_verdict_requires_screening_grade_dcf(self):
        row = {"RI Valid": True, "DCF Reliable": True, "DCF V/P": 1.5, "RI V/P": 1.8,
               "RI Conservative V/P": 1.2, "Implied Omega": 0.5, "Accruals": 0.0,
               "FLEV": 0.5, "ROE": 0.2, "Cost of Equity": 0.1, "Valuation Quality": "OK"}
        args = (1.2, 1.2, 1.0, 1.5, 0.8, 0.1, 2.0)
        self.assertEqual(NS["classify_verdict"](row, *args), "STRONG CANDIDATE")
        row["DCF Reliable"] = False
        self.assertEqual(NS["classify_verdict"](row, *args), "WATCH")

    def test_missing_risk_inputs_never_become_candidate(self):
        row = {"RI Valid": True, "DCF V/P": 1.5, "RI V/P": 1.8, "RI Conservative V/P": 1.2,
               "Implied Omega": 0.5, "Accruals": float("nan"), "FLEV": 0.5,
               "ROE": 0.2, "Cost of Equity": 0.1}
        verdict = NS["classify_verdict"](row, 1.2, 1.2, 1.0, 1.5, 0.8, 0.1, 2.0)
        self.assertEqual(verdict, "INSUFFICIENT DATA")

    def test_fiscal_periods_align_by_date_not_column_position(self):
        income = pd.DataFrame({pd.Timestamp("2025-12-31"): [100.0], pd.Timestamp("2024-12-31"): [80.0]}, index=["Net Income Common Stockholders"])
        balance = pd.DataFrame({pd.Timestamp("2024-12-31"): [900.0], pd.Timestamp("2023-12-31"): [800.0], pd.Timestamp("2025-12-31"): [1100.0]}, index=["Stockholders Equity"])
        aligned = NS["aligned_statement_periods"](income, balance)
        self.assertEqual(aligned[0][0], pd.Timestamp("2025-12-31"))
        self.assertEqual(pick(balance, ["Stockholders Equity"], aligned[0][2]), 1100.0)
        self.assertEqual(pick(balance, ["Stockholders Equity"], aligned[0][3]), 900.0)

    def test_missing_beta_does_not_silently_become_one(self):
        data = self.synthetic_company(beta=None)
        with self.assertRaisesRegex(ValueError, "beta is missing"):
            NS["discount_rates"](data, self.model_params())

    def test_tax_normalization_uses_median_not_latest_only(self):
        inc = pd.DataFrame({pd.Timestamp("2025-12-31"): [100, 40], pd.Timestamp("2024-12-31"): [100, 20], pd.Timestamp("2023-12-31"): [100, 30]}, index=["Pretax Income", "Tax Provision"])
        rate, source = NS["normalized_tax_rate"](inc)
        self.assertAlmostEqual(rate, 0.30)
        self.assertIn("median", source)

    def test_dcf_refuses_missing_minority_interest(self):
        data = self.synthetic_company()
        data["years"][0]["minority"] = None
        with self.assertRaisesRegex(ValueError, "minority interest is unavailable"):
            NS["dcf_valuation"](data, 0.09, self.path, 0.025, margin=0.20, tax=0.20,
                                capex_ratio=0.05, dep_ratio=0.03, nwc_ratio=0.10, term_roic=0.12)

    def test_liabilities_are_derived_from_balance_sheet_identity_when_missing(self):
        balance = pd.DataFrame({pd.Timestamp("2025-12-31"): [1000.0, 700.0]},
                               index=["Total Assets", "Total Equity Gross Minority Interest"])
        value, source = NS["total_liabilities_for_period"](balance, 0)
        self.assertEqual(value, 300.0)
        self.assertIn("derived", source)

    def test_missing_yahoo_minority_can_be_zero_assumed_for_exploratory_dcf(self):
        data = self.company_case("MSFT")
        for year in data["years"]:
            year["minority"] = 0.0
            year["minority_assumed_zero"] = True
            year["minority_source"] = "not separately reported by Yahoo; assumed zero pending filing verification"
        result = NS["analyze_company"](data, self.analysis_params())
        self.assertTrue(math.isfinite(result["DCF V/P"]))
        self.assertFalse(result["DCF Reliable"])
        self.assertTrue(any("not separately reported by Yahoo" in warning for warning in result["warnings"]))

    def test_normalized_beta_dcf_value_is_reported_separately(self):
        data = self.company_case("MSFT")
        result = NS["analyze_company"](data, self.analysis_params())
        self.assertTrue(math.isfinite(result["DCF V/P at Normalized WACC"]))
        self.assertAlmostEqual(result["Normalized Beta Sensitivity"], 0.67 * data["beta"] + 0.33)


if __name__ == "__main__" and TEST_MODE:
    unittest.main(verbosity=2)
