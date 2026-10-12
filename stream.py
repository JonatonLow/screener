import math
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Penman Stock Screener",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Penman Residual-Income + DCF Stock Screener")

st.markdown(
    """
This screener combines:
- a residual-income valuation based on Penman-style operating analysis
- an FCFF enterprise DCF valuation

It reports both values side by side and preserves the original screener flow.
"""
)

st.caption(
    "Educational research tool — not investment advice. "
    "Yahoo Finance data may contain missing or inconsistent accounting fields."
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def pick(df, names, col=0):
    """
    Find the first available accounting item from a list of
    possible Yahoo Finance row names.
    """
    if df is None or df.empty:
        return None
    if col >= df.shape[1]:
        return None

    for name in names:
        if name in df.index:
            value = df.loc[name].iloc[col]
            if pd.notna(value):
                return safe_float(value)
    return None


def fmt_pct(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1%}"


def fmt_num(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.2f}"


def clamp(value, lower, upper):
    if value is None:
        return None
    return max(lower, min(upper, value))


def as_positive_spending(value):
    """
    Convert negative reported capital expenditure into positive spending.
    """
    if value is None:
        return None
    value = safe_float(value)
    if value is None:
        return None
    return abs(value) if value < 0 else value


# ============================================================
# FETCH DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol):
    ticker = yf.Ticker(symbol)

    try:
        info = ticker.info
    except Exception:
        info = {}

    try:
        balance_sheet = ticker.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        income_statement = ticker.financials
    except Exception:
        income_statement = pd.DataFrame()

    try:
        cash_flow = ticker.cashflow
    except Exception:
        cash_flow = pd.DataFrame()

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    if price is None:
        try:
            history = ticker.history(period="5d")
            if not history.empty:
                price = float(history["Close"].dropna().iloc[-1])
        except Exception:
            pass

    # --------------------------------------------------------
    # Shares outstanding
    # --------------------------------------------------------
    shares = info.get("sharesOutstanding")
    if shares is None:
        shares = pick(balance_sheet, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 0)

    if shares is None:
        shares = pick(income_statement, ["Weighted Average Shares Basic", "Weighted Average Shares Diluted"], 0)

    # --------------------------------------------------------
    # Balance sheet and cashflow values
    # --------------------------------------------------------
    years = []

    for col in [0, 1]:
        y = {
            "assets": pick(balance_sheet, ["Total Assets"], col),
            "liab": pick(balance_sheet, ["Total Liabilities Net Minority Interest", "Total Liabilities"], col),
            "debt": pick(balance_sheet, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"], col),
            "cash": pick(balance_sheet, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial"], col),
            "equity": pick(balance_sheet, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"], col),
            "current_assets": pick(balance_sheet, ["Total Current Assets"], col),
            "current_liabilities": pick(balance_sheet, ["Total Current Liabilities"], col),
        }
        years.append(y)

    # --------------------------------------------------------
    # Income statement
    # --------------------------------------------------------
    revenue = pick(income_statement, ["Total Revenue", "Revenue"], 0)
    ebit = pick(income_statement, ["EBIT", "Operating Income"], 0)
    pretax = pick(income_statement, ["Pretax Income"], 0)
    tax = pick(income_statement, ["Tax Provision", "Income Tax Expense"], 0)
    interest = pick(income_statement, ["Interest Expense", "Interest Expense Non Operating"], 0)
    net_income = pick(income_statement, ["Net Income Common Stockholders", "Net Income", "Net Income Including Noncontrolling Interests"], 0)

    # --------------------------------------------------------
    # Cash flow
    # --------------------------------------------------------
    cfo = pick(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)
    dividends = pick(cash_flow, ["Cash Dividends Paid", "Common Stock Dividend Paid"], 0)
    depreciation = pick(cash_flow, ["Depreciation", "Depreciation And Amortization", "D&A", "Depreciation, Depletion And Amortization"], 0)
    capex_raw = pick(cash_flow, ["Capital Expenditure", "Capital Expenditures"], 0)
    capex = as_positive_spending(capex_raw)

    # --------------------------------------------------------
    # DCF-specific derived values
    # --------------------------------------------------------
    current_nwc = None
    prior_nwc = None

    if years[0]["current_assets"] is not None and years[0]["current_liabilities"] is not None:
        current_nwc = years[0]["current_assets"] - years[0]["current_liabilities"]

    if len(years) > 1 and years[1]["current_assets"] is not None and years[1]["current_liabilities"] is not None:
        prior_nwc = years[1]["current_assets"] - years[1]["current_liabilities"]

    effective_tax_rate = None
    if pretax is not None and pretax != 0 and tax is not None:
        effective_tax_rate = max(0.0, min(0.45, tax / pretax))

    fcff_base = None
    if ebit is not None and effective_tax_rate is not None:
        fcff_base = ebit * (1 - effective_tax_rate) + (depreciation or 0) - (capex or 0)
        if current_nwc is not None and prior_nwc is not None:
            fcff_base = fcff_base - (current_nwc - prior_nwc)

    return {
        "ticker": symbol.upper(),
        "price": safe_float(price),
        "shares": safe_float(shares),
        "sector": info.get("sector", ""),
        "years": years,
        "revenue": safe_float(revenue),
        "ebit": safe_float(ebit),
        "pretax": safe_float(pretax),
        "tax": safe_float(tax),
        "interest": safe_float(interest),
        "ni": safe_float(net_income),
        "cfo": safe_float(cfo),
        "div": safe_float(dividends),
        "depreciation": safe_float(depreciation),
        "capex": safe_float(capex),
        "nwc": safe_float(current_nwc),
        "prior_nwc": safe_float(prior_nwc),
        "fcff_base": safe_float(fcff_base),
        "effective_tax_rate": safe_float(effective_tax_rate),
    }


# ============================================================
# HISTORICAL QUALITY / GROWTH
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def historical_analysis(data):
    ticker = yf.Ticker(data["ticker"])

    try:
        income = ticker.financials
    except Exception:
        income = pd.DataFrame()

    try:
        balance = ticker.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    result = {
        "ROE_5Y": None,
        "Earnings_CAGR": None,
        "FCF_Yield": None,
    }

    if income.empty or balance.empty:
        return result

    earnings_row = None
    for name in ("Net Income Common Stockholders", "Net Income"):
        if name in income.index:
            earnings_row = income.loc[name]
            break

    if earnings_row is not None:
        earnings = [safe_float(earnings_row.iloc[i]) for i in range(len(earnings_row))]
        equity_names = ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]

        roe_values = []
        for i in range(min(5, len(earnings), max(0, balance.shape[1] - 1))):
            equity_previous = pick(balance, equity_names, i + 1)
            earnings_value = earnings[i]
            if earnings_value is not None and equity_previous is not None and equity_previous > 0:
                roe_values.append(earnings_value / equity_previous)

        if roe_values:
            result["ROE_5Y"] = float(pd.Series(roe_values).median())

        valid_earnings = [(i, value) for i, value in enumerate(earnings) if value is not None and value > 0]
        if len(valid_earnings) >= 3:
            newest_index, newest = valid_earnings[0]
            oldest_index, oldest = valid_earnings[-1]
            years = oldest_index - newest_index
            if years > 0:
                result["Earnings_CAGR"] = (newest / oldest) ** (1 / years) - 1

    try:
        cash_flow = ticker.cashflow
    except Exception:
        cash_flow = pd.DataFrame()

    cfo = pick(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)
    capex = pick(cash_flow, ["Capital Expenditure", "Capital Expenditures"], 0)
    capex = as_positive_spending(capex)
    price = data["price"]
    shares = data["shares"]

    if cfo is not None and capex is not None and price is not None and shares is not None and price > 0 and shares > 0:
        result["FCF_Yield"] = (cfo + capex) / (price * shares)

    return result


# ============================================================
# PENMAN OPERATING ANALYSIS
# ============================================================

def operating_analysis(data):
    y0 = data["years"][0]
    y1 = data["years"][1]

    def calculate_noa(year):
        operating_assets = (year["assets"] - year["cash"])
        operating_liabilities = (year["liab"] - year["debt"])
        return operating_assets - operating_liabilities

    def calculate_nfo(year):
        return year["debt"] - year["cash"]

    noa0 = calculate_noa(y0)
    noa1 = calculate_noa(y1)
    nfo0 = calculate_nfo(y0)
    nfo1 = calculate_nfo(y1)

    average_noa = (noa0 + noa1) / 2
    average_nfo = (nfo0 + nfo1) / 2

    if data["pretax"] is not None and data["tax"] is not None and data["pretax"] > 0:
        tax_rate = max(0.0, min(0.40, data["tax"] / data["pretax"]))
    else:
        tax_rate = 0.21

    operating_income_after_tax = data["ebit"] * (1 - tax_rate)
    interest = abs(data["interest"] or 0)
    net_financial_expense = interest * (1 - tax_rate)

    rnoa = (operating_income_after_tax / average_noa) if average_noa and average_noa > 0 else None
    nbc = (net_financial_expense / average_nfo) if average_nfo and average_nfo > 0 else None
    flev = (nfo0 / y0["equity"]) if y0["equity"] and y0["equity"] > 0 else None
    spread = (rnoa - nbc) if rnoa is not None and nbc is not None else None

    average_assets = (y0["assets"] + y1["assets"]) / 2
    if data["cfo"] is not None and average_assets and average_assets > 0:
        accruals = (data["ni"] - data["cfo"]) / average_assets
    else:
        accruals = None

    return {
        "tax_rate": tax_rate,
        "RNOA": rnoa,
        "NBC": nbc,
        "FLEV": flev,
        "spread": spread,
        "accruals": accruals,
    }


# ============================================================
# RESIDUAL INCOME VALUATION
# ============================================================

def residual_income_valuation(data, required_return, omega, terminal_growth, forecast_years=5):
    y0 = data["years"][0]
    y1 = data["years"][1]

    current_book_value = y0["equity"]
    previous_book_value = y1["equity"]

    net_income = data["ni"]
    shares = data["shares"]
    price = data["price"]

    if current_book_value is None or current_book_value <= 0:
        raise ValueError("invalid current book equity")
    if previous_book_value is None or previous_book_value <= 0:
        raise ValueError("invalid prior book equity")
    if net_income is None:
        raise ValueError("missing net income")
    if shares is None or shares <= 0:
        raise ValueError("missing shares outstanding")
    if price is None or price <= 0:
        raise ValueError("missing market price")
    if required_return <= terminal_growth:
        raise ValueError("required return must exceed terminal growth")

    historical_roe = net_income / previous_book_value
    roe_base = clamp(historical_roe, -0.10, 0.50)

    if data["div"] is not None and net_income > 0 and data["div"] != 0:
        payout_ratio = max(0.0, min(1.0, abs(data["div"]) / net_income))
    else:
        payout_ratio = 0.0

    book_value = current_book_value
    present_value_residual_income = 0.0
    forecast = []
    last_residual_income = 0.0

    for year in range(1, forecast_years + 1):
        forecast_roe = required_return + (roe_base - required_return) * (omega ** year)
        beginning_book_value = book_value
        earnings = forecast_roe * beginning_book_value
        residual_income = earnings - (required_return * beginning_book_value)
        dividends = earnings * payout_ratio
        ending_book_value = beginning_book_value + earnings - dividends
        pv_residual_income = residual_income / ((1 + required_return) ** year)
        present_value_residual_income += pv_residual_income

        forecast.append(
            {
                "Year": year,
                "Beginning BV": beginning_book_value,
                "ROE": forecast_roe,
                "Earnings": earnings,
                "Residual Income": residual_income,
                "Ending BV": ending_book_value,
                "PV Residual Income": pv_residual_income,
            }
        )

        book_value = ending_book_value
        last_residual_income = residual_income

    terminal_residual_income = last_residual_income * (1 + terminal_growth)
    terminal_value = terminal_residual_income / (required_return - terminal_growth)
    pv_terminal = terminal_value / ((1 + required_return) ** forecast_years)

    equity_value = current_book_value + present_value_residual_income + pv_terminal
    value_per_share = equity_value / shares
    market_cap = price * shares
    price_to_book = market_cap / current_book_value
    value_to_price = value_per_share / price
    terminal_percentage = (pv_terminal / equity_value) if equity_value != 0 else None

    return {
        "book_value": current_book_value,
        "historical_roe": historical_roe,
        "payout": payout_ratio,
        "value_per_share": value_per_share,
        "V/P": value_to_price,
        "P/B": price_to_book,
        "equity_value": equity_value,
        "PV_RI": present_value_residual_income,
        "PV_terminal": pv_terminal,
        "terminal_percentage": terminal_percentage,
        "forecast": forecast,
    }


# ============================================================
# IMPLIED ROE PERSISTENCE
# ============================================================

def calculate_implied_omega(data, required_return, terminal_growth, target_vp=1.0):
    """Find the ROE persistence needed to reach a target V/P."""
    def model_vp(omega):
        try:
            value_to_price = residual_income_valuation(
                data,
                required_return=required_return,
                omega=omega,
                terminal_growth=terminal_growth,
                forecast_years=5,
            )["V/P"]
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError):
            return None

        if pd.isna(value_to_price):
            return None
        return float(value_to_price)

    vp_low = model_vp(0.0)
    vp_high = model_vp(1.0)

    if vp_low is None or vp_high is None:
        return None
    if vp_low >= target_vp:
        return 0.0
    if vp_high < target_vp:
        return None

    low = 0.0
    high = 1.0
    for _ in range(50):
        middle = (low + high) / 2
        vp_middle = model_vp(middle)
        if vp_middle is None:
            return None
        if vp_middle < target_vp:
            low = middle
        else:
            high = middle
    return (low + high) / 2


# ============================================================
# DISCOUNTED CASH FLOW VALUATION
# ============================================================

def dcf_valuation(
    data,
    wacc,
    terminal_growth,
    revenue_growth,
    operating_margin,
    tax_rate,
    capex_ratio,
    dep_ratio,
    nwc_ratio,
    forecast_years=5,
):
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth")

    base_revenue = data["revenue"]
    base_ebit = data["ebit"]
    base_nwc = data["nwc"]
    base_depreciation = data["depreciation"]
    base_capex = data["capex"]
    net_debt = ((data["years"][0]["debt"] or 0) - (data["years"][0]["cash"] or 0))
    shares = data["shares"]
    price = data["price"]

    if base_revenue is None or base_revenue <= 0:
        raise ValueError("missing revenue for DCF")
    if base_ebit is None:
        raise ValueError("missing EBIT for DCF")
    if shares is None or shares <= 0:
        raise ValueError("missing shares outstanding")
    if price is None or price <= 0:
        raise ValueError("missing market price")

    base_fcff = data["fcff_base"]
    if base_fcff is None:
        base_fcff = base_ebit * (1 - tax_rate) + (base_depreciation or 0) - (base_capex or 0)

    revenue = base_revenue
    nwc_prev = base_nwc if base_nwc is not None else revenue * nwc_ratio
    forecast = []
    pv_fcff = 0.0

    for year in range(1, forecast_years + 1):
        revenue = revenue * (1 + revenue_growth)
        ebit = revenue * operating_margin
        depreciation = revenue * dep_ratio
        capex = revenue * capex_ratio
        nwc = revenue * nwc_ratio
        delta_nwc = nwc - nwc_prev
        fcff = ebit * (1 - tax_rate) + depreciation - capex - delta_nwc

        pv_fcff_year = fcff / ((1 + wacc) ** year)
        pv_fcff += pv_fcff_year

        forecast.append(
            {
                "Year": year,
                "Revenue": revenue,
                "EBIT": ebit,
                "Tax Rate": tax_rate,
                "D&A": depreciation,
                "CapEx": capex,
                "NWC": nwc,
                "ΔNWC": delta_nwc,
                "FCFF": fcff,
                "PV FCFF": pv_fcff_year,
            }
        )
        nwc_prev = nwc

    terminal_fcff = forecast[-1]["FCFF"] * (1 + terminal_growth)
    terminal_value = terminal_fcff / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)
    enterprise_value = pv_fcff + pv_terminal
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares
    value_to_price = value_per_share / price

    return {
        "base_revenue": base_revenue,
        "base_ebit": base_ebit,
        "base_fcff": base_fcff,
        "net_debt": net_debt,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": value_per_share,
        "V/P": value_to_price,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "forecast": forecast,
        "PV_FCFF": pv_fcff,
        "PV_terminal": pv_terminal,
        "terminal_fcff": terminal_fcff,
    }


# ============================================================
# COMPLETE COMPANY ANALYSIS
# ============================================================

def analyze_company(
    data,
    required_return,
    omega,
    terminal_growth,
    wacc,
    dcf_terminal_growth,
    dcf_revenue_growth,
    dcf_operating_margin,
    dcf_tax_rate,
    dcf_capex_ratio,
    dcf_dep_ratio,
    dcf_nwc_ratio,
    dcf_forecast_years=5,
):
    y0 = data["years"][0]
    y1 = data["years"][1]

    required_fields = [
        y0["assets"], y0["liab"], y0["debt"], y0["cash"], y0["equity"],
        y1["assets"], y1["liab"], y1["debt"], y1["cash"], y1["equity"],
        data["ebit"], data["ni"], data["price"], data["shares"],
    ]

    if any(x is None for x in required_fields):
        raise ValueError("missing required financial data")

    if y0["equity"] <= 0:
        raise ValueError("non-positive book equity")

    operating = operating_analysis(data)
    base = residual_income_valuation(
        data,
        required_return=required_return,
        omega=omega,
        terminal_growth=terminal_growth,
        forecast_years=5,
    )

    dcf = dcf_valuation(
        data,
        wacc=wacc,
        terminal_growth=dcf_terminal_growth,
        revenue_growth=dcf_revenue_growth,
        operating_margin=dcf_operating_margin,
        tax_rate=dcf_tax_rate,
        capex_ratio=dcf_capex_ratio,
        dep_ratio=dcf_dep_ratio,
        nwc_ratio=dcf_nwc_ratio,
        forecast_years=dcf_forecast_years,
    )

    historical = historical_analysis(data)
    implied_omega = calculate_implied_omega(
        data,
        required_return=required_return,
        terminal_growth=terminal_growth,
        target_vp=1.0,
    )

    warnings = []
    if data["effective_tax_rate"] is None:
        warnings.append("Effective operating tax rate unavailable; using user assumption for DCF.")
    if data["capex"] is None:
        warnings.append("Capex unavailable; DCF uses the user assumption for capex/revenue.")
    if data["depreciation"] is None:
        warnings.append("Depreciation unavailable; DCF uses the user assumption for D&A/revenue.")
    if data["nwc"] is None:
        warnings.append("Working capital unavailable; DCF uses the user assumption for NWC/revenue.")

    return {
        "Ticker": data["ticker"],
        "Price": data["price"],
        "P/B": base["P/B"],
        "ROE": base["historical_roe"],
        "Cost of Equity": required_return,
        "WACC": wacc,
        "ROE 5Y": historical["ROE_5Y"],
        "Earnings CAGR": historical["Earnings_CAGR"],
        "FCF Yield": historical["FCF_Yield"],
        "Implied Omega": implied_omega,
        "RNOA": operating["RNOA"],
        "NBC": operating["NBC"],
        "FLEV": operating["FLEV"],
        "Spread": operating["spread"],
        "Accruals": operating["accruals"],

        "RI Value": base["value_per_share"],
        "RI V/P": base["V/P"],
        "RI Upside": base["V/P"] - 1.0,
        "DCF Value": dcf["value_per_share"],
        "DCF V/P": dcf["V/P"],
        "DCF Upside": dcf["V/P"] - 1.0,

        "Net Debt": dcf["net_debt"],
        "Diluted Shares": data["shares"],
        "Forecast Years": dcf_forecast_years,
        "Terminal %": base["terminal_percentage"],

        "warnings": warnings,
        "_base": base,
        "_dcf": dcf,
        "_data": data,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Residual income assumptions")

    required_return = st.number_input(
        "Cost of equity",
        min_value=0.01,
        max_value=0.30,
        value=0.09,
        step=0.005,
        format="%.3f",
    )

    omega = st.slider(
        "ROE persistence (ω)",
        min_value=0.00,
        max_value=1.00,
        value=0.70,
        step=0.05,
    )

    terminal_growth = st.number_input(
        "Residual income terminal growth",
        min_value=-0.05,
        max_value=0.08,
        value=0.02,
        step=0.005,
        format="%.3f",
    )

    st.divider()

    st.header("DCF assumptions")

    wacc = st.number_input(
        "WACC",
        min_value=0.04,
        max_value=0.30,
        value=0.09,
        step=0.005,
        format="%.3f",
    )

    dcf_terminal_growth = st.number_input(
        "FCFF terminal growth",
        min_value=-0.03,
        max_value=0.08,
        value=0.025,
        step=0.005,
        format="%.3f",
    )

    dcf_forecast_years = st.slider(
        "Forecast years",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    dcf_revenue_growth = st.number_input(
        "Revenue growth",
        min_value=-0.10,
        max_value=0.25,
        value=0.05,
        step=0.005,
        format="%.3f",
    )

    dcf_operating_margin = st.number_input(
        "EBIT margin",
        min_value=-0.20,
        max_value=0.50,
        value=0.18,
        step=0.005,
        format="%.3f",
    )

    dcf_tax_rate = st.number_input(
        "Operating tax rate",
        min_value=0.00,
        max_value=0.40,
        value=0.22,
        step=0.005,
        format="%.3f",
    )

    dcf_capex_ratio = st.number_input(
        "CapEx / revenue",
        min_value=0.00,
        max_value=0.30,
        value=0.04,
        step=0.005,
        format="%.3f",
    )

    dcf_dep_ratio = st.number_input(
        "D&A / revenue",
        min_value=0.00,
        max_value=0.20,
        value=0.03,
        step=0.005,
        format="%.3f",
    )

    dcf_nwc_ratio = st.number_input(
        "NWC / revenue",
        min_value=0.00,
        max_value=0.60,
        value=0.10,
        step=0.005,
        format="%.3f",
    )

    st.divider()

    st.header("Screening rules")

    min_ri_vp = st.number_input(
        "Minimum RI V/P",
        min_value=0.50,
        max_value=5.00,
        value=1.25,
        step=0.05,
    )

    min_dcf_vp = st.number_input(
        "Minimum DCF V/P",
        min_value=0.50,
        max_value=5.00,
        value=1.10,
        step=0.05,
    )

    min_conservative_vp = st.number_input(
        "Minimum conservative RI V/P",
        min_value=0.50,
        max_value=3.00,
        value=1.00,
        step=0.05,
    )

    max_accruals = st.number_input(
        "Maximum accruals",
        min_value=0.00,
        max_value=0.50,
        value=0.10,
        step=0.01,
        format="%.2f",
    )

    max_flev = st.number_input(
        "Maximum FLEV",
        min_value=0.00,
        max_value=10.00,
        value=2.00,
        step=0.25,
    )


# ============================================================
# TICKER INPUT
# ============================================================

st.subheader("Enter stocks")

ticker_text = st.text_area(
    "Ticker symbols",
    value=("MSFT COR VRSK CME KO AAPL GOOGL META AMZN"),
    height=100,
    help="Separate tickers using spaces, commas, or new lines.",
)

run_screener = st.button(
    "🚀 Run Screener",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN SCREENER
# ============================================================

if run_screener:

    if required_return <= terminal_growth:
        st.error("Cost of equity must be greater than the RI terminal growth rate.")
        st.stop()

    if wacc <= dcf_terminal_growth:
        st.error("WACC must be greater than the DCF terminal growth rate.")
        st.stop()

    symbols = ticker_text.replace(",", " ").replace("\n", " ").split()
    symbols = list(dict.fromkeys(x.upper() for x in symbols if x.strip()))
    if not symbols:
        st.warning("Enter at least one ticker.")
        st.stop()

    progress = st.progress(0)
    results = []
    errors = []

    for i, symbol in enumerate(symbols):
        try:
            data = fetch_data(symbol)

            sector = (data.get("sector", "") or "").lower()

            if "financial" in sector:
                errors.append(f"{symbol}: skipped — financial companies need a different accounting reformulation.")
            else:
                result = analyze_company(
                    data,
                    required_return=required_return,
                    omega=omega,
                    terminal_growth=terminal_growth,
                    wacc=wacc,
                    dcf_terminal_growth=dcf_terminal_growth,
                    dcf_revenue_growth=dcf_revenue_growth,
                    dcf_operating_margin=dcf_operating_margin,
                    dcf_tax_rate=dcf_tax_rate,
                    dcf_capex_ratio=dcf_capex_ratio,
                    dcf_dep_ratio=dcf_dep_ratio,
                    dcf_nwc_ratio=dcf_nwc_ratio,
                    dcf_forecast_years=dcf_forecast_years,
                )
                results.append(result)
        except Exception as error:
            errors.append(f"{symbol}: {str(error)}")
        progress.progress((i + 1) / len(symbols))

    if errors:
        with st.expander(f"⚠️ {len(errors)} stock(s) skipped"):
            for error in errors:
                st.write(error)

    if not results:
        st.error("No stocks could be analyzed.")
        st.stop()

    df = pd.DataFrame(results)

    def get_verdict(row):
        ri_vp = row["RI V/P"]
        dcf_vp = row["DCF V/P"]
        implied_omega = row["Implied Omega"]

        if (
            ri_vp >= 1.50
            and dcf_vp >= 1.10
            and pd.notna(implied_omega)
            and implied_omega <= 0.70
        ):
            return "STRONG CANDIDATE"

        if (
            ri_vp >= min_ri_vp
            and dcf_vp >= min_dcf_vp
            and (
                pd.isna(row["ROE"])
                or row["ROE"] > required_return
            )
            and (
                pd.isna(row["Accruals"])
                or row["Accruals"] <= max_accruals
            )
            and (
                pd.isna(row["FLEV"])
                or row["FLEV"] <= max_flev
            )
        ):
            return "CANDIDATE"

        if ri_vp >= 1.10 or dcf_vp >= 1.10:
            return "WATCH"

        return "NO"

    df["Verdict"] = df.apply(get_verdict, axis=1)

    df = df.sort_values(["RI V/P", "DCF V/P"], ascending=False).reset_index(drop=True)

    candidate_count = df["Verdict"].isin(["STRONG CANDIDATE", "CANDIDATE"]).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks analyzed", len(df))
    col2.metric("Candidates", int(candidate_count))
    col3.metric("Highest RI V/P", f"{df['RI V/P'].max():.2f}x")
    col4.metric("Median RI V/P", f"{df['RI V/P'].median():.2f}x")

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------
    st.subheader("📈 Screening results")

    display_columns = [
        "Ticker",
        "Price",
        "P/B",
        "ROE",
        "Cost of Equity",
        "WACC",
        "ROE 5Y",
        "Earnings CAGR",
        "FCF Yield",
        "RNOA",
        "NBC",
        "FLEV",
        "Spread",
        "Accruals",
        "Implied Omega",
        "RI Value",
        "RI V/P",
        "RI Upside",
        "DCF Value",
        "DCF V/P",
        "DCF Upside",
        "Net Debt",
        "Diluted Shares",
        "Forecast Years",
        "Terminal %",
        "Verdict",
    ]

    display = df[display_columns].copy()

    st.dataframe(
        display.style.format(
            {
                "Price": "${:.2f}",
                "P/B": "{:.2f}x",
                "ROE": "{:.1%}",
                "Cost of Equity": "{:.1%}",
                "WACC": "{:.1%}",
                "ROE 5Y": "{:.1%}",
                "Earnings CAGR": "{:.1%}",
                "FCF Yield": "{:.1%}",
                "RNOA": "{:.1%}",
                "NBC": "{:.1%}",
                "FLEV": "{:.2f}",
                "Spread": "{:.1%}",
                "Accruals": "{:.1%}",
                "Implied Omega": "{:.2f}",
                "RI Value": "${:,.2f}",
                "RI V/P": "{:.2f}x",
                "RI Upside": "{:.1%}",
                "DCF Value": "${:,.2f}",
                "DCF V/P": "{:.2f}x",
                "DCF Upside": "{:.1%}",
                "Net Debt": "${:,.2f}",
                "Diluted Shares": "{:,.0f}",
                "Forecast Years": "{:,.0f}",
                "Terminal %": "{:.1%}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv,
        file_name="penman_dcf_screen_results.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # COMPANY DEEP DIVE
    # --------------------------------------------------------
    st.divider()
    st.subheader("🔎 Company deep dive")

    selected_ticker = st.selectbox("Select a company", df["Ticker"].tolist())
    selected = df.loc[df["Ticker"] == selected_ticker].iloc[0]

    base = selected["_base"]
    dcf = selected["_dcf"]

    left, right = st.columns(2)

    with left:
        st.markdown("### Valuation")
        st.write(f"**Market price:** ${selected['Price']:.2f}")
        st.write(f"**Residual Income value:** ${selected['RI Value']:.2f}")
        st.write(f"**RI V/P:** {selected['RI V/P']:.2f}x")
        st.write(f"**RI upside:** {selected['RI Upside']:.1%}")
        st.write(f"**DCF equity value:** ${selected['DCF Value']:.2f}")
        st.write(f"**DCF V/P:** {selected['DCF V/P']:.2f}x")
        st.write(f"**DCF upside:** {selected['DCF Upside']:.1%}")
        st.write(f"**P/B:** {selected['P/B']:.2f}x")
        st.write(f"**Terminal value contribution (RI):** {fmt_pct(selected['Terminal %'])}")

    with right:
        st.markdown("### Penman / DCF diagnostics")
        st.write(f"**ROE:** {fmt_pct(selected['ROE'])}")
        st.write(f"**RNOA:** {fmt_pct(selected['RNOA'])}")
        st.write(f"**NBC:** {fmt_pct(selected['NBC'])}")
        st.write(f"**FLEV:** {fmt_num(selected['FLEV'])}")
        st.write(f"**Spread:** {fmt_pct(selected['Spread'])}")
        st.write(f"**Accruals:** {fmt_pct(selected['Accruals'])}")
        st.write(f"**Cost of equity:** {fmt_pct(selected['Cost of Equity'])}")
        st.write(f"**WACC:** {fmt_pct(selected['WACC'])}")
        st.write(f"**Net debt:** ${selected['Net Debt']:.2f}")
        st.write(f"**Diluted shares:** {selected['Diluted Shares']:.0f}")

    if selected["Verdict"] == "STRONG CANDIDATE":
        st.success("STRONG CANDIDATE — passes the stronger RI and DCF checks.")
    elif selected["Verdict"] == "CANDIDATE":
        st.success("CANDIDATE — passes the current screening rules.")
    elif selected["Verdict"] == "WATCH":
        st.info("WATCH — valuation is interesting but does not pass all screening rules.")
    else:
        st.warning("NO — does not pass all screening rules.")

    # --------------------------------------------------------
    # RI forecast
    # --------------------------------------------------------
    st.subheader("📊 Five-year residual-income forecast")
    ri_forecast_df = pd.DataFrame(base["forecast"])
    st.dataframe(
        ri_forecast_df.style.format(
            {
                "Beginning BV": "{:,.0f}",
                "ROE": "{:.1%}",
                "Earnings": "{:,.0f}",
                "Residual Income": "{:,.0f}",
                "Ending BV": "{:,.0f}",
                "PV Residual Income": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # DCF forecast
    # --------------------------------------------------------
    st.subheader("💸 Five-year FCFF forecast")
    dcf_forecast_df = pd.DataFrame(dcf["forecast"])
    st.dataframe(
        dcf_forecast_df.style.format(
            {
                "Revenue": "{:,.0f}",
                "EBIT": "{:,.0f}",
                "Tax Rate": "{:.1%}",
                "D&A": "{:,.0f}",
                "CapEx": "{:,.0f}",
                "NWC": "{:,.0f}",
                "ΔNWC": "{:,.0f}",
                "FCFF": "{:,.0f}",
                "PV FCFF": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    warnings = selected.get("warnings", [])
    if warnings:
        with st.expander("⚠️ Data-quality warnings"):
            for w in warnings:
                st.warning(w)

    with st.expander("📚 How to interpret these two valuations"):
        st.markdown(
            """
### Residual income
The RI model values equity by adding current book equity to the present value of expected residual income.

Residual income = Net Income - r × Beginning Book Value

This is a clean-equity approach and is most useful when:
- ROE is a meaningful driver of value
- book value is informative
- earnings quality is stable

### DCF / FCFF
The DCF model values the firm using free cash flow to the firm and then subtracts net debt to get equity value.

FCFF = EBIT × (1 - tax rate) + D&A - CapEx - ΔNWC

This is an enterprise-value approach and is more sensitive to:
- WACC
- terminal growth
- working-capital assumptions
- capex and depreciation assumptions

### Why values differ
A large difference usually means the market is pricing:
- more operating cash generation than book-based earnings suggest (DCF)
- or more persistence in ROE and excess returns than the DCF model implies (RI)

Neither valuation should be treated as a guaranteed buy.
"""
        )

    with st.expander("📊 Sensitivity analysis"):
        st.subheader("RI sensitivity: cost of equity vs terminal RI growth")
        ri_rows = []
        for required in [0.06, 0.08, 0.10, 0.12, 0.14]:
            for g in [-0.01, 0.00, 0.02, 0.04]:
                try:
                    ri = residual_income_valuation(
                        selected["_data"],
                        required_return=required,
                        omega=omega,
                        terminal_growth=g,
                        forecast_years=5,
                    )["value_per_share"]
                    ri_rows.append({"Cost of Equity": required, "RI Growth": g, "RI Value": ri})
                except Exception:
                    pass
        ri_sensitivity = pd.DataFrame(ri_rows).pivot_table(index="Cost of Equity", columns="RI Growth", values="RI Value", aggfunc="first")
        st.dataframe(ri_sensitivity.style.format("{:,.2f}"), use_container_width=True)

        st.subheader("DCF sensitivity: WACC vs FCFF terminal growth")
        dcf_rows = []
        for w in [0.06, 0.08, 0.10, 0.12]:
            for g in [0.00, 0.02, 0.04]:
                try:
                    d = dcf_valuation(
                        selected["_data"],
                        wacc=w,
                        terminal_growth=g,
                        revenue_growth=dcf_revenue_growth,
                        operating_margin=dcf_operating_margin,
                        tax_rate=dcf_tax_rate,
                        capex_ratio=dcf_capex_ratio,
                        dep_ratio=dcf_dep_ratio,
                        nwc_ratio=dcf_nwc_ratio,
                        forecast_years=dcf_forecast_years,
                    )["value_per_share"]
                    dcf_rows.append({"WACC": w, "FCFF Growth": g, "DCF Value": d})
                except Exception:
                    pass
        dcf_sensitivity = pd.DataFrame(dcf_rows).pivot_table(index="WACC", columns="FCFF Growth", values="DCF Value", aggfunc="first")
        st.dataframe(dcf_sensitivity.style.format("{:,.2f}"), use_container_width=True)

    with st.expander("✅ Formula validation example"):
        st.markdown(
            """
A simple check:

- Beginning book value: $100
- Required return: 10%
- Net income: $12
- Dividend: $2

Residual income in year 1:

RI1 = NI1 - r × B0 = 12 - 0.10 × 100 = 2

Ending book value:

B1 = B0 + NI1 - Dividends = 100 + 12 - 2 = 110

This is exactly how the RI model computes the first forecast row.
"""
        )

        validate_df = pd.DataFrame(
            [
                {"Beginning BV": 100, "NI": 12, "Dividend": 2, "Required Return": 0.10, "RI": 2, "Ending BV": 110}
            ]
        )
        st.dataframe(validate_df.style.format({"Beginning BV": "${:,.0f}", "NI": "${:,.0f}", "Dividend": "${:,.0f}", "Required Return": "{:.0%}", "RI": "${:,.0f}", "Ending BV": "${:,.0f}"}), use_container_width=True)

else:
    st.info("Enter tickers above and click **Run Screener**.")
