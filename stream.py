import streamlit as st
import pandas as pd
import yfinance as yf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Penman Stock Screener",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Penman Residual-Income Stock Screener")

st.markdown(
    """
This screener uses a **residual-income valuation model** together with
Penman-style operating analysis.

It evaluates:

- Residual-income intrinsic value
- V/P valuation ratio
- P/B
- ROE
- RNOA
- Net borrowing cost
- Financial leverage
- Operating spread
- Accruals
- Conservative valuation
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

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
    )

    if price is None:

        try:

            history = ticker.history(
                period="5d"
            )

            if not history.empty:

                price = float(
                    history["Close"]
                    .dropna()
                    .iloc[-1]
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Shares outstanding
    # --------------------------------------------------------

    shares = info.get(
        "sharesOutstanding"
    )

    if shares is None:

        shares = pick(
            balance_sheet,
            [
                "Ordinary Shares Number",
                "Share Issued",
            ],
            0,
        )

    # --------------------------------------------------------
    # Balance sheet
    # --------------------------------------------------------

    years = []

    for col in [0, 1]:

        years.append(

            {
                "assets": pick(
                    balance_sheet,
                    [
                        "Total Assets",
                    ],
                    col,
                ),

                "liab": pick(
                    balance_sheet,
                    [
                        "Total Liabilities Net Minority Interest",
                        "Total Liabilities",
                    ],
                    col,
                ),

                "debt": pick(
                    balance_sheet,
                    [
                        "Total Debt",
                        "Long Term Debt And Capital Lease Obligation",
                        "Long Term Debt",
                    ],
                    col,
                ),

                "cash": pick(
                    balance_sheet,
                    [
                        "Cash Cash Equivalents And Short Term Investments",
                        "Cash And Cash Equivalents",
                        "Cash Financial",
                    ],
                    col,
                ),

                "equity": pick(
                    balance_sheet,
                    [
                        "Stockholders Equity",
                        "Common Stock Equity",
                        "Total Equity Gross Minority Interest",
                    ],
                    col,
                ),
            }
        )

    # --------------------------------------------------------
    # Income statement
    # --------------------------------------------------------

    ebit = pick(
        income_statement,
        [
            "EBIT",
            "Operating Income",
        ],
    )

    pretax = pick(
        income_statement,
        [
            "Pretax Income",
        ],
    )

    tax = pick(
        income_statement,
        [
            "Tax Provision",
        ],
    )

    interest = pick(
        income_statement,
        [
            "Interest Expense",
            "Interest Expense Non Operating",
        ],
    )

    net_income = pick(
        income_statement,
        [
            "Net Income Common Stockholders",
            "Net Income",
            "Net Income Including Noncontrolling Interests",
        ],
    )

    # --------------------------------------------------------
    # Cash flow
    # --------------------------------------------------------

    cfo = pick(
        cash_flow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ],
    )

    dividends = pick(
        cash_flow,
        [
            "Cash Dividends Paid",
            "Common Stock Dividend Paid",
        ],
    )

    return {

        "ticker": symbol.upper(),

        "price": safe_float(price),

        "shares": safe_float(shares),

        "sector": info.get(
            "sector",
            "",
        ),

        "years": years,

        "ebit": safe_float(ebit),

        "pretax": safe_float(pretax),

        "tax": safe_float(tax),

        "interest": safe_float(interest),

        "ni": safe_float(net_income),

        "cfo": safe_float(cfo),

        "div": safe_float(dividends),
    }


# ============================================================
# PENMAN OPERATING ANALYSIS
# ============================================================

def operating_analysis(data):

    y0 = data["years"][0]
    y1 = data["years"][1]

    # --------------------------------------------------------
    # Net operating assets
    #
    # NOA =
    # Operating Assets - Operating Liabilities
    #
    # Operating Assets =
    # Total Assets - Cash
    #
    # Operating Liabilities =
    # Total Liabilities - Debt
    # --------------------------------------------------------

    def calculate_noa(year):

        operating_assets = (
            year["assets"]
            - year["cash"]
        )

        operating_liabilities = (
            year["liab"]
            - year["debt"]
        )

        return (
            operating_assets
            - operating_liabilities
        )

    # --------------------------------------------------------
    # Net financial obligations
    # --------------------------------------------------------

    def calculate_nfo(year):

        return (
            year["debt"]
            - year["cash"]
        )

    noa0 = calculate_noa(y0)
    noa1 = calculate_noa(y1)

    nfo0 = calculate_nfo(y0)
    nfo1 = calculate_nfo(y1)

    average_noa = (
        noa0 + noa1
    ) / 2

    average_nfo = (
        nfo0 + nfo1
    ) / 2

    # --------------------------------------------------------
    # Tax rate
    # --------------------------------------------------------

    if (
        data["pretax"] is not None
        and data["tax"] is not None
        and data["pretax"] > 0
    ):

        tax_rate = max(
            0.0,
            min(
                0.40,
                data["tax"]
                / data["pretax"],
            ),
        )

    else:

        tax_rate = 0.21

    # --------------------------------------------------------
    # Operating income after tax
    # --------------------------------------------------------

    operating_income_after_tax = (
        data["ebit"]
        * (1 - tax_rate)
    )

    # --------------------------------------------------------
    # Net financial expense after tax
    # --------------------------------------------------------

    interest = abs(
        data["interest"] or 0
    )

    net_financial_expense = (
        interest
        * (1 - tax_rate)
    )

    # --------------------------------------------------------
    # RNOA
    # --------------------------------------------------------

    if average_noa > 0:

        rnoa = (
            operating_income_after_tax
            / average_noa
        )

    else:

        rnoa = None

    # --------------------------------------------------------
    # Net borrowing cost
    # --------------------------------------------------------

    if average_nfo > 0:

        nbc = (
            net_financial_expense
            / average_nfo
        )

    else:

        nbc = None

    # --------------------------------------------------------
    # Financial leverage
    # --------------------------------------------------------

    if y0["equity"] > 0:

        flev = (
            nfo0
            / y0["equity"]
        )

    else:

        flev = None

    # --------------------------------------------------------
    # Spread
    #
    # Spread = RNOA - NBC
    # --------------------------------------------------------

    if (
        rnoa is not None
        and nbc is not None
    ):

        spread = rnoa - nbc

    else:

        spread = None

    # --------------------------------------------------------
    # Accruals
    #
    # Accruals = (Net Income - CFO) / Average Assets
    # --------------------------------------------------------

    average_assets = (
        y0["assets"]
        + y1["assets"]
    ) / 2

    if (
        data["cfo"] is not None
        and average_assets > 0
    ):

        accruals = (
            data["ni"]
            - data["cfo"]
        ) / average_assets

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

def residual_income_valuation(
    data,
    required_return,
    omega,
    terminal_growth,
    forecast_years=5,
):

    """
    Residual income valuation.

    V0 =
        B0
        + PV(RI1)
        + PV(RI2)
        + ...
        + PV(RIT)
        + PV(Terminal RI)

    RI_t =
        Earnings_t
        - r * Beginning Book Value_t

    or:

        RI_t =
        (ROE_t - r) * Beginning Book Value_t

    ROE fades toward the required return.

        ROE_t =
        r + (ROE_0 - r) * omega^t

    Book value follows the clean-surplus relation:

        BV_t =
        BV_(t-1)
        + Earnings_t
        - Dividends_t
    """

    y0 = data["years"][0]
    y1 = data["years"][1]

    current_book_value = y0["equity"]
    previous_book_value = y1["equity"]

    net_income = data["ni"]
    shares = data["shares"]
    price = data["price"]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        current_book_value is None
        or current_book_value <= 0
    ):

        raise ValueError(
            "invalid current book equity"
        )

    if (
        previous_book_value is None
        or previous_book_value <= 0
    ):

        raise ValueError(
            "invalid prior book equity"
        )

    if net_income is None:

        raise ValueError(
            "missing net income"
        )

    if (
        shares is None
        or shares <= 0
    ):

        raise ValueError(
            "missing shares outstanding"
        )

    if (
        price is None
        or price <= 0
    ):

        raise ValueError(
            "missing market price"
        )

    if required_return <= terminal_growth:

        raise ValueError(
            "required return must exceed terminal growth"
        )

    # --------------------------------------------------------
    # Historical ROE
    #
    # Earnings / beginning book value
    # --------------------------------------------------------

    historical_roe = (
        net_income
        / previous_book_value
    )

    # Prevent an extreme ROE from completely dominating
    # the model when book equity is unusually small.
    roe_base = max(
        -0.10,
        min(
            0.50,
            historical_roe,
        ),
    )

    # --------------------------------------------------------
    # Historical payout ratio
    # --------------------------------------------------------

    if (
        data["div"] is not None
        and net_income > 0
        and data["div"] != 0
    ):

        payout_ratio = max(
            0.0,
            min(
                1.0,
                abs(data["div"])
                / net_income,
            ),
        )

    else:

        # If dividends aren't available,
        # assume 0% payout.
        payout_ratio = 0.0

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    book_value = current_book_value

    present_value_residual_income = 0.0

    forecast = []

    last_residual_income = 0.0

    for year in range(
        1,
        forecast_years + 1,
    ):

        # ----------------------------------------------------
        # ROE persistence
        #
        # omega = 1.0:
        # ROE remains near its current level.
        #
        # omega = 0:
        # ROE immediately moves toward r.
        # ----------------------------------------------------

        forecast_roe = (
            required_return
            + (
                roe_base
                - required_return
            )
            * (
                omega ** year
            )
        )

        beginning_book_value = book_value

        # ----------------------------------------------------
        # Earnings
        # ----------------------------------------------------

        earnings = (
            forecast_roe
            * beginning_book_value
        )

        # ----------------------------------------------------
        # Residual income
        # ----------------------------------------------------

        residual_income = (
            earnings
            - (
                required_return
                * beginning_book_value
            )
        )

        # ----------------------------------------------------
        # Dividends
        # ----------------------------------------------------

        dividends = (
            earnings
            * payout_ratio
        )

        # ----------------------------------------------------
        # Ending book value
        #
        # Clean surplus:
        #
        # BV_t =
        # BV_(t-1) + NI_t - Dividends_t
        # ----------------------------------------------------

        ending_book_value = (
            beginning_book_value
            + earnings
            - dividends
        )

        # ----------------------------------------------------
        # Discount residual income
        # ----------------------------------------------------

        pv_residual_income = (
            residual_income
            / (
                (1 + required_return)
                ** year
            )
        )

        present_value_residual_income += (
            pv_residual_income
        )

        forecast.append(
            {
                "Year": year,

                "Beginning BV":
                    beginning_book_value,

                "ROE":
                    forecast_roe,

                "Earnings":
                    earnings,

                "Residual Income":
                    residual_income,

                "Ending BV":
                    ending_book_value,

                "PV Residual Income":
                    pv_residual_income,
            }
        )

        book_value = ending_book_value

        last_residual_income = (
            residual_income
        )

    # --------------------------------------------------------
    # Terminal residual income
    #
    # RI_(T+1) = RI_T * (1 + g)
    #
    # Terminal Value =
    # RI_(T+1) / (r - g)
    # --------------------------------------------------------

    terminal_residual_income = (
        last_residual_income
        * (1 + terminal_growth)
    )

    terminal_value = (
        terminal_residual_income
        / (
            required_return
            - terminal_growth
        )
    )

    pv_terminal = (
        terminal_value
        / (
            (1 + required_return)
            ** forecast_years
        )
    )

    # --------------------------------------------------------
    # Total equity value
    # --------------------------------------------------------

    equity_value = (
        current_book_value
        + present_value_residual_income
        + pv_terminal
    )

    value_per_share = (
        equity_value
        / shares
    )

    market_cap = (
        price
        * shares
    )

    price_to_book = (
        market_cap
        / current_book_value
    )

    value_to_price = (
        value_per_share
        / price
    )

    terminal_percentage = (
        pv_terminal
        / equity_value
        if equity_value != 0
        else None
    )

    return {

        "book_value":
            current_book_value,

        "historical_roe":
            historical_roe,

        "payout":
            payout_ratio,

        "value_per_share":
            value_per_share,

        "V/P":
            value_to_price,

        "P/B":
            price_to_book,

        "equity_value":
            equity_value,

        "PV_RI":
            present_value_residual_income,

        "PV_terminal":
            pv_terminal,

        "terminal_percentage":
            terminal_percentage,

        "forecast":
            forecast,
    }


# ============================================================
# COMPLETE COMPANY ANALYSIS
# ============================================================

def analyze_company(
    data,
    required_return,
    omega,
    terminal_growth,
):

    y0 = data["years"][0]
    y1 = data["years"][1]

    required_fields = [

        y0["assets"],
        y0["liab"],
        y0["debt"],
        y0["cash"],
        y0["equity"],

        y1["assets"],
        y1["liab"],
        y1["debt"],
        y1["cash"],
        y1["equity"],

        data["ebit"],
        data["ni"],
        data["price"],
        data["shares"],
    ]

    if any(
        x is None
        for x in required_fields
    ):

        raise ValueError(
            "missing required financial data"
        )

    if y0["equity"] <= 0:

        raise ValueError(
            "non-positive book equity"
        )

    # --------------------------------------------------------
    # Operating analysis
    # --------------------------------------------------------

    operating = operating_analysis(
        data
    )

    # --------------------------------------------------------
    # Base valuation
    # --------------------------------------------------------

    base = residual_income_valuation(

        data,

        required_return=
            required_return,

        omega=
            omega,

        terminal_growth=
            terminal_growth,

        forecast_years=5,
    )

    # --------------------------------------------------------
    # Conservative valuation
    #
    # Required return +2%
    # ROE persistence reduced
    # --------------------------------------------------------

    conservative = residual_income_valuation(

        data,

        required_return=
            required_return + 0.02,

        omega=
            omega * 0.60,

        terminal_growth=
            terminal_growth,

        forecast_years=5,
    )

    return {

        "Ticker":
            data["ticker"],

        "Price":
            data["price"],

        "P/B":
            base["P/B"],

        "ROE":
            base["historical_roe"],

        "RNOA":
            operating["RNOA"],

        "NBC":
            operating["NBC"],

        "FLEV":
            operating["FLEV"],

        "Spread":
            operating["spread"],

        "Accruals":
            operating["accruals"],

        "Base Value":
            base["value_per_share"],

        "V/P":
            base["V/P"],

        "Conservative Value":
            conservative[
                "value_per_share"
            ],

        "V/P Cons.":
            conservative["V/P"],

        "Terminal %":
            base["terminal_percentage"],

        "_base":
            base,

        "_conservative":
            conservative,

        "_data":
            data,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model assumptions")

    required_return = st.number_input(

        "Cost of equity",

        min_value=0.01,

        max_value=0.30,

        value=0.09,

        step=0.005,

        format="%.3f",

        help=(
            "Required return on equity."
        ),
    )

    omega = st.slider(

        "ROE persistence (ω)",

        min_value=0.00,

        max_value=1.00,

        value=0.70,

        step=0.05,

        help=(
            "Higher values assume excess ROE "
            "persists for longer."
        ),
    )

    terminal_growth = st.number_input(

        "Terminal RI growth",

        min_value=-0.05,

        max_value=0.08,

        value=0.02,

        step=0.005,

        format="%.3f",
    )

    st.divider()

    st.header("Screening rules")

    min_vp = st.number_input(

        "Minimum base V/P",

        min_value=0.50,

        max_value=5.00,

        value=1.25,

        step=0.05,
    )

    min_conservative_vp = st.number_input(

        "Minimum conservative V/P",

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

    value=(
        "MSFT COR VRSK CME KO "
        "AAPL GOOGL META AMZN"
    ),

    height=100,

    help=(
        "Separate tickers using spaces, commas, "
        "or new lines."
    ),
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

    # --------------------------------------------------------
    # Validate assumptions
    # --------------------------------------------------------

    if required_return <= terminal_growth:

        st.error(
            "Cost of equity must be greater "
            "than terminal growth."
        )

        st.stop()

    # --------------------------------------------------------
    # Parse tickers
    # --------------------------------------------------------

    symbols = (
        ticker_text
        .replace(",", " ")
        .replace("\n", " ")
        .split()
    )

    symbols = list(
        dict.fromkeys(
            x.upper()
            for x in symbols
            if x.strip()
        )
    )

    if not symbols:

        st.warning(
            "Enter at least one ticker."
        )

        st.stop()

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    progress = st.progress(0)

    results = []

    errors = []

    for i, symbol in enumerate(symbols):

        try:

            data = fetch_data(
                symbol
            )

            # Financial firms are skipped because
            # their accounting structure is different.
            sector = (
                data.get(
                    "sector",
                    ""
                )
                or ""
            ).lower()

            if "financial" in sector:

                errors.append(
                    f"{symbol}: skipped — "
                    "financial companies need "
                    "a different reformulation."
                )

            else:

                result = analyze_company(

                    data,

                    required_return=
                        required_return,

                    omega=
                        omega,

                    terminal_growth=
                        terminal_growth,
                )

                results.append(
                    result
                )

        except Exception as error:

            errors.append(
                f"{symbol}: {str(error)}"
            )

        progress.progress(
            (i + 1)
            / len(symbols)
        )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    if errors:

        with st.expander(
            f"⚠️ {len(errors)} stock(s) skipped"
        ):

            for error in errors:

                st.write(
                    error
                )

    if not results:

        st.error(
            "No stocks could be analyzed."
        )

        st.stop()

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Apply screen
    # --------------------------------------------------------

    def get_verdict(row):

        if row["V/P"] < min_vp:

            return "NO"

        if (
            row["V/P Cons."]
            < min_conservative_vp
        ):

            return "NO"

        if (
            pd.notna(row["ROE"])
            and row["ROE"]
            <= required_return
        ):

            return "NO"

        if (
            pd.notna(row["Accruals"])
            and row["Accruals"]
            > max_accruals
        ):

            return "NO"

        if (
            pd.notna(row["FLEV"])
            and row["FLEV"]
            > max_flev
        ):

            return "NO"

        return "BUY"

    df["Verdict"] = df.apply(
        get_verdict,
        axis=1,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        "V/P",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    buy_count = (
        df["Verdict"]
        == "BUY"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Stocks analyzed",
        len(df),
    )

    col2.metric(
        "BUY",
        int(buy_count),
    )

    col3.metric(
        "Highest V/P",
        f"{df['V/P'].max():.2f}x",
    )

    col4.metric(
        "Median V/P",
        f"{df['V/P'].median():.2f}x",
    )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    st.subheader(
        "📈 Screening results"
    )

    display_columns = [

        "Ticker",

        "Price",

        "P/B",

        "ROE",

        "RNOA",

        "NBC",

        "FLEV",

        "Spread",

        "Accruals",

        "Base Value",

        "V/P",

        "Conservative Value",

        "V/P Cons.",

        "Terminal %",

        "Verdict",
    ]

    display = df[
        display_columns
    ].copy()

    st.dataframe(

        display.style.format(

            {
                "Price":
                    "${:.2f}",

                "P/B":
                    "{:.2f}x",

                "ROE":
                    "{:.1%}",

                "RNOA":
                    "{:.1%}",

                "NBC":
                    "{:.1%}",

                "FLEV":
                    "{:.2f}",

                "Spread":
                    "{:.1%}",

                "Accruals":
                    "{:.1%}",

                "Base Value":
                    "${:.2f}",

                "V/P":
                    "{:.2f}x",

                "Conservative Value":
                    "${:.2f}",

                "V/P Cons.":
                    "{:.2f}x",

                "Terminal %":
                    "{:.1%}",
            },
            na_rep="—",
        ),

        use_container_width=True,

        hide_index=True,
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    csv = display.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(

        "⬇️ Download results as CSV",

        data=csv,

        file_name=
            "penman_screen_results.csv",

        mime=
            "text/csv",
    )

    # ========================================================
    # COMPANY DEEP DIVE
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 Company deep dive"
    )

    selected_ticker = st.selectbox(

        "Select a company",

        df["Ticker"].tolist(),
    )

    selected = next(

        x
        for x in results
        if x["Ticker"]
        == selected_ticker
    )

    base = selected["_base"]

    # --------------------------------------------------------
    # Valuation
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### Valuation"
        )

        st.write(
            f"**Market price:** "
            f"${selected['Price']:.2f}"
        )

        st.write(
            f"**Base intrinsic value:** "
            f"${selected['Base Value']:.2f}"
        )

        st.write(
            f"**Base V/P:** "
            f"{selected['V/P']:.2f}x"
        )

        st.write(
            f"**Conservative value:** "
            f"${selected['Conservative Value']:.2f}"
        )

        st.write(
            f"**Conservative V/P:** "
            f"{selected['V/P Cons.']:.2f}x"
        )

        st.write(
            f"**P/B:** "
            f"{selected['P/B']:.2f}x"
        )

        st.write(
            f"**Terminal value contribution:** "
            f"{fmt_pct(selected['Terminal %'])}"
        )

    # --------------------------------------------------------
    # Penman analysis
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### Penman analysis"
        )

        st.write(
            f"**ROE:** "
            f"{fmt_pct(selected['ROE'])}"
        )

        st.write(
            f"**RNOA:** "
            f"{fmt_pct(selected['RNOA'])}"
        )

        st.write(
            f"**NBC:** "
            f"{fmt_pct(selected['NBC'])}"
        )

        st.write(
            f"**FLEV:** "
            f"{fmt_num(selected['FLEV'])}"
        )

        st.write(
            f"**Spread:** "
            f"{fmt_pct(selected['Spread'])}"
        )

        st.write(
            f"**Accruals:** "
            f"{fmt_pct(selected['Accruals'])}"
        )

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    if selected["Verdict"] == "BUY":

        st.success(
            "BUY — passes the current screening rules."
        )

    else:

        st.warning(
            "NO — does not pass all screening rules."
        )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    st.subheader(
        "📊 Five-year residual-income forecast"
    )

    forecast_df = pd.DataFrame(
        base["forecast"]
    )

    st.dataframe(

        forecast_df.style.format(

            {
                "Beginning BV":
                    "{:,.0f}",

                "ROE":
                    "{:.1%}",

                "Earnings":
                    "{:,.0f}",

                "Residual Income":
                    "{:,.0f}",

                "Ending BV":
                    "{:,.0f}",

                "PV Residual Income":
                    "{:,.0f}",
            }
        ),

        use_container_width=True,

        hide_index=True,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    with st.expander(
        "📚 What do these numbers mean?"
    ):

        st.markdown(
            """
### V/P

The model's estimated intrinsic value divided by the current price.

For example:

**V/P = 1.50x**

means the model estimates intrinsic value at approximately
150% of the current market price.

That does **not** mean the stock will rise 50%.

---

### Residual income

Residual income is:

**RI = Net Income − Required Return × Beginning Book Value**

or:

**RI = (ROE − r) × Beginning Book Value**

A company creates residual income when its return on equity
exceeds the return shareholders require.

---

### ROE

ROE measures how much profit the company generates relative
to shareholder book equity.

---

### RNOA

RNOA focuses on the profitability of the company's operations
rather than how it finances those operations.

---

### FLEV

Financial leverage measures the contribution of net financial
obligations relative to equity.

High leverage can make ROE look better while increasing risk.

---

### Spread

Approximately:

**Spread = RNOA − NBC**

A positive spread means the company's operating return exceeds
its net borrowing cost.

---

### Accruals

A simple earnings-quality indicator:

**Accruals = (Net Income − Operating Cash Flow) / Average Assets**

Very high positive accruals can be a warning that accounting
earnings are running ahead of cash generation.

---

### Important

A high V/P should be treated as a **research candidate**, not
an automatic buy.

The valuation is sensitive to:

- Cost of equity
- ROE persistence
- Terminal growth
- Payout
- Book value
- Accounting quality
- Data quality

Always investigate why the market price differs from the model.
"""
        )

else:

    st.info(
        "Enter tickers above and click **Run Screener**."
    )
