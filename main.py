# Copyright (c) 2024 PogChan Github
# All rights reserved.
import cloudscraper
import streamlit as st
import yfinance as yf
import requests
import random
from datetime import datetime, timedelta
import pandas as pd
from tickers import *
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from indicators import *
import numpy as np

load_dotenv()

apiUrl = st.secrets["API"]
baseURL = st.secrets["BASEAPI"]

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
]


@st.cache_data(ttl=60*10)
def get_options_chain(symbol, expiration):
    """
    Retrieves options chain data for a single expiration date using yfinance.
    Preserves the original structure for compatibility:
        {
          "options": {
            "YYYY-MM-DD": {
              "c": { <strike_str>: {...fields...}, ... },
              "p": { <strike_str>: {...fields...}, ... }
            }
          }
        }

    Fields include:
      b  -> bid
      a  -> ask
      oi -> openInterest
      v  -> volume
      iv -> impliedVolatility
      itm -> inTheMoney (boolean)
      chg -> change
      pctChg -> percentChange
      lp  -> lastPrice
    """

    try:
        ticker = yf.Ticker(symbol)
    except Exception as e:
        st.error(f"Error creating yfinance Ticker for {symbol}: {e}")
        return None

    # Check if the expiration date is valid
    expiration_dates = ticker.options
    if not expiration_dates or expiration not in expiration_dates:
        st.error(f"Invalid expiration date for {symbol}. Available dates: {', '.join(expiration_dates)}")
        return None

    # Retrieve the options chain for the specified expiration date
    try:
        chain = ticker.option_chain(expiration)
        calls_df = chain.calls
        puts_df = chain.puts
    except Exception as e:
        st.error(f"Failed to retrieve option chain for {symbol} {expiration}: {e}")
        return None

    # Build the structure for the single expiration date
    data = {"options": {expiration: {"c": {}, "p": {}}}}

    # Build "c" dictionary for calls
    c_dict = {}
    for _, row in calls_df.iterrows():
        strike_str = f"{row['strike']:.2f}"
        c_dict[strike_str] = {
            "b": float(row['bid']) if not pd.isna(row['bid']) else 0.0,
            "a": float(row['ask']) if not pd.isna(row['ask']) else 0.0,
            "oi": float(row['openInterest']) if not pd.isna(row['openInterest']) else 0.0,
            "v": float(row['volume']) if not pd.isna(row['volume']) else 0.0,
            "iv": float(row.get('impliedVolatility', 0.0)) if not pd.isna(row.get('impliedVolatility', 0.0)) else 0.0,
            "itm": bool(row.get('inTheMoney', False)),
            "chg": float(row.get('change', 0.0)) if not pd.isna(row.get('change', 0.0)) else 0.0,
            "pctChg": float(row.get('percentChange', 0.0)) if not pd.isna(row.get('percentChange', 0.0)) else 0.0,
            "lp": float(row.get('lastPrice', 0.0)) if not pd.isna(row.get('lastPrice', 0.0)) else 0.0
        }

    # Build "p" dictionary for puts
    p_dict = {}
    for _, row in puts_df.iterrows():
        strike_str = f"{row['strike']:.2f}"
        p_dict[strike_str] = {
            "b": float(row['bid']) if not pd.isna(row['bid']) else 0.0,
            "a": float(row['ask']) if not pd.isna(row['ask']) else 0.0,
            "oi": float(row['openInterest']) if not pd.isna(row['openInterest']) else 0.0,
            "v": float(row['volume']) if not pd.isna(row['volume']) else 0.0,
            "iv": float(row.get('impliedVolatility', 0.0)) if not pd.isna(row.get('impliedVolatility', 0.0)) else 0.0,
            "itm": bool(row.get('inTheMoney', False)),
            "chg": float(row.get('change', 0.0)) if not pd.isna(row.get('change', 0.0)) else 0.0,
            "pctChg": float(row.get('percentChange', 0.0)) if not pd.isna(row.get('percentChange', 0.0)) else 0.0,
            "lp": float(row.get('lastPrice', 0.0)) if not pd.isna(row.get('lastPrice', 0.0)) else 0.0
        }

    # Assign dictionaries to the data structure
    data["options"][expiration] = {"c": c_dict, "p": p_dict}

    return data
@st.cache_data(ttl=60*10)
def get_options_chains(symbols, expiration):
    result = {}

    # Create multi-ticker object once
    all_tickers = yf.Tickers(" ".join(symbols))

    for symbol in symbols:
        try:
            ticker = all_tickers.tickers[symbol]

            # Expiration check
            if expiration not in ticker.options:
                result[symbol] = None
                continue

            # Batch fetch options data
            chain = ticker.option_chain(expiration)

            # Vectorized processing with proper tuple access
            def process_chain(df):
                processed = {}
                for row in df.itertuples():
                    strike_str = f"{row.strike:.2f}"
                    processed[strike_str] = {
                        "b": float(row.bid) if not pd.isna(row.bid) else 0.0,
                        "a": float(row.ask) if not pd.isna(row.ask) else 0.0,
                        "oi": float(row.openInterest) if not pd.isna(row.openInterest) else 0.0,
                        "v": float(row.volume) if not pd.isna(row.volume) else 0.0,
                        "iv": float(row.impliedVolatility) if not pd.isna(row.impliedVolatility) else 0.0,
                        "itm": bool(row.inTheMoney),
                        "chg": float(row.change) if not pd.isna(row.change) else 0.0,
                        "pctChg": float(row.percentChange) if not pd.isna(row.percentChange) else 0.0,
                        "lp": float(row.lastPrice) if not pd.isna(row.lastPrice) else 0.0
                    }
                return processed

            result[symbol] = {
                "options": {
                    expiration: {
                        "c": process_chain(chain.calls),
                        "p": process_chain(chain.puts)
                    }
                }
            }

        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")
            result[symbol] = None

    return result

# find stock price currnet
def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    try:
        price = ticker.history(period="1d")['Close'].iloc[-1]
        return price
    except Exception as e:
        st.error(f"Failed to fetch stock price for {symbol}: {e}")
        return None

def get_next_fridays(n=10):
    """Get the next `num_fridays` Fridays starting from today."""
    today = datetime.datetime.now()
    fridays = []
    # find el next fridiossss
    days_until_next_friday = (4 - today.weekday() + 7) % 7
    next_friday = today + timedelta(days=days_until_next_friday)

    for i in range(n):
        fridays.append(next_friday + timedelta(weeks=i))

    return [friday.strftime('%Y-%m-%d') for friday in fridays]


def analyze_options_chain(data, exp_date, stock_price):
    """Analyze call and put premiums with dynamic increments."""
    if exp_date not in data.get("options", {}):
        return {"call_premium": 0, "put_premium": 0, "call_heatmap": {}, "put_heatmap": {}}

    call_data = data["options"][exp_date]["c"]
    put_data = data["options"][exp_date]["p"]

    def process_options(data, stock_price, max_strikes=10):
        """Process options data dynamically around stock price."""
        premiums = {}
        strikes = sorted([float(strike) for strike in data.keys()])

        # find strike increments so i know wtf is the increment
        increments = [round(strikes[i + 1] - strikes[i], 2) for i in range(len(strikes) - 1)]
        increment = max(set(increments), key=increments.count) if increments else 1

        # get all the strikes thats good within increments, both itm and atm
        valid_strikes = [
            strike for strike in strikes
            if stock_price - (increment * max_strikes / 2) <= strike <= stock_price + (increment * max_strikes / 2)
        ]

        # get premiums
        for strike in valid_strikes:

            strike_key = f"{strike:.2f}"
            info = data.get(strike_key, {})

            # make sure its got the actual b a oi info
            if 'b' in info and 'a' in info and 'oi' in info:
                mid_price = (info.get("b", 0) + info.get("a", 0)) / 2
                total_premium = round(mid_price * info.get("oi", 0)) * 100
                premiums[float(strike_key)] = total_premium
            else:
                st.write(f"Invalid data for strike {strike_key}: {info}")

        return premiums

    # get the heat map so i can get top 5 lkater
    call_heatmap = process_options(call_data, stock_price)
    put_heatmap = process_options(put_data, stock_price)

    # calc total call and puts
    call_premium = sum(call_heatmap.values())
    put_premium = sum(put_heatmap.values())

    return {
        "call_premium": round(call_premium),
        "put_premium": round(put_premium),
        "call_heatmap": call_heatmap,
        "put_heatmap": put_heatmap,
    }



def compare_atm_mispricing(data, exp_date, stock_price):
    """
    Finds the ATM strike (closest to current stock_price),
    retrieves that strike’s call and put,
    computes mid-price for each, and returns the difference.

    Returns a dict with:
      {
        "atm_strike": float,
        "call_mid": float,
        "put_mid": float,
        "difference": float,   # call_mid - put_mid
      }
    or None if data is missing.
    """
    nothing = {
        "atm_strike": 0,
        "call_mid": 0,
        "put_mid": 0,
        "difference": 0,
        "precentage": 0,
        "direction": 'Options Chain Not Valid'
    }

    # make sure the exp date is there
    if exp_date not in data.get("options", {}):
        return nothing

    calls_dict = data["options"][exp_date].get("c", {})
    puts_dict  = data["options"][exp_date].get("p", {})

    #fetch all available strikes
    all_strikes = sorted(list(set(float(s) for s in calls_dict.keys()) |
                              set(float(s) for s in puts_dict.keys())))

    if not all_strikes:
        return nothing

    #find the strike closest to stock_price (ATM)
    atm_strike = min(all_strikes, key=lambda strike: abs(strike - stock_price))
    atm_strike_str = f"{atm_strike:.2f}"

    #retrieve the call/put info if it exists
    call_info = calls_dict.get(atm_strike_str, {})
    put_info  = puts_dict.get(atm_strike_str, {})

    call_iv = call_info.get("iv", 0)
    put_iv = put_info.get("iv", 0)

    #compute the mid-price for each if possible
    def get_mid_price(info):
        # we need bid/ask
        if "b" in info and "a" in info:
            return (info["b"] + info["a"]) / 2
        return 0.0

    call_mid = get_mid_price(call_info)
    put_mid  = get_mid_price(put_info)

    # 6) The difference might show mispricing (call - put)
    difference = call_mid - put_mid
    percentage = (difference / ((call_mid + put_mid) / 2)) * 100 if call_mid + put_mid > 0 else 0
    return {
        "atm_strike": atm_strike,
        "call_mid": call_mid,
        "put_mid": put_mid,
        "difference": difference,
        "precentage": percentage,
        "direction": "Bearish" if percentage > 20 else "Bullish" if percentage < -20 else "Neutral",
        "call_iv": call_iv,
        "put_iv": put_iv,
        "avg_iv": (call_iv + put_iv) / 2 if call_iv and put_iv else 0
    }

def main():
    st.title("✨ EFI Imbalance Screener v2.0")

    st.markdown(
        """
        **Welcome to the Options Scanner by PogChan!**
        Select a sector, customize tickers, and choose an expiration date to analyze options data with a beautiful interface.
        """
    )

    # selection drop down from ticker grops like industries and sectors
    sector_keys = list(sectors.keys())
    selected_sector = st.selectbox("📊 Select a Sector:", sector_keys)

    # custom text box
    default_tickers = ", ".join(sectors[selected_sector])
    tickers_input = st.text_area(
        "📝 Tickers (comma-separated) ^SPX:",
        value=default_tickers,
        help="You can customize the tickers here. Separate each ticker with a comma."
    )

    # make sure we get the tickers rihgt
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    # fridays selection date
    expiration_dates_set = set()

    # Loop through tickers to fetch expiration dates
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            expiration_dates = ticker.options
            if expiration_dates:
                expiration_dates_set.update(expiration_dates)
        except Exception as e:
            st.warning(f"Unable to fetch expiration dates for {symbol}: {e}")

    # Convert set to sorted list for dropdown
    expiration_dates_list = sorted(expiration_dates_set)

    # Streamlit dropdown for expiration dates
    expTopCols = st.columns(2)
    selected_expiration = expTopCols[0].selectbox(
        "📅 Select an Options Expiration Date:",
        expiration_dates_list + ['Custom Date']
    )

    top_n = expTopCols[1].number_input('🔝How many top strikes to display?', min_value=1, value=5)

    if selected_expiration == 'Custom Date':
        custom_date = st.date_input('📆 Select a custom date'
                                    , datetime.datetime.now() + timedelta(days=7))
        selected_expiration = custom_date.strftime('%Y-%m-%d')

    if "runAnalysis" not in st.session_state:
        st.session_state.runAnalysis = False

    if "top_n" not in st.session_state:
        st.session_state.top_n = 5

    def run_analysis_callback():
        st.session_state.runAnalysis = True

    st.button("🚀 Run Analysis", on_click=run_analysis_callback)

    if st.session_state.runAnalysis:
        st.markdown("### 📈 Analyzing Options Data...")
        results = []

        chains = get_options_chains(tickers, selected_expiration)

        for symbol, data in chains.items():
            with st.spinner(f"🔍 Analyzing {symbol}..."):
                # stock price fetch yfinance
                stock_price = get_stock_price(symbol)
                if stock_price is None:
                    st.error(f"⚠️ Skipping {symbol} due to missing stock price.")
                    continue

                #options chain fetch
                if not data:
                    st.error(f"⚠️ No valid options data for {symbol} {selected_expiration}.")
                    continue


                # get hte call put premium stuff from the chain parsed
                result = analyze_options_chain(data, selected_expiration, stock_price)

                #get the results into a final little mapping
                call_premium = result["call_premium"]
                put_premium = result["put_premium"]
                put_call_ratio = put_premium / call_premium if call_premium > 0 else float("inf")

                # displays
                st.markdown(f"#### **{symbol}**")
                st.markdown(
                    f"""
                    - **Stock Price:** ${stock_price:,.2f}
                    - **Call Premium:** ${call_premium:,}
                    - **Put Premium:** ${put_premium:,}
                    - **Put-to-Call Ratio:** {put_call_ratio:.2f}
                    """
                )
                atm_mispricing = compare_atm_mispricing(data, selected_expiration, stock_price)

                # st.write(atm_mispricing)
                avg_iv = atm_mispricing.get("avg_iv", 0) * 100


                st.markdown(f"**ATM Strike:** {atm_mispricing['atm_strike']}")
                st.markdown(
                    f"""
                    - {"🐻" if atm_mispricing['direction'] == 'Bearish' else "🐂" if atm_mispricing['direction'] == 'Bullish' else "😐"} Potential {atm_mispricing['direction']} Mispricing
                    - {atm_mispricing['call_mid']:.2f} ATM Call - {atm_mispricing['put_mid']:.2f} ATM Put
                    - Difference: {atm_mispricing['difference']:.2f} ({atm_mispricing['precentage']:.2f}%)
                    - ATM IV: {avg_iv:.2f}%
                    """)

                results.append({
                    "Symbol": symbol,
                    "Stock Price": stock_price,
                    "Sum Call Premium": call_premium,
                    "Sum Put Premium": put_premium,
                    "Put-To-Call Ratio": put_call_ratio,
                    "call_heatmap": result["call_heatmap"],
                    "put_heatmap": result["put_heatmap"],
                    "ATM Premium Difference": atm_mispricing['difference'],
                    "Percentage Significance": atm_mispricing['precentage'],
                })

                if top_n != st.session_state.top_n:
                    st.session_state.top_n = top_n


                plotChartOI(symbol, data, selected_expiration, top_n=top_n)

                plot_volatility_comparison(symbol, avg_iv)

                stock_seasonality(symbol)


                st.markdown("##### Top 5 Call Heatmap Strikes")
                call_heatmap_data = pd.DataFrame(
                    sorted(result["call_heatmap"].items(), key=lambda x: x[1], reverse=True)[:5],
                    columns=["Strike Price", "Premium"]
                )
                call_heatmap_data["Premium"] = call_heatmap_data["Premium"].apply(lambda x: f"${x:,}")
                st.table(call_heatmap_data)

                st.markdown("##### Top 5 Put Heatmap Strikes")
                put_heatmap_data = pd.DataFrame(
                    sorted(result["put_heatmap"].items(), key=lambda x: x[1], reverse=True)[:5],
                    columns=["Strike Price", "Premium"]
                )
                put_heatmap_data["Premium"] = put_heatmap_data["Premium"].apply(lambda x: f"${x:,}")
                st.table(put_heatmap_data)

        # final
        if results:
            st.markdown("### 🏆 Final Results")
            df = pd.DataFrame(results).sort_values("Put-To-Call Ratio", ascending=False)
            st.dataframe(df[["Symbol", "Stock Price", "Sum Call Premium", "Sum Put Premium", "Put-To-Call Ratio", "ATM Premium Difference", "Percentage Significance"]])
        else:
            st.write("⚠️ No data available for the selected tickers and expiration.")

# from py_vollib.black_scholes.implied_volatility import implied_volatility

# def fetch_risk_free_rate():
#     """
#     Fetches the risk-free rate (13-week or 10-year Treasury yield) from Yahoo Finance.
#     Tries to retrieve the latest available data over the past 5 days.

#     Returns:
#         float: Risk-free rate as a decimal.
#     """
#     try:
#         # Fetch 13-week Treasury yield
#         treasury_data = yf.Ticker("^IRX")
#         recent_data = treasury_data.history(period="5d")["Close"].dropna()
#         if not recent_data.empty:
#             rate = recent_data.iloc[-1] / 100  # Convert to decimal
#             print(f"Fetched 13-week Treasury yield: {rate:.2%}")
#             return rate
#     except Exception as e:
#         print(f"Error fetching ^IRX: {e}")

#     try:
#         # Fallback to 10-year Treasury yield
#         treasury_data = yf.Ticker("^TNX")
#         recent_data = treasury_data.history(period="5d")["Close"].dropna()
#         if not recent_data.empty:
#             rate = recent_data.iloc[-1] / 100  # Convert to decimal
#             print(f"Fetched 10-year Treasury yield: {rate:.2%}")
#             return rate
#     except Exception as e:
#         print(f"Error fetching ^TNX: {e}")

#     # Default risk-free rate
#     default_rate = 0.01
#     print(f"Using default risk-free rate: {default_rate:.2%}")
#     return default_rate


# def calculate_iv(underlying_price, strike_price, time_to_expiration, bid_price, ask_price, option_type='c'):
#     """
#     Calculates implied volatility for an option.

#     Parameters:
#         underlying_price (float): Current price of the underlying asset.
#         strike_price (float): Strike price of the option.
#         time_to_expiration (float): Time to expiration in years.
#         bid_price (float): Bid price of the option.
#         ask_price (float): Ask price of the option.
#         option_type (str): 'c' for call, 'p' for put.

#     Returns:
#         float: Implied volatility as a percentage.
#     """
#     market_price = (bid_price + ask_price) / 2
#     risk_free_rate = fetch_risk_free_rate()
#     print(f"Market Price: ${market_price:.2f}, Risk-Free Rate: {risk_free_rate:.2%}")
#     try:
#         iv = implied_volatility(market_price, underlying_price, strike_price, time_to_expiration, risk_free_rate, option_type)
#         return iv * 100  # Convert to percentage
#     except Exception as e:
#         print(f"Error calculating implied volatility: {e}")
#         return None

if __name__ == "__main__":
    main()

    # # Example usage
    # underlying_price = 229.98  # Current price of the underlying
    # strike_price = 230  # Option strike price
    # time_to_expiration = 10 / 365  # Time to expiration in years
    # bid_price = 2.76  # Bid price of the option
    # ask_price = 2.85  # Ask price of the option

    # iv = calculate_iv(underlying_price, strike_price, time_to_expiration, bid_price, ask_price, option_type='c')
    # if iv is not None:
    #     print(f"Implied Volatility: {iv:.2f}%")
    # else:
    #     print("Failed to calculate implied volatility.")
