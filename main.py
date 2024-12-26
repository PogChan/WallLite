# Copyright (c) 2024 PogChan Github
# All rights reserved.

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
from OIChart import *

load_dotenv()

apiUrl = st.secrets["API"]
baseURL = st.secrets["BASEAPI"]

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)",
]

# run options chain
@st.cache_data(ttl=60*60*8)
def get_options_chain(symbol):
    time.sleep(1)
    url = f"{baseURL}?stock={symbol.upper()}&reqId={random.randint(1, 1000000)}"
    headers = {
        'User-Agent': random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        'Referer': apiUrl,
        'Accept': 'application/json',
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch options chain for {symbol}. Status code: {response.status_code}")
        return None

def fetch_ticker_data(symbol):
    return {symbol: get_options_chain(symbol)}

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
    today = datetime.now()
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

def main():
    st.title("✨ EFI Imbalance Screener")

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
        "📝 Tickers (comma-separated):",
        value=default_tickers,
        help="You can customize the tickers here. Separate each ticker with a comma."
    )

    # make sure we get the tickers rihgt
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    # fridays selection date
    expiration_dates = get_next_fridays()
    expTopCols = st.columns(2)
    selected_expiration = expTopCols[0].selectbox("📅 Select an Options Expiration Date:", expiration_dates)
    top_n = expTopCols[1].number_input('🔝How many top strikes to display?', min_value=1, value=5)

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

        for symbol in tickers:
            with st.spinner(f"🔍 Analyzing {symbol}..."):
                # stock price fetch yfinance
                stock_price = get_stock_price(symbol)
                if stock_price is None:
                    st.write(f"⚠️ Skipping {symbol} due to missing stock price.")
                    continue

                #options chain fetch
                data = get_options_chain(symbol)
                if not data:
                    st.write(f"⚠️ No valid options data for {symbol}.")
                    continue


                # get hte call put premium stuff from the chain parsed
                result = analyze_options_chain(data, selected_expiration, stock_price)

                #get the results into a final little mapping
                call_premium = result["call_premium"]
                put_premium = result["put_premium"]
                put_call_ratio = put_premium / call_premium if call_premium > 0 else float("inf")

                results.append({
                    "symbol": symbol,
                    "stock_price": stock_price,
                    "call_premium": call_premium,
                    "put_premium": put_premium,
                    "put_call_ratio": put_call_ratio,
                    "call_heatmap": result["call_heatmap"],
                    "put_heatmap": result["put_heatmap"],
                })

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
                if top_n != st.session_state.top_n:
                    st.session_state.top_n = top_n

                plotChartOI(symbol, data, selected_expiration, top_n=top_n)

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
            df = pd.DataFrame(results).sort_values("put_call_ratio", ascending=False)
            st.dataframe(df[["symbol", "stock_price", "call_premium", "put_premium", "put_call_ratio"]])
        else:
            st.write("⚠️ No data available for the selected tickers and expiration.")

if __name__ == "__main__":
    main()
