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
load_dotenv()
import os 

apiUrl = os.getenv("API")
# User agent pool
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)",
]

# Cached function to fetch options chain
@st.cache_data(ttl=43200)
def get_options_chain(symbol):
    time.sleep(1)
    url = f"{apiUrl}ajax/getOptions?stock={symbol.upper()}&reqId={random.randint(1, 1000000)}"
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

# Fetch stock price from Yahoo Finance
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
    # Find the next Friday
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

    def process_options(data, stock_price, max_strikes=15):
        """Process options data dynamically around stock price."""
        premiums = {}
        strikes = sorted([float(strike) for strike in data.keys()])

        # Determine strike increment
        increments = [round(strikes[i + 1] - strikes[i], 2) for i in range(len(strikes) - 1)]
        increment = max(set(increments), key=increments.count) if increments else 1

        # Get valid strikes within range
        valid_strikes = [
            strike for strike in strikes
            if stock_price - (increment * max_strikes / 2) <= strike <= stock_price + (increment * max_strikes / 2)
        ]

        # Process premiums for valid strikes
        for strike in valid_strikes:
            # Use the original string representation for lookup
            strike_key = f"{strike:.2f}"
            info = data.get(strike_key, {})
   
            # Check if 'b' and 'a' keys exist and are valid
            if 'b' in info and 'a' in info and 'oi' in info:
                mid_price = (info.get("b", 0) + info.get("a", 0)) / 2
                total_premium = round(mid_price * info.get("oi", 0)) * 100  # Round premium to whole number
                premiums[float(strike_key)] = total_premium
            else:
                st.write(f"Invalid data for strike {strike_key}: {info}")

        return premiums

    # Process call and put data
    call_heatmap = process_options(call_data, stock_price)
    put_heatmap = process_options(put_data, stock_price)

    # Calculate totals
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

    # Dropdown for sectors
    sector_keys = list(sectors.keys())
    selected_sector = st.selectbox("📊 Select a Sector:", sector_keys)

    # Text box for tickers
    default_tickers = ", ".join(sectors[selected_sector])
    tickers_input = st.text_area(
        "📝 Tickers (comma-separated):",
        value=default_tickers,
        help="You can customize the tickers here. Separate each ticker with a comma."
    )

    # Parse the tickers
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    # Dropdown for expiration dates
    expiration_dates = get_next_fridays()
    selected_expiration = st.selectbox("📅 Select an Options Expiration Date:", expiration_dates)

    # Button to run analysis
    if st.button("🚀 Run Analysis"):
        st.markdown("### 📈 Analyzing Options Data...")
        results = []

        for symbol in tickers:
            with st.spinner(f"🔍 Analyzing {symbol}..."):
                # Fetch stock price
                stock_price = get_stock_price(symbol)
                if stock_price is None:
                    st.write(f"⚠️ Skipping {symbol} due to missing stock price.")
                    continue

                # Fetch options chain
                data = get_options_chain(symbol)
                if not data:
                    st.write(f"⚠️ No valid options data for {symbol}.")
                    continue

                # Analyze options chain
                result = analyze_options_chain(data, selected_expiration, stock_price)

                # Append results
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

                # Display partial result with better formatting
                st.markdown(f"#### **{symbol}**")
                st.markdown(
                    f"""
                    - **Stock Price:** ${stock_price:,.2f}  
                    - **Call Premium:** ${call_premium:,}  
                    - **Put Premium:** ${put_premium:,}  
                    - **Put-to-Call Ratio:** {put_call_ratio:.2f}
                    """
                )

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

        # Final summary
        if results:
            st.markdown("### 🏆 Final Results")
            df = pd.DataFrame(results).sort_values("put_call_ratio", ascending=False)
            st.dataframe(df[["symbol", "stock_price", "call_premium", "put_premium", "put_call_ratio"]])
        else:
            st.write("⚠️ No data available for the selected tickers and expiration.")

# Run the app
if __name__ == "__main__":
    main()
