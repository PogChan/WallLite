import streamlit as st
from datetime import datetime, timedelta
import calendar
from tickers import * 
import requests
import random
import time

# Helper function to get next Fridays
def get_next_fridays(n=10):
    """Return the next `n` Fridays from today."""
    today = datetime.today()
    fridays = []
    for i in range(n):
        next_friday = today + timedelta((calendar.FRIDAY - today.weekday()) % 7 + 7 * i)
        fridays.append(next_friday.strftime('%Y-%m-%d'))
    return fridays

# API request session and user-agents
session = requests.Session()
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
]

@st.cache_data(ttl=43200)
def get_options_chain(symbol):
    """Fetch options chain from API."""
    url = f"https://www.optionsprofitcalculator.com/ajax/getOptions?stock={symbol.upper()}&reqId={random.randint(1, 1000000)}"
    headers = {
        'User-Agent': random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        'Referer': 'https://www.optionsprofitcalculator.com/',
        'Accept': 'application/json',
    }
    time.sleep(1)  # To avoid being rate-limited
    response = session.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch options chain for {symbol}. Status code: {response.status_code}")
        return None


def analyze_tickers(tickers, exp_date):
    """Analyze tickers for heatmap and PC flow."""
    results = []
    for symbol in tickers:
        st.write(f"Fetching data for {symbol}...")
        option_chain = get_options_chain(symbol)
        if not option_chain or 'options' not in option_chain:
            st.write(f"No data available for {symbol}")
            continue

        stock_price = float(option_chain.get('stock_price', 0))
        call_data = option_chain['options']['call']
        put_data = option_chain['options']['put']

        call_heatmap = {}
        put_heatmap = {}
        total_call_oi = 0
        total_put_oi = 0

        # Process calls
        for strike, data in call_data.items():
            strike_price = float(strike)
            open_interest = int(data.get('oi', 0))
            mid_price = float(data.get('last', 0))
            if stock_price >= strike_price:
                total_call_oi += open_interest
            call_heatmap[strike_price] = mid_price * open_interest

        # Process puts
        for strike, data in put_data.items():
            strike_price = float(strike)
            open_interest = int(data.get('oi', 0))
            mid_price = float(data.get('last', 0))
            if stock_price <= strike_price:
                total_put_oi += open_interest
            put_heatmap[strike_price] = mid_price * open_interest

        total_call_value = sum(call_heatmap.values())
        total_put_value = sum(put_heatmap.values())
        put_call_ratio = total_put_value / total_call_value if total_call_value > 0 else float('inf')

        # Store results
        results.append({
            "symbol": symbol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "total_call_value": total_call_value,
            "total_put_value": total_put_value,
            "put_call_ratio": put_call_ratio,
        })

    return results

# Streamlit App
def main():
    st.title("Options Scanner Interface")
    st.markdown(
        """
        **Select a sector**, customize tickers, and choose an expiration date to run the options analysis.
        """
    )
    
    # Dropdown for sectors
    sector_keys = list(sectors.keys())
    selected_sector = st.selectbox("Select a Sector:", sector_keys)
    
    # Text box for tickers
    default_tickers = ", ".join(sectors[selected_sector])
    tickers_input = st.text_area(
        "Tickers (comma-separated):",
        value=default_tickers,
        help="You can customize the tickers here. Separate each ticker with a comma."
    )
    
    # Parse the tickers
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    
 
    
    # Dropdown for expiration dates
    expiration_dates = get_next_fridays()
    selected_expiration = st.selectbox("Select an Options Expiration Date:", expiration_dates)
    
    # Button to run analysis
    if st.button("Run Analysis"):
        # Placeholder for options analysis logic
        st.markdown("### Running Options Analysis...")
        st.write(f"Selected Sector: {selected_sector}")
        st.write(f"Selected Tickers: {tickers}")
        st.write(f"Selected Expiration Date: {selected_expiration}")
        
        # Simulate analysis (replace this with actual logic)
        st.markdown("Analysis complete! Displaying results...")
        st.write(f"Results for tickers: {', '.join(tickers)} on expiration: {selected_expiration}")
        # Add your analysis logic here, e.g., calling an API or running calculations.

# Run the app
if __name__ == "__main__":
    main()