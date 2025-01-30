import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
import datetime as datetime

def plotChartOI(symbol, data, exp_date, top_n=5):
    #Download 1 month of data its free
    df = yf.download(symbol, period="1mo", interval="1d")
    if df.empty:
        st.warning(f"No price data for {symbol}.")
        return

    # Flatten columns if multi-level from yfinance
    if hasattr(df.columns, "droplevel") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(-1)

    # chain is verified but we want to just dobule check the exp exists
    if exp_date not in data.get("options", {}):
        st.warning(f"No options data found for {exp_date}.")
        return

    calls_dict = data["options"][exp_date].get("c", {})
    puts_dict  = data["options"][exp_date].get("p", {})

    # get the data for the actual options parsing
    def parse_chain(chain, opt_type):
        """
         {
           "type":      "call" or "put",
           "strike":    float,
           "oi":        float,
           "volume":    float,
           "totalValue": float  # = OI * (bid+ask)/2 * 100
         }
        Skip zero OI or zero bid/ask.
        """
        parsed = []
        for strike_str, info in chain.items():
            if not all(k in info for k in ("oi", "v", "b", "a")):
                continue

            strike = float(strike_str)
            oi     = info["oi"]
            vol    = info["v"]
            bid    = info["b"]
            ask    = info["a"]

            if oi <= 0:
                continue  # skip zero OI
            if bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2
            else:
                mid_price = 0

            total_val = oi * mid_price * 100

            parsed.append({
                "type": opt_type,
                "strike": strike,
                "oi": oi,
                "volume": vol,
                "totalValue": total_val
            })
        return parsed

    calls = parse_chain(calls_dict, "call")
    puts  = parse_chain(puts_dict,  "put")

    #Sort & pick top_n by OI, top_n by Volume
    top_calls_oi     = sorted(calls, key=lambda x: x["oi"],     reverse=True)[:top_n]
    top_calls_volume = sorted(calls, key=lambda x: x["volume"], reverse=True)[:top_n]
    top_puts_oi      = sorted(puts,  key=lambda x: x["oi"],     reverse=True)[:top_n]
    top_puts_volume  = sorted(puts,  key=lambda x: x["volume"], reverse=True)[:top_n]

    #{"type":"call"/"put","strike", "oi","volume","totalValue","metric":"oi"/"volume"}
    lines = []

    # calls by OI -> green
    for row in top_calls_oi:
        lines.append({**row, "metric": "oi"})
    # puts by OI -> red
    for row in top_puts_oi:
        lines.append({**row, "metric": "oi"})
    # calls by Volume -> orange
    for row in top_calls_volume:
        lines.append({**row, "metric": "volume"})
    # puts by Volume -> blue
    for row in top_puts_volume:
        lines.append({**row, "metric": "volume"})

    if not lines:
        st.warning("No OI/Volume data found.")
        return

    # -------------------------------------------------------------------------
    # Toggle calls/puts/both
    # -------------------------------------------------------------------------
    display_choice = st.selectbox(
            "Show Which Bars?",
            ["Both Calls & Puts", "Calls Only", "Puts Only"],
            key=symbol
    )

    # Filter the lines based on user choice
    filtered_lines = []
    for line in lines:
        if display_choice == "Calls Only" and line["type"] == "call":
            filtered_lines.append(line)
        elif display_choice == "Puts Only" and line["type"] == "put":
            filtered_lines.append(line)
        elif display_choice == "Both Calls & Puts":
            filtered_lines.append(line)

    if not filtered_lines:
        st.warning(f"No {display_choice} data found.")
        return

    # Sort from largest to smallest so the largest bars are drawn first
    #      and the smallest bars are drawn last (on top).
    def get_value(row):
        return row["oi"] if row["metric"] == "oi" else row["volume"]
    filtered_lines = sorted(filtered_lines, key=get_value, reverse=True)

    #this is actually fire they have it lol
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol
        )
    )

    min_date = df.index.min()
    max_date = df.index.max()
    total_days = (max_date - min_date).days
    if total_days < 1:
        total_days = 1

    # the largest will be the scale here.
    all_vals = [2000]
    for r in lines:
        if r["metric"] == "oi":
            all_vals.append(r["oi"])
        else:
            all_vals.append(r["volume"])

    min_val = min(all_vals) if all_vals else 0
    max_val = max(all_vals) if all_vals else 1

    def unify_normalize(v):
        # If all values are the same, fallback to 1
        if max_val == min_val:
            return 1
        return (v - min_val) / (max_val - min_val)

    # prevent overlapwith tiny price offset
    offset_map = {
       ("call","oi"):     0.03,
       ("put","oi"):     -0.03,
       ("call","volume"): 0.05,
       ("put","volume"): -0.05
    }

    # anchro each bar near the right side (max_date),
    # then extend left by bar_length_days, clamping at 90% of chart width.
    day_offset = 0.0

    for entry in filtered_lines:
        typ    = entry["type"]       # "call" or "put"
        strike = entry["strike"]
        oi     = entry["oi"]
        vol    = entry["volume"]
        tval   = entry["totalValue"]
        metric = entry["metric"]     # "oi" or "volume"

        if   (typ == "call" and metric=="oi"):       color = "green"
        elif (typ == "put"  and metric=="oi"):       color = "red"
        elif (typ == "call" and metric=="volume"):   color = "orange"
        else:                                        color = "blue"

        raw_value = oi if metric == "oi" else vol
        scale = unify_normalize(raw_value)

        #well the thing is that we need to scale the bar length based off the days so its ezpz
        bar_length_days = scale * (0.5 * total_days)

        # clamp so we don't go off the chart entirely
        bar_length_days = min(bar_length_days, 0.9 * total_days)

        # anchor each bar near the right side, shifting each line horizontally
        x1 = max_date - timedelta(days=day_offset)

        x0 = x1 - timedelta(days=bar_length_days)

        # tiny offset in price so lines at same strike won't overlap
        y_offset = offset_map.get((typ, metric), 0.0)
        y0 = strike + y_offset
        y1 = strike + y_offset

        #get the bar added
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color=color, width=6),
            opacity=0.7
        )

        # 5) add an invisible scatter for hover
        mid_time = x0 + (x1 - x0)/2
        hover_text = (
            f"<b>{typ.upper()} {metric.upper()}</b><br>"
            f"Strike: {strike}<br>"
            f"OI: {oi}<br>"
            f"Volume: {vol}<br>"
            f"Total Premium: ${tval:,.0f}<br>"
        )
        fig.add_trace(
            go.Scatter(
                x=[mid_time],
                y=[(y0 + y1)/2],
                mode="markers",
                marker=dict(size=10, color=color, opacity=0),
                hovertemplate=hover_text
            )
        )

        # day_offset += 0.7  # shift next bar left by 0.7 day to avoid clumping tbh

    # LEGENDS DATAS THESE ARE INVISIBLE
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="green", width=6),
        name="Call OI"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="red", width=6),
        name="Put OI"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="orange", width=6),
        name="Call Volume"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="blue", width=6),
        name="Put Volume"
    ))

    # Layout
    fig.update_layout(
        title=f"{symbol.upper()} - {exp_date}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=800
    )

    fig.update_layout(
        title=(
            f"{symbol.upper()} — {exp_date}<br>"
            f"Top {top_n} OI & Volume (Calls/Puts)"
        ),
        xaxis_title="Date",
        yaxis_title="Price (Strike)",
        xaxis_rangeslider_visible=False,
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_historical_volatility(symbol, window=20):
    """
    Calculate n-day historical volatility (annualized)
    Returns: Series with dates and volatility values
    """
    try:
        # Get enough data to account for the rolling window
        df = yf.download(symbol, period=f"2y", interval="1d")
        if df.empty:
            st.warning(f"No price data for {symbol}.")
            return

        # Flatten columns if multi-level from yfinance
        if hasattr(df.columns, "droplevel") and len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel(-1)

        returns = np.log(df['Close']).diff()
        hv = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized as percentage
        return hv.dropna()
    except Exception as e:
        st.error(f"Historical Volatility Error: {e}")
        return pd.Series()


def plot_volatility_comparison(symbol, avg_iv):
    """
    Plots historical volatility vs average IV for selected expiration
    """
    hv_series = calculate_historical_volatility(symbol)

    # Plot volatility comparison
    if hv_series.empty:
        st.warning("Could not calculate historical volatility")
        return

    fig = go.Figure()

    # Historical Volatility Line
    fig.add_trace(go.Scatter(
        x=hv_series.index,
        y=hv_series,
        mode='lines',
        name='20D Historical Volatility',
        line=dict(color='#1f77b4', width=2)
    ))

    # Average IV Horizontal Line
    fig.add_trace(go.Scatter(
        x=[hv_series.index[0], hv_series.index[-1]],
        y=[avg_iv, avg_iv],
        mode='lines',
        name=f'ATM IV (Avg)',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"{symbol} Volatility Comparison",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
        showlegend=True,
        height=400
    )
    st.plotly_chart(fig,
                use_container_width=True)
    return fig




def stock_seasonality(ticker, start_date='2013-01-01', 
                      exclude_years=[2020], smooth_factor=3,
                      confidence_band=True, show_current_year=True):
    """
    Enhanced seasonality analysis with probabilistic forecasting and regime awareness
    
    Improvements:
    1. Better normalization using Z-score detrending
    2. Volatility-adjusted smoothing
    3. Current year comparison
    4. Confidence bands with percentile ranges
    5. Residual seasonality detection
    """
    
    # Fetch and prepare data
    data = yf.download(ticker, start=start_date)
    if data.empty:
        st.warning(f"No price data for {ticker}.")
        return

    # Get as business days and fill in, if we have NaN days in business days jsut forward fill from prev day. 
    data = data.asfreq('B').ffill().dropna()
    data['daily_return'] = data['Adj Close'].pct_change().dropna()
    
    # Excludes the date
    data['year'] = data.index.year
    data['day_of_year'] = data.index.dayofyear
    data = data[~data['year'].isin(exclude_years)]
    
    # This ensures that volatility in each year is normalzied. that we dont have periods of low volatility be comapred iwth periods of hgih vaoliltiy 
    def normalize_group(group):
        mu = group.mean()
        std = group.std()
        return (group - mu) / std
    
    normalized_returns = data.groupby('year')['daily_return'].transform(normalize_group)
    data['norm_return'] = normalized_returns
    
    # Gets the uptrend or downtrend as you get cumulative sum per year. 1+ 2+ 3 + 4 with increasing means the trend keeps going up, strong!
    data['cumulative_index'] = data.groupby('year')['norm_return'].cumsum()
    
    # Convert to common year dates
    base_year = 2020  # Leap year for alignment
    data['date'] = data['day_of_year'].apply(lambda x: pd.to_datetime(f'{base_year}-{x:03d}', format='%Y-%j'))

    # Calculate probabilistic paths
    daily_stats = data.groupby('date')['cumulative_index'].agg(
        ['median', 'mean', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
    ).rename(columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'})

    # Volatility-adjusted smoothing
    def adaptive_smooth(series, window):
        window = max(int(window), 3)  # Ensure minimum window size
        return series.rolling(
            window=window,
            win_type='gaussian',
            center=True,
            min_periods=1
        ).mean(std=2)
    
    daily_stats['smooth_median'] = adaptive_smooth(daily_stats['median'], window=smooth_factor)

    # Current year comparison
    current_year = datetime.datetime.now().year
    current_data = data[data['year'] == current_year].copy()

    today = datetime.datetime.today()
    start_1m = (today - datetime.timedelta(days=15)).strftime(f'{base_year}-%m-%d')
    end_1m   = (today + datetime.timedelta(days=15)).strftime(f'{base_year}-%m-%d')
    start_3m = (today - datetime.timedelta(days=60)).strftime(f'{base_year}-%m-%d')
    end_3m   = (today + datetime.timedelta(days=60)).strftime(f'{base_year}-%m-%d')
    # Visualization
    fig = go.Figure()
    
    # Historical years
    for year in data['year'].unique():
        if year == current_year: continue  # Handle current year separately
        year_data = data[data['year'] == year]
        fig.add_trace(go.Scatter(
            x=year_data['date'],
            y=year_data['cumulative_index'],
            mode='lines',
            line=dict(width=0.7, color='lightgray'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Confidence bands
    if confidence_band:
        fig.add_trace(go.Scatter(
            x=daily_stats.index,
            y=daily_stats['q75'],
            line=dict(width=0),
            hoverinfo='skip',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_stats.index,
            y=daily_stats['q25'],
            fill='tonexty',
            fillcolor='rgba(100,100,100,0.2)',
            line=dict(width=0),
            name='25-75% Percentile',
            hoverinfo='skip'
        ))

    # Smoothed median path
    fig.add_trace(go.Scatter(
        x=daily_stats.index,
        y=daily_stats['smooth_median'],
        mode='lines',
        line=dict(color='navy', width=3),
        name='Median Seasonal Path',
        hovertemplate="<b>%{x|%b %d}</b><br>Index: %{y:.2f}<extra></extra>"
    ))

    # Current year overlay
    if show_current_year and not current_data.empty:
        current_data = current_data[current_data['date'] <= datetime.datetime.now().strftime(f'{base_year}-%m-%d')]
        fig.add_trace(go.Scatter(
            x=current_data['date'],
            y=current_data['cumulative_index'],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=5),
            name=f'{current_year} Actual',
            hovertemplate="<b>%{x|%b %d}</b><br>Current: %{y:.2f}<extra></extra>"
        ))

        # Enhanced layout for vertical scaling
        fig.update_layout(
            title=f'{ticker} Seasonality Analysis',
            xaxis=dict(
                type='date',
                rangeslider=dict(visible=True),
                range=[f'{base_year}-01-01', f'{base_year}-12-31'],
                tickformat='%b %d',
                autorange=True,
                fixedrange=False  # Allow horizontal zoom/pan
            ),
            yaxis=dict(
                title='Normalized Index',
                zerolinecolor='gray',
                zerolinewidth=1,
                fixedrange=False,  # Allow vertical zoom/pan
                autorange=True,
                automargin=True
            ),
            hovermode='x unified',
            dragmode='zoom',  # Start in zoom mode
            margin=dict(t=40, b=20, l=40, r=20),
        )

        # Add range selector buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Zoom 1M",
                            method="relayout",
                            args=[{"xaxis.range": [f'{base_year}-01-01', f'{base_year}-02-01']}]),
                        dict(label="Zoom 3M",
                            method="relayout",
                            args=[{"xaxis.range": [f'{base_year}-01-01', f'{base_year}-04-01']}]),
                        dict(label="Reset",
                            method="relayout",
                            args=[{"xaxis.autorange": True, "yaxis.autorange": True}])
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )

        # Enable all zoom/pan features
        config = {
            'scrollZoom': True,
            'modeBarButtonsToAdd': [
                'vscrollzoom',
                'hscrollzoom',
                'togglespikelines',
                'resetScale2d'
            ]
        }

        st.plotly_chart(fig, use_container_width=True, config=config)
        return fig
