import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta

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

def pc_check(symbol, data, top_n=5):
    """
    pc_check:
      - Aggregates the volume across all expirations in the options chain data.
      - Groups the volume by strike and option type (call/put).
      - Plots a candlestick chart for the underlying stock along with volume bars
        (calls in orange, puts in blue) indicating where the volume has been placed.
      - Hover text includes the aggregated volume, total premium value (based on volume),
        and a breakdown of which expiration(s) contributed that volume.
    """
    # Download stock price data (e.g., 1 month) for the background candlestick chart
    df = yf.download(symbol, period="1mo", interval="1d")
    if df.empty:
        st.warning(f"No price data for {symbol}.")
        return

    # Flatten columns if necessary
    if hasattr(df.columns, "droplevel") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(-1)

    if "options" not in data:
        st.warning("No options data found.")
        return

    # Aggregate volume data across all expirations by (option type, strike)
    aggregated = {}  # key: (option_type, strike), value: dict with cumulative volume and breakdown
    for exp_date, exp_data in data["options"].items():
        # Process call options for this expiration
        calls = exp_data.get("c", {})
        for strike_str, info in calls.items():
            if not all(k in info for k in ("v", "b", "a")):
                continue
            try:
                strike = float(strike_str)
            except Exception:
                continue
            vol = info.get("v", 0)
            bid = info.get("b", 0)
            ask = info.get("a", 0)
            mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
            total_val = vol * mid_price * 100
            key = ("call", strike)
            if key not in aggregated:
                aggregated[key] = {"volume": 0, "totalValue": 0, "exp_breakdown": {}}
            aggregated[key]["volume"] += vol
            aggregated[key]["totalValue"] += total_val
            aggregated[key]["exp_breakdown"][exp_date] = aggregated[key]["exp_breakdown"].get(exp_date, 0) + vol

        # Process put options for this expiration
        puts = exp_data.get("p", {})
        for strike_str, info in puts.items():
            if not all(k in info for k in ("v", "b", "a")):
                continue
            try:
                strike = float(strike_str)
            except Exception:
                continue
            vol = info.get("v", 0)
            bid = info.get("b", 0)
            ask = info.get("a", 0)
            mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
            total_val = vol * mid_price * 100
            key = ("put", strike)
            if key not in aggregated:
                aggregated[key] = {"volume": 0, "totalValue": 0, "exp_breakdown": {}}
            aggregated[key]["volume"] += vol
            aggregated[key]["totalValue"] += total_val
            aggregated[key]["exp_breakdown"][exp_date] = aggregated[key]["exp_breakdown"].get(exp_date, 0) + vol

    # Convert aggregated dictionary to a list of dictionaries for plotting
    aggregated_list = []
    for (option_type, strike), values in aggregated.items():
        aggregated_list.append({
            "type": option_type,
            "strike": strike,
            "volume": values["volume"],
            "totalValue": values["totalValue"],
            "exp_breakdown": values["exp_breakdown"]
        })

    # Separate calls and puts and select the top_n items (by volume)
    calls_agg = [item for item in aggregated_list if item["type"] == "call"]
    puts_agg  = [item for item in aggregated_list if item["type"] == "put"]

    top_calls_volume = sorted(calls_agg, key=lambda x: x["volume"], reverse=True)[:top_n]
    top_puts_volume  = sorted(puts_agg, key=lambda x: x["volume"], reverse=True)[:top_n]

    # Prepare the lines to be drawn (only volume-based)
    lines = []
    # For calls
    for row in top_calls_volume:
        sorted_exp = sorted(row["exp_breakdown"].items(), key=lambda x: x[1], reverse=True)[:top_n]

        exp_info = "<br>".join([f"{exp}: {vol}" for exp, vol in sorted_exp])
        row_copy = row.copy()
        row_copy["metric"] = "volume"
        row_copy["hover_exp"] = exp_info
        lines.append(row_copy)
    # For puts
    for row in top_puts_volume:
        sorted_exp = sorted(row["exp_breakdown"].items(), key=lambda x: x[1], reverse=True)[:top_n]

        exp_info = "<br>".join([f"{exp}: {vol}" for exp, vol in sorted_exp])
        row_copy = row.copy()
        row_copy["metric"] = "volume"
        row_copy["hover_exp"] = exp_info
        lines.append(row_copy)

    if not lines:
        st.warning("No volume data found.")
        return

    # Allow user to filter the bars (calls only, puts only, or both)
    display_choice = st.selectbox(
        "Show Which Bars?",
        ["Both Calls & Puts", "Calls Only", "Puts Only"],
        key=symbol + "_pc_check"
    )

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

    # Sort the filtered lines by volume (largest first)
    filtered_lines = sorted(filtered_lines, key=lambda row: row["volume"], reverse=True)

    # Build the Plotly chart with the stock candlesticks in the background
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

    # Determine the scaling of the volume bars
    all_vals = [2000]  # default to ensure nonzero range
    for r in filtered_lines:
        all_vals.append(r["volume"])
    min_val = min(all_vals)
    max_val = max(all_vals)

    def unify_normalize(v):
        if max_val == min_val:
            return 1
        return (v - min_val) / (max_val - min_val)

    # Offset mapping for clarity (prevents overlapping bars at the same strike)
    offset_map = {
        ("call", "volume"): 0.05,
        ("put", "volume"): -0.05
    }

    day_offset = 0.0

    for entry in filtered_lines:
        typ = entry["type"]      # "call" or "put"
        strike = entry["strike"]
        vol = entry["volume"]
        tval = entry["totalValue"]
        metric = entry["metric"]  # always "volume" here

        # Choose color: calls (orange) and puts (blue)
        color = "orange" if typ == "call" else "blue"
        raw_value = vol
        scale = unify_normalize(raw_value)


        bar_length_days = scale * (0.5 * total_days)
        bar_length_days = min(bar_length_days, 0.9 * total_days)

        # Anchor each bar near the right side of the chart
        x1 = max_date - timedelta(days=day_offset)
        x0 = x1 - timedelta(days=bar_length_days)

        # Apply a small vertical offset so bars at the same strike don't overlap
        y_offset = offset_map.get((typ, metric), 0.0)
        y0 = strike + y_offset
        y1 = strike + y_offset

        # Draw the bar as a line shape
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color=color, width=6),
            opacity=0.7
        )

        # Add an invisible scatter for hover text
        mid_time = x0 + (x1 - x0) / 2
        hover_text = (
            f"<b>{typ.upper()} Volume</b><br>"
            f"Strike: {strike}<br>"
            f"Volume: {vol}<br>"
            f"Total Value: ${tval:,.0f}<br>"
            f"Exp Breakdown:<br>{entry['hover_exp']}"
        )
        fig.add_trace(
            go.Scatter(
                x=[mid_time],
                y=[(y0 + y1) / 2],
                mode="markers",
                marker=dict(size=10, color=color, opacity=0),
                hovertemplate=hover_text
            )
        )
        # Optionally, adjust day_offset to avoid clumping (uncomment if needed)
        # day_offset += 0.7

    # Add invisible traces for the legend
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

    # Layout settings
    fig.update_layout(
        title=f"{symbol.upper()} - Aggregate Volume Across Expirations",
        xaxis_title="Date",
        yaxis_title="Price (Strike)",
        xaxis_rangeslider_visible=False,
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)
