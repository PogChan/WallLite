import requests
import pandas as pd
import streamlit as st
import calendar
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pytz
import yfinance as yf
from main import *


def get_next_fridays(n=10, startDate = datetime.now()):
    """Get the next `num_fridays` Fridays starting from today."""
    fridays = []
    # find el next fridiossss
    days_until_next_friday = (4 - startDate.weekday() + 7) % 7
    next_friday = startDate + timedelta(days=days_until_next_friday)

    for i in range(n):
        fridays.append(next_friday + timedelta(weeks=i))

    return [friday.strftime('%Y-%m-%d') for friday in fridays]


def get_next_opex():
    now = datetime.now()

    # Get the third Friday of this month
    first_day_this_month = datetime(now.year, now.month, 1)
    third_friday_this_month = get_next_fridays(3, first_day_this_month)[-1]

    # If today is before this month's third Friday, return it
    if now.strftime("%Y-%m-%d") < third_friday_this_month:
        return third_friday_this_month

    # Otherwise, return next month's third Friday
    next_month = now.month + 1 if now.month < 12 else 1
    year = now.year if now.month < 12 else now.year + 1
    first_day_next_month = datetime(year, next_month, 1)

    third_friday_next_month = get_next_fridays(3, first_day_next_month)[-1]
    return third_friday_next_month


def getHistoricalOHLC(symbol, period ='60d'):
    # Create a Ticker object
    ticker = yf.Ticker(symbol)
    # Fetch historical data for the last 60 days
    df = ticker.history(period=period, interval="1d")
    # Select only required columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df

# ---------------------------------------------------------------------------
# Function: Plot Chart with Options OI/Volume
# ---------------------------------------------------------------------------
def plotChartOI(symbol, data, exp_date, top_n=5):
    # Download 1 month of data from Alpha Vantage
    df = getHistoricalOHLC(symbol)

    if df.empty:
        st.warning(f"No price data for {symbol}.")
        return

    # (The rest of your options-chain parsing code remains unchanged)
    if exp_date not in data.get("options", {}):
        st.warning(f"No options data found for {exp_date}.")
        return

    calls_dict = data["options"][exp_date].get("c", {})
    puts_dict  = data["options"][exp_date].get("p", {})

    def parse_chain(chain, opt_type):
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

    # Sort & pick top_n by OI and Volume
    top_calls_oi     = sorted(calls, key=lambda x: x["oi"],     reverse=True)[:top_n]
    top_calls_volume = sorted(calls, key=lambda x: x["volume"], reverse=True)[:top_n]
    top_puts_oi      = sorted(puts,  key=lambda x: x["oi"],     reverse=True)[:top_n]
    top_puts_volume  = sorted(puts,  key=lambda x: x["volume"], reverse=True)[:top_n]

    lines = []
    for row in top_calls_oi:
        lines.append({**row, "metric": "oi"})
    for row in top_puts_oi:
        lines.append({**row, "metric": "oi"})
    for row in top_calls_volume:
        lines.append({**row, "metric": "volume"})
    for row in top_puts_volume:
        lines.append({**row, "metric": "volume"})

    if not lines:
        st.warning("No OI/Volume data found.")
        return

    display_choice = st.selectbox(
        "Show Which Bars?",
        ["Both Calls & Puts", "Calls Only", "Puts Only"],
        key=symbol
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

    def get_value(row):
        return row["oi"] if row["metric"] == "oi" else row["volume"]
    filtered_lines = sorted(filtered_lines, key=get_value, reverse=True)

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

    all_vals = [2000]
    for r in lines:
        all_vals.append(r["oi"] if r["metric"] == "oi" else r["volume"])
    min_val = min(all_vals) if all_vals else 0
    max_val = max(all_vals) if all_vals else 1

    def unify_normalize(v):
        if max_val == min_val:
            return 1
        return (v - min_val) / (max_val - min_val)

    offset_map = {
       ("call","oi"):     0.03,
       ("put","oi"):     -0.03,
       ("call","volume"): 0.05,
       ("put","volume"): -0.05
    }

    day_offset = 0.0
    for entry in filtered_lines:
        typ    = entry["type"]
        strike = entry["strike"]
        oi     = entry["oi"]
        vol    = entry["volume"]
        tval   = entry["totalValue"]
        metric = entry["metric"]

        if   (typ == "call" and metric=="oi"):       color = "green"
        elif (typ == "put"  and metric=="oi"):         color = "red"
        elif (typ == "call" and metric=="volume"):     color = "orange"
        else:                                          color = "blue"

        raw_value = oi if metric == "oi" else vol
        scale = unify_normalize(raw_value)
        bar_length_days = scale * (0.5 * total_days)
        bar_length_days = min(bar_length_days, 0.9 * total_days)

        x1 = max_date - timedelta(days=day_offset)
        x0 = x1 - timedelta(days=bar_length_days)
        y_offset = offset_map.get((typ, metric), 0.0)
        y0 = strike + y_offset
        y1 = strike + y_offset

        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color=color, width=6),
            opacity=0.7
        )

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
        # Optionally adjust day_offset if needed:
        # day_offset += 0.7

    # Invisible traces for the legend
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="green", width=6),
                             name="Call OI"))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="red", width=6),
                             name="Put OI"))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="orange", width=6),
                             name="Call Volume"))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="blue", width=6),
                             name="Put Volume"))

    fig.update_layout(
        title=f"{symbol.upper()} — {exp_date}<br>Top {top_n} OI & Volume (Calls/Puts)",
        xaxis_title="Date",
        yaxis_title="Price (Strike)",
        xaxis_rangeslider_visible=False,
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Function: Options Volume Check (Aggregate across expirations)
# ---------------------------------------------------------------------------
def pc_check(symbol, data, top_n=5):
    if symbol == "SPX":
        symbol ="^SPX"
    df = getHistoricalOHLC(symbol)
    if df.empty:
        st.warning(f"No price data for {symbol}.")
        return

    if "options" not in data:
        st.warning("No options data found.")
        return

    aggregated = {}
    # (Assuming today_date and now are defined elsewhere or can be defined as needed)


    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    today_date = now.strftime("%Y-%m-%d")

    for exp_date, exp_data in data["options"].items():
        if exp_date == today_date and now.hour >= 16:
            continue
        #if 2025-03-21W > 2025-03-21 then we remove the W and is it equal then we skip it. if its not then we go next.
        if exp_date > get_next_opex() and exp_data.replace('W', '') > get_next_opex():
            break


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

    aggregated_list = []
    for (option_type, strike), values in aggregated.items():
        aggregated_list.append({
            "type": option_type,
            "strike": strike,
            "volume": values["volume"],
            "totalValue": values["totalValue"],
            "exp_breakdown": values["exp_breakdown"]
        })

    calls_agg = [item for item in aggregated_list if item["type"] == "call"]
    puts_agg  = [item for item in aggregated_list if item["type"] == "put"]

    top_calls_volume = sorted(calls_agg, key=lambda x: x["volume"], reverse=True)[:top_n]
    top_puts_volume  = sorted(puts_agg, key=lambda x: x["volume"], reverse=True)[:top_n]

    lines = []
    for row in top_calls_volume:
        sorted_exp = sorted(row["exp_breakdown"].items(), key=lambda x: x[1], reverse=True)[:top_n]
        exp_info = "<br>".join([f"{exp}: {vol}" for exp, vol in sorted_exp])
        row_copy = row.copy()
        row_copy["metric"] = "volume"
        row_copy["hover_exp"] = exp_info
        lines.append(row_copy)
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

    filtered_lines = sorted(filtered_lines, key=lambda row: row["volume"], reverse=True)

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

    all_vals = [2000]
    for r in filtered_lines:
        all_vals.append(r["volume"])
    min_val = min(all_vals)
    max_val = max(all_vals)

    def unify_normalize(v):
        if max_val == min_val:
            return 1
        return (v - min_val) / (max_val - min_val)

    offset_map = {
        ("call", "volume"): 0.05,
        ("put", "volume"): -0.05
    }

    day_offset = 0.0
    for entry in filtered_lines:
        typ = entry["type"]
        strike = entry["strike"]
        vol = entry["volume"]
        tval = entry["totalValue"]
        metric = entry["metric"]

        color = "orange" if typ == "call" else "blue"
        raw_value = vol
        scale = unify_normalize(raw_value)
        bar_length_days = scale * (0.5 * total_days)
        bar_length_days = min(bar_length_days, 0.9 * total_days)

        x1 = max_date - timedelta(days=day_offset)
        x0 = x1 - timedelta(days=bar_length_days)
        y_offset = offset_map.get((typ, metric), 0.0)
        y0 = strike + y_offset
        y1 = strike + y_offset

        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color=color, width=6),
            opacity=0.7
        )

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
        # Optionally, adjust day_offset if needed:
        # day_offset += 0.7

    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="orange", width=6),
                             name="Call Volume"))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="blue", width=6),
                             name="Put Volume"))

    fig.update_layout(
        title=f"{symbol.upper()} - Aggregate Volume Across Expirations",
        xaxis_title="Date",
        yaxis_title="Price (Strike)",
        xaxis_rangeslider_visible=False,
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Function: Plot Aggregated OI & Volume Across Expirations (using Alpha Vantage)
# ---------------------------------------------------------------------------
def plotAggregateOI(symbol, data, top_n=5, default_expiration=None):
    if default_expiration is None:
        default_expiration = get_next_opex()

    df = getHistoricalOHLC(symbol)
    if df.empty:
        st.warning(f"No price data for {symbol}.")
        return

    try:
        default_exp_date = datetime.strptime(default_expiration, "%Y-%m-%d").date()
    except Exception as e:
        st.warning(f"Invalid default expiration date: {default_expiration}")
        return

    aggregated = {}


    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    today_date = now.strftime("%Y-%m-%d")

    for exp_str, exp_data in data.get("options", {}).items():


        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except Exception:
            continue

        if exp_date == today_date and now.hour >= 15:
            continue
        if exp_date > default_exp_date:
            break

        for opt_key, opt_type in [("c", "call"), ("p", "put")]:
            chain = exp_data.get(opt_key, {})
            for strike_str, info in chain.items():
                if not all(k in info for k in ("oi", "v", "b", "a")):
                    continue
                try:
                    strike = float(strike_str)
                except Exception:
                    continue

                oi = info.get("oi", 0)
                vol = info.get("v", 0)
                bid = info.get("b", 0)
                ask = info.get("a", 0)
                if oi <= 0:
                    continue
                mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                total_val = oi * mid_price * 100

                key = (opt_type, strike)
                if key not in aggregated:
                    aggregated[key] = {"oi": 0, "volume": 0, "totalValue": 0, "breakdown": {}}
                aggregated[key]["oi"] += oi
                aggregated[key]["volume"] += vol
                aggregated[key]["totalValue"] += total_val
                if exp_str not in aggregated[key]["breakdown"]:
                    aggregated[key]["breakdown"][exp_str] = {"oi": 0, "volume": 0}
                aggregated[key]["breakdown"][exp_str]["oi"] += oi
                aggregated[key]["breakdown"][exp_str]["volume"] += vol

    aggregated_list = []
    for (opt_type, strike), values in aggregated.items():
        aggregated_list.append({
            "type": opt_type,
            "strike": strike,
            "oi": values["oi"],
            "volume": values["volume"],
            "totalValue": values["totalValue"],
            "breakdown": values["breakdown"]
        })

    top_calls_oi     = sorted([d for d in aggregated_list if d["type"] == "call"], key=lambda x: x["oi"], reverse=True)[:top_n]
    top_calls_volume = sorted([d for d in aggregated_list if d["type"] == "call"], key=lambda x: x["volume"], reverse=True)[:top_n]
    top_puts_oi      = sorted([d for d in aggregated_list if d["type"] == "put"],  key=lambda x: x["oi"], reverse=True)[:top_n]
    top_puts_volume  = sorted([d for d in aggregated_list if d["type"] == "put"],  key=lambda x: x["volume"], reverse=True)[:top_n]

    def build_hover_breakdown(breakdown):
        lines = []
        for exp, vals in sorted(breakdown.items()):
            lines.append(f"{exp}: OI={vals['oi']}, Vol={vals['volume']}")
        return "<br>".join(lines)

    lines = []
    for row in top_calls_oi:
        row_copy = row.copy()
        row_copy["metric"] = "oi"
        row_copy["hover_breakdown"] = build_hover_breakdown(row["breakdown"])
        lines.append(row_copy)
    for row in top_puts_oi:
        row_copy = row.copy()
        row_copy["metric"] = "oi"
        row_copy["hover_breakdown"] = build_hover_breakdown(row["breakdown"])
        lines.append(row_copy)
    for row in top_calls_volume:
        row_copy = row.copy()
        row_copy["metric"] = "volume"
        row_copy["hover_breakdown"] = build_hover_breakdown(row["breakdown"])
        lines.append(row_copy)
    for row in top_puts_volume:
        row_copy = row.copy()
        row_copy["metric"] = "volume"
        row_copy["hover_breakdown"] = build_hover_breakdown(row["breakdown"])
        lines.append(row_copy)

    if not lines:
        st.warning("No aggregated OI/Volume data found.")
        return

    display_choice = st.selectbox(
        "Show Which Bars?",
        ["Both Calls & Puts", "Calls Only", "Puts Only"],
        key=symbol + "_aggregate"
    )
    filtered_lines = []
    for line in lines:
        if display_choice == "Calls Only" and line["type"] != "call":
            continue
        elif display_choice == "Puts Only" and line["type"] != "put":
            continue
        filtered_lines.append(line)

    if not filtered_lines:
        st.warning(f"No {display_choice} data found.")
        return

    def get_value(row):
        return row["oi"] if row["metric"] == "oi" else row["volume"]
    filtered_lines = sorted(filtered_lines, key=get_value, reverse=True)

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

    all_vals = [2000]
    for r in filtered_lines:
        all_vals.append(r["oi"] if r["metric"] == "oi" else r["volume"])
    min_val = min(all_vals) if all_vals else 0
    max_val = max(all_vals) if all_vals else 1
    def unify_normalize(v):
        if max_val == min_val:
            return 1
        return (v - min_val) / (max_val - min_val)

    offset_map = {
       ("call","oi"):     0.03,
       ("put","oi"):     -0.03,
       ("call","volume"): 0.05,
       ("put","volume"): -0.05
    }

    day_offset = 0.0
    for entry in filtered_lines:
        typ    = entry["type"]
        strike = entry["strike"]
        oi     = entry["oi"]
        vol    = entry["volume"]
        total_val = entry["totalValue"]
        metric = entry["metric"]

        if   (typ == "call" and metric=="oi"):
            color = "green"
        elif (typ == "put"  and metric=="oi"):
            color = "red"
        elif (typ == "call" and metric=="volume"):
            color = "orange"
        else:
            color = "blue"

        raw_value = oi if metric == "oi" else vol
        scale = unify_normalize(raw_value)
        bar_length_days = scale * (0.5 * total_days)
        bar_length_days = min(bar_length_days, 0.9 * total_days)

        x1 = max_date - timedelta(days=day_offset)
        x0 = x1 - timedelta(days=bar_length_days)
        y_offset = offset_map.get((typ, metric), 0.0)
        y0 = strike + y_offset
        y1 = strike + y_offset

        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color=color, width=6),
            opacity=0.7
        )

        mid_time = x0 + (x1 - x0) / 2
        hover_text = (
            f"<b>{typ.upper()} {metric.upper()}</b><br>"
            f"Strike: {strike}<br>"
            f"Aggregated OI: {oi}<br>"
            f"Aggregated Volume: {vol}<br>"
            f"Total Premium: ${total_val:,.0f}<br>"
            f"Breakdown:<br>{entry['hover_breakdown']}"
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
        # Optionally adjust day_offset if needed:
        # day_offset += 0.7

    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="green", width=6),
                             name="Call OI"))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="red", width=6),
                             name="Put OI"))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="orange", width=6),
                             name="Call Volume"))
    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode="lines", line=dict(color="blue", width=6),
                             name="Put Volume"))

    fig.update_layout(
        title=f"{symbol.upper()} - Aggregated OI & Volume (Expirations ≤ {default_expiration})",
        xaxis_title="Date",
        yaxis_title="Price (Strike)",
        xaxis_rangeslider_visible=False,
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)
