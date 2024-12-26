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
       ("call","volume"): 0.01,
       ("put","oi"):     -0.01,
       ("put","volume"): -0.03
    }

    # anchro each bar near the right side (max_date),
    # then extend left by bar_length_days, clamping at 90% of chart width.
    day_offset = 0.0

    for entry in lines:
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
