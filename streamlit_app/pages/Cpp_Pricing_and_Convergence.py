from __future__ import annotations

import os
import streamlit as st

from src.pricers import OptionSpec, BMS
from linker_cpp import discrete_models

st.set_page_config(
    page_title = "C++ Pricing",
    layout = "wide",
)

st.title("High-Performance C++ Pricing")
st.markdown("### Convergence speed and metrics")

with st.sidebar:
    st.header("Contract and market inputs")

    option_type = st.selectbox(
        "Option type",
        options=["Call", "Put"],
    )

    exercise_style = st.selectbox(
        "Exercise style",
        options=["European", "American"],
    )

    spot = st.number_input(
        "Spot",
        min_value=0.01,
        value=100.0,
        step=0.01,
    )

    strike = st.number_input(
        "Strike",
        min_value=0.01,
        value=100.0,
        step=0.01,
    )

    maturity = st.number_input(
        "Maturity",
        min_value=0.01,
        value=1.0,
        step=0.01,
    )

    rate = st.number_input(
        "Risk-free rate",
        value=0.05,
        step=0.005,
        format="%.4f",
    )

    dividend_yield = st.number_input(
        "Dividend yield",
        value=0.0,
        step=0.005,
        format="%.4f",
    )

    volatility = st.number_input(
        "Volatility",
        min_value=0.0001,
        value=0.20,
        step=0.01,
        format="%.4f",
    )

    steps = st.slider(
        "Tree steps",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=1000
    )

spec = OptionSpec(
    spot=spot,
    strike=strike,
    maturity=maturity,
    rate=rate,
    volatility=volatility,
    steps=steps,
    option_type=option_type,
    exercise_style=exercise_style,
    dividend_yield=dividend_yield,
)

if option_type == "Call":
    payoff = discrete_models.PayoffCall(strike)
else:
    payoff = discrete_models.PayoffPut(strike)

if exercise_style == "European":
    cpp_exercise_style = discrete_models.ExerciseStyle.European
else:
    cpp_exercise_style = discrete_models.ExerciseStyle.American

option = discrete_models.VanillaOption(payoff,
        maturity,
        cpp_exercise_style
)

market = discrete_models.MarketData(spot,
        rate,
        dividend_yield,
        volatility
)

config = discrete_models.TreeConfig(
    int(steps)
)

config_shifted = discrete_models.TreeConfig(
    int(steps) + 1
)

crr_price = discrete_models.CRR_cpp_price(
    option,
    market,
    config,
)

crr_price_shifted =discrete_models.CRR_cpp_price(
    option,
    market,
    config_shifted,
)

jr_price = discrete_models.JR_cpp_price(
    option,
    market,
    config,
)

tian_price = discrete_models.TIAN93_cpp_price(
    option,
    market,
    config,
)

if option_type == "Call":
    bms_price = BMS(spec)[0]
else: bms_price = BMS(spec)[1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "CRR averaged",
        f"{(crr_price+crr_price_shifted)/2:.8f}",
    )

with col2:
    st.metric(
        "Jarrow-Rudd",
        f"{jr_price:.8f}",
    )

with col3:
    st.metric(
        "Tian 1993",
        f"{tian_price:.8f}",
    )

with col4:
    st.metric(
        "BMS",
        f"{bms_price:.8f}",
 )