from __future__ import annotations

import os
import streamlit as st

from src.pricers import OptionSpec, BMS, BAW_call, BAW_put, JZ_Pricer
from linker_cpp import discrete_models
from src.visuals import (plot_price_convergence, plot_absolute_error, plot_relative_error,
                         plot_signed_error, plot_convergence_performance, plot_crr_smoothing)
import matplotlib.pyplot as plt
from time import perf_counter

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
        min_value=0.00,
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
        min_value=1001,
        max_value=50001,
        value=5001,
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

krconfig = discrete_models.KRConfig()

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

lr_price = discrete_models.LR_cpp_price(
    option,
    market,
    config,
)

tian99_price = discrete_models.TIAN99_cpp_price(
    option,
    market,
    config,
)

kr_price = discrete_models.KR_cpp_price(
    option,
    market,
    config,

)

benchmarks = {}

t_graph = 0 #Diag global variable
if exercise_style == "European":
    if option_type == "Call":
        bms_price = BMS(spec)[0]
    else: bms_price = BMS(spec)[1]
    benchmarks["BMS"] = bms_price
    benchmark = bms_price
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

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
            "Leisen-Reimer",
            f"{lr_price:.8f}",
        )

    with col5:
        st.metric(
            "Tian 1999",
            f"{tian99_price:.8f}",
        )

    with col6:
        st.metric(
            "Kramrad-Ritchken",
            f"{kr_price:.8f}",
        )

    with col7:
        st.metric(
            "BMS",
            f"{bms_price:.8f}",
     )
    steps_grid = [5, 11, 15, 25, 51, 101, 251, 1001, 2501, 5001, 10001]

    results = discrete_models.Binomial_Convergence(
        option,
        market,
        steps_grid,
    )

    results_trino = discrete_models.Trinomial_Convergence(
        option,
        market,
        steps_grid,
        krconfig,
    )

    t_graph = perf_counter()
    fig_price = plot_price_convergence(
        results,
        results_trino,
        benchmarks=benchmarks,
    )

    fig_abs = plot_absolute_error(
        results,
        benchmark,
        results_trino,
    )

    fig_rel = plot_relative_error(
        results,
        benchmark,
        results_trino,
    )

    plot_build_time = perf_counter() - t_graph

    st.pyplot(fig_price, width='content')
    st.pyplot(fig_abs, width='content')
    st.pyplot(fig_rel, width='content')

    plt.close(fig_price)
    plt.close(fig_abs)
    plt.close(fig_rel)

else:
    if option_type == "Call":
        baw_price = BAW_call(spec, spot)
        jz_price = JZ_Pricer(spec, spot, 1)
        benchmarks["BAW"] = baw_price
        benchmarks["Ju-Zhong"] = jz_price
    else:
        baw_price = BAW_put(spec, spot)
        jz_price = JZ_Pricer(spec, spot, -1)
        benchmarks["BAW"] = baw_price
        benchmarks["Ju-Zhong"] = jz_price

    if (option_type == "Call") and (spec.dividend_yield <= 0.00):
        col1, col2, col3, col4,  = st.columns(4)

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

    with col1:
        st.metric(
            "CRR averaged",
            f"{(crr_price + crr_price_shifted) / 2:.8f}",
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
            "Leisen-Reimer",
            f"{lr_price:.8f}",
        )

    with col5:
        st.metric(
            "Tian 1999",
            f"{tian99_price:.8f}",
        )

    with col6:
        st.metric(
            "Kramrad-Ritchken",
            f"{kr_price:.8f}",
        )

    if not (option_type == "Call" and spec.dividend_yield == 0.00):
        with col7:
            st.metric(
                "BAW",
                f"{baw_price:.8f}",
            )

        with col8:
            st.metric(
                "JZ",
                f"{jz_price:.8f}",
            )

    steps_grid = [5, 11, 15, 25, 51, 101, 251, 1001, 2501, 5001, 10001]

    results = discrete_models.Binomial_Convergence(
        option,
        market,
        steps_grid,
    )

    results_trino = discrete_models.Trinomial_Convergence(
        option,
        market,
        steps_grid,
        krconfig,
    )

    t_graph = perf_counter()
    fig_price = plot_price_convergence(
        results,
        results_trino,
        benchmarks=benchmarks,
    )
    plot_build_time = perf_counter() - t_graph

    st.pyplot(fig_price, width='content')
    plt.close(fig_price)

t0 = perf_counter()
results = discrete_models.Binomial_Convergence(
    option,
    market,
    steps_grid,
)
threaded_time = perf_counter() - t0

t0 = perf_counter()
results_unthreaded = discrete_models.Binomial_Convergence_Unthreaded(
    option,
    market,
    steps_grid,
)
unthreaded_time = perf_counter() - t0
speedup = unthreaded_time / threaded_time

with st.expander("Advanced diagnostics"):

    if exercise_style == "European":
        fig_signed = plot_signed_error(
            results,
            benchmark,
            results_trino
        )

        st.pyplot(fig_signed, width='content')
        plt.close(fig_signed)

        fig_crr = plot_crr_smoothing(
            results,
            benchmark,
        )

        st.pyplot(fig_crr)
        plt.close(fig_crr)

    col3, col4 = st.columns(2)

    col3.metric(
        "Speedup",
        f"{speedup:.2f}×",
    )

    col4.metric(
        "Plot generation",
        f"{plot_build_time * 1000:.0f} ms",
    )

    fig_perf = plot_convergence_performance(
        threaded_time,
        unthreaded_time,
    )

    st.pyplot(fig_perf)
    plt.close(fig_perf)