from __future__ import annotations

from unittest import case

import matplotlib.pyplot as plt
import streamlit as st

from src.pricers import OptionSpec, build_binomial_tree
from src.visuals import add_spec_box, render_binomial_tree

st.set_page_config(
    page_title = "Tree Explorer",
    layout = "wide",
)

st.title("Discrete-Time Option Pricing")
st.markdown("### Tree exploration")

with st.sidebar:
    st.header("Contract and market inputs")

    model_label = st.selectbox(
        "Model",
        options=["CRR", "Jarrow-Rudd", "Tian-1993", "Leisen-Reimer", "Tian-1999"],
    )

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

    if model_label == "Leisen-Reimer":
        steps = st.slider(
            "Tree steps",
            min_value=1,
            max_value=19,
            value=5,
            step=2,
        )
    else:
        steps = st.slider(
            "Tree steps",
            min_value=1,
            max_value=20,
            value=5,
        )

    display_mode = st.selectbox(
        "Node display",
        options=["combined", "spot", "option"],
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
    model=model_label,
    dividend_yield=dividend_yield,
)

try:
    result = build_binomial_tree(spec)

except ValueError as exc:
    st.error(str(exc))
    st.stop()


option_price = float(result.option_tree[0, 0])

metric_col, parameter_col = st.columns([1, 3])

with metric_col:
    st.metric(
        label="Option price",
        value=f"{option_price:.6f}",
    )

with parameter_col:
    if spec.model == "Tian-1999":
        st.write(
            {
                "u": result.u,
                "d": result.d,
                "p": result.p,
                "lambda": result.the_lambda,
            }
        )
    else:
        st.write(
            {
                "u": result.u,
                "d": result.d,
                "p": result.p,
            }
        )


fig, ax = render_binomial_tree(
    spec=spec,
    spot_tree=result.spot_tree,
    option_tree=result.option_tree,
    exercise_tree=result.exercise_tree,
    mode=display_mode,
    decimals=3,
    fontsize=7,
    node_width=1.55,
    node_height=0.95,
    x_spacing=1.8,
    y_spacing=1.25,
    title=(
        f"{model_label} · "
        f"{exercise_style.capitalize()} "
        f"{option_type.capitalize()}"
    ),
)

add_spec_box(
    ax=ax,
    spec=spec,
    model_name=model_label,
    u=result.u,
    d=result.d,
    p=result.p,
    x=-0.12,
    y=0.98,
    fontsize=9,
)

st.pyplot(fig, width="stretch")
plt.close(fig)