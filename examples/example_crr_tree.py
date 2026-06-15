import matplotlib.pyplot as plt
from src.visuals import render_binomial_tree, add_spec_box
from src.pricers import crr_tree, OptionSpec

spec = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    steps=12,
    option_type="put",
    exercise_style="american",
    dividend_yield=0.0,
)

result = crr_tree(spec)

fig, ax = render_binomial_tree(
    spec=spec,
    spot_tree=result.spot_tree,
    option_tree=result.option_tree,
    exercise_tree=result.exercise_tree,
    mode="combined",
    decimals=3,
    fontsize=7,
    node_width=1.55,
    node_height=0.95,
    x_spacing=1.8,
    y_spacing=1.25,
    title="CRR American Put",
)

add_spec_box(
    ax=ax,
    spec=spec,
    model_name="CRR",
    u=result.u,
    d=result.d,
    p=result.p,
    fontsize=9,
    x=-0.50,
    y=0.98,
)

plt.show()