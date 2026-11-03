from src.pricers import Trinomial_Kamrad_Ritchken_price, Trinomial_Kamrad_Ritchken_tree, OptionSpec, KRConfig, lr_price
from src.visuals import render_trinomial_tree, add_spec_box
import matplotlib.pyplot as plt

spec = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    steps=11,
    option_type="Put",
    exercise_style="American",
    dividend_yield=0.12,
)

config = KRConfig()

price = Trinomial_Kamrad_Ritchken_price(spec, config)
result = Trinomial_Kamrad_Ritchken_tree(spec, config)
compare = lr_price(spec)
print(price)
print(compare)

fig, ax = render_trinomial_tree(
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
    title="Trinomial American Put",
)

add_spec_box(
    ax=ax,
    spec=spec,
    model_name="Trinomial",
    u=result.u,
    m=result.m,
    d=result.d,
    p_u=result.p_u,
    p_m=result.p_m,
    p_d=result.p_d,
    fontsize=9,
    x=-0.01,
    y=0.98,
)

plt.show()
