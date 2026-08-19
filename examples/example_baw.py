from src.pricers import BAW_call
from src.pricers import OptionSpec

spec_crr_call = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    steps=1001,
    option_type="Call",
    exercise_style="American",
    dividend_yield=0.1,
)

BAW_price = BAW_call(spec_crr_call,100)
print(BAW_price)
