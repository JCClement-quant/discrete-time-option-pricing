from src.pricers import BAW_quadratic
from src.pricers import OptionSpec
import math

def test_baw_american_call():
    spec_crr_call = OptionSpec(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            steps=1000,
            option_type="Call",
            exercise_style="American",
            dividend_yield=0.1,
    )

    q1, q2 = BAW_quadratic(spec_crr_call)

    assert math.isfinite(q1)
    assert math.isfinite(q2)

    assert q1 < 0.0
    assert q2 > 0.0
    assert q1 < q2