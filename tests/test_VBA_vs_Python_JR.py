from src.pricers import OptionSpec, jr_price

def test_jr_regression_case_1():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=150,
        option_type="call",
        exercise_style="american",
        model="jarrow-rudd",
        dividend_yield=0.0,
    )
    price = jr_price(spec)
    print(price)
    expected = 10.4454568011477 # VBA value
    assert abs(price - expected) < 1e-6