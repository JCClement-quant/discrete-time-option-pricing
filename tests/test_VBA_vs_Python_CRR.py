from src.pricers import OptionSpec, crr_price


def test_crr_regression_case_1():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=150,
        option_type="put",
        exercise_style="american",
        model="crr",
        dividend_yield=0.0,
    )
    price = crr_price(spec)
    print(price)
    expected = 6.085043264 # VBA value
    assert abs(price - expected) < 1e-6