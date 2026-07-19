from src.pricers import OptionSpec, crr_price, jr_price

def test_option_spec_creation():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.2,
        steps=10,
        option_type="Call",
        exercise_style="European",
        model="CRR",
    )
    assert spec.spot == 100.0
    assert spec.model == "CRR"


def test_crr_price_is_positive_for_atm_call():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Call",
        exercise_style="European",
        model="CRR",
        dividend_yield=0.0,
    )
    price = crr_price(spec)
    assert price > 0.0


def test_american_put_is_not_cheaper_than_european_put():
    european_spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Put",
        exercise_style="European",
        model="CRR",
        dividend_yield=0.0,
    )

    american_spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Put",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.0,
    )

    european_price = crr_price(european_spec)
    american_price = crr_price(american_spec)

    assert american_price >= european_price

def test_option_jr_spec_creation():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.2,
        steps=10,
        option_type="Call",
        exercise_style="European",
        model="Jarrow-Rudd",
    )
    assert spec.spot == 100.0
    assert spec.model == "Jarrow-Rudd"

def test_jr_price_is_positive_for_atm_call():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Call",
        exercise_style="European",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )
    price = jr_price(spec)
    assert price > 0.0

def test_jr_american_put_is_not_cheaper_than_european_put():
    european_spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Put",
        exercise_style="European",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )

    american_spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Put",
        exercise_style="American",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )
    european_price = jr_price(european_spec)
    american_price = jr_price(american_spec)

    assert american_price >= european_price