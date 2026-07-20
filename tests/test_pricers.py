from src.pricers import OptionSpec, crr_price, jr_price, tian93_price, build_binomial_tree
import pytest

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

def test_option_tian93_spec_creation():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.2,
        steps=10,
        option_type="Call",
        exercise_style="European",
        model="Tian-1993",
    )
    assert spec.spot == 100.0
    assert spec.model == "Tian-1993"

def test_tian93_price_is_positive_for_atm_call():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Call",
        exercise_style="European",
        model="Tian-1993",
        dividend_yield=0.0,
    )
    price = jr_price(spec)
    assert price > 0.0

def test_tian93_american_put_is_not_cheaper_than_european_put():
    european_spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Put",
        exercise_style="European",
        model="Tian-1993",
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
        model="Tian-1993",
        dividend_yield=0.0,
    )
    european_price = tian93_price(european_spec)
    american_price = tian93_price(american_spec)

    assert american_price >= european_price

def test_binomial_models_are_consistent():
    spec_crr = OptionSpec(
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
    spec_jr = OptionSpec(
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
    spec_tian93 = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=50,
        option_type="Put",
        exercise_style="American",
        model="Tian-1993",
        dividend_yield=0.0,
    )

    crr = crr_price(spec_crr)
    jr = jr_price(spec_jr)
    tian93 = tian93_price(spec_tian93)

    assert abs(crr - jr) < 0.05
    assert abs(crr - tian93) < 0.05

def test_build_binomial_tree_dispatches_crr():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=5,
        option_type="Put",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.0,
    )

    result = build_binomial_tree(spec)

    assert result.option_tree.shape == (6, 6)

def test_build_binomial_tree_dispatches_jr():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=5,
        option_type="Put",
        exercise_style="American",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )

    result = build_binomial_tree(spec)

    assert result.option_tree.shape == (6, 6)

def test_build_binomial_tree_dispatches_tian93():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=5,
        option_type="Put",
        exercise_style="American",
        model="Tian-1993",
        dividend_yield=0.0,
    )

    result = build_binomial_tree(spec)

    assert result.option_tree.shape == (6, 6)


def test_build_binomial_tree_rejects_unknown_model():
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=5,
        option_type="Call",
        exercise_style="European",
        model="unknown",
        dividend_yield=0.0,
    )
    with pytest.raises(ValueError, match="Unsupported binomial model"):
        build_binomial_tree(spec)