from src.pricers import OptionSpec, crr_price, jr_price, tian93_price, build_binomial_tree, lr_price, tian99_price
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

def test_american_call_without_dividend_matches_european():
    european = OptionSpec(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Call",
        exercise_style="European",
        model="CRR",
        dividend_yield=0.0,
    )

    american = OptionSpec(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Call",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.0,
    )

    assert crr_price(american) == pytest.approx(
        crr_price(european),
        abs=1e-10,
    )

def test_american_call_with_high_dividend_exceeds_european():
    european = OptionSpec(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Call",
        exercise_style="European",
        model="CRR",
        dividend_yield=0.15,
    )

    american = OptionSpec(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Call",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.15,
    )

    european_price = crr_price(european)
    american_price = crr_price(american)

    assert american_price >= european_price
    assert american_price > european_price + 1.0
    assert american_price >= american.spot - american.strike

def test_american_call_decreases_when_dividend_yield_increases():
    without_dividend = OptionSpec(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Call",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.0,
    )

    with_dividend = OptionSpec(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Call",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.15,
    )

    assert crr_price(with_dividend) < crr_price(without_dividend)

@pytest.mark.parametrize(
    ("model", "pricer"),
    [
        ("CRR", crr_price),
        ("Jarrow-Rudd", jr_price),
        ("Tian-1993", tian93_price),
    ],
)
def test_american_call_reacts_to_dividend_yield(model, pricer):
    common = dict(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Call",
        exercise_style="American",
        model=model,
    )

    no_dividend = OptionSpec(
        **common,
        dividend_yield=0.0,
    )

    high_dividend = OptionSpec(
        **common,
        dividend_yield=0.15,
    )

    assert pricer(high_dividend) < pricer(no_dividend)

@pytest.mark.parametrize(
    ("model", "pricer"),
    [
        ("CRR", crr_price),
        ("Jarrow-Rudd", jr_price),
        ("Tian-1993", tian93_price),
    ],
)
def test_american_put_reacts_to_dividend_yield(model, pricer):
    common = dict(
        spot=150.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.20,
        steps=1_000,
        option_type="Put",
        exercise_style="American",
        model=model,
    )

    no_dividend = OptionSpec(
        **common,
        dividend_yield=0.0,
    )

    high_dividend = OptionSpec(
        **common,
        dividend_yield=0.15,
    )

    assert pricer(high_dividend) > pricer(no_dividend)

@pytest.mark.parametrize(
    ("model", "pricer"),
    [
        ("CRR", crr_price),
        ("Jarrow-Rudd", jr_price),
        ("Tian-1993", tian93_price),
    ],
)
def test_american_put_is_not_worth_less_than_european_put(
    model,
    pricer,
):
    common = dict(
        spot=80.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        steps=1_000,
        option_type="Put",
        model=model,
        dividend_yield=0.0,
    )

    european = OptionSpec(
        **common,
        exercise_style="European",
    )

    american = OptionSpec(
        **common,
        exercise_style="American",
    )

    european_price = pricer(european)
    american_price = pricer(american)

    assert american_price >= european_price

    
def test_binomial_models_are_consistent():
    spec_crr = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=51,
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
        steps=51,
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
        steps=51,
        option_type="Put",
        exercise_style="American",
        model="Tian-1993",
        dividend_yield=0.0,
    )

    spec_lr = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=51,
        option_type="Put",
        exercise_style="American",
        model="Leisen-Reimer",
        dividend_yield=0.0,
    )

    spec_tian99 = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=51,
        option_type="Put",
        exercise_style="American",
        model="Tian-1999",
        dividend_yield=0.0,
    )

    crr = crr_price(spec_crr)
    jr = jr_price(spec_jr)
    tian93 = tian93_price(spec_tian93)
    lr = lr_price(spec_lr)
    tian99 = tian99_price(spec_tian99)

    assert abs(crr - jr) < 0.05
    assert abs(crr - tian93) < 0.05
    assert abs(crr - lr) < 0.05
    assert abs(crr - tian99) < 0.05

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
