import numpy as np
from scipy.stats import norm
from src.pricers import OptionSpec, crr_price, jr_price, tian93_price
import pytest

N = 2000
@pytest.mark.slow

def BMS(spec: OptionSpec) -> tuple[float, float]:
    sigma_sqrt_t = spec.volatility * np.sqrt(spec.maturity)

    d1 = (
         np.log(spec.spot / spec.strike)
        + (
        spec.rate
        - spec.dividend_yield
        + 0.5 * spec.volatility ** 2
         ) * spec.maturity
     ) / sigma_sqrt_t

    d2 = d1 - sigma_sqrt_t

    call_bms = (
        norm.cdf(d1)
        * spec.spot
        * np.exp(-spec.dividend_yield * spec.maturity)
        - norm.cdf(d2)
        * spec.strike
        * np.exp(-spec.rate * spec.maturity)
    )

    put_bms = (
        norm.cdf(-d2)
        * spec.strike
        * np.exp(-spec.rate * spec.maturity)
        - norm.cdf(-d1)
        * spec.spot
        * np.exp(-spec.dividend_yield * spec.maturity)
    )
    return float(call_bms), float(put_bms)

def test_crr_european_call_model_is_consistent():
    spec_crr_call = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Call",
        exercise_style="European",
        model="CRR",
        dividend_yield=0.0,
    )
    print(crr_price(spec_crr_call))
    print(BMS(spec_crr_call)[0])
    # assert np.abs(crr_price(spec_crr_call) - BMS(spec_crr_call)[0]) <= 1e-2

def test_jr_european_call_model_is_consistent():
    spec_jr_call = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Call",
        exercise_style="European",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )
    print(jr_price(spec_jr_call))
    print(BMS(spec_jr_call)[0])
    #assert np.abs(jr_price(spec_jr_call) - BMS(spec_jr_call)[0]) <= 1e-2

def test_tian93_european_call_model_is_consistent():
    spec_tian93_call = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Call",
        exercise_style="European",
        model="Tian-1993",
        dividend_yield=0.0,
    )
    print(tian93_price(spec_tian93_call))
    print(BMS(spec_tian93_call)[0])
    #assert np.abs(tian93_price(spec_tian93_call) - BMS(spec_tian93_call)[0]) <= 1e-2

def test_crr_european_put_model_is_consistent():
    spec_crr_put = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Put",
        exercise_style="European",
        model="CRR",
        dividend_yield=0.0,
    )
    print(crr_price(spec_crr_put))
    print(BMS(spec_crr_put)[1])
    #assert np.abs(crr_price(spec_crr_put) - BMS(spec_crr_put)[0]) <= 1e-2

def test_jr_european_put_model_is_consistent():
    spec_jr_put = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Put",
        exercise_style="European",
        model="Jarrow-Rudd",
        dividend_yield=0.0,
    )
    print(jr_price(spec_jr_put))
    print(BMS(spec_jr_put)[1])
    #assert np.abs(jr_price(spec_jr_put) - BMS(spec_jr_put)[0]) <= 1e-2

def test_tian93_european_put_model_is_consistent():
    spec_tian93_put = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Put",
        exercise_style="European",
        model="Tian-1993",
        dividend_yield=0.0,
    )
    print(tian93_price(spec_tian93_put))
    print(BMS(spec_tian93_put)[1])
    #assert np.abs(tian93_price(spec_tian93_put) - BMS(spec_tian93_put)[0]) <= 1e-2




