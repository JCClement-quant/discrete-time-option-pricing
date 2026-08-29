from src.pricers import (BAW_quadratic, BAW_put, BAW_call, BAW_call_boundary, BAW_call_critical_spot,
                        BAW_put_critical_spot, BAW_put_boundary, BMS, BMS_d1_d2, BMS_at_spot)
from src.pricers import OptionSpec
import math, numpy as np
from scipy.stats import norm

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


def test_baw_american_put():
    spec_crr_put = OptionSpec(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            steps=1000,
            option_type="Put",
            exercise_style="American",
            dividend_yield=0.1,
    )
    spec_crr_put_eu = OptionSpec(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            steps=1000,
            option_type="Put",
            exercise_style="European",
            dividend_yield=0.1,
    )


    eps = 1e-4
    q1, q2 = BAW_quadratic(spec_crr_put)
    S_star = BAW_put_critical_spot(spec_crr_put, q1)
    value_at_boundary = BAW_put(spec_crr_put, S_star)
    intrinsic_at_boundary = spec_crr_put.strike - S_star
    bms_put = BMS(spec_crr_put_eu)[1]
    intrinsic = spec_crr_put.strike - spec_crr_put.spot

    delta = (
                    BAW_put(spec_crr_put, S_star + eps)
                    - BAW_put(spec_crr_put, S_star - eps)
            ) / (2.0 * eps)

    # ---------------------------------------------------------
    # Exponents / boundary
    # ---------------------------------------------------------

    assert math.isfinite(q1)
    assert math.isfinite(q2)

    assert q1 < 0.0
    assert q2 > 0.0
    assert q1 < q2

    assert 0.0 < S_star < spec_crr_put.strike

    # ---------------------------------------------------------
    # Value matching: CONTINUATION side
    # ---------------------------------------------------------

    d1_star = BMS_d1_d2(
        spec_crr_put,
        S_star,
    )[0]

    p_bms_star = BMS_at_spot(
        spec_crr_put_eu,
        S_star,
    )[1]

    A1 = (
            -S_star / q1
            * (
                    1.0
                    - np.exp(
                -spec_crr_put.dividend_yield
                * spec_crr_put.maturity
            )
                    * norm.cdf(-d1_star)
            )
    )

    continuation_at_boundary = (
            p_bms_star + A1
    )

    intrinsic_at_boundary = (
            spec_crr_put.strike - S_star
    )

    assert abs(
        continuation_at_boundary
        - intrinsic_at_boundary
    ) < eps

    # ---------------------------------------------------------
    # Smooth pasting
    # ---------------------------------------------------------

    delta = (
                    BAW_put(spec_crr_put, S_star + eps)
                    - BAW_put(spec_crr_put, S_star - eps)
            ) / (2.0 * eps)

    assert abs(delta + 1.0) < eps

    # ---------------------------------------------------------
    # American dominance at CURRENT spot
    # ---------------------------------------------------------

    baw_put = BAW_put(
        spec_crr_put,
        spec_crr_put.spot,
    )

    bms_put = BMS_at_spot(
        spec_crr_put_eu,
        spec_crr_put_eu.spot,
    )[1]

    intrinsic = max(
        spec_crr_put.strike - spec_crr_put.spot,
        0.0,
    )

    assert baw_put >= bms_put
    assert baw_put >= intrinsic

    # ---------------------------------------------------------
    # Exercise branch below S*
    # ---------------------------------------------------------

    spot_exercise = S_star - eps

    assert abs(
        BAW_put(spec_crr_put, spot_exercise)
        - (spec_crr_put.strike - spot_exercise)
    ) < eps