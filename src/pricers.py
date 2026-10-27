from dataclasses import dataclass
from typing import Literal
from scipy.stats import norm
import math
import numpy as np

from src.payoffs import vanilla_payoff
from src.validation import validate_option_inputs
from src.functions import peizer_pratt_inversion
from scipy.optimize import brentq

OptionType = Literal["Call", "Put"]
ExerciseStyle = Literal["European", "American"]
ModelType = Literal["CRR", "Jarrow-Rudd", "Tian-1993", "Leisen-Reimer"]

@dataclass(frozen=True)
class OptionSpec:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    steps: int
    option_type: OptionType
    exercise_style: ExerciseStyle
    model: ModelType = "Jarrow-Rudd"
    dividend_yield: float = 0.0

@dataclass
class BinomialTreeResult:
    spot_tree: np.ndarray
    option_tree: np.ndarray
    exercise_tree: np.ndarray
    intrinsic_tree: np.ndarray
    u: float
    d: float
    p: float
    dt: float
    discount: float

def crr_price(spec: OptionSpec) -> float:
    """
    Price a vanilla option with a Cox-Ross-Rubinstein recombining tree.
    """
    result = crr_tree(spec)
    return float(result.option_tree[0, 0])

def crr_tree(spec: OptionSpec) -> BinomialTreeResult:
    """
    Build the CRR spot tree and option value tree.

    Returns
    -------
    BinomialTreeResult
        Full tree result, suitable for visualization and diagnostics.
    """
    validate_option_inputs(
        spot=spec.spot,
        strike=spec.strike,
        maturity=spec.maturity,
        rate=spec.rate,
        volatility=spec.volatility,
        steps=spec.steps,
        option_type=spec.option_type,
        exercise_style=spec.exercise_style,
        model="CRR",
    )

    if spec.volatility == 0.0:
        raise ValueError("CRR tree is not defined for zero volatility in this implementation.")

    dt = spec.maturity / spec.steps
    carry_rate = spec.rate - spec.dividend_yield

    u = math.exp(spec.volatility * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(carry_rate * dt)
    p = (growth - d) / (u - d)
    discount = math.exp(-spec.rate * dt)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability out of bounds: p={p:.6f}. "
            "Check inputs or time-step granularity."
        )

    n = spec.steps

    spot_tree = np.full((n + 1, n + 1), np.nan)
    option_tree = np.full((n + 1, n + 1), np.nan)
    intrinsic_tree = np.full((n + 1, n + 1), np.nan)
    exercise_tree = np.full((n + 1, n + 1), False, dtype=bool)

    # Build spot tree
    for i in range(n + 1):
        for j in range(i + 1):
            spot_tree[i, j] = spec.spot * (u ** j) * (d ** (i - j))

    # Terminal payoff
    for j in range(n + 1):
        intrinsic = vanilla_payoff(spot_tree[n, j], spec.strike, spec.option_type)
        option_tree[n, j] = intrinsic
        intrinsic_tree[n, j] = intrinsic

    allow_early_exercise = spec.exercise_style == "American"

    # Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (
                p * option_tree[i + 1, j + 1] + (1.0 - p) * option_tree[i + 1, j]
            )

            intrinsic_value = vanilla_payoff(spot_tree[i, j], spec.strike, spec.option_type)
            intrinsic_tree[i, j] = intrinsic_value

            if allow_early_exercise:
                option_tree[i, j] = max(continuation_value, intrinsic_value)
                exercise_tree[i, j] = intrinsic_value > continuation_value
            else:
                option_tree[i, j] = continuation_value

    return BinomialTreeResult(
        spot_tree=spot_tree,
        option_tree=option_tree,
        exercise_tree=exercise_tree,
        intrinsic_tree=intrinsic_tree,
        u=u,
        d=d,
        p=p,
        dt=dt,
        discount=discount,
    )

def jr_price(spec: OptionSpec) -> float:
    """
    Price a vanilla option with a Jarrow-Rudd recombining tree.
    """
    result = jr_tree(spec)
    return float(result.option_tree[0, 0])

def jr_tree(spec: OptionSpec) -> BinomialTreeResult:
    """
    Build the JR spot tree and option value tree.

    Returns
    -------
    BinomialTreeResult
        Full tree result, suitable for visualization and diagnostics.
    """
    validate_option_inputs(
        spot=spec.spot,
        strike=spec.strike,
        maturity=spec.maturity,
        rate=spec.rate,
        volatility=spec.volatility,
        steps=spec.steps,
        option_type=spec.option_type,
        exercise_style=spec.exercise_style,
        model="Jarrow-Rudd",
    )

    if spec.volatility == 0.0:
        raise ValueError("JR tree is not defined for zero volatility in this implementation.")

    dt = spec.maturity / spec.steps
    carry_rate = spec.rate - spec.dividend_yield

    u = math.exp((carry_rate - 0.5 * spec.volatility**2) * dt + spec.volatility * math.sqrt(dt))
    d = math.exp((carry_rate - 0.5 * spec.volatility**2) * dt - spec.volatility * math.sqrt(dt))
    p = 0.5
    discount = math.exp(-spec.rate * dt)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability out of bounds: p={p:.6f}. "
            "Check inputs or time-step granularity."
        )

    n = spec.steps

    spot_tree = np.full((n + 1, n + 1), np.nan)
    option_tree = np.full((n + 1, n + 1), np.nan)
    intrinsic_tree = np.full((n + 1, n + 1), np.nan)
    exercise_tree = np.full((n + 1, n + 1), False, dtype=bool)

    # Build spot tree
    for i in range(n + 1):
        for j in range(i + 1):
            spot_tree[i, j] = spec.spot * (u ** j) * (d ** (i - j))

    # Terminal payoff
    for j in range(n + 1):
        intrinsic = vanilla_payoff(spot_tree[n, j], spec.strike, spec.option_type)
        option_tree[n, j] = intrinsic
        intrinsic_tree[n, j] = intrinsic

    allow_early_exercise = spec.exercise_style == "American"

    # Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (
                p * option_tree[i + 1, j + 1] + (1.0 - p) * option_tree[i + 1, j]
            )

            intrinsic_value = vanilla_payoff(spot_tree[i, j], spec.strike, spec.option_type)
            intrinsic_tree[i, j] = intrinsic_value

            if allow_early_exercise:
                option_tree[i, j] = max(continuation_value, intrinsic_value)
                exercise_tree[i, j] = intrinsic_value > continuation_value
            else:
                option_tree[i, j] = continuation_value

    return BinomialTreeResult(
        spot_tree=spot_tree,
        option_tree=option_tree,
        exercise_tree=exercise_tree,
        intrinsic_tree=intrinsic_tree,
        u=u,
        d=d,
        p=p,
        dt=dt,
        discount=discount,
    )

def tian93_price(spec: OptionSpec) -> float:
    """
    Price a vanilla option with a Tian 1993 recombining tree.
    """
    result = tian93_tree(spec)
    return float(result.option_tree[0, 0])

def tian93_tree(spec: OptionSpec) -> BinomialTreeResult:
    """
    Build the Tian 1993 spot tree and option value tree.

    Returns
    -------
    BinomialTreeResult
        Full tree result, suitable for visualization and diagnostics.
    """
    validate_option_inputs(
        spot=spec.spot,
        strike=spec.strike,
        maturity=spec.maturity,
        rate=spec.rate,
        volatility=spec.volatility,
        steps=spec.steps,
        option_type=spec.option_type,
        exercise_style=spec.exercise_style,
        model="Tian-1993",
    )

    if spec.volatility == 0.0:
        raise ValueError("Tian 1993 tree is not defined for zero volatility in this implementation.")

    dt = spec.maturity / spec.steps
    carry_rate = spec.rate - spec.dividend_yield

    h = np.exp((spec.volatility**2) * dt)
    u = 0.5 * np.exp(carry_rate * dt) * h * ( h + 1 + np.sqrt(h**2+2*h-3))
    d = 0.5 * np.exp(carry_rate * dt) * h * ( h + 1 - np.sqrt(h**2+2*h-3))
    p = (np.exp(carry_rate * dt) - d)/(u - d)
    discount = math.exp(-spec.rate * dt)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability out of bounds: p={p:.6f}. "
            "Check inputs or time-step granularity."
        )

    n = spec.steps

    spot_tree = np.full((n + 1, n + 1), np.nan)
    option_tree = np.full((n + 1, n + 1), np.nan)
    intrinsic_tree = np.full((n + 1, n + 1), np.nan)
    exercise_tree = np.full((n + 1, n + 1), False, dtype=bool)

    # Build spot tree
    for i in range(n + 1):
        for j in range(i + 1):
            spot_tree[i, j] = spec.spot * (u ** j) * (d ** (i - j))

    # Terminal payoff
    for j in range(n + 1):
        intrinsic = vanilla_payoff(spot_tree[n, j], spec.strike, spec.option_type)
        option_tree[n, j] = intrinsic
        intrinsic_tree[n, j] = intrinsic

    allow_early_exercise = spec.exercise_style == "American"

    # Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (
                p * option_tree[i + 1, j + 1] + (1.0 - p) * option_tree[i + 1, j]
            )

            intrinsic_value = vanilla_payoff(spot_tree[i, j], spec.strike, spec.option_type)
            intrinsic_tree[i, j] = intrinsic_value

            if allow_early_exercise:
                option_tree[i, j] = max(continuation_value, intrinsic_value)
                exercise_tree[i, j] = intrinsic_value > continuation_value
            else:
                option_tree[i, j] = continuation_value

    return BinomialTreeResult(
        spot_tree=spot_tree,
        option_tree=option_tree,
        exercise_tree=exercise_tree,
        intrinsic_tree=intrinsic_tree,
        u=u,
        d=d,
        p=p,
        dt=dt,
        discount=discount,
    )

def lr_price(spec: OptionSpec) -> float:
    """
    Price a vanilla option with a Leisen-Reimer recombining tree.
    """
    result = lr_tree(spec)
    return float(result.option_tree[0, 0])

def lr_tree(spec: OptionSpec) -> BinomialTreeResult:
    """
    Build the Leisen-Reimer spot tree and option value tree.

    Returns
    -------
    BinomialTreeResult
        Full tree result, suitable for visualization and diagnostics.
    """
    validate_option_inputs(
        spot=spec.spot,
        strike=spec.strike,
        maturity=spec.maturity,
        rate=spec.rate,
        volatility=spec.volatility,
        steps=spec.steps,
        option_type=spec.option_type,
        exercise_style=spec.exercise_style,
        model="Leisen-Reimer",
    )

    if spec.volatility == 0.0:
        raise ValueError("Leisen-Riemer tree is not defined for zero volatility in this implementation.")

    if spec.steps % 2 == 0:
        raise ValueError("Leisen-Riemer tree requires an odd number of steps for convergence.")

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

    p_prime = peizer_pratt_inversion(d1, spec.steps)
    p = peizer_pratt_inversion(d2, spec.steps)

    dt = spec.maturity / spec.steps
    carry_rate = spec.rate - spec.dividend_yield
    a = np.exp(carry_rate * dt)
    u = a * p_prime / p
    d = (a - p * u) / (1 - p)
    discount = math.exp(-spec.rate * dt)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability out of bounds: p={p:.6f}. "
            "Check inputs or time-step granularity."
        )

    n = spec.steps

    spot_tree = np.full((n + 1, n + 1), np.nan)
    option_tree = np.full((n + 1, n + 1), np.nan)
    intrinsic_tree = np.full((n + 1, n + 1), np.nan)
    exercise_tree = np.full((n + 1, n + 1), False, dtype=bool)

    # Build spot tree
    for i in range(n + 1):
        for j in range(i + 1):
            spot_tree[i, j] = spec.spot * (u ** j) * (d ** (i - j))

    # Terminal payoff
    for j in range(n + 1):
        intrinsic = vanilla_payoff(spot_tree[n, j], spec.strike, spec.option_type)
        option_tree[n, j] = intrinsic
        intrinsic_tree[n, j] = intrinsic

    allow_early_exercise = spec.exercise_style == "American"

    # Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (
                p * option_tree[i + 1, j + 1] + (1.0 - p) * option_tree[i + 1, j]
            )

            intrinsic_value = vanilla_payoff(spot_tree[i, j], spec.strike, spec.option_type)
            intrinsic_tree[i, j] = intrinsic_value

            if allow_early_exercise:
                option_tree[i, j] = max(continuation_value, intrinsic_value)
                exercise_tree[i, j] = intrinsic_value > continuation_value
            else:
                option_tree[i, j] = continuation_value

    return BinomialTreeResult(
        spot_tree=spot_tree,
        option_tree=option_tree,
        exercise_tree=exercise_tree,
        intrinsic_tree=intrinsic_tree,
        u=u,
        d=d,
        p=p,
        dt=dt,
        discount=discount,
    )


def build_binomial_tree(spec: OptionSpec):
    match spec.model:
        case "CRR":
            return crr_tree(spec)

        case "Jarrow-Rudd":
            return jr_tree(spec)

        case "Tian-1993":
            return tian93_tree(spec)

        case "Leisen-Reimer":
            return lr_tree(spec)

        case _:
            raise ValueError(
                f"Unsupported binomial model: {spec.model!r}"
            )

def BMS_d1_d2(spec: OptionSpec, spot: float) -> tuple[float, float]:
    """
    Calculates BMS d1 and d2 parameters
    :param spec : option spec
    :param spot : float, spot input

    Returns
    -------
    Tuple[float, float]
    d1,d2
    """

    sigma_sqrt_t = (spec.volatility * np.sqrt(spec.maturity))

    d1 = (
        np.log(spot / spec.strike)
        + (
            spec.rate
            - spec.dividend_yield
            + 0.5 * spec.volatility**2
        ) * spec.maturity
    ) / sigma_sqrt_t

    d2 = d1 - sigma_sqrt_t

    return float(d1), float(d2)

def BMS_at_spot(spec: OptionSpec, spot :float) -> tuple[float, float]:
    """
    Calculates BMS price fo a given spot
    :param spec : option spec
    :param spot : float, spot input
    Returns
    -------
    Tuple[float, float]
    call[0], put [1]
    """
    d1, d2 = BMS_d1_d2(spec, spot)

    call_bms = (
        norm.cdf(d1)
         * spot
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
        * spot
        * np.exp(-spec.dividend_yield * spec.maturity)
    )
    return float(call_bms), float(put_bms)

def BMS(spec: OptionSpec) -> tuple[float, float]:
    """
    BMS wrapper public interface.
    :param spec : option spec
    Returns
    -------
    Tuple[float, float]
    call BMS_at_spot(spec)[0], put BMS_at_spot(spec)[1]
    """
    if spec.exercise_style != "European":
        raise ValueError(
            "BMS is inaccurate for American options."
        )

    return BMS_at_spot(
        spec,
        spec.spot,
    )

def BAW_quadratic(spec: OptionSpec) -> tuple[float, float]:
    """

    BAW parameters setter
    :param spec : option spec
    Returns
    -------
    Tuple[float, float]
    Q1, Q2 from the Euler-Cauchy EDP
    """
    if spec.exercise_style != "American":
        raise ValueError("BAW is intended for American options only.")

    tau = spec.maturity
    r = spec.rate
    q = spec.dividend_yield
    sigma = spec.volatility

    if tau <= 0.0:
        raise ValueError("Time to maturity must be positive.")

    if sigma <= 0.0:
        raise ValueError("Volatility must be positive.")

    b = r - q

    M = 2.0 * r / sigma ** 2
    N = 2.0 * b / sigma ** 2

    kappa = 1.0 - math.exp(-r * tau)

    # M / kappa = 2 * r / (sigma^2 * (1 - e^(-rT)))
    # if r == 0 then 1 - e^0 = 0 so kappa = "0/0"
    if abs(r) < 1e-12:
        M_over_kappa = 2.0 / (sigma ** 2 * tau) # 1 - e^(-r * tau) --> r * tau when (r->0)
                                                # so M/kappa --> 2 / (sigma^2 * tau)
    else:
        M_over_kappa = M / kappa

    discriminant = ((N - 1.0) ** 2 + 4.0 * M_over_kappa)

    Q1 = ( - (N - 1.0) - math.sqrt(discriminant)) / 2.0

    Q2 = ( - (N - 1.0) + math.sqrt(discriminant)) / 2.0

    return Q1, Q2

def BAW_call_boundary(critical_spot :float, spec: OptionSpec, Q2: float) -> float:
    """
    :param spec : option spec
    :param critical_spot: from BAW_call_critical_spot(spec, Q2)
    :param Q2: from BAW_quadratic(spec)

    Returns
    -------
    float
    f(S_star) << epsilon
    """
    d1, _ = BMS_d1_d2(spec, critical_spot)
    call_bms, _ = BMS_at_spot(spec, critical_spot)
    assert critical_spot > 0.0
    assert Q2 > 0.0

    return (critical_spot - spec.strike - call_bms - critical_spot / Q2
                * (1.0 - np.exp(-spec.dividend_yield * spec.maturity) * norm.cdf(d1)))

def BAW_call_critical_spot(spec: OptionSpec,Q2: float,) -> float:
    """
      :param spec : option spec
      :param Q2: from BAW_quadratic(spec)

      Returns
      -------
      float
      f(S_star) << epsilon
    """

    r = spec.rate
    q = spec.dividend_yield
    sigma = spec.volatility

    b = r - q

    M = 2.0 * r / sigma ** 2
    N = 2.0 * b / sigma ** 2

    # Seed
    QPerp = 0.5 * ((-1.0) * (N - 1.0) + np.sqrt((N - 1.0) ** 2 + 4.0 * M))
    S_star_steady = spec.strike / (1.0 - 1.0 / QPerp)

    lower_bound = spec.strike
    upper_bound = S_star_steady

    f_lower = BAW_call_boundary(lower_bound, spec, Q2)
    f_upper = BAW_call_boundary(upper_bound, spec, Q2)
    security = upper_bound
    security *=50

    while f_lower * f_upper > 0.0 and upper_bound < security:
        upper_bound *= 2.0
        f_upper = BAW_call_boundary(upper_bound, spec, Q2)

    if f_lower * f_upper > 0.0:
        raise RuntimeError(
            "Unable to bracket the BAW call critical spot."
        )

    critical_spot = brentq(
        lambda s: BAW_call_boundary(
            s,
            spec,
            Q2,
        ),
        lower_bound,
        upper_bound,
    )

    return float(critical_spot)

def BAW_call(spec: OptionSpec, spot :float) -> float:
    """
    :param spec : option spec
    :param spot : float, input spot

    Returns
    -------
    float
    BAW call price
    """
    if spec.dividend_yield <= 0.0:
        return BMS_at_spot(spec, spot)[0]

    _ , Q2 = BAW_quadratic(spec)
    S_star = BAW_call_critical_spot(spec, Q2)
    c_bms = BMS_at_spot(spec, spot)[0]
    d1_at_S_star = BMS_d1_d2(spec, S_star)[0]

    A2 = (S_star / Q2) * (1.0 - np.exp(-spec.dividend_yield * spec.maturity) * norm.cdf(d1_at_S_star))

    if spot < S_star:
        return c_bms + A2 * (spot/S_star) ** Q2
    else:
        return spot - spec.strike

def BAW_put_boundary(critical_spot: float, spec: OptionSpec, Q1: float) -> float:
    """
    :param spec : option spec
    :param critical_spot: from BAW_put_critical_spot(spec, Q2)
    :param Q1: from BAW_quadratic(spec)

    Returns
    -------
    float
    f(S_star) << epsilon
    """
    d1, _ = BMS_d1_d2(spec, critical_spot)
    _, put_bms = BMS_at_spot(spec, critical_spot)
    assert critical_spot > 0.0
    assert Q1 < 0.0

    return (spec.strike - critical_spot - put_bms + critical_spot / Q1
            * (1.0 - np.exp(-spec.dividend_yield * spec.maturity) * norm.cdf(-d1)))

def BAW_put_critical_spot(spec: OptionSpec,Q1: float,) -> float:
    """
      :param spec : option spec
      :param Q1: from BAW_quadratic(spec)

      Returns
      -------
      float
      f(S_star) << epsilon
    """

    r = spec.rate
    q = spec.dividend_yield
    sigma = spec.volatility

    b = r - q

    M = 2.0 * r / sigma ** 2
    N = 2.0 * b / sigma ** 2

    # Seed
    QPerp = 0.5 * ((-1.0) * (N - 1.0) - np.sqrt((N - 1.0) ** 2 + 4.0 * M))
    S_star_steady = spec.strike / (1.0 - 1.0 / QPerp)

    assert 0.0 < S_star_steady < spec.strike

    lower_bound = S_star_steady
    upper_bound = spec.strike

    f_lower = BAW_put_boundary(lower_bound, spec, Q1)
    f_upper = BAW_put_boundary(upper_bound, spec, Q1)

    floor = spec.strike * 1e-10

    while f_lower * f_upper > 0.0 and lower_bound > floor:
        lower_bound *= 0.5

        f_lower = BAW_put_boundary(
            lower_bound, spec, Q1
        )

    if f_lower * f_upper > 0.0:
        raise RuntimeError(
            "Unable to bracket the BAW put critical spot. "
            f"f(lower)={f_lower}, "
            f"f(upper)={f_upper}, "
            f"lower={lower_bound}, "
            f"upper={upper_bound}"
        )

    critical_spot = brentq(
        lambda s: BAW_put_boundary(
            s,
            spec,
            Q1,
        ),
        lower_bound,
        upper_bound,
    )

    assert 0.0 < critical_spot < spec.strike

    return float(critical_spot)

def BAW_put(spec: OptionSpec, spot :float) -> float:
    """
    :param spec : option spec
    :param spot : float, input spot

    Returns
    -------
    float
    BAW put price
    """

    Q1, _ = BAW_quadratic(spec)
    S_star = BAW_put_critical_spot(spec, Q1)
    p_bms = BMS_at_spot(spec, spot)[1]
    negative_d1_at_S_star = - BMS_d1_d2(spec, S_star)[0]

    A1 = -(S_star / Q1) * (1.0 - np.exp(-spec.dividend_yield * spec.maturity) * norm.cdf(negative_d1_at_S_star))
    assert A1 > 0.0

    if spot > S_star:
        return p_bms + A1 * (spot/S_star) ** Q1
    else:
        return spec.strike - spot

@dataclass
class JZParams:
    h: float
    alpha: float
    beta: float
    the_lambda: float
    the_lambda_prime: float
    S_star: float
    hA : float
    b: float
    c: float

def JZ_parameters(spec: OptionSpec, phi: int) -> JZParams:

    tau = spec.maturity
    r = spec.rate
    q = spec.dividend_yield
    sigma = spec.volatility

    h = 1.0 - math.exp(-r * tau)

    alpha = 2.0 * r / sigma**2
    beta = 2.0 * (r - q) / sigma**2

    root = math.sqrt(
        (beta - 1.0)**2
        + 4.0 * alpha / h
    )

    lam = 0.5 * (
        -(beta - 1.0)
        + phi * root
    )

    lam_prime = (
        -phi
        * alpha
        / (h**2 * root)
    )

    q1, q2 = BAW_quadratic(spec)

    if phi == 1:
        S_star = BAW_call_critical_spot(spec, q2)
    else:
        S_star = BAW_put_critical_spot(spec, q1)

    call_eur, put_eur = BMS_at_spot(spec, S_star)
    VE_star = call_eur if phi == 1 else put_eur

    hA = phi * (S_star - spec.strike) - VE_star

    d1_star, d2_star = BMS_d1_d2(spec, S_star)

    denom = 2.0 * lam + beta - 1.0

    b = (
        (1.0 - h)
        * alpha
        * lam_prime
        / (2.0 * denom)
    )

    DVe = (
        S_star
        * norm.pdf(d1_star)
        * sigma
        * math.exp((r - q) * tau)
        / (2.0 * r * math.sqrt(tau))
        -
        phi
        * q
        * S_star
        * norm.cdf(phi * d1_star)
        * math.exp((r - q) * tau)
        / r
        +
        phi
        * spec.strike
        * norm.cdf(phi * d2_star)
    )

    c = (
        -(1.0 - h)
        * alpha
        / denom
        * (
            DVe / hA
            + 1.0 / h
            + lam_prime / denom
        )
    )

    return JZParams(
        h=h,
        alpha=alpha,
        beta=beta,
        the_lambda=lam,
        the_lambda_prime=lam_prime,
        S_star=S_star,
        hA=hA,
        b=b,
        c=c,
    )

def JZ_Pricer(spec: OptionSpec, spot:float, phi:int) -> float:
    params = JZ_parameters(spec, phi)

    if phi * (params.S_star - spot) <= 0:
        return phi * (spot - spec.strike)

    call_eur, put_eur = BMS_at_spot(spec, spot)
    european_price = call_eur if phi == 1 else put_eur

    log_ratio = math.log(spot / params.S_star)

    chi = (
        params.b * log_ratio**2
        + params.c * log_ratio
    )

    premium = (
        params.hA
        * (spot / params.S_star) ** params.the_lambda
        / (1.0 - chi)
    )

    return european_price + premium