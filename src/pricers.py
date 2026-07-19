from dataclasses import dataclass
from typing import Literal
import math
import numpy as np

from src.payoffs import vanilla_payoff
from src.validation import validate_option_inputs

OptionType = Literal["Call", "Put"]
ExerciseStyle = Literal["European", "American"]
ModelType = Literal["CRR", "Jarrow-Rudd"]


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
class CRRTreeResult:
    spot_tree: np.ndarray
    option_tree: np.ndarray
    exercise_tree: np.ndarray
    intrinsic_tree: np.ndarray
    u: float
    d: float
    p: float
    dt: float
    discount: float

@dataclass
class JRTreeResult:
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

def crr_tree(spec: OptionSpec) -> CRRTreeResult:
    """
    Build the CRR spot tree and option value tree.

    Returns
    -------
    CRRTreeResult
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

    return CRRTreeResult(
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

def jr_tree(spec: OptionSpec) -> JRTreeResult:
    """
    Build the JR spot tree and option value tree.

    Returns
    -------
    JRTreeResult
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
    d = math.exp((carry_rate - 0.5 * spec.volatility**2) * dt- spec.volatility * math.sqrt(dt))
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

    return JRTreeResult(
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