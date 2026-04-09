from dataclasses import dataclass
from typing import Literal

import math

from src.payoffs import vanilla_payoff
from src.validation import validate_option_inputs

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]
ModelType = Literal["crr", "jarrow-rudd"]


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
    model: ModelType = "crr"
    dividend_yield: float = 0.0


def crr_price(spec: OptionSpec) -> float:
    """
    Price a vanilla option with a Cox-Ross-Rubinstein recombining tree.

    Parameters
    ----------
    spec : OptionSpec
        Option specification and model inputs.

    Returns
    -------
    float
        Present value of the option.
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
        model="crr",
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

    values = []
    for j in range(spec.steps + 1):
        terminal_spot = spec.spot * (u ** j) * (d ** (spec.steps - j))
        values.append(vanilla_payoff(terminal_spot, spec.strike, spec.option_type))

    allow_early_exercise = spec.exercise_style == "american"

    for i in range(spec.steps - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (p * values[j + 1] + (1.0 - p) * values[j])

            if allow_early_exercise:
                node_spot = spec.spot * (u ** j) * (d ** (i - j))
                exercise_value = vanilla_payoff(node_spot, spec.strike, spec.option_type)
                values[j] = max(continuation_value, exercise_value)
            else:
                values[j] = continuation_value

    return values[0]