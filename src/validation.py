from typing import Literal

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]
ModelType = Literal["crr", "jarrow-rudd"]


def validate_option_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    steps: int,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    model: ModelType,
) -> None:
    """
    Validate core inputs for discrete-time vanilla option pricing.

    Notes
    -----
    Negative rates are allowed.
    Zero volatility is allowed at validation stage, although some
    pricing kernels may choose to reject it explicitly.
    """
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")

    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")

    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive")

    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")

    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a strictly positive integer")

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if exercise_style not in ("european", "american"):
        raise ValueError("exercise_style must be 'european' or 'american'")

    if model not in ("crr", "jarrow-rudd"):
        raise ValueError("model must be 'crr' or 'jarrow-rudd'")