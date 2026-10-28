from typing import Literal

OptionType = Literal["Call", "Put"]
ExerciseStyle = Literal["European", "American"]
ModelType = Literal["CRR", "Jarrow-Rudd", "Tian-1993", "Leisen-Reimer"]


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

    if rate < 0.0:
        raise ValueError(
            "Negative interest rates are not supported yet."
        )

    if volatility <= 0.0:
        raise ValueError("volatility must be non-negative")

    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a strictly positive integer")

    if option_type not in ("Call", "Put"):
        raise ValueError("option_type must be 'Call' or 'Put'")

    if exercise_style not in ("European", "American"):
        raise ValueError("exercise_style must be 'European' or 'American'")

    if model not in ("CRR", "Jarrow-Rudd", "Tian-1993", "Leisen-Reimer", "Tian-1999"):
        raise ValueError("model must be 'CRR', 'Jarrow-Rudd' or 'Tian-1993' or 'Leisen-Reimer' or 'Tian-1999'")