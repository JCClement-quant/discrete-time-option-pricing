from typing import Literal

OptionType = Literal["call", "put"]


def call_payoff(spot: float, strike: float) -> float:
    """Return the payoff of a vanilla call option."""
    return max(spot - strike, 0.0)


def put_payoff(spot: float, strike: float) -> float:
    """Return the payoff of a vanilla put option."""
    return max(strike - spot, 0.0)


def vanilla_payoff(spot: float, strike: float, option_type: OptionType) -> float:
    """Dispatch vanilla payoff computation by option type."""
    if option_type == "call":
        return call_payoff(spot, strike)
    if option_type == "put":
        return put_payoff(spot, strike)
    raise ValueError(f"Unsupported option_type: {option_type}")