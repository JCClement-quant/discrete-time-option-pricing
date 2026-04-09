from datetime import date
from typing import Optional

def make_date(day: int, month: int, year: int) -> date:
    """
    Build a Python date from day, month, year integers.
    """
    return date(year, month, day)


def year_fraction(start_date: date, end_date: date, basis: float = 365.0) -> float:
    """
    Compute a simple ACT/basis year fraction between two dates.

    Parameters
    ----------
    start_date : date
        Valuation date.
    end_date : date
        Maturity date.
    basis : float, optional
        Day-count denominator, by default 365.0. Industry standard.

    Returns
    -------
    float
        Year fraction.

    Raises
    ------
    ValueError
        If the maturity date is not strictly after the valuation date.
    """
    delta_days = (end_date - start_date).days
    if delta_days <= 0:
        raise ValueError("Option expired.")
    return delta_days / basis

def time_to_maturity_from_components(
    day: int,
    month: int,
    year: int,
    valuation_date: Optional[date] = None,  # more modern : valuation_date: date | None = None,
    basis: float = 365.0,
) -> float:
    """
    Build a maturity date from day/month/year and return time to maturity in years.
    """
    if valuation_date is None:
        effective_valuation_date: date = date.today()
    else:
        effective_valuation_date = valuation_date

    maturity_date = make_date(day, month, year)
    return year_fraction(effective_valuation_date, maturity_date, basis=basis)