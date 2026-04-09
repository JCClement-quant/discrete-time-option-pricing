from datetime import date

import pytest

from src.dates import make_date, year_fraction, time_to_maturity_from_components


def test_make_date():
    d = make_date(1, 1, 2028)
    assert d == date(2028, 1, 1)


def test_year_fraction_positive():
    start = date(2026, 4, 5)
    end = date(2028, 1, 1)
    result = year_fraction(start, end, basis=365.0)
    assert result > 0.0


def test_year_fraction_rejects_expired_option():
    start = date(2026, 4, 5)
    end = date(2026, 4, 5)
    with pytest.raises(ValueError, match="expired"):
        year_fraction(start, end)


def test_time_to_maturity_from_components():
    valuation_date = date(2026, 4, 5)
    result = time_to_maturity_from_components(
        day=1,
        month=1,
        year=2028,
        valuation_date=valuation_date,
        basis=365.0,
    )
    assert result > 1.0