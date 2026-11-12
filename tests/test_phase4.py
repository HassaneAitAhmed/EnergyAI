from datetime import date

import pytest

from energyai.phase4 import _dates


def test_dates_are_inclusive() -> None:
    assert _dates(date(2024, 6, 2), date(2024, 6, 4)) == [
        date(2024, 6, 2),
        date(2024, 6, 3),
        date(2024, 6, 4),
    ]


def test_dates_reject_reverse_range() -> None:
    with pytest.raises(ValueError, match="earlier"):
        _dates(date(2024, 6, 4), date(2024, 6, 2))
