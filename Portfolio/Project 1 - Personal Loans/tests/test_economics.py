"""Every money figure the memo quotes, pinned so a code change cannot move it silently."""

import numpy as np
import pytest

from loan_targeting.economics import (
    break_even_threshold,
    campaign_profit,
    per_thousand,
    should_contact,
)

# 1,000 customers at the data's 9.6% base rate.
ACCEPTED = np.array([1] * 96 + [0] * 904)


def test_break_even_is_ten_percent():
    assert break_even_threshold() == pytest.approx(0.10)


def test_doubled_contact_cost_moves_break_even_to_twenty_percent():
    assert break_even_threshold(contact_cost=1000.0) == pytest.approx(0.20)


@pytest.mark.parametrize(("borrow_anyway", "expected"), [(0.0, 0.10), (0.25, 0.1333), (0.50, 0.20)])
def test_would_borrow_anyway_raises_break_even(borrow_anyway, expected):
    assert break_even_threshold(borrow_anyway=borrow_anyway) == pytest.approx(expected, abs=1e-4)


def test_borrow_anyway_of_one_is_rejected():
    with pytest.raises(ValueError):
        break_even_threshold(borrow_anyway=1.0)


def test_contact_rule_includes_the_threshold():
    assert should_contact([0.05, 0.10, 0.30], 0.10).tolist() == [False, True, True]


def test_call_everyone_loses_twenty_thousand_per_thousand():
    called = np.ones(1000, dtype=bool)
    assert per_thousand(campaign_profit(called, ACCEPTED), 1000) == pytest.approx(-20_000)


def test_call_nobody_earns_nothing():
    called = np.zeros(1000, dtype=bool)
    assert campaign_profit(called, ACCEPTED) == 0


def test_perfect_foresight_ceiling():
    assert campaign_profit(ACCEPTED == 1, ACCEPTED) == pytest.approx(432_000)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="differ in length"):
        campaign_profit(np.ones(10, dtype=bool), ACCEPTED)
