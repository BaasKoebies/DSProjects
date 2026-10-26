"""The money behind the calling decision.

All amounts are in rand and are assumptions, stated in the memo:
calling a customer costs R500, and an accepted loan earns R5,000 margin.
Profit is measured against calling nobody, which earns R0.

Would-borrow-anyway share (s): the data has no control group, so we cannot see
who would have taken a loan without a call. If a share s of acceptors would
have borrowed anyway, a call only earns margin on the other (1 - s). s = 0
means every loan is caused by the call. A uniform s flatters targeting, because
the customers most likely to accept may also be the most likely to borrow
anyway; the memo says so.
"""

import numpy as np
import numpy.typing as npt

CONTACT_COST = 500.0
MARGIN = 5000.0


def break_even_threshold(contact_cost: float = CONTACT_COST, margin: float = MARGIN, borrow_anyway: float = 0.0) -> float:
    """Acceptance probability at which a call exactly pays for itself.

    A call earns p * (1 - s) * margin and costs contact_cost, so it pays when
    p >= contact_cost / ((1 - s) * margin). R500 / R5,000 gives 10%.
    """
    if not 0.0 <= borrow_anyway < 1.0:
        raise ValueError(f"borrow_anyway must be in [0, 1), got {borrow_anyway}")
    return contact_cost / ((1.0 - borrow_anyway) * margin)


def should_contact(p: npt.ArrayLike, threshold: float) -> npt.NDArray[np.bool_]:
    """Contact rule: call every customer whose calibrated probability is at or above the threshold."""
    return np.asarray(p) >= threshold


def campaign_profit(
    called: npt.ArrayLike,
    accepted: npt.ArrayLike,
    contact_cost: float = CONTACT_COST,
    margin: float = MARGIN,
    borrow_anyway: float = 0.0,
) -> float:
    """Total profit of a calling rule, against calling nobody.

    called: True for each customer the rule calls.
    accepted: 1 if that customer accepted the offer, else 0.
    Each call costs contact_cost; each accepted call earns (1 - s) * margin.
    """
    called = np.asarray(called, dtype=bool)
    accepted = np.asarray(accepted)
    if called.shape != accepted.shape:
        raise ValueError(f"called and accepted differ in length: {called.shape} vs {accepted.shape}")
    earned = (1.0 - borrow_anyway) * margin * accepted[called].sum()
    return float(earned - contact_cost * called.sum())


def per_thousand(total: float, customers: int) -> float:
    """Scale a profit to 'per 1,000 customers', the memo's headline unit."""
    return total / customers * 1000.0
