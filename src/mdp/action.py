from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass(frozen=True)
class InventoryAction:
    """
    Represents a replenishment decision.

    Attributes:
        order_quantities: Tuple (q_0, q_1) where q_j ∈ [0, Q_max]
    """

    order_quantities: Tuple[int, int]

    def __post_init__(self):
        """Validate action."""
        if len(self.order_quantities) != 2:
            raise ValueError("Action must have exactly 2 order quantities")
        for q in self.order_quantities:
            if q < 0:
                raise ValueError(f"Order quantity cannot be negative: {q}")


class ActionSpace:
    """
    Discrete action space for inventory management.

    Each product can be ordered in quantities from 0 to Q_max.
    Total action space size: (Q_max + 1)²
    """

    def __init__(self, Q_max: int = 20):
        """
        Initialize action space.

        Args:
            Q_max: Maximum order quantity per product (default: 20)
        """
        if Q_max <= 0:
            raise ValueError(f"Q_max must be positive, got {Q_max}")

        self.Q_max = Q_max
        self.possible_quantities = list(range(Q_max + 1))

        # Generate all possible actions
        self.actions: List[InventoryAction] = []
        self.action_to_index: dict = {}

        idx = 0
        for q0 in self.possible_quantities:
            for q1 in self.possible_quantities:
                action = InventoryAction(order_quantities=(q0, q1))
                self.actions.append(action)
                self.action_to_index[action] = idx
                idx += 1

        self.n = len(self.actions)

    def sample(
        self, random_state: Optional[np.random.Generator] = None
    ) -> InventoryAction:
        """Sample a random action."""
        if random_state is None:
            random_state = np.random.default_rng()
        idx = random_state.integers(0, self.n)
        return self.actions[idx]

    def get_action(self, index: int) -> InventoryAction:
        """Get action by index."""
        if not 0 <= index < self.n:
            raise ValueError(f"Index {index} out of range [0, {self.n})")
        return self.actions[index]

    def get_index(self, action: InventoryAction) -> int:
        """Get index of an action."""
        if action not in self.action_to_index:
            raise ValueError(f"Action {action} not in action space")
        return self.action_to_index[action]

    def is_valid(self, action: InventoryAction) -> bool:
        """Check if action is valid."""
        return (
            0 <= action.order_quantities[0] <= self.Q_max
            and 0 <= action.order_quantities[1] <= self.Q_max
        )


def order_both_products(quantity_0: int, quantity_1: int) -> InventoryAction:
    """
    Convenience function to create an action.

    Args:
        quantity_0: Order quantity for product 0
        quantity_1: Order quantity for product 1

    Returns:
        InventoryAction object
    """
    return InventoryAction(order_quantities=(quantity_0, quantity_1))


def no_order_action() -> InventoryAction:
    """Create a no-order action (0, 0)."""
    return InventoryAction(order_quantities=(0, 0))
