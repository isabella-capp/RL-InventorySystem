from dataclasses import dataclass
from typing import List, Tuple



@dataclass(frozen=True)
class Action:
    """
    Represents a replenishment decision for N products.

    Design Note: This is an MDP abstraction - the agent's decision.
    The simulation components (SupplierManager) convert this to orders.

    Attributes:
        order_quantities: Tuple (q_0, q_1, ..., q_n) where q_j ∈ [0, Q_max]

    Note: Length determines number of products. Assignment uses 2, but design
    supports N products for future scalability.
    """

    order_quantities: Tuple[int, ...]

    @property
    def num_products(self) -> int:
        """Number of products in this action."""
        return len(self.order_quantities)


class ActionSpace:
    """
    Discrete action space for inventory management.

    Each product can be ordered in quantities from 0 to Q_max.
    Total action space size: (Q_max + 1)²
    """

    def __init__(self, Q_max: int = 30):
        """
        Initialize action space.

        Args:
            Q_max: Maximum order quantity per product (default: 42)
        """
        if Q_max <= 0:
            raise ValueError(f"Q_max must be positive, got {Q_max}")

        self.Q_max = Q_max
        self.possible_quantities = list(range(Q_max + 1))

        # Generate all possible actions
        self.actions: List[Action] = []

        for q0 in self.possible_quantities:
            for q1 in self.possible_quantities:
                action = Action(order_quantities=(q0, q1))
                self.actions.append(action)

        self.n = len(self.actions)

    def get_action(self, index: int) -> Action:
        """Get action by index."""
        if not 0 <= index < self.n:
            raise ValueError(f"Index {index} out of range [0, {self.n})")
        return self.actions[index]


def order_both_products(quantity_0: int, quantity_1: int) -> Action:
    """
    Convenience function to create an action.

    Args:
        quantity_0: Order quantity for product 0
        quantity_1: Order quantity for product 1

    Returns:
        Action object
    """
    return Action(order_quantities=(quantity_0, quantity_1))
