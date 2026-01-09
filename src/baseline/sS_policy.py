from dataclasses import dataclass
from typing import Tuple

from src.mdp import Action, State


@dataclass
class sSPolicy:
    """
    (s,S) policy for N-product inventory system.

    Rule: If inventory_position ≤ s, order (S - IP) units

    Parameters:
        params: Tuple of (s, S) for each product
                e.g., ((9, 21), (8, 20)) for 2 products
                s = reorder point (when to order)
                S = order-up-to level (target inventory position)
    """

    params: Tuple[Tuple[int, int], ...]  # ((s_0, S_0), (s_1, S_1), ...)

    def __post_init__(self):
        """Validate parameters."""
        for i, (s, S) in enumerate(self.params):
            if S <= s:
                raise ValueError(f"Product {i}: S ({S}) must be > s ({s})")

    @property
    def num_products(self) -> int:
        """Number of products this policy manages."""
        return len(self.params)

    def get_s_min(self, product_id: int) -> int:
        """Get reorder point for a product."""
        return self.params[product_id][0]

    def get_s_max(self, product_id: int) -> int:
        """Get order-up-to level for a product."""
        return self.params[product_id][1]

    def __call__(self, state: State) -> Action:
        """
        Make ordering decision based on current state.

        Args:
            state: Current inventory state

        Returns:
            Action with order quantities for all products
        """
        if state.num_products != self.num_products:
            raise ValueError(
                f"State has {state.num_products} products, "
                f"policy configured for {self.num_products}"
            )

        order_quantities = []
        for i in range(self.num_products):
            ip = state.get_inventory_position(i)
            s_min, s_max = self.params[i]
            q = self._get_order(ip, s_min, s_max)
            order_quantities.append(q)

        return Action(tuple(order_quantities))

    def _get_order(self, inventory_position: int, s_min: int, s_max: int) -> int:
        """Get order quantity for a single product."""
        if inventory_position <= s_min:
            return max(0, s_max - inventory_position)
        return 0

    def as_dict(self) -> dict:
        """Return policy parameters as dictionary."""
        return {
            f"product_{i}": {"s": self.params[i][0], "S": self.params[i][1]}
            for i in range(self.num_products)
        }

    def __str__(self) -> str:
        products = ", ".join(
            f"P{i}=(s_min={self.params[i][0]}, s_max={self.params[i][1]})"
            for i in range(self.num_products)
        )
        return f"(s,S) Policy: {products}"

    def __repr__(self) -> str:
        return f"sSPolicy(params={self.params})"


def create_sS_policy(
    *product_params: Tuple[int, int],
) -> sSPolicy:
    """
    Factory function to create (s,S) policy.

    Args:
        *product_params: (s, S) tuples for each product

    Returns:
        Configured sSPolicy instance

    Example:
        policy = create_sS_policy((9, 21), (8, 20))
    """
    return sSPolicy(params=product_params)
