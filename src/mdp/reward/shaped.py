from ..action import InventoryAction
from ..state import InventoryState
from .base import CostParameters
from .standard import StandardRewardFunction


class ShapedRewardFunction(StandardRewardFunction):
    """
    Reward shaping variant that adds potential-based shaping.

    Shaped reward: R'(s,a,s') = R(s,a,s') + γ*Φ(s') - Φ(s)

    where Φ is a potential function. This can speed up learning without
    changing the optimal policy (Ng et al., 1999).

    This is an example of the Template Method pattern - base calculation
    from parent, with additional shaping logic.
    """

    def __init__(self, cost_params: CostParameters, gamma: float = 0.99):
        super().__init__(cost_params)
        self.gamma = gamma

    def potential(self, state: InventoryState) -> float:
        """
        Potential function based on inventory position.

        Higher potential for better inventory positions (less likely to stockout).
        """
        # Simple potential: reward being near target inventory
        target_inventory = 50
        potential = 0.0

        for product_id in range(2):
            inv_position = state.get_inventory_position(product_id)
            # Negative quadratic penalty for deviation from target
            deviation = abs(inv_position - target_inventory)
            potential -= 0.01 * (deviation**2)

        return potential

    def __call__(
        self, state: InventoryState, action: InventoryAction, next_state: InventoryState
    ) -> float:
        """Calculate shaped reward."""
        base_reward = super().__call__(state, action, next_state)
        shaping = self.gamma * self.potential(next_state) - self.potential(state)
        return base_reward + shaping
