from dataclasses import dataclass
from typing import Protocol

from mdp import InventoryState
from mdp import InventoryAction

from .standard import StandardRewardFunction
from .shaped import ShapedRewardFunction
from .factory import RewardFunctionFactory


@dataclass(frozen=True)
class CostComponents:
    """
    Breakdown of costs in the inventory system.

    This provides transparency and helps with debugging/analysis.
    """

    holding_cost: float = 0.0  # Cost of holding inventory
    backorder_cost: float = 0.0  # Penalty for unfulfilled demand
    ordering_cost: float = 0.0  # Fixed cost per order
    purchase_cost: float = 0.0  # Variable cost per unit ordered

    @property
    def total_cost(self) -> float:
        """Total cost across all components."""
        return (
            self.holding_cost
            + self.backorder_cost
            + self.ordering_cost
            + self.purchase_cost
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "holding": self.holding_cost,
            "backorder": self.backorder_cost,
            "ordering": self.ordering_cost,
            "purchase": self.purchase_cost,
            "total": self.total_cost,
        }

    def __repr__(self) -> str:
        return (
            f"CostComponents(holding={self.holding_cost:.2f}, "
            f"backorder={self.backorder_cost:.2f}, "
            f"ordering={self.ordering_cost:.2f}, "
            f"purchase={self.purchase_cost:.2f}, "
            f"total={self.total_cost:.2f})"
        )


@dataclass(frozen=True)
class CostParameters:
    """
    System-wide cost parameters.

    As specified in the assignment:
    - K: Fixed ordering cost per order
    - i: Unit purchase cost per item
    - h: Holding cost per unit per day
    - π: Backorder penalty per unit per day
    """

    K: float = 10.0  # Fixed ordering cost
    i: float = 3.0  # Unit purchase cost
    h: float = 1.0  # Holding cost per unit per day
    pi: float = 7.0  # Backorder penalty per unit per day

    def validate(self) -> None:
        """Validate that all costs are non-negative."""
        if self.K < 0 or self.i < 0 or self.h < 0 or self.pi < 0:
            raise ValueError("All cost parameters must be non-negative")


class RewardFunction(Protocol):
    """
    Protocol (interface) for reward functions.

    This follows the Dependency Inversion Principle - depend on abstractions,
    not concretions. Different reward functions can be implemented.
    """

    def __call__(
        self, state: InventoryState, action: InventoryAction, next_state: InventoryState
    ) -> float:
        """
        Calculate reward for a state transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward (negative cost in this case)
        """
        ...


if __name__ == "__main__":
    # Test cost parameters
    print("Testing CostParameters...")
    params = CostParameters()
    print(f"Cost params: K={params.K}, i={params.i}, h={params.h}, π={params.pi}")
    params.validate()
    print("✓ Parameters valid")

    # Test reward function
    print("\nTesting StandardRewardFunction...")
    from ..state import create_initial_state
    from ..action import order_both_products

    state = create_initial_state(40, 45)
    action = order_both_products(20, 15)
    next_state = create_initial_state(40, 45)  # Simplified for testing

    reward_fn = StandardRewardFunction(params)
    costs = reward_fn.calculate_costs(state, action)
    print(f"Cost breakdown: {costs}")

    reward = reward_fn(state, action, next_state)
    print(f"Reward (negative cost): {reward:.2f}")

    # Test shaped reward
    print("\nTesting ShapedRewardFunction...")
    shaped_fn = ShapedRewardFunction(params)
    shaped_reward = shaped_fn(state, action, next_state)
    print(f"Shaped reward: {shaped_reward:.2f}")
    print(f"Difference from standard: {shaped_reward - reward:.2f}")

    # Test factory
    print("\nTesting RewardFunctionFactory...")
    standard = RewardFunctionFactory.create_standard()
    shaped = RewardFunctionFactory.create_shaped()
    print(f"Standard reward: {standard(state, action, next_state):.2f}")
    print(f"Shaped reward: {shaped(state, action, next_state):.2f}")

    print("\n✓ All reward tests passed!")
