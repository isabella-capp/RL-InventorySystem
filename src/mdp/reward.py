from dataclasses import dataclass
from typing import Optional, Protocol
from src.mdp.state import Observation
from src.mdp.action import InventoryAction


@dataclass(frozen=True)
class CostParameters:
    """
    Cost parameters for the inventory system.

    From assignment specification:
    - K: Fixed ordering cost (setup cost)
    - i: Incremental cost per unit (purchase cost)
    - h: Holding cost per unit per day
    - pi: Shortage penalty per unit per day (backorder cost)
    """

    K: float = 10.0  # Fixed ordering cost
    i: float = 3.0  # Unit purchase cost
    h: float = 1.0  # Holding cost per unit per day
    pi: float = 7.0  # Shortage penalty per unit per day


@dataclass(frozen=True)
class CostComponents:
    """Breakdown of costs for analysis."""

    ordering_cost: float
    holding_cost: float
    shortage_cost: float

    @property
    def total_cost(self) -> float:
        """Get total cost."""
        return self.ordering_cost + self.holding_cost + self.shortage_cost


class RewardFunction(Protocol):
    """Protocol for reward functions."""

    def __call__(self, obs: Observation, action: InventoryAction) -> float:
        """
        Calculate reward for taking action in observation.

        Args:
            obs: Next observation (after action and events)
            action: Action taken

        Returns:
            Reward (negative cost)
        """
        ...

    def calculate_costs(
        self, obs: Observation, action: InventoryAction
    ) -> CostComponents:
        """
        Calculate detailed cost breakdown.

        Args:
            obs: Next observation
            action: Action taken

        Returns:
            Detailed cost components
        """
        ...


class StandardRewardFunction:
    """
    Standard reward function as specified in the assignment.

    R(t) = -[C_order(t) + C_holding(t) + C_shortage(t)]

    Where:
    - C_order = K·𝟙{q>0} + i·q  (for each product)
    - C_holding = h·max(0, I)  (for each product)
    - C_shortage = π·max(0, -I)  (for each product)
    """

    def __init__(self, params: Optional[CostParameters] = None):
        """
        Initialize reward function.

        Args:
            params: Cost parameters (uses defaults if None)
        """
        self.params = params or CostParameters()

    def calculate_costs(
        self, obs: Observation, action: InventoryAction
    ) -> CostComponents:
        """
        Calculate detailed cost breakdown.

        Args:
            obs: Next observation (after action effects)
            action: Action taken

        Returns:
            Detailed cost components
        """
        # Ordering costs (K + i·q for each product ordered)
        ordering_cost = 0.0
        for j in range(2):
            q = action.order_quantities[j]
            if q > 0:
                ordering_cost += self.params.K + self.params.i * q

        # Holding costs (h·I⁺ for each product)
        holding_cost = 0.0
        for j in range(2):
            on_hand = obs.get_on_hand_inventory(j)
            holding_cost += self.params.h * on_hand

        # Shortage costs (π·I⁻ for each product)
        shortage_cost = 0.0
        for j in range(2):
            backorders = obs.get_backorders(j)
            shortage_cost += self.params.pi * backorders

        return CostComponents(
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
        )

    def __call__(self, obs: Observation, action: InventoryAction) -> float:
        """
        Calculate reward (negative cost).

        Args:
            obs: Next observation
            action: Action taken

        Returns:
            Reward = -total_cost
        """
        costs = self.calculate_costs(obs, action)
        return -costs.total_cost


# Default reward function factory
def create_default_reward_function() -> StandardRewardFunction:
    """Create reward function with default parameters."""
    return StandardRewardFunction(CostParameters())
