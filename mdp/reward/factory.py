from typing import Optional
from .base import CostParameters
from .standard import StandardRewardFunction
from .shaped import ShapedRewardFunction


class RewardFunctionFactory:
    """
    Factory for creating different reward function configurations.
    """

    @staticmethod
    def create_standard(
        cost_params: Optional[CostParameters] = None,
    ) -> StandardRewardFunction:
        """Create standard cost-based reward function."""
        if cost_params is None:
            cost_params = CostParameters()
        return StandardRewardFunction(cost_params)

    @staticmethod
    def create_shaped(
        cost_params: Optional[CostParameters] = None, gamma: float = 0.99
    ) -> ShapedRewardFunction:
        """Create shaped reward function."""
        if cost_params is None:
            cost_params = CostParameters()
        return ShapedRewardFunction(cost_params, gamma)
