from src.mdp import ActionSpaceFactory, CostParameters
from .gym_env import InventoryEnvironment


class InventoryEnvironmentFactory:
    """
    Factory for creating pre-configured environments.

    Follows the Factory Pattern for convenient environment creation.
    """

    @staticmethod
    def create_default() -> InventoryEnvironment:
        """Create environment with default settings."""
        return InventoryEnvironment()

    @staticmethod
    def create_quick_training() -> InventoryEnvironment:
        """Create environment optimized for quick training (coarse actions)."""
        action_space = ActionSpaceFactory.create_coarse_action_space()
        return InventoryEnvironment(
            action_space_config=action_space, max_steps_per_episode=50
        )

    @staticmethod
    def create_detailed() -> InventoryEnvironment:
        """Create environment with fine-grained actions."""
        action_space = ActionSpaceFactory.create_fine_action_space()
        return InventoryEnvironment(
            action_space_config=action_space, max_steps_per_episode=100
        )

    @staticmethod
    def create_with_custom_costs(
        K: float, i: float, h: float, pi: float
    ) -> InventoryEnvironment:
        """Create environment with custom cost parameters."""
        cost_params = CostParameters(K=K, i=i, h=h, pi=pi)
        return InventoryEnvironment(cost_params=cost_params)
