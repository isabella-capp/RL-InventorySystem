from src.mdp import InventoryAction, InventoryState

from .base import CostComponents, CostParameters


class StandardRewardFunction:
    """
    Standard cost-based reward function for inventory management.

    Reward = -1 * (holding_cost + backorder_cost + ordering_cost + purchase_cost)

    The negative sign converts minimization (costs) to maximization (rewards),
    which is the standard RL formulation.
    """

    def __init__(self, cost_params: CostParameters):
        """
        Args:
            cost_params: Cost parameters for the system
        """
        cost_params.validate()
        self.params = cost_params

    def calculate_costs(
        self, state: InventoryState, action: InventoryAction
    ) -> CostComponents:
        """
        Calculate detailed cost breakdown.

        This method computes costs based on:
        1. Current inventory levels (holding cost)
        2. Current backorders (backorder penalty)
        3. Action taken (ordering and purchase costs)
        """
        # Holding cost: h * inventory_level for each product
        holding_cost = sum(
            max(0, inv) * self.params.h for inv in state.inventory_levels
        )

        # Backorder cost: π * backorder_level for each product
        backorder_cost = sum(bo * self.params.pi for bo in state.backorders)

        # Ordering cost: K per product if order placed (quantity > 0)
        num_orders = sum(1 for qty in action.order_quantities if qty > 0)
        ordering_cost = num_orders * self.params.K

        # Purchase cost: i * quantity for each product
        purchase_cost = sum(qty * self.params.i for qty in action.order_quantities)

        return CostComponents(
            holding_cost=holding_cost,
            backorder_cost=backorder_cost,
            ordering_cost=ordering_cost,
            purchase_cost=purchase_cost,
        )

    def __call__(
        self, state: InventoryState, action: InventoryAction, next_state: InventoryState
    ) -> float:
        """
        Calculate reward (negative cost).

        Note: We use the current state for costs, as costs are incurred
        based on the current inventory position and the action taken.
        """
        costs = self.calculate_costs(state, action)
        return -costs.total_cost
