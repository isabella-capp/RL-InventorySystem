from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Observation:
    """
    Single time-step observation of the inventory system.

    Represents the system state at one decision epoch.

    Attributes:
        net_inventory: Tuple of net inventory levels (I_0, I_1) where:
            - Positive: on-hand inventory
            - Negative: backorders (unfulfilled demand)
        outstanding_orders: Tuple of outstanding order quantities (O_0, O_1)
    """

    net_inventory: Tuple[int, int]
    outstanding_orders: Tuple[int, int]

    def to_array(self) -> NDArray[np.float32]:
        """Convert observation to numpy array [I_0, O_0, I_1, O_1]."""
        return np.array(
            [
                self.net_inventory[0],
                self.outstanding_orders[0],
                self.net_inventory[1],
                self.outstanding_orders[1],
            ],
            dtype=np.float32,
        )

    def get_on_hand_inventory(self, product_id: int) -> int:
        """Get on-hand inventory for a product (max(0, net_inventory))."""
        return max(0, self.net_inventory[product_id])

    def get_backorders(self, product_id: int) -> int:
        """Get backorders for a product (max(0, -net_inventory))."""
        return max(0, -self.net_inventory[product_id])

    def get_inventory_position(self, product_id: int) -> int:
        """
        Get inventory position for a product.

        IP = Net_Inventory + Outstanding_Orders

        Note: Net_Inventory already accounts for backorders (I - B),
        so IP = (I - B) + O = I - B + O
        """
        return self.net_inventory[product_id] + self.outstanding_orders[product_id]


@dataclass(frozen=True)
class InventoryState:
    """
    Full MDP state using frame stacking.

    State is a sequence of k+1 recent observations to approximate Markov property.

    Attributes:
        observations: List of observations [o_t, o_{t-1}, ..., o_{t-k}]
            Ordered from most recent (index 0) to oldest (index k)
    """

    observations: Tuple[Observation, ...]  # Immutable sequence

    def __post_init__(self):
        """Validate state."""
        if len(self.observations) == 0:
            raise ValueError("State must contain at least one observation")

    @property
    def k(self) -> int:
        """Get frame stack depth (number of historical frames)."""
        return len(self.observations) - 1

    @property
    def current_observation(self) -> Observation:
        """Get the most recent observation (o_t)."""
        return self.observations[0]

    def to_array(self) -> NDArray[np.float32]:
        """
        Convert state to flat numpy array.

        Returns:
            1D array of shape ((k+1) * 4,) containing all stacked observations
        """
        return np.concatenate([obs.to_array() for obs in self.observations])

    @property
    def shape(self) -> Tuple[int]:
        """Get shape of state array."""
        return (len(self.observations) * 4,)

    def get_on_hand_inventory(self, product_id: int) -> int:
        """Get current on-hand inventory for a product."""
        return self.current_observation.get_on_hand_inventory(product_id)

    def get_backorders(self, product_id: int) -> int:
        """Get current backorders for a product."""
        return self.current_observation.get_backorders(product_id)

    def get_outstanding_orders(self, product_id: int) -> int:
        """Get current outstanding orders for a product."""
        return self.current_observation.outstanding_orders[product_id]

    def get_inventory_position(self, product_id: int) -> int:
        """Get current inventory position for a product."""
        return self.current_observation.get_inventory_position(product_id)


class StateSpace:
    """
    Configuration for the state space.

    Defines bounds, normalization parameters, and utility functions.
    """

    def __init__(
        self,
        k: int = 3,
        net_inventory_min: int = -100,
        net_inventory_max: int = 200,
        max_outstanding: int = 150,
    ):
        """
        Initialize state space configuration.

        Args:
            k: Number of historical frames to stack (default: 3)
            net_inventory_min: Minimum net inventory (backorder limit)
            net_inventory_max: Maximum net inventory (on-hand limit)
            max_outstanding: Maximum outstanding orders
        """
        self.k = k
        self.net_inventory_min = net_inventory_min
        self.net_inventory_max = net_inventory_max
        self.max_outstanding = max_outstanding

        # State dimension: (k+1) observations × 4 features per observation
        self.dim = (k + 1) * 4

    @property
    def shape(self) -> Tuple[int]:
        """Get shape of state space."""
        return (self.dim,)

    def is_valid_observation(self, obs: Observation) -> bool:
        """Check if observation is within bounds."""
        for net_inv, outstanding in zip(obs.net_inventory, obs.outstanding_orders):
            if net_inv < self.net_inventory_min or net_inv > self.net_inventory_max:
                return False
            if outstanding < 0 or outstanding > self.max_outstanding:
                return False
        return True

    def is_valid_state(self, state: InventoryState) -> bool:
        """Check if state is valid."""
        if len(state.observations) != self.k + 1:
            return False
        return all(self.is_valid_observation(obs) for obs in state.observations)

    def sample_observation(
        self, random_state: Optional[np.random.Generator] = None
    ) -> Observation:
        """Sample a random valid observation."""
        if random_state is None:
            random_state = np.random.default_rng()

        net_inv_0 = random_state.integers(
            self.net_inventory_min, self.net_inventory_max + 1
        )
        net_inv_1 = random_state.integers(
            self.net_inventory_min, self.net_inventory_max + 1
        )
        out_0 = random_state.integers(0, self.max_outstanding + 1)
        out_1 = random_state.integers(0, self.max_outstanding + 1)

        return Observation(
            net_inventory=(int(net_inv_0), int(net_inv_1)),
            outstanding_orders=(int(out_0), int(out_1)),
        )

    def sample_state(
        self, random_state: Optional[np.random.Generator] = None
    ) -> InventoryState:
        """Sample a random valid state with frame stacking."""
        observations = tuple(
            self.sample_observation(random_state) for _ in range(self.k + 1)
        )
        return InventoryState(observations=observations)


def create_observation(
    net_inventory_0: int,
    net_inventory_1: int,
    outstanding_0: int = 0,
    outstanding_1: int = 0,
) -> Observation:
    """
    Convenience function to create an observation.

    Args:
        net_inventory_0: Net inventory for product 0
        net_inventory_1: Net inventory for product 1
        outstanding_0: Outstanding orders for product 0
        outstanding_1: Outstanding orders for product 1

    Returns:
        Observation object
    """
    return Observation(
        net_inventory=(net_inventory_0, net_inventory_1),
        outstanding_orders=(outstanding_0, outstanding_1),
    )


def create_initial_state(
    net_inventory_0: int = 50, net_inventory_1: int = 50, k: int = 3
) -> InventoryState:
    """
    Create an initial state with the same observation repeated k+1 times.

    This is used at the start of an episode before history is built up.

    Args:
        net_inventory_0: Initial net inventory for product 0
        net_inventory_1: Initial net inventory for product 1
        k: Number of historical frames

    Returns:
        Initial state with stacked observations
    """
    initial_obs = create_observation(net_inventory_0, net_inventory_1, 0, 0)
    observations = tuple(initial_obs for _ in range(k + 1))
    return InventoryState(observations=observations)


def update_state_with_observation(
    state: InventoryState, new_observation: Observation
) -> InventoryState:
    """
    Create new state by adding a new observation and dropping the oldest.

    Args:
        state: Current state
        new_observation: New observation to add

    Returns:
        New state with updated observation stack
    """
    # Add new observation at the front, drop the last one
    new_observations = (new_observation,) + state.observations[:-1]
    return InventoryState(observations=new_observations)
