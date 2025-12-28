from typing import Protocol

import numpy as np

from .events import SystemParameters


class DemandGenerator(Protocol):
    """
    Protocol for demand generation strategies.

    Following the Strategy Pattern - different demand patterns can be plugged in.
    """

    def generate_demand(
        self, product_id: int, random_state: np.random.Generator
    ) -> int:
        """Generate demand quantity for a product."""
        ...

    def generate_interarrival_time(self, random_state: np.random.Generator) -> float:
        """Generate time until next customer arrival."""
        ...


class LeadTimeGenerator(Protocol):
    """
    Protocol for lead time generation.

    Following the Strategy Pattern.
    """

    def generate_lead_time(
        self, product_id: int, random_state: np.random.Generator
    ) -> float:
        """Generate lead time for an order."""
        ...


class StandardDemandGenerator:
    """
    Standard demand generator following assignment specifications.

    Implements the DemandGenerator protocol.
    """

    def __init__(self, params: SystemParameters):
        params.validate()
        self.params = params

        # Store demand distributions
        self.demand_dists = {
            0: (params.demand_0_values, params.demand_0_probs),
            1: (params.demand_1_values, params.demand_1_probs),
        }

    def generate_demand(
        self, product_id: int, random_state: np.random.Generator
    ) -> int:
        """Generate demand quantity for a product."""
        values, probs = self.demand_dists[product_id]
        return int(random_state.choice(values, p=probs))

    def generate_interarrival_time(self, random_state: np.random.Generator) -> float:
        """Generate exponential inter-arrival time."""
        return random_state.exponential(1.0 / self.params.lambda_arrival)


class StandardLeadTimeGenerator:
    """
    Standard lead time generator following assignment specifications.

    Implements the LeadTimeGenerator protocol.
    """

    def __init__(self, params: SystemParameters):
        params.validate()
        self.params = params

        # Store lead time ranges
        self.lead_time_ranges = {
            0: (params.lead_time_0_min, params.lead_time_0_max),
            1: (params.lead_time_1_min, params.lead_time_1_max),
        }

    def generate_lead_time(
        self, product_id: int, random_state: np.random.Generator
    ) -> float:
        """Generate uniform lead time for a product."""
        min_lt, max_lt = self.lead_time_ranges[product_id]
        return random_state.uniform(min_lt, max_lt)
