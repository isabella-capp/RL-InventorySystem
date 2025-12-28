from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from enum import Enum


class EventType(Enum):
    """Types of events in the simulation."""

    CUSTOMER_ARRIVAL = "customer_arrival"
    ORDER_ARRIVAL = "order_arrival"


@dataclass
class Event:
    """
    Discrete event representation.

    Following the Data Transfer Object (DTO) pattern.
    """

    time: float
    event_type: EventType
    data: Dict[str, Any]

    def __lt__(self, other: "Event") -> bool:
        """Required for heap queue operations."""
        return self.time < other.time


@dataclass
class OutstandingOrder:
    """Represents an order in transit."""

    product_id: int
    quantity: int
    order_time: float
    arrival_time: float


@dataclass
class SystemParameters:
    """
    System-wide parameters for simulation.

    As specified in assignment:
    - Demand inter-arrival: Exponential(λ=0.1)
    - Product 0: Demand ~ {1:1/6, 2:1/3, 3:1/3, 4:1/6}, LeadTime ~ U(0.5, 1.0)
    - Product 1: Demand ~ {5:1/8, 4:1/2, 3:1/4, 2:1/8}, LeadTime ~ U(0.2, 0.7)
    """

    lambda_arrival: float = 0.1

    # Product 0 demand distribution
    demand_0_values: Tuple[int, ...] = (1, 2, 3, 4)
    demand_0_probs: Tuple[float, ...] = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
    lead_time_0_min: float = 0.5
    lead_time_0_max: float = 1.0

    # Product 1 demand distribution
    demand_1_values: Tuple[int, ...] = (5, 4, 3, 2)
    demand_1_probs: Tuple[float, ...] = (1 / 8, 1 / 2, 1 / 4, 1 / 8)
    lead_time_1_min: float = 0.2
    lead_time_1_max: float = 0.7

    @classmethod
    def create_default(cls) -> "SystemParameters":
        """Factory method for default parameters."""
        return cls()

    def validate(self) -> None:
        """Validate parameter consistency."""
        # Check demand distributions sum to 1
        if not np.isclose(sum(self.demand_0_probs), 1.0):
            raise ValueError("Product 0 demand probabilities must sum to 1")
        if not np.isclose(sum(self.demand_1_probs), 1.0):
            raise ValueError("Product 1 demand probabilities must sum to 1")

        # Check lead time bounds
        if self.lead_time_0_min >= self.lead_time_0_max:
            raise ValueError("Product 0 lead time min must be less than max")
        if self.lead_time_1_min >= self.lead_time_1_max:
            raise ValueError("Product 1 lead time min must be less than max")
