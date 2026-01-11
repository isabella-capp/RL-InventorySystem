from dataclasses import dataclass
from typing import List

import numpy as np
import simpy

from src.simulation.product import Product
from src.simulation.warehouse import Warehouse


@dataclass
class PendingOrder:
    """Simple data structure for tracking pending orders."""

    product_id: int
    quantity: int
    lead_time: float


class SupplierManager:
    """
    Manages interactions with suppliers.
    """

    def __init__(
        self,
        products: List[Product],
        warehouse: Warehouse,
        env: simpy.Environment,
        rng: np.random.Generator,
    ):
        """
        Initialize supplier manager.

        Args:
            products: List of products managed
            warehouse: Warehouse to deliver to
            env: SimPy environment for scheduling
            rng: Random number generator for lead times
        """
        self.products = {p.product_id: p for p in products}
        self.warehouse = warehouse
        self.env = env
        self.rng = rng

    def place_order(self, product_id: int, quantity: int) -> None:
        """
        Place a replenishment order with the supplier.

        Args:
            product_id: Which product to order
            quantity: How many units to order
        """
        if product_id not in self.products:
            raise ValueError(f"Unknown product ID: {product_id}")

        if quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {quantity}")

        # Get product info
        product = self.products[product_id]

        # Sample lead time
        lead_time = product.sample_lead_time(self.rng)

        # Create order
        order = PendingOrder(
            product_id=product_id,
            quantity=quantity,
            lead_time=lead_time,
        )

        # Update warehouse outstanding
        self.warehouse.place_order(product_id, quantity)

        # Schedule delivery
        self.env.process(self._delivery_process(order))

    def _delivery_process(self, order: PendingOrder):
        """
        SimPy process for order delivery.

        Args:
            order: Order to deliver
        """
        # Wait for lead time
        yield self.env.timeout(order.lead_time)

        # Deliver to warehouse
        self.warehouse.receive_shipment(order.product_id, order.quantity)
