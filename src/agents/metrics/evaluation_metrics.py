from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.mdp import CostParameters


class EvaluationMetrics:
    """Class for generating evaluation-related plots for RL agents."""

    def __init__(self, figsize: tuple = (12, 5)):
        """
        Initialize EvaluationPlots.

        Args:
            figsize: Default figure size for plots.
            cost_params: Cost parameters for the inventory system.
        """
        self.figsize = figsize
        self.cost_params = CostParameters()

    def plot_welch_procedure(
        self,
        test_episodes: List[Dict],
        window_size: int = 25,
        title: str = "Warm-up Period Analysis",
        show: bool = True,
    ) -> tuple:
        """
        Plot Welch's Graphical Procedure for warm-up detection.

        Args:
            test_episodes: List of episode data dictionaries with 'total_daily_cost'.
            window_size: Half-width of window (total window = 2*window_size + 1).
            title: Plot title.
            show: Whether to display the plot.

        Returns:
            Tuple of (n_days, n_reps, warmup_detected).
        """
        # Extract daily costs into a 2D array: days x replications
        daily_costs_matrix = np.array(
            [ep["total_daily_cost"] for ep in test_episodes]
        ).T

        n_days = daily_costs_matrix.shape[0]
        n_reps = daily_costs_matrix.shape[1]

        # Average across replications for each day
        Y_bar = np.mean(daily_costs_matrix, axis=1)

        # Apply centered moving average
        Y_smoothed = self._compute_moving_average(Y_bar.tolist(), window_size)
        smoothed_days = np.arange(len(Y_smoothed))

        # Detect warmup using second half as steady-state estimate
        if len(Y_smoothed) < 100:
            warmup_detected = len(Y_smoothed) // 2
        else:
            # Use second half as steady-state estimate
            steady_state_start = len(Y_smoothed) // 2
            ss_mean = np.mean(Y_smoothed[steady_state_start:])
            threshold = 0.05 * ss_mean  # 5% tolerance

            # Find first point where 50-day window stays within tolerance
            warmup_detected = None
            for i in range(50, len(Y_smoothed) - 50):
                window = Y_smoothed[i : i + 50]
                if all(abs(val - ss_mean) <= threshold for val in window):
                    warmup_detected = i
                    break

            # Fallback if no plateau found
            if warmup_detected is None:
                warmup_detected = n_days // 2

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Raw average
        ax = axes[0]
        ax.plot(
            range(n_days), Y_bar, "b-", linewidth=1, alpha=0.7, label="Mean Daily Cost"
        )
        ax.set_xlabel("Day (Time Step)", fontsize=11)
        ax.set_ylabel("Average Daily Cost ($)", fontsize=11)
        ax.set_title(
            "Average Daily Cost Across Replications", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Right: Smoothed
        ax = axes[1]
        ax.plot(
            smoothed_days,
            Y_smoothed,
            "b-",
            linewidth=2,
            label=f"Moving Average (w={window_size})",
        )
        ax.axhline(
            y=ss_mean,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Steady-State Mean: ${ss_mean:.2f}",
        )
        # Mark detected warmup period
        ax.axvline(
            x=warmup_detected,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Suggested Warmup: {warmup_detected} days",
        )
        ax.set_xlabel("Day (Time Step)", fontsize=11)
        ax.set_ylabel("Smoothed Daily Cost ($)", fontsize=11)
        ax.set_title(
            "Welch's Procedure: Identify Warm-up Period", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if show:
            plt.show()

        # Print summary
        print("=" * 60)
        print(f"📊 Welch's Procedure Summary:")
        print("=" * 60)
        print(f"   Total replications: {n_reps}")
        print(f"\n✅ Suggested Warmup Length: {warmup_detected} days")
        print(f"   (Detected where curve stays within 5% of steady-state)")

        print(f"\n💡 Tip: Visually verify the green line on the right plot.")
        print(f"   Adjust manually if needed based on your domain knowledge.")

        return n_days, n_reps, warmup_detected

    def _compute_moving_average(self, data: list, window_size: int) -> list:
        """
        Compute centered moving average with expanding window at start.

        Args:
            data: Daily cost values
            window_size: Half-width of window (total window = 2*window_size + 1)

        Returns:
            Smoothed values
        """
        moving_avg = []

        for i in range(len(data) - window_size):
            if i < window_size:
                # Use expanding window at start
                moving_avg.append(sum(data[: 2 * i + 1]) / (2 * i + 1))
            else:
                # Use full centered window
                window = data[i - window_size : i + window_size + 1]
                moving_avg.append(sum(window) / len(window))

        return moving_avg

    def plot_cost_breakdown_by_product(
        self,
        test_episodes: List[Dict],
        warmup_length: int,
        n_days: int,
        title: str = "Cost Component Breakdown by Product",
        show: bool = True,
    ) -> None:
        """
        Plot cost component breakdown separated by product.

        Args:
            test_episodes: List of episode data dictionaries.
            warmup_length: Number of warm-up days to exclude.
            n_days: Total number of days per episode.
            title: Plot title.
            show: Whether to display the plot.

        Returns:
            matplotlib Figure object.
        """
        costs_p1, costs_p2 = self._compute_costs_by_product(
            test_episodes, warmup_length
        )

        # Calculate daily means (divide by number of steady-state days)
        n_ss_days = n_days - warmup_length
        p1_ord = np.mean(costs_p1["ordering"]) / n_ss_days
        p1_hold = np.mean(costs_p1["holding"]) / n_ss_days
        p1_short = np.mean(costs_p1["shortage"]) / n_ss_days
        p2_ord = np.mean(costs_p2["ordering"]) / n_ss_days
        p2_hold = np.mean(costs_p2["holding"]) / n_ss_days
        p2_short = np.mean(costs_p2["shortage"]) / n_ss_days

        # Plot
        x = np.arange(3)
        width = 0.35
        labels = [
            "Ordering\n(K=$10 + i=$3)",
            "Holding\n(h=$1/unit/day)",
            "Shortage\n(π=$7/unit/day)",
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(
            x - width / 2,
            [p1_ord, p1_hold, p1_short],
            width,
            label="Product 1",
            color="steelblue",
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + width / 2,
            [p2_ord, p2_hold, p2_short],
            width,
            label="Product 2",
            color="forestgreen",
            edgecolor="black",
        )

        ax.set_ylabel("Average Daily Cost ($)", fontsize=12)
        ax.set_title(
            f"{title} (Steady-State, days {warmup_length+1}-{n_days})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"${height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"${height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        if show:
            plt.show()

        # Print daily cost breakdown
        total_p1 = p1_ord + p1_hold + p1_short
        total_p2 = p2_ord + p2_hold + p2_short
        total_all = total_p1 + total_p2

        print("=" * 60)
        print(f"📊 Steady-State Average Daily Cost (days {warmup_length+1}-{n_days}):")
        print("=" * 60)
        print(f"\n  Product 1: ${total_p1:.2f} / day")
        print(f"    Ordering: ${p1_ord:.2f} ({p1_ord/total_p1*100:.1f}%)")
        print(f"    Holding:  ${p1_hold:.2f} ({p1_hold/total_p1*100:.1f}%)")
        print(f"    Shortage: ${p1_short:.2f} ({p1_short/total_p1*100:.1f}%)")
        print(f"\n  Product 2: ${total_p2:.2f} / day")
        print(f"    Ordering: ${p2_ord:.2f} ({p2_ord/total_p2*100:.1f}%)")
        print(f"    Holding:  ${p2_hold:.2f} ({p2_hold/total_p2*100:.1f}%)")
        print(f"    Shortage: ${p2_short:.2f} ({p2_short/total_p2*100:.1f}%)\n")
        print("-" * 60)
        print(f"\n  TOTAL: ${total_all:.2f} / day")

    def _compute_costs_by_product(self, episodes: List[Dict], warmup: int) -> tuple:
        """Compute cost components separately for each product."""
        costs_p1 = {"ordering": [], "holding": [], "shortage": []}
        costs_p2 = {"ordering": [], "holding": [], "shortage": []}
        params = self.cost_params

        for ep in episodes:
            q0_ss = ep["q0"][warmup:]
            q1_ss = ep["q1"][warmup:]
            inv0_ss = ep["net_inv_0"][warmup:]
            inv1_ss = ep["net_inv_1"][warmup:]

            # Product 1 costs
            ord_p1 = sum(params.K + params.i * q for q in q0_ss if q > 0)
            hold_p1 = sum(params.h * max(0, inv) for inv in inv0_ss)
            short_p1 = sum(params.pi * max(0, -inv) for inv in inv0_ss)

            # Product 2 costs
            ord_p2 = sum(params.K + params.i * q for q in q1_ss if q > 0)
            hold_p2 = sum(params.h * max(0, inv) for inv in inv1_ss)
            short_p2 = sum(params.pi * max(0, -inv) for inv in inv1_ss)

            costs_p1["ordering"].append(ord_p1)
            costs_p1["holding"].append(hold_p1)
            costs_p1["shortage"].append(short_p1)
            costs_p2["ordering"].append(ord_p2)
            costs_p2["holding"].append(hold_p2)
            costs_p2["shortage"].append(short_p2)

        return costs_p1, costs_p2

    def plot_operational_timeseries(
        self,
        episode_data: Dict,
        title: str = "Operational Time Series",
        show: bool = True,
        start_day: int = 0,
        end_day: Optional[int] = None,
        max_days: int = 200,
    ) -> None:
        """
        Plot operational time series showing inventory, orders, and demand.

        Args:
            episode_data: Episode data dictionary with inventory, order, and demand data.
            title: Plot title.
            show: Whether to display the plot.
            start_day: Starting day for the plot (default: 0).
            end_day: Ending day for the plot (default: None, uses start_day + max_days).
            max_days: Maximum number of days to plot if end_day is None (default: 200).

        Returns:
            matplotlib Figure object.
        """
        # Determine the range of days to plot
        total_days = len(episode_data["net_inv_0"])
        if end_day is None:
            end_day = min(start_day + max_days, total_days)
        else:
            end_day = min(end_day, total_days)

        # Slice the data
        day_slice = slice(start_day, end_day)
        days = np.arange(start_day, end_day)

        has_demand = "demand_0" in episode_data and "demand_1" in episode_data

        n_rows = 3 if has_demand else 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), sharex=True)

        # Product 1 - Net Inventory
        ax = axes[0, 0]
        ax.plot(days, episode_data["net_inv_0"][day_slice], "b-", linewidth=1.5)
        ax.axhline(
            y=0, color="k", linestyle="-", linewidth=2, label="Stockout threshold"
        )
        ax.fill_between(
            days,
            episode_data["net_inv_0"][day_slice],
            0,
            where=[x < 0 for x in episode_data["net_inv_0"][day_slice]],
            color="red",
            alpha=0.3,
            label="Backlog",
        )
        ax.set_ylabel("Net Inventory (P1)", fontsize=11)
        ax.set_title("Product 1: Inventory Level", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Product 1 - Replenishment
        ax = axes[1, 0]
        ax.bar(
            days, episode_data["q0"][day_slice], color="steelblue", alpha=0.8, width=1.0
        )
        ax.set_ylabel("Order Quantity (P1)", fontsize=11)
        ax.set_title("Product 1: Replenishment Actions", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if not has_demand:
            ax.set_xlabel("Day", fontsize=11)

        # Product 2 - Net Inventory
        ax = axes[0, 1]
        ax.plot(days, episode_data["net_inv_1"][day_slice], "g-", linewidth=1.5)
        ax.axhline(
            y=0, color="k", linestyle="-", linewidth=2, label="Stockout threshold"
        )
        ax.fill_between(
            days,
            episode_data["net_inv_1"][day_slice],
            0,
            where=[x < 0 for x in episode_data["net_inv_1"][day_slice]],
            color="red",
            alpha=0.3,
            label="Backlog",
        )
        ax.set_ylabel("Net Inventory (P2)", fontsize=11)
        ax.set_title("Product 2: Inventory Level", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Product 2 - Replenishment
        ax = axes[1, 1]
        ax.bar(
            days,
            episode_data["q1"][day_slice],
            color="forestgreen",
            alpha=0.8,
            width=1.0,
        )
        ax.set_ylabel("Order Quantity (P2)", fontsize=11)
        ax.set_title("Product 2: Replenishment Actions", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if not has_demand:
            ax.set_xlabel("Day", fontsize=11)

        # Demand plots (if available)
        if has_demand:
            # Product 1 - Demand
            ax = axes[2, 0]
            ax.bar(
                days,
                episode_data["demand_0"][day_slice],
                color="coral",
                alpha=0.8,
                width=1.0,
            )
            ax.axhline(
                y=np.mean(episode_data["demand_0"][day_slice]),
                color="darkred",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {np.mean(episode_data['demand_0'][day_slice]):.1f}",
            )
            ax.set_xlabel("Day", fontsize=11)
            ax.set_ylabel("Daily Demand (P1)", fontsize=11)
            ax.set_title("Product 1: Customer Demand", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

            # Product 2 - Demand
            ax = axes[2, 1]
            ax.bar(
                days,
                episode_data["demand_1"][day_slice],
                color="darkorange",
                alpha=0.8,
                width=1.0,
            )
            ax.axhline(
                y=np.mean(episode_data["demand_1"][day_slice]),
                color="darkred",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {np.mean(episode_data['demand_1'][day_slice]):.1f}",
            )
            ax.set_xlabel("Day", fontsize=11)
            ax.set_ylabel("Daily Demand (P2)", fontsize=11)
            ax.set_title("Product 2: Customer Demand", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"{title} (Days {start_day}-{end_day})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if show:
            plt.show()

    def plot_inventory_histogram(
        self,
        test_episodes: List[Dict],
        title: str = "Inventory Distribution - Risk Profile",
        show: bool = True,
        warmup_length: int = 0,
    ) -> None:
        """
        Plot inventory distribution histogram.

        Args:
            test_episodes: List of episode data dictionaries.
            title: Plot title.
            show: Whether to display the plot.
            warmup_length: Number of days to exclude from the beginning (default: 0).

        Returns:
            matplotlib Figure object.
        """
        # Collect all inventory levels (excluding warmup)
        all_inv_p1 = []
        all_inv_p2 = []
        for ep in test_episodes:
            all_inv_p1.extend(ep["net_inv_0"][warmup_length:])
            all_inv_p2.extend(ep["net_inv_1"][warmup_length:])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Product 1 Histogram
        ax = axes[0]
        bins = np.linspace(min(all_inv_p1) - 5, max(all_inv_p1) + 5, 40)
        n, bin_edges, patches = ax.hist(
            all_inv_p1, bins=bins, edgecolor="black", alpha=0.7
        )
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            patch.set_facecolor("red" if bin_center < 0 else "green")
        ax.axvline(x=0, color="k", linestyle="-", linewidth=2)
        ax.set_xlabel("Net Inventory Level", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(
            "Product 1: Inventory Distribution", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")
        legend_elements = [
            Patch(facecolor="red", label="Backlog (I < 0)"),
            Patch(facecolor="green", label="On-Hand (I > 0)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Product 2 Histogram
        ax = axes[1]
        bins = np.linspace(min(all_inv_p2) - 5, max(all_inv_p2) + 5, 40)
        n, bin_edges, patches = ax.hist(
            all_inv_p2, bins=bins, edgecolor="black", alpha=0.7
        )
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            patch.set_facecolor("red" if bin_center < 0 else "green")
        ax.axvline(x=0, color="k", linestyle="-", linewidth=2)
        ax.set_xlabel("Net Inventory Level", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(
            "Product 2: Inventory Distribution", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if show:
            plt.show()

        # Print service level stats
        stockout_p1 = sum(1 for x in all_inv_p1 if x < 0) / len(all_inv_p1)
        stockout_p2 = sum(1 for x in all_inv_p2 if x < 0) / len(all_inv_p2)

        print("=" * 60)
        print(f"⚙️Service Level (% days without stockout):")
        print("=" * 60)
        print(f"  Product 1: {(1-stockout_p1)*100:.1f}%")
        print(f"  Product 2: {(1-stockout_p2)*100:.1f}%")

    def plot_daily_cost_analysis(
        self,
        test_episodes: List[Dict],
        warmup_length: int = 0,
        title: str = "Daily Cost Analysis",
        show: bool = True,
    ) -> None:
        """
        Plot aggregated daily cost statistics across all episodes.

        Args:
            test_episodes: List of episode data dictionaries.
            warmup_length: Number of days to exclude from the beginning.
            title: Plot title.
            show: Whether to display the plot.
        """
        # Extract daily costs matrix (days x episodes) excluding warmup
        daily_costs_matrix = np.array(
            [ep["total_daily_cost"][warmup_length:] for ep in test_episodes]
        ).T

        n_days = daily_costs_matrix.shape[0]
        days = np.arange(warmup_length, warmup_length + n_days)

        # Compute statistics
        mean_daily = np.mean(daily_costs_matrix, axis=1)
        std_daily = np.std(daily_costs_matrix, axis=1)
        percentile_25 = np.percentile(daily_costs_matrix, 25, axis=1)
        percentile_75 = np.percentile(daily_costs_matrix, 75, axis=1)
        min_daily = np.min(daily_costs_matrix, axis=1)
        max_daily = np.max(daily_costs_matrix, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Mean with confidence band
        ax = axes[0]
        ax.plot(days, mean_daily, "b-", linewidth=2, label="Mean")
        ax.fill_between(
            days,
            mean_daily - std_daily,
            mean_daily + std_daily,
            alpha=0.3,
            color="blue",
            label="±1 Std Dev",
        )
        ax.axhline(
            y=np.mean(mean_daily),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Overall Mean: ${np.mean(mean_daily):.2f}",
        )
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Daily Cost ($)", fontsize=11)
        ax.set_title(
            "Mean Daily Cost with Standard Deviation", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Right: Percentiles with min/max
        ax = axes[1]
        ax.plot(days, mean_daily, "b-", linewidth=2, label="Median (Mean)")
        ax.fill_between(
            days,
            percentile_25,
            percentile_75,
            alpha=0.4,
            color="green",
            label="25th-75th Percentile",
        )
        ax.fill_between(
            days,
            min_daily,
            max_daily,
            alpha=0.2,
            color="gray",
            label="Min-Max Range",
        )
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Daily Cost ($)", fontsize=11)
        ax.set_title(
            "Daily Cost Distribution (Percentiles)", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"{title} (Steady-State: Days {warmup_length+1}-{warmup_length+n_days})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if show:
            plt.show()

        # Print summary statistics
        print("=" * 60)
        print(f"📊 Daily Cost Statistics (Steady-State):")
        print("=" * 60)
        print(f"  Mean:   ${np.mean(mean_daily):.2f} ± ${np.mean(std_daily):.2f}")
        print(f"  Median: ${np.median(mean_daily):.2f}")
        print(f"  Min:    ${np.min(min_daily):.2f}")
        print(f"  Max:    ${np.max(max_daily):.2f}")
        print(f"  Range:  ${np.max(max_daily) - np.min(min_daily):.2f}")

    def print_evaluation_statistics(
        self,
        test_episodes: List[Dict],
        warmup_length: int,
    ) -> Dict:
        """
        Compute and print comprehensive evaluation statistics.

        Args:
            test_episodes: List of episode data dictionaries.
            warmup_length: Number of days to exclude from the beginning.

        Returns:
            Dictionary with computed statistics.
        """
        # Extract steady-state costs
        all_daily_costs = np.concatenate(
            [ep["total_daily_cost"][warmup_length:] for ep in test_episodes]
        )
        episode_totals = np.array(
            [sum(ep["total_daily_cost"][warmup_length:]) for ep in test_episodes]
        )

        n_ss_days = len(test_episodes[0]["total_daily_cost"]) - warmup_length

        # Compute statistics
        stats = {
            "daily_mean": np.mean(all_daily_costs),
            "daily_std": np.std(all_daily_costs, ddof=1),
            "daily_median": np.median(all_daily_costs),
            "daily_min": np.min(all_daily_costs),
            "daily_max": np.max(all_daily_costs),
            "episode_mean": np.mean(episode_totals),
            "episode_std": np.std(episode_totals, ddof=1),
            "n_steady_state_days": n_ss_days,
        }

        # Confidence intervals (95%)
        stats["daily_ci"] = 1.96 * stats["daily_std"] / np.sqrt(len(all_daily_costs))
        stats["episode_ci"] = 1.96 * stats["episode_std"] / np.sqrt(len(episode_totals))

        # Print brief summary
        print("=" * 60)
        print(f"📊 COST SUMMARY ({len(test_episodes)} episodes, {n_ss_days} steady-state days)")
        print("=" * 60)
        print(f"  Daily Cost: ${stats['daily_mean']:.2f} ± ${stats['daily_ci']:.2f} (95% CI)")
        print(f"  Episode Total: ${stats['episode_mean']:.2f} ± ${stats['episode_ci']:.2f}")

        return stats
