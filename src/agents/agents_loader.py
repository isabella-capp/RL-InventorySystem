from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from stable_baselines3 import DQN, PPO

from src.agents.base import Agent
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.environment import InventoryEnvironment


class AgentsLoader:
    """
    Load and manage trained RL agents.

    Supports multiple agent types and provides a unified interface
    for loading pre-trained models.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
    ):
        """
        Initialize agent loader.

        Args:
            models_dir: Directory containing saved models (default: ./models)
        """
        self.models_dir = Path(models_dir) if models_dir else Path("./models")

        self.agent_types = {
            "dqn": self._load_dqn,
            "ppo": self._load_ppo,
        }

        self._loaded_agents: Dict[str, Tuple[Agent, Dict]] = {}

    def load_agent(
        self, model_id: str, agent_type: str = "dqn"
    ) -> Tuple[Agent, Dict[str, int]]:
        """
        Load a trained agent with its environment metadata.

        Args:
            model_id: Model identifier (e.g., 'dqn_baseline', 'dqn_20260108_104709')
            agent_type: Type of agent ('dqn', 'a2c', 'ppo', etc.)

        Returns:
            Tuple of (agent instance, metadata dict with k and Q_max)

        Raises:
            ValueError: If agent_type not supported
            FileNotFoundError: If model file doesn't exist
        """
        # Check cache
        cache_key = f"{agent_type}_{model_id}"
        if cache_key in self._loaded_agents:
            print(f"✅ Using cached agent: {cache_key}")
            agent, metadata = self._loaded_agents[cache_key]
            return agent, metadata

        # Load agent
        if agent_type not in self.agent_types:
            raise ValueError(
                f"Unknown agent type '{agent_type}'. "
                f"Available: {list(self.agent_types.keys())}"
            )

        print(f"📥 Loading {agent_type.upper()} agent: {model_id}...")
        agent, metadata = self.agent_types[agent_type](model_id)

        # Cache and return
        self._loaded_agents[cache_key] = (agent, metadata)
        print("✅ Agent loaded successfully")
        return agent, metadata

    def _load_generic(
        self, model_id: str, agent_class: type, model_class: type
    ) -> Tuple[Agent, Dict[str, int]]:
        """Generic method to load any agent type with metadata."""

        model_path = self.models_dir / f"{model_id}"
        metadata_path = self.models_dir / f"{model_id}_metadata.json"

        if not model_path.exists() and not Path(str(model_path) + ".zip").exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        agent_env = InventoryEnvironment(
            k=metadata["k"],
            Q_max=metadata["Q_max"],
            episode_length=metadata.get("episode_length", 1000),
            random_seed=42,
        )

        agent = agent_class(env=agent_env, verbose=0)

        agent.model = model_class.load(str(model_path), env=agent_env)

        return agent, metadata

    def _load_dqn(self, model_id: str) -> Tuple[Agent, Dict[str, int]]:
        """Load DQN agent from disk with metadata."""
        return self._load_generic(model_id, DQNAgent, DQN)

    def _load_ppo(self, model_id: str) -> Tuple[Agent, Dict[str, int]]:
        """Load PPO agent from disk with metadata."""
        return self._load_generic(model_id, PPOAgent, PPO)

    def list_available_models(self, agent_type: Optional[str] = None) -> list[str]:
        """
        List available trained models.

        Args:
            agent_type: Filter by agent type (default: all types)

        Returns:
            List of available model IDs
        """
        if not self.models_dir.exists():
            print(f"⚠️  Models directory not found: {self.models_dir}")
            return []

        models = []

        # Search for .zip files (SB3 format)
        for model_file in self.models_dir.glob("*.zip"):
            model_id = model_file.stem

            # Filter by agent type if specified
            if agent_type is None or model_id.startswith(agent_type):
                models.append(model_id)

        return sorted(models)

    def __repr__(self) -> str:
        return (
            f"AgentsLoader(models_dir={self.models_dir}, "
            f"supported_types={list(self.agent_types.keys())})"
        )
