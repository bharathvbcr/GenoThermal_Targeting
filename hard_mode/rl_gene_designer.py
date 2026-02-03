# hard_mode/rl_gene_designer.py

"""
The In-Silico Cell: RL-Driven Design of Thermo-Genetic Circuits
Module 1: Custom RL Environment & AlphaGenome Judge

The RL agent builds DNA sequences one nucleotide at a time.
The "Judge" scores completed sequences via the AlphaGenome API
(tumour expression minus healthy expression).

Compute split:
  - Environment (step/reset/obs):  LOCAL  (trivial array ops)
  - Reward evaluation:             API    (AlphaGenome predict_sequence_fitness)
  - Fallback reward:               LOCAL  (motif heuristic, for dev/CI)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import os
import sys
import random

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RL_Gene_Designer")

# ---------------------------------------------------------------------------
# Import project API client
# ---------------------------------------------------------------------------
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

try:
    from alphagenome_utils import AlphaGenomeClient
    _AG_AVAILABLE = True
except ImportError:
    _AG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Judge — delegates to AlphaGenome API or local fallback
# ---------------------------------------------------------------------------
class SequenceJudge:
    """
    Evaluates a completed DNA sequence and returns a scalar reward.

    API mode:   Calls AlphaGenomeClient.predict_sequence_fitness()
                (tumour CAGE signal minus healthy CAGE signal).
    Local mode: Fast motif heuristic (TATA, HSE, GC penalty).
    """

    def __init__(self, api_key=None, mode="Auto"):
        """
        Args:
            api_key: Optional API key.
            mode: "API", "Local", or "Auto" (uses API if possible).
        """
        self._client = None
        force_local = (mode == "Local")
        
        if _AG_AVAILABLE:
            self._client = AlphaGenomeClient(api_key=api_key, force_local=force_local)
            if self._client._mode == "API":
                logger.info("Judge: Using AlphaGenome API for reward.")
            else:
                logger.info("Judge: Using local heuristic reward.")
                self._client = None
        else:
            logger.info("Judge: alphagenome_utils not found — local heuristic.")

    def score(self, dna_sequence):
        """Return a scalar fitness reward for the completed sequence."""
        if self._client is not None:
            try:
                return self._client.predict_sequence_fitness(dna_sequence)
            except Exception as e:
                logger.warning(f"API error, falling back to local: {e}")
        return self._local_score(dna_sequence)

    @staticmethod
    def _local_score(dna):
        """Fast local heuristic (no network)."""
        score = 0.0
        if "TATA" in dna:
            score += 5.0
        if "GAA" in dna and "TTC" in dna:
            score += 8.0
        gc = (dna.count('G') + dna.count('C')) / max(len(dna), 1)
        if gc > 0.7:
            score -= 5.0
        score += random.uniform(-1, 1)
        return score


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------
class PromoterDesignEnv(gym.Env):
    """
    Custom Gymnasium environment for DNA promoter design.

    State:   int32 array of length target_length (-1 = empty, 0-3 = ACGT)
    Action:  Discrete(4)  — 0=A, 1=C, 2=G, 3=T
    Reward:  Sparse — only at episode end (full sequence evaluated by Judge)

    All env logic runs locally (array ops).  Only the reward call
    hits the API (once per episode).
    """

    def __init__(self, target_length=50, api_key=None, mode="Auto"):
        super().__init__()
        self.target_length = target_length
        self.current_step = 0
        self.sequence = []

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1, high=3, shape=(target_length,), dtype=np.int32
        )

        self.judge = SequenceJudge(api_key=api_key, mode=mode)
        logger.info(f"RL Environment initialised. Target length: {target_length}bp")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.sequence = [-1] * self.target_length
        return np.array(self.sequence, dtype=np.int32), {}

    def step(self, action):
        self.sequence[self.current_step] = action
        self.current_step += 1

        terminated = (self.current_step >= self.target_length)
        reward = 0.0

        if terminated:
            dna = self._indices_to_string(self.sequence)
            reward = self.judge.score(dna)
            if reward > 5.0:
                logger.info(f"HIGH FITNESS: {dna[:30]}... | {reward:.2f}")

        obs = np.array(self.sequence, dtype=np.int32)
        return obs, reward, terminated, False, {"length": self.current_step}

    def _indices_to_string(self, indices):
        mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        return "".join(mapping.get(i, 'N') for i in indices if i != -1)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Testing RL Environment (random agent) ---")
    env = PromoterDesignEnv(target_length=20)
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
    print(f"DNA:    {env._indices_to_string(obs)}")
    print(f"Reward: {reward:.4f}")
    print("Test complete.")
