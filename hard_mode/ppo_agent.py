# hard_mode/ppo_agent.py

"""
The In-Silico Cell: RL-Driven Design of Thermo-Genetic Circuits
Module 2: The PPO Agent (Writer)

This script trains a Reinforcement Learning agent (Proximal Policy Optimization)
to write DNA sequences that maximize the 'Fitness Reward' provided by the environment.
"""

import os
import sys
import gymnasium as gym
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Logging Setup — defined BEFORE the import fallback so the except branch can log safely
# (previously logger was referenced here before it existed -> NameError on the fallback path).
logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("PPO_Trainer")

# Import our Custom Environment (ensure 'hard_mode' is on the python path or accessible)
try:
    from rl_gene_designer import PromoterDesignEnv
except ImportError:
    logger.warning("rl_gene_designer not on path; retrying from hard_mode/ subdirectory.")
    sys.path.append(os.path.join(os.getcwd(), 'hard_mode'))
    from rl_gene_designer import PromoterDesignEnv
    logger.info("rl_gene_designer loaded via hard_mode/ path.")

# --- CUSTOM CALLBACK ---
class ProgressCallback(BaseCallback):
    """
    Logs training progress and saves the best model found so far.
    """
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        # Check every 1000 steps
        if self.n_calls % 1000 == 0:
            # Retrieve training rewards (approximate)
            # Access the underlying environment info if needed, or rely on PPO's built-in logging
            # Here we just log that we are still running.
            logger.info(f"Step {self.n_calls}: Training in progress...")
        return True

# --- TRAINING PIPELINE ---
def train_agent(total_timesteps=10000):
    logger.info("Initializing PPO Training Pipeline...")

    logger.info("train_agent: total_timesteps=%d", total_timesteps)
    env = DummyVecEnv([lambda: PromoterDesignEnv(target_length=200)])
    logger.info("DummyVecEnv created (target_length=200).")

    n_steps = min(2048, total_timesteps)
    batch_size = min(64, n_steps)
    logger.info("PPO config: n_steps=%d, batch_size=%d, lr=0.0003, gamma=0.99", n_steps, batch_size)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,
        tensorboard_log="./outputs/ppo_gene_tensorboard/"
    )
    logger.info("PPO model built (MlpPolicy).")

    logger.info("Starting PPO Learning Loop (%d timesteps)...", total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=ProgressCallback())
    logger.info("PPO training complete.")

    model.save("hard_mode/best_promoter_agent")
    logger.info("Model saved to 'hard_mode/best_promoter_agent.zip'")
    
    return model

# --- INFERENCE / GENERATION ---
def generate_sequence(model, length=200):
    """
    Uses the trained agent to write a new DNA sequence.
    """
    logger.info("Generating new promoter design with trained agent (length=%d)...", length)

    env = PromoterDesignEnv(target_length=length)
    obs, _ = env.reset()
    logger.debug("Environment reset; starting deterministic rollout.")

    done = False
    step_count = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)

        if isinstance(action, np.ndarray):
            action = int(action.item())

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated

    dna = env._indices_to_string(env.sequence)
    logger.info("Generation complete: %d steps, fitness=%.4f", step_count, reward)
    logger.info("Generated sequence (first 30bp): %s...", dna[:30])
    return dna, reward

if __name__ == "__main__":
    trained_model = train_agent()

    best_dna, score = generate_sequence(trained_model, length=200)

    logger.info("--- PPO Agent Design Complete ---")
    logger.info("Generated Sequence (200bp): %s", best_dna)
    logger.info("Predicted Fitness Score: %.2f", score)
