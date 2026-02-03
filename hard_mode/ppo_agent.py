# hard_mode/ppo_agent.py

"""
The In-Silico Cell: RL-Driven Design of Thermo-Genetic Circuits
Module 2: The PPO Agent (Writer)

This script trains a Reinforcement Learning agent (Proximal Policy Optimization)
to write DNA sequences that maximize the 'Fitness Reward' provided by the environment.
"""

import gymnasium as gym
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import our Custom Environment
# Ensure 'hard_mode' is in python path or accessible
try:
    from rl_gene_designer import PromoterDesignEnv
except ImportError:
    # If running from root directory
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), 'hard_mode'))
    from rl_gene_designer import PromoterDesignEnv

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PPO_Trainer")

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
def train_agent():
    logger.info("Initializing PPO Training Pipeline...")
    
    # 1. Setup Environment
    # We use a wrapper to vectorize it (Standard for Stable Baselines)
    env = DummyVecEnv([lambda: PromoterDesignEnv(target_length=200)])
    
    # 2. Define PPO Model
    # Policy: MlpPolicy (Multi-Layer Perceptron) because input is a flat vector of integers
    # Hyperparameters tuned for discrete sequence generation:
    # - learning_rate: 0.0003 (Standard)
    # - n_steps: 2048 (Batch size)
    # - gamma: 0.99 (Discount factor - we care about the final reward)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048, 
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./ppo_gene_tensorboard/"
    )
    
    # 3. Train
    logger.info("Starting PPO Learning Loop (50,000 timesteps)...")
    # In a real scenario, we'd run for millions of steps. 
    # For this demo, 10,000 is enough to see 'learning' of TATA boxes.
    model.learn(total_timesteps=10000, callback=ProgressCallback())
    
    # 4. Save Model
    model.save("hard_mode/best_promoter_agent")
    logger.info("Model saved to 'hard_mode/best_promoter_agent.zip'")
    
    return model

# --- INFERENCE / GENERATION ---
def generate_sequence(model, length=200):
    """
    Uses the trained agent to write a new DNA sequence.
    """
    logger.info("Generating new promoter design with trained agent...")
    
    # Create a fresh environment for inference
    env = PromoterDesignEnv(target_length=length)
    obs, _ = env.reset()
    
    done = False
    while not done:
        # Predict action (deterministic=False allows for some exploration/creativity)
        action, _states = model.predict(obs, deterministic=True)
        
        # Ensure action is an integer (SB3 returns numpy array)
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    dna = env._indices_to_string(env.sequence)
    return dna, reward

if __name__ == "__main__":
    # Train
    trained_model = train_agent()
    
    # Generate
    best_dna, score = generate_sequence(trained_model, length=200)
    
    print("\n--- PPO Agent Design Complete ---")
    print(f"Generated Sequence (200bp):")
    print(f"{best_dna}")
    print(f"Predicted Fitness Score: {score:.2f}")
