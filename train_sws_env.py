import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from typing import List, Dict, Tuple, Any, Optional
from StageWiseHENGymEnv import StageWiseHENGymEnv


def main():
    """Main function to train and test the PPO agent."""
    
    # --- 1. Define the HEN Problem ---
    hot_streams = [[443, 333, 30], [423, 303, 15]]
    cold_streams = [[293, 408, 20], [353, 413, 40]]
    num_stages = max(len(hot_streams), len(cold_streams)) # A good heuristic

    # --- 2. Instantiate the Environment ---
    env = StageWiseHENGymEnv(
                streams_filepath="data/example1/streams.csv",
                utilities_filepath="data/example1/utilities.csv",
                matches_cost_filepath="data/example1/matches_cost.csv",
                forbidden_matches_filepath="data/example1/forbidden_matches.csv",
                num_stages=3,
                min_deltaT=10.0
            )
    
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment check passed!")

    # --- 3. Define and Train the PPO Agent ---
    device = 'mps' if th.backends.mps.is_available() else 'auto'
    print(f"Using device: {device}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./hen_ppo_tensorboard/",
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    # Training duration. This environment is complex and benefits from longer training.
    total_timesteps = 200000 
    print(f"\nStarting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    model.save("ppo_hen_model_stagewise")
    print("Training complete. Model saved as ppo_hen_model_stagewise.zip")

    # --- 4. Test the Trained Agent ---
    print("\n--- Testing Trained Agent ---")
    
    obs, _ = env.reset()
    
    for i in range(num_stages + 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            print("\nFinal network design:")
            env.render()
            break

if __name__ == '__main__':
    main()
