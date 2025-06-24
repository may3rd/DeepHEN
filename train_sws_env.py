import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from typing import List, Dict, Tuple, Any, Optional

# ==============================================================================
#  Helper Class to Store Problem Data
# ==============================================================================
class HENProblem:
    """
    Data class for the Heat Exchanger Network (HEN) problem definition.
    This class holds the initial stream data and cost parameters.
    """
    def __init__(
        self, 
        hot_streams: np.ndarray, 
        cold_streams: np.ndarray,
        hot_utility_cost_rate: float = 120.0,
        cold_utility_cost_rate: float = 80.0,
        exchanger_fixed_cost: float = 0.0,
        exchanger_area_cost_coeff: float = 1000.0,
        exchanger_area_cost_exponent: float = 0.6
    ):
        self.hot_streams = np.array(hot_streams, dtype=np.float64)
        self.cold_streams = np.array(cold_streams, dtype=np.float64)
        self.n_hot = len(self.hot_streams)
        self.n_cold = len(self.cold_streams)
        
        self.hot_utility_cost_rate = hot_utility_cost_rate
        self.cold_utility_cost_rate = cold_utility_cost_rate
        self.exchanger_fixed_cost = exchanger_fixed_cost
        self.exchanger_area_cost_coeff = exchanger_area_cost_coeff
        self.exchanger_area_cost_exponent = exchanger_area_cost_exponent
        
        self.total_hot_duty = (self.hot_streams[:, 0] - self.hot_streams[:, 1]) * self.hot_streams[:, 2]
        self.total_cold_duty = (self.cold_streams[:, 1] - self.cold_streams[:, 0]) * self.cold_streams[:, 2]
        
        self.max_duty = max(np.max(self.total_hot_duty), np.max(self.total_cold_duty)) if self.total_hot_duty.size > 0 and self.total_cold_duty.size > 0 else 1.0
        
        self.max_temp = np.max(self.hot_streams[:, 0]) if self.n_hot > 0 else 0
        self.min_temp = np.min(self.cold_streams[:, 0]) if self.n_cold > 0 else 0
        self.temp_range = self.max_temp - self.min_temp if self.max_temp > self.min_temp else 1.0


# ==============================================================================
#  The Final Stage-Wise Environment
# ==============================================================================
class StageWiseHENGymEnv(gym.Env):
    """
    A Gymnasium environment for HEN synthesis based on the stage-wise 
    superstructure model proposed by Yee and Grossmann (1990).
    This version includes normalized observations for stable training.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        hot_streams: List[List[float]],
        cold_streams: List[List[float]],
        num_stages: int,
        min_deltaT: float = 10.0,
        **kwargs # Pass cost parameters to HENProblem
    ):
        super().__init__()
        self.problem = HENProblem(np.array(hot_streams), np.array(cold_streams), **kwargs)
        self.num_stages = num_stages
        self.min_deltaT = min_deltaT
        self.tolerance = 1e-3

        # Observation space: Inlet Temps, Remaining Duties, Driving Forces, Time
        obs_size = (self.problem.n_hot + self.problem.n_cold) + \
                   (self.problem.n_hot + self.problem.n_cold) + \
                   (self.problem.n_hot * self.problem.n_cold) + 1 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Action space: Normalized heat loads for the current stage matrix
        action_size = self.problem.n_hot * self.problem.n_cold
        self.action_space = spaces.Box(low=0, high=1, shape=(action_size,), dtype=np.float32)
        self.max_q_value = self.problem.max_duty # Use max duty for scaling action

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        self.network_design = [np.zeros((self.problem.n_hot, self.problem.n_cold)) for _ in range(self.num_stages)]
        
        self.hot_temps_grid = np.zeros((self.problem.n_hot, self.num_stages + 1))
        self.cold_temps_grid = np.zeros((self.problem.n_cold, self.num_stages + 1))
        self._solve_temperature_grid() 
        
        self.current_stage_idx = 0
        
        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        """Constructs a fully NORMALIZED observation vector for the current stage."""
        obs_stage_idx = min(self.current_stage_idx, self.num_stages -1)

        # Get temperatures and normalize them
        hot_inlet_temps = self.hot_temps_grid[:, obs_stage_idx]
        cold_inlet_temps = self.cold_temps_grid[:, obs_stage_idx + 1]
        norm_hot_inlet_temps = (hot_inlet_temps - self.problem.min_temp) / self.problem.temp_range
        norm_cold_inlet_temps = (cold_inlet_temps - self.problem.min_temp) / self.problem.temp_range

        # Get remaining duties and normalize them
        final_hot_temps = self.hot_temps_grid[:, -1]
        final_cold_temps = self.cold_temps_grid[:, 0]
        hot_fcp = self.problem.hot_streams[:, 2]
        cold_fcp = self.problem.cold_streams[:, 2]
        hot_duty_remaining = (final_hot_temps - self.problem.hot_streams[:, 1]) * hot_fcp
        cold_duty_remaining = (self.problem.cold_streams[:, 1] - final_cold_temps) * cold_fcp
        
        norm_hot_duty_rem = hot_duty_remaining / (self.problem.max_duty + 1e-6)
        norm_cold_duty_rem = cold_duty_remaining / (self.problem.max_duty + 1e-6)
        
        # Add feasible driving forces
        t_hot_in_stage = self.hot_temps_grid[:, obs_stage_idx]
        t_cold_in_stage = self.cold_temps_grid[:, obs_stage_idx + 1]
        hot_temps_matrix = t_hot_in_stage[:, np.newaxis]
        cold_temps_matrix = t_cold_in_stage[np.newaxis, :]
        feasible_driving_force = np.maximum(0, hot_temps_matrix - cold_temps_matrix - self.min_deltaT)
        norm_driving_forces = feasible_driving_force / (self.problem.temp_range + 1e-6)

        # Get normalized time feature
        time_feature = np.array([self.current_stage_idx / self.num_stages])
        
        obs = np.concatenate([
            norm_hot_inlet_temps, 
            norm_cold_inlet_temps, 
            norm_hot_duty_rem, 
            norm_cold_duty_rem, 
            norm_driving_forces.flatten(),
            time_feature
        ])
        return obs.astype(np.float32).clip(0, 1)

    def _get_info(self) -> Dict:
        return { "current_stage": self.current_stage_idx }

    def _calculate_tac(self):
        """Helper to calculate the total CAPEX and OPEX for the current network design."""
        total_capex = 0
        # Calculate CAPEX by iterating through all placed exchangers
        for k, q_matrix in enumerate(self.network_design):
            t_hot_in, t_hot_out = self.hot_temps_grid[:, k], self.hot_temps_grid[:, k+1]
            t_cold_in, t_cold_out = self.cold_temps_grid[:, k+1], self.cold_temps_grid[:, k]
            for i in range(self.problem.n_hot):
                for j in range(self.problem.n_cold):
                    q_ij = q_matrix[i, j]
                    if q_ij > self.tolerance:
                        dT1, dT2 = t_hot_in[i] - t_cold_out[j], t_hot_out[i] - t_cold_in[j]
                        if dT1 < self.min_deltaT - self.tolerance or dT2 < self.min_deltaT - self.tolerance:
                            return float('inf'), float('inf') # Return infinite cost if invalid
                        lmtd = (dT1 * dT2 * (dT1 + dT2) / 2.0)**(1/3.0) if dT1 * dT2 > 0 else 1e-6
                        area = q_ij / (lmtd + 1e-6)
                        total_capex += (self.problem.exchanger_fixed_cost + 
                                        self.problem.exchanger_area_cost_coeff * (area ** self.problem.exchanger_area_cost_exponent))
        
        # Calculate OPEX
        hot_fcp = self.problem.hot_streams[:, 2]
        cold_fcp = self.problem.cold_streams[:, 2]
        final_hot_temps = self.hot_temps_grid[:, -1]
        final_cold_temps = self.cold_temps_grid[:, 0]
        cold_utility_needed = np.sum(np.maximum(0, (final_hot_temps - self.problem.hot_streams[:,1]) * hot_fcp))
        hot_utility_needed = np.sum(np.maximum(0, (self.problem.cold_streams[:,1] - final_cold_temps) * cold_fcp))
        opex = (hot_utility_needed * self.problem.hot_utility_cost_rate + 
                cold_utility_needed * self.problem.cold_utility_cost_rate)
                
        return total_capex, opex

    def _solve_temperature_grid(self):
        """Recalculates the entire temperature grid based on the current network design."""
        self.hot_temps_grid[:, 0] = self.problem.hot_streams[:, 0]
        self.cold_temps_grid[:, -1] = self.problem.cold_streams[:, 0]
        hot_fcp = self.problem.hot_streams[:, 2]
        cold_fcp = self.problem.cold_streams[:, 2]
        for k in range(self.num_stages):
            q_out_hot = np.sum(self.network_design[k], axis=1)
            self.hot_temps_grid[:, k + 1] = self.hot_temps_grid[:, k] - q_out_hot / hot_fcp
        for k in range(self.num_stages - 1, -1, -1):
            q_in_cold = np.sum(self.network_design[k], axis=0)
            self.cold_temps_grid[:, k] = self.cold_temps_grid[:, k + 1] + q_in_cold / cold_fcp

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Processes the design for one stage."""
        
        # --- FIX: Calculate TAC based on the state *before* this action ---
        capex_before, opex_before = self._calculate_tac()
        tac_before = capex_before + opex_before
        
        # De-normalize the agent's action and update the network design
        q_matrix = (action.reshape(self.problem.n_hot, self.problem.n_cold)) * self.max_q_value
        self.network_design[self.current_stage_idx] = q_matrix

        # Recalculate all temperatures based on the new design
        self._solve_temperature_grid()

        # --- FIX: Re-validate the entire network for thermodynamic feasibility ---
        capex_after, opex_after = self._calculate_tac()
        tac_after = capex_after + opex_after

        if tac_after == float('inf'): # Check if the network is now infeasible
             return self._get_observation(), -1e7, True, False, self._get_info()

        # The reward is the reduction in total annual cost
        reward = tac_before - tac_after
        
        self.current_stage_idx += 1
        done = self.current_stage_idx >= self.num_stages
            
        return self._get_observation(), reward, done, False, self._get_info()
    
    def render(self, mode='human'):
        """Renders the current state or a final summary of the environment."""
        if self.current_stage_idx >= self.num_stages:
            print("\n" + "="*50)
            print("---       EPISODE FINISHED: FINAL NETWORK SUMMARY       ---")
            print("="*50)
            
            total_capex, opex = self._calculate_tac()

            for k, q_matrix in enumerate(self.network_design):
                print(f"\n--- Stage {k} Details ---")
                
                t_hot_in_s, t_hot_out_s = self.hot_temps_grid[:, k], self.hot_temps_grid[:, k+1]
                t_cold_in_s, t_cold_out_s = self.cold_temps_grid[:, k+1], self.cold_temps_grid[:, k]
                
                print("Hot Streams:")
                for i in range(self.problem.n_hot):
                    print(f"  H{i}: Tin = {t_hot_in_s[i]:.2f}, Tout = {t_hot_out_s[i]:.2f}")
                
                print("Cold Streams:")
                for j in range(self.problem.n_cold):
                    print(f"  C{j}: Tin = {t_cold_in_s[j]:.2f}, Tout = {t_cold_out_s[j]:.2f}")
                
                print("Heat Exchangers (q_ij > 0):")
                has_exchanger = False
                for i in range(self.problem.n_hot):
                    for j in range(self.problem.n_cold):
                        if q_matrix[i,j] > self.tolerance:
                            print(f"  - Match H{i}-C{j}: Heat Duty = {q_matrix[i,j]:.2f} kW")
                            has_exchanger = True
                if not has_exchanger: print("  - No exchangers in this stage.")
            
            print("\n" + "-"*50)
            print("Final Cost Summary:")
            print(f"  - Total Capital Cost (CAPEX): ${total_capex:,.2f}")
            print(f"  - Total Operating Cost (OPEX): ${opex:,.2f}")
            print(f"  - Total Annual Cost (TAC):    ${(total_capex + opex):,.2f}")
            print("-"*50)

        else:
            print(f"--- Stage {self.current_stage_idx} ---")
            print("Inlet Hot Temps:", self.hot_temps_grid[:, self.current_stage_idx])
            print("Inlet Cold Temps:", self.cold_temps_grid[:, self.current_stage_idx + 1])


def main():
    """Main function to train and test the PPO agent."""
    
    # --- 1. Define the HEN Problem ---
    hot_streams = [[443, 333, 30], [423, 303, 15]]
    cold_streams = [[293, 408, 20], [353, 413, 40]]
    num_stages = max(len(hot_streams), len(cold_streams)) # A good heuristic

    # --- 2. Instantiate the Environment ---
    env = StageWiseHENGymEnv(
        hot_streams, 
        cold_streams, 
        num_stages, 
        min_deltaT=10.0,
        exchanger_fixed_cost=1000.0
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
