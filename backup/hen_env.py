import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json

class HENProblem:
    """Data class for the HEN problem definition."""
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
        self.hot_streams_initial_pristine = np.array(hot_streams, dtype=np.float64)
        self.cold_streams_initial_pristine = np.array(cold_streams, dtype=np.float64)
        self.n_hot = len(self.hot_streams_initial_pristine)
        self.n_cold = len(self.cold_streams_initial_pristine)
        self.hot_utility_cost_rate = hot_utility_cost_rate
        self.cold_utility_cost_rate = cold_utility_cost_rate
        self.exchanger_fixed_cost = exchanger_fixed_cost
        self.exchanger_area_cost_coeff = exchanger_area_cost_coeff
        self.exchanger_area_cost_exponent = exchanger_area_cost_exponent
        
        # These will be set during environment reset
        self.hot_streams_initial = self.hot_streams_initial_pristine.copy()
        self.cold_streams_initial = self.cold_streams_initial_pristine.copy()
        self.total_hot_duty = np.zeros(self.n_hot)
        self.total_cold_duty = np.zeros(self.n_cold)

    def update_duties(self):
        """Recalculates total duties based on the current initial streams."""
        self.total_hot_duty = (self.hot_streams_initial[:, 0] - self.hot_streams_initial[:, 1]) * self.hot_streams_initial[:, 2]
        self.total_cold_duty = (self.cold_streams_initial[:, 1] - self.cold_streams_initial[:, 0]) * self.cold_streams_initial[:, 2]


class HENGymEnv(gym.Env):
    """
    An advanced Gymnasium environment for HEN design with modern RL features.
    
    Includes:
    - Dense, step-wise rewards based on TAC reduction.
    - Action masking to identify valid moves.
    - A time-aware observation space.
    - Controllable stochasticity for robust training.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        hot_streams: List[List[float]],
        cold_streams: List[List[float]],
        max_steps: int = 20,
        min_deltaT: float = 10.0,
        # --- NEW: Parameters to control stochasticity ---
        stochastic_mode: bool = False,
        temp_noise_std_dev: float = 0.5, # Absolute temperature noise
        fcp_noise_std_dev: float = 0.01, # Proportional FCp noise (1%)
        # --- ---
        **kwargs # Pass other cost parameters to HENProblem
    ):
        super().__init__()
        self.problem = HENProblem(np.array(hot_streams), np.array(cold_streams), **kwargs)
        self.max_steps = max_steps
        self.min_deltaT = min_deltaT
        self.tolerance = 1e-3
        # --- NEW: Store stochasticity settings ---
        self.stochastic_mode = stochastic_mode
        self.temp_noise_std_dev = temp_noise_std_dev
        self.fcp_noise_std_dev = fcp_noise_std_dev
        
        self.action_space = spaces.Dict({
            'i': spaces.Discrete(self.problem.n_hot),
            'j': spaces.Discrete(self.problem.n_cold),
            'Q_ratio': spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        })

        # --- NEW: Added 1 to obs_len for the time feature ---
        obs_len = (
            self.problem.n_hot + self.problem.n_cold + 
            self.problem.n_hot + self.problem.n_cold + 
            self.problem.n_hot * self.problem.n_cold + 
            3 + 2 + 1 # Added time feature
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32)
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        # --- NEW: Stochasticity implementation ---
        if self.stochastic_mode:
            # Add small random noise to the initial problem for this episode
            self.problem.hot_streams_initial = self.problem.hot_streams_initial_pristine.copy()
            self.problem.cold_streams_initial = self.problem.cold_streams_initial_pristine.copy()
            
            # Temperature noise
            self.problem.hot_streams_initial[:, :2] += self.np_random.normal(0, self.temp_noise_std_dev, size=(self.problem.n_hot, 2))
            self.problem.cold_streams_initial[:, :2] += self.np_random.normal(0, self.temp_noise_std_dev, size=(self.problem.n_cold, 2))
            
            # FCp noise
            self.problem.hot_streams_initial[:, 2] *= self.np_random.normal(1.0, self.fcp_noise_std_dev, size=self.problem.n_hot)
            self.problem.cold_streams_initial[:, 2] *= self.np_random.normal(1.0, self.fcp_noise_std_dev, size=self.problem.n_cold)
        
        # Update duties based on potentially noisy initial conditions
        self.problem.update_duties()
        
        # Initialize state
        self.step_count = 0
        self.exchanger_list = []
        self.cumulative_capex = 0.0
        self.hot_duty_remaining = self.problem.total_hot_duty.copy()
        self.cold_duty_remaining = self.problem.total_cold_duty.copy()
        self.current_hot_temps = self.problem.hot_streams_initial[:, 0].copy()
        self.current_cold_temps = self.problem.cold_streams_initial[:, 0].copy()
        self.last_action_tuple = (-1, -1, 0.0)
        
        obs = self._make_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: Dict[str, Any]):
        # --- NEW: Reward shaping - get cost *before* the action ---
        cost_before_action = self._get_info()["total_cost_with_penalty"]

        self.step_count += 1
        i, j = action['i'], action['j']
        Q_ratio = np.clip(float(action['Q_ratio']), 0, 1)
        self.last_action_tuple = (i, j, Q_ratio)
        
        T_hot_in, T_cold_in = self.current_hot_temps[i], self.current_cold_temps[j]
        Fh, Fc = self.problem.hot_streams_initial[i, 2], self.problem.cold_streams_initial[j, 2]
        max_Q_duty = min(self.hot_duty_remaining[i], self.cold_duty_remaining[j])
        
        # Check for invalid moves based on current state
        if max_Q_duty <= self.tolerance: return self._handle_invalid_step(-2.0, "No remaining duty.")
        Q = Q_ratio * max_Q_duty
        if Q < self.tolerance: return self._handle_invalid_step(-0.5, "Negligible heat transfer.")
        T_hot_out, T_cold_out = T_hot_in - Q / Fh, T_cold_in + Q / Fc
        if T_hot_in < T_cold_out + self.min_deltaT or T_hot_out < T_cold_in + self.min_deltaT:
            return self._handle_invalid_step(-10.0, "Temperature violation.")

        # If move is valid, update state
        self.hot_duty_remaining[i] -= Q
        self.cold_duty_remaining[j] -= Q
        self.current_hot_temps[i], self.current_cold_temps[j] = T_hot_out, T_cold_out
        dT1, dT2 = T_hot_in - T_cold_out, T_hot_out - T_cold_in
        lmtd = (dT1 * dT2 * (dT1 + dT2) / 2.0)**(1/3.0)
        area = Q / (lmtd + 1e-6)
        
        exchanger_cost = self.problem.exchanger_fixed_cost + self.problem.exchanger_area_cost_coeff * (area ** self.problem.exchanger_area_cost_exponent)
        self.cumulative_capex += exchanger_cost
        self.exchanger_list.append({'match': (i, j), 'Q': Q, 'Area': area, 'Cost': exchanger_cost})
        
        # --- NEW: Reward is the immediate cost reduction from this single step ---
        cost_after_action = self._get_info()["total_cost_with_penalty"]
        reward = cost_before_action - cost_after_action

        done = bool(self.step_count >= self.max_steps or (np.sum(self.hot_duty_remaining) < self.tolerance and np.sum(self.cold_duty_remaining) < self.tolerance))
        
        return self._make_obs(), reward, done, False, self._get_info()

    def _handle_invalid_step(self, penalty: float, reason: str):
        info = self._get_info()
        info["invalid_step_reason"] = reason
        # Invalid actions have a direct penalty. The reward is just this penalty.
        return self._make_obs(), penalty, False, False, info

    def _get_action_mask(self) -> np.ndarray:
        """
        --- NEW: Action Masking implementation ---
        Calculates a mask of valid (i, j) pairs.
        Returns a (n_hot, n_cold) boolean array where True means the action is valid.
        """
        hot_duty_mask = self.hot_duty_remaining > self.tolerance
        cold_duty_mask = self.cold_duty_remaining > self.tolerance
        
        # Create a broadcastable mask
        duty_mask = hot_duty_mask[:, np.newaxis] & cold_duty_mask[np.newaxis, :]
        
        # Temperature feasibility mask
        temp_mask = self.current_hot_temps[:, np.newaxis] > (self.current_cold_temps[np.newaxis, :] + self.min_deltaT)
        
        return duty_mask & temp_mask

    def _get_info(self):
        hot_utility_duty = np.sum(self.hot_duty_remaining)
        cold_utility_duty = np.sum(self.cold_duty_remaining)
        opex = hot_utility_duty * self.problem.hot_utility_cost_rate + cold_utility_duty * self.problem.cold_utility_cost_rate
        capex = self.cumulative_capex
        tac = capex + opex
        pinch_penalty = 0.0 # Pinch penalties are not used in dense reward scheme but kept for info
        
        return {
            "hot_utility_needed": hot_utility_duty, "cold_utility_needed": cold_utility_duty,
            "exchangers_placed": len(self.exchanger_list), "CAPEX": capex, "OPEX": opex, "TAC": tac,
            "pinch_penalty": pinch_penalty, "total_cost_with_penalty": tac + pinch_penalty, 
            "action_mask": self._get_action_mask(), # Include mask in info
            "network": self.exchanger_list
        }

    def _make_obs(self):
        # --- NEW: Time feature added to observation ---
        time_feature = np.array([self.step_count / self.max_steps])

        r_hot = self.hot_duty_remaining / (self.problem.total_hot_duty + 1e-6)
        r_cold = self.cold_duty_remaining / (self.problem.total_cold_duty + 1e-6)
        
        initial_hot_temps, target_hot_temps = self.problem.hot_streams_initial[:, 0], self.problem.hot_streams_initial[:, 1]
        t_hot_progress = (initial_hot_temps - self.current_hot_temps) / (initial_hot_temps - target_hot_temps + 1e-6)
        
        initial_cold_temps, target_cold_temps = self.problem.cold_streams_initial[:, 0], self.problem.cold_streams_initial[:, 1]
        t_cold_progress = (self.current_cold_temps - initial_cold_temps) / (target_cold_temps - initial_cold_temps + 1e-6)
        
        hot_temps_matrix = self.current_hot_temps[:, np.newaxis]
        cold_temps_matrix = self.current_cold_temps[np.newaxis, :]
        feasible_driving_force = np.maximum(0, hot_temps_matrix - cold_temps_matrix - self.min_deltaT)
        max_temp_diff = np.max(initial_hot_temps) - np.min(initial_cold_temps)
        normalized_driving_forces = feasible_driving_force / (max_temp_diff + 1e-6)
        
        last_i, last_j, last_Qr = self.last_action_tuple
        last_act_norm = np.array([
            2 * (last_i / (self.problem.n_hot - 1) if self.problem.n_hot > 1 else 0.5) - 1,
            2 * (last_j / (self.problem.n_cold - 1) if self.problem.n_cold > 1 else 0.5) - 1,
            last_Qr
        ])
        
        max_capex_heuristic = self.problem.n_hot * self.problem.n_cold * (self.problem.exchanger_fixed_cost + self.problem.exchanger_area_cost_coeff * (100**self.problem.exchanger_area_cost_exponent))
        cum_capex_norm = np.array([min(self.cumulative_capex / (max_capex_heuristic + 1e-6), 1.0)])
        
        total_q_recovered = np.sum(self.problem.total_hot_duty) - np.sum(self.hot_duty_remaining)
        q_rec_norm = np.array([total_q_recovered / (np.sum(self.problem.total_hot_duty) + 1e-6)])

        obs = np.concatenate([
            r_hot, r_cold, t_hot_progress, t_cold_progress, normalized_driving_forces.flatten(),
            last_act_norm, cum_capex_norm, q_rec_norm, time_feature
        ]).clip(-1.0, 1.0)
        return obs.astype(np.float32)

    def render(self, mode='human'):
        if mode == 'human':
            info = self._get_info()
            last_i, last_j, last_Qr = self.last_action_tuple
            hot_temps_str = np.array2string(self.current_hot_temps, precision=2, separator=', ')
            cold_temps_str = np.array2string(self.current_cold_temps, precision=2, separator=', ')
            
            print(f"Step {self.step_count:02d} | Action: ({last_i}, {last_j}, Qr={last_Qr:.2f}) | "
                  f"TAC: {info['TAC']:,.2f} (CAPEX: {info['CAPEX']:,.2f}, OPEX: {info['OPEX']:,.2f})")
            print(f"  -> Temps (H): {hot_temps_str} | Hot Utility: {info['hot_utility_needed']:.2f}")
            print(f"  -> Temps (C): {cold_temps_str} | Cold Utility: {info['cold_utility_needed']:.2f}")
            if "invalid_step_reason" in info and info["invalid_step_reason"]:
                print(f"  -> Invalid Step: {info['invalid_step_reason']}")

def replay_expert_solution(filepath: str) -> Tuple[HENGymEnv, List[Dict]]:
    """
    Reads an expert solution from a JSON file, creates the appropriate
    environment, and applies the action sequence.

    Args:
        filepath (str): The path to the expert solution JSON file.

    Returns:
        A tuple containing:
        - The HENGymEnv instance in its final state.
        - A list of the 'info' dictionaries from each step for analysis.
    """
    print(f"Loading expert solution from: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)

    # 1. Extract the problem definition to create the exact environment
    problem_def = data['problem_definition']
    
    env = HENGymEnv(
        hot_streams=problem_def['hot_streams'],
        cold_streams=problem_def['cold_streams'],
        max_steps=problem_def['max_steps'],
        min_deltaT=problem_def['min_deltaT'],
        # cost_parameters=problem_def['cost_parameters']
    )
    
    # 2. Reset the environment to get the initial state
    obs, info = env.reset()
    history = [info]
    print(f"Problem '{data['problem_name']}' loaded. Replaying {len(data['expert_action_sequence'])} actions.")

    # 3. Loop through and apply each action from the expert sequence
    for i, action in enumerate(data['expert_action_sequence']):
        print(f"Step {i+1}: {action}")
        obs, reward, done, truncated, info = env.step(action)
        history.append(info)
        env.render()
        # You could add a env.render() here if you want to see the output at each step
        if done:
            print("  -> Episode finished.")
            break
            
    print("\nReplay complete.")
    final_info = history[-1]
    print(f"Final TAC: {final_info.get('TAC', 'N/A'):,.2f}")
    
    return env, history

if __name__ == "__main__":
    
    final_env, run_history = replay_expert_solution('expert_solution.json')
    exit(0)
    
    # Define streams
    hot_streams  = [(443,333,30),(423,303,15)]
    cold_streams = [(293,408,20),(353,413,40)]
    
    action_list = []
    
    # Test step
    max_step = 10
    # Instantiate env
    env = HENGymEnv(hot_streams, cold_streams, max_steps=max_step, min_deltaT=10.0, stochastic_mode=False)
    obs, _ = env.reset()
    print("Initial obs:", obs)
    env.render()
    for step in range(max_step):
        action = {
          "i": env.action_space["i"].sample(),
          "j": env.action_space["j"].sample(),
          "Q_ratio": float(env.action_space["Q_ratio"].sample()),
        }
        # record action list
        action_list.append((obs, action))
        obs, r, done, _, info = env.step(action)
        env.render()
        if done:
            break
    