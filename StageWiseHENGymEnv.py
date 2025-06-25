import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json
import os
import csv

# ==============================================================================
#  Helper Functions
# ==============================================================================
def calculate_lmtd(dt1: float, dt2: float) -> float:
    return (dt1 * dt2 * (dt1 + dt2) / 2.0)**(1/3.0)

# ==============================================================================
#  Updated HENProblem Class with Data Loading from CSV
# ==============================================================================
class HENProblem:
    """
    Data class for HEN problem definition.
    This version loads all stream, utility, and cost data from external CSV files.
    """
    def __init__(
        self, 
        streams_filepath: str,
        utilities_filepath: str,
        matches_cost_filepath: Optional[str] = None,
        forbidden_matches_filepath: Optional[str] = None, # <-- NEW
        default_u: float = 0.8,
        default_fixed_cost: float = 0.0,
        default_area_coeff: float = 1000.0,
        default_area_exp: float = 0.6,
        **kwargs
    ):
        self.streams_filepath = streams_filepath
        self.utilities_filepath = utilities_filepath
        self.matches_cost_filepath = matches_cost_filepath
        self.forbidden_matches_filepath = forbidden_matches_filepath # <-- NEW
        
        # Default values
        self.default_u = default_u
        self.default_cost_params = {
            'fixed_cost': default_fixed_cost,
            'area_coeff': default_area_coeff,
            'area_exp': default_area_exp
        }
        
        self._load_data_from_files()

        self.n_hot = len(self.hot_streams)
        self.n_cold = len(self.cold_streams)
        
        self.total_hot_duty = (self.hot_streams[:, 1] - self.hot_streams[:, 2]) * self.hot_streams[:, 3]
        self.total_cold_duty = (self.cold_streams[:, 2] - self.cold_streams[:, 1]) * self.cold_streams[:, 3]
        self.max_duty = max(np.max(self.total_hot_duty), np.max(self.total_cold_duty)) if self.n_hot > 0 and self.n_cold > 0 else 1.0
        self.max_temp = np.max(self.hot_streams[:, 1]) if self.n_hot > 0 else 0
        self.min_temp = np.min(self.cold_streams[:, 1]) if self.n_cold > 0 else 0
        self.temp_range = self.max_temp - self.min_temp if self.max_temp > self.min_temp else 1.0
        
    def _load_data_from_files(self):
        """Parses all required CSV files and ensures all possible matches have cost data."""
        # --- Load Streams ---
        hot_streams_list, cold_streams_list = [], []
        with open(self.streams_filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stream_data = [row['Name'], float(row['Tin']), float(row['Tout']), float(row['Fcp']), float(row['h'])]
                if row['Type'].lower() == 'hot': hot_streams_list.append(stream_data)
                else: cold_streams_list.append(stream_data)
        self.hot_streams = np.array(hot_streams_list, dtype=object)
        self.cold_streams = np.array(cold_streams_list, dtype=object)

        # --- Load Utilities ---
        hot_utils_list, cold_utils_list = [], []
        with open(self.utilities_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                util_data = {
                    'name': row['Name'],
                    'tin': float(row['Tin']),
                    'tout': float(row['Tout']),
                    'cost': float(row['Cost']),
                    'h': float(row['h'])
                    }
                if row['Type'].lower() == 'hot': hot_utils_list.append(util_data)
                else: cold_utils_list.append(util_data)
        self.hot_utilities = hot_utils_list
        self.cold_utilities = cold_utils_list

        # --- Load Match-Specific Costs ---
        self.match_costs = {}
        if self.matches_cost_filepath and os.path.exists(self.matches_cost_filepath):
            with open(self.matches_cost_filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['Hot_Stream'], row['Cold_Stream'])
                    self.match_costs[key] = {
                        'fixed_cost': float(row['Fixed_Cost_Unit']),
                        'area_coeff': float(row['Area_Cost_Coeff']),
                        'area_exp': float(row['Area_Cost_Exp'])
                        }
        
        # --- NEW: Load Forbidden Matches ---
        self.forbidden_matches = set()
        if self.forbidden_matches_filepath and os.path.exists(self.forbidden_matches_filepath):
            with open(self.forbidden_matches_filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.forbidden_matches.add((row['Hot_Stream'], row['Cold_Stream']))
        
        # --- Populate missing match combinations with default values ---
        hot_stream_ids = self.hot_streams[:, 0]
        cold_stream_ids = self.cold_streams[:, 0]
        hot_utility_ids = [hu['name'] for hu in self.hot_utilities]
        cold_utility_ids = [cu['name'] for cu in self.cold_utilities]
        
        for h_id in hot_stream_ids:
            for c_id in cold_stream_ids:
                if (h_id, c_id) not in self.match_costs: self.match_costs[(h_id, c_id)] = self.default_cost_params
        for h_id in hot_stream_ids:
            for cu_id in cold_utility_ids:
                if (h_id, cu_id) not in self.match_costs: self.match_costs[(h_id, cu_id)] = self.default_cost_params
        for hu_id in hot_utility_ids:
            for c_id in cold_stream_ids:
                if (hu_id, c_id) not in self.match_costs: self.match_costs[(hu_id, c_id)] = self.default_cost_params

    def get_cost_params(self, hot_id: str, cold_id: str) -> dict:
        return self.match_costs.get((hot_id, cold_id), self.default_cost_params)

    def get_u_value(self, h_hot: float, h_cold: float) -> float:
        if h_hot <= 0 or h_cold <= 0: return self.default_u
        return (1/h_hot + 1/h_cold)**-1


# ==============================================================================
#  The Final Stage-Wise Environment with Data-Driven Logic
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
        streams_filepath: str,
        utilities_filepath: str,
        matches_cost_filepath: str,
        num_stages: int,
        min_deltaT: float = 10.0,
        **kwargs
    ):
        super().__init__()
        self.problem = HENProblem(streams_filepath, utilities_filepath, matches_cost_filepath, **kwargs)
        self.num_stages = num_stages
        self.min_deltaT = min_deltaT
        self.tolerance = 1e-3

        # --- UPDATE: Added forbidden match mask to observation size ---
        obs_size = (self.problem.n_hot + self.problem.n_cold) * 2 + \
                   (self.problem.n_hot * self.problem.n_cold) * 2 + 1 + 1 # Temps + Duties + Driving Forces + Forbidden Mask + Time + Num Stages
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        action_size = self.problem.n_hot * self.problem.n_cold
        self.action_space = spaces.Box(low=0, high=1, shape=(action_size,), dtype=np.float32)
        self.max_q_value = self.problem.max_duty

        self.network_design = None
        self.hot_temps_grid = None
        self.cold_temps_grid = None
        self.current_stage_idx = None


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        self.network_design = [np.zeros((self.problem.n_hot, self.problem.n_cold)) for _ in range(self.num_stages)]
        self.hot_temps_grid = np.zeros((self.problem.n_hot, self.num_stages + 1))
        self.cold_temps_grid = np.zeros((self.problem.n_cold, self.num_stages + 1))
        self._solve_temperature_grid()
        
        _, opex = self._calculate_tac()
        
        self.initial_tac = opex
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
        # --- NEW: Added normalized num_stages feature ---
        norm_num_stages = np.array([self.num_stages / 10.0]) # Normalize by a reasonable max
        
        # --- NEW: Create and add forbidden match mask ---
        forbidden_mask = np.zeros((self.problem.n_hot, self.problem.n_cold))
        hot_ids, cold_ids = self.problem.hot_streams[:, 0], self.problem.cold_streams[:, 0]
        for i, h_id in enumerate(hot_ids):
            for j, c_id in enumerate(cold_ids):
                if (h_id, c_id) in self.problem.forbidden_matches:
                    forbidden_mask[i, j] = 1.0
        
        obs = np.concatenate([
            norm_hot_inlet_temps,
            norm_cold_inlet_temps,
            norm_hot_duty_rem,
            norm_cold_duty_rem,
            norm_driving_forces.flatten(),
            forbidden_mask.flatten(),
            time_feature,
            norm_num_stages
        ])
        
        return obs.astype(np.float32).clip(0, 1)

    def _get_info(self) -> Dict:
        return { "current_stage": self.current_stage_idx }

    def _calculate_tac(self):
        """Helper to calculate the total CAPEX and OPEX for the current network design."""
        total_capex = 0
        
        hot_ids, cold_ids = self.problem.hot_streams[:, 0], self.problem.cold_streams[:, 0]
        hot_fcp, cold_fcp = self.problem.hot_streams[:, 3], self.problem.cold_streams[:, 3]
        hot_h, cold_h = self.problem.hot_streams[:, 4], self.problem.cold_streams[:, 4]
        
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
                            return float('inf'), float('inf')
                        # --- UPDATE: Use data-driven U and cost params ---
                        U = self.problem.get_u_value(hot_h[i], cold_h[j])
                        cost_params = self.problem.get_cost_params(hot_ids[i], cold_ids[j])
                        lmtd = calculate_lmtd(dT1, dT2) if dT1 * dT2 > 0 else 1e-6
                        area = q_ij / (U * lmtd + 1e-6)
                        total_capex += (cost_params['fixed_cost'] + cost_params['area_coeff'] * (area ** cost_params['area_exp']))
        
        # Calculate utility requirements and costs
        opex = 0.0
        hot_fcp, cold_fcp = self.problem.hot_streams[:, 3], self.problem.cold_streams[:, 3]
        final_hot_temps, final_cold_temps = self.hot_temps_grid[:, -1], self.cold_temps_grid[:, 0]
        
        # Cold utility for remaining hot stream duties
        for i in range(self.problem.n_hot):
            duty_needed = (final_hot_temps[i] - self.problem.hot_streams[i, 2]) * hot_fcp[i]
            if duty_needed > self.tolerance:
                best_utility_tac = float('inf')
                for cu in self.problem.cold_utilities:
                    dT1 = final_hot_temps[i] - cu['tout']
                    dT2 = self.problem.hot_streams[i, 2] - cu['tin']
                    if dT1 >= self.min_deltaT and dT2 >= self.min_deltaT:
                        op_cost = cu['cost'] * duty_needed
                        lmtd = calculate_lmtd(dT1, dT2)
                        U = self.problem.get_u_value(hot_h[i], cu['h'])
                        cost_params = self.problem.get_cost_params(hot_ids[i], cu['name'])
                        area = duty_needed / (U * lmtd + 1e-6)
                        capex_cost = cost_params['fixed_cost'] + cost_params['area_coeff'] * (area ** cost_params['area_exp'])
                        if capex_cost + op_cost < best_utility_tac:
                            best_utility_tac = capex_cost + op_cost
                if best_utility_tac != float('inf'):
                    total_capex += best_utility_tac - (cu['cost'] * duty_needed) # Add only the capex part
                    opex += (cu['cost'] * duty_needed)
                else: return float('inf'), float('inf') # No feasible utility

        # Hot utility for remaining cold stream duties
        for j in range(self.problem.n_cold):
            duty_needed = (self.problem.cold_streams[j, 2] - final_cold_temps[j]) * cold_fcp[j]
            if duty_needed > self.tolerance:
                best_utility_tac = float('inf')
                for hu in self.problem.hot_utilities:
                    dT1 = hu['tin'] - self.problem.cold_streams[j, 2]
                    dT2 = hu['tout'] - final_cold_temps[j]
                    if dT1 >= self.min_deltaT and dT2 >= self.min_deltaT:
                        op_cost = hu['cost'] * duty_needed
                        lmtd = calculate_lmtd(dT1, dT2)
                        U = self.problem.get_u_value(hu['h'], cold_h[j])
                        cost_params = self.problem.get_cost_params(hu['name'], cold_ids[j])
                        area = duty_needed / (U * lmtd + 1e-6)
                        capex_cost = cost_params['fixed_cost'] + cost_params['area_coeff'] * (area ** cost_params['area_exp'])
                        if capex_cost + op_cost < best_utility_tac:
                            best_utility_tac = capex_cost + op_cost
                if best_utility_tac != float('inf'):
                    total_capex += best_utility_tac - (hu['cost'] * duty_needed)
                    opex += (hu['cost'] * duty_needed)
                else: return float('inf'), float('inf')
        
        return total_capex, opex

    def _solve_temperature_grid(self):
        """Recalculates the entire temperature grid based on the current network design."""
        self.hot_temps_grid[:, 0] = self.problem.hot_streams[:, 1]
        self.cold_temps_grid[:, -1] = self.problem.cold_streams[:, 1]
        
        hot_fcp = self.problem.hot_streams[:, 3]
        cold_fcp = self.problem.cold_streams[:, 3]
        
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
        
        # --- NEW: Check for forbidden matches before applying the action ---
        hot_ids = self.problem.hot_streams[:, 0]
        cold_ids = self.problem.cold_streams[:, 0]
        for i in range(self.problem.n_hot):
            for j in range(self.problem.n_cold):
                if q_matrix[i, j] > self.tolerance:
                    if (hot_ids[i], cold_ids[j]) in self.problem.forbidden_matches:
                        return self._get_observation(), -1e7, True, False, self._get_info()
        
        self.network_design[self.current_stage_idx] = q_matrix

        # Recalculate all temperatures based on the new design
        self._solve_temperature_grid()

        # --- FIX: Re-validate the entire network for thermodynamic feasibility ---
        capex_after, opex_after = self._calculate_tac()
        tac_after = capex_after + opex_after

        if tac_after == float('inf'): # Check if the network is now infeasible
            return self._get_observation(), -10.0, True, False, self._get_info()

        # The reward is the reduction in total annual cost
        reward = (tac_before - tac_after) / (self.initial_tac + 1e-6) if self.initial_tac > 0 else 0
        
        self.current_stage_idx += 1
        done = self.current_stage_idx >= self.num_stages
            
        return self._get_observation(), reward, done, False, self._get_info()
    
    def render(self, mode='human'):
        """Renders the current state or a final summary of the environment."""
        if mode == 'human':
            if self.current_stage_idx >= self.num_stages:
                head_str = "---       EPISODE FINISHED: FINAL NETWORK SUMMARY       ---"
                print("\n" + "="*len(head_str))
                print(head_str)
                print("="*len(head_str))
                
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
        else:
            raise NotImplementedError
        

# ==============================================================================
#  Updated Function to Generate Dataset from Expert Solutions
# ==============================================================================
def generate_dataset_from_json(json_filepath: str, output_path: str, verbose: bool = False):
    """
    Reads a JSON file containing a list of expert HEN solutions,
    unrolls them into (observation, action) pairs, and saves them.

    Args:
        json_filepath (str): Path to the input JSON file.
        output_path (str): Path to save the generated dataset (e.g., 'dataset.npz').
        verbose (bool): If True, prints each generated pair and final state.
    """
    print(f"Loading expert solutions from '{json_filepath}'...")
    with open(json_filepath, 'r') as f:
        expert_solutions = json.load(f)

    all_observations = []
    all_actions = []

    for problem_data in expert_solutions:
        print(f"Processing problem: {problem_data['problem_name']}")
        
        # --- UPDATE: Instantiate environment using file paths from the JSON ---
        prob_def = problem_data['problem_definition']
        env = StageWiseHENGymEnv(
            streams_filepath=prob_def['streams_filepath'],
            utilities_filepath=prob_def['utilities_filepath'],
            matches_cost_filepath=prob_def.get('matches_cost_filepath', None),
            forbidden_matches_filepath=prob_def.get('forbidden_matches_filepath', None),
            num_stages=prob_def['num_stages'],
            min_deltaT=prob_def['min_deltaT'],
            **prob_def.get('cost_parameters', {})
        )
        
        obs, _ = env.reset()

        for stage_k, q_matrix_list in enumerate(problem_data['expert_action_sequence']):
            current_obs = obs
            
            q_matrix = np.array(q_matrix_list)
            normalized_action = (q_matrix / env.max_q_value).flatten()
            
            if verbose:
                print("-" * 30)
                print(f"STAGE {stage_k}:")
                print("  Observation (State):")
                print(f"    {np.round(current_obs, 3)}")
                print("  Normalized Action:")
                print(f"    {np.round(normalized_action, 3)}")
                print("-" * 30)
            
            all_observations.append(current_obs)
            all_actions.append(normalized_action)
            
            obs, _, done, _, _ = env.step(normalized_action)

            if done:
                break
    
        if verbose:
            # --- UPDATE: Render the final state for verification ---
            print("\nFinal network state from expert actions:")
            env.render()
            print("="*60)

    print(f"\nGenerated a total of {len(all_observations)} (observation, action) pairs.")
    
    np.savez_compressed(
        output_path, 
        observations=np.array(all_observations), 
        actions=np.array(all_actions)
    )
    print(f"Dataset saved to '{output_path}'")
    

if __name__ == '__main__':
    # --- Create Dummy CSV files for demonstration ---
    from stable_baselines3.common.env_checker import check_env
    
    # with open('streams.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Name', 'Type', 'Tin', 'Tout', 'Fcp', 'h'])
    #     writer.writerow(['H1', 'Hot', '443', '333', '30', '0.8'])
    #     writer.writerow(['H2', 'Hot', '423', '303', '15', '0.7'])
    #     writer.writerow(['C1', 'Cold', '293', '408', '20', '0.6'])
    #     writer.writerow(['C2', 'Cold', '353', '413', '40', '0.5'])
    # with open('utilities.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Name', 'Type', 'Tin', 'Tout', 'Cost', 'h'])
    #     writer.writerow(['W1', 'Cold', '293', '333', '80', '1.0'])
    #     writer.writerow(['S1', 'Hot', '450', '449', '120', '1.2'])
    # with open('matches_cost.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Hot_Stream', 'Cold_Stream', 'Fixed_Cost_Unit', 'Area_Cost_Coeff', 'Area_Cost_Exp'])
    #     writer.writerow(['H1', 'C1', '1000', '1200', '0.6'])
    
    # --- Instantiate and test the new data-driven environment ---
    env = StageWiseHENGymEnv(
                streams_filepath="data/example1/streams.csv",
                utilities_filepath="data/example1/utilities.csv",
                matches_cost_filepath="data/example1/matches_cost.csv",
                forbidden_matches_filepath="data/example1/forbidden_matches.csv",
                num_stages=3,
                min_deltaT=10.0
            )
    check_env(env)
    print("Environment check passed!")
    obs, _ = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Observation space size:", env.observation_space.shape[0])

    print("Initial Observation:", obs)
    env.render()
    
    action_vector_s0 = np.zeros(env.action_space.shape)
    action_vector_s0[0 * env.problem.n_cold + 1] = 2400/3300 
    action_vector_s0[1 * env.problem.n_cold + 0] = 900/3300
    
    print("\nTaking a sample action for Stage 0...")
    obs, reward, done, _, info = env.step(action_vector_s0)
    
    print("Action taken:")
    print(action_vector_s0.reshape(env.problem.n_hot, env.problem.n_cold))
    # --- FIX: Corrected syntax error in print statement ---
    print("After action 0:", obs)
    env.render()

    action_vector_s1 = np.zeros(env.action_space.shape)
    action_vector_s1[0 * env.problem.n_cold + 0] = 900/3300
    
    print("\nTaking a sample action for Stage 1...")
    obs, reward, done, _, info = env.step(action_vector_s1)
    
    print("Action taken:")
    print(action_vector_s1.reshape(env.problem.n_hot, env.problem.n_cold))
    # --- FIX: Corrected syntax error in print statement ---
    print("After action 1:", obs)
    env.render()

    action_vector_s2 = np.zeros(env.action_space.shape)
    action_vector_s2[1 * env.problem.n_cold + 0] = 300/3300
    
    print("\nTaking a sample action for Stage 2...")
    obs, reward, done, _, info = env.step(action_vector_s2)
    
    print("Action taken:")
    print(action_vector_s2.reshape(env.problem.n_hot, env.problem.n_cold))
    # --- FIX: Corrected syntax error in print statement ---
    print("After action 2:", obs)
    env.render()
