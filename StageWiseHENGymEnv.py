import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
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
#  Updated HENProblem Class with Pandas and Validation
# ==============================================================================
class HENProblem:
    """
    Data class for Heat Exchanger Network (HEN) problem definition.
    Loads stream, utility, match-specific cost, and forbidden match data from CSV files
    using pandas for efficient parsing. Includes comprehensive input validation and
    detailed documentation for CSV file formats.

    Attributes:
        streams_filepath (str): Path to the CSV file containing stream data.
        utilities_filepath (str): Path to the CSV file containing utility data.
        matches_cost_filepath (Optional[str]): Path to the CSV file with match-specific cost data.
        forbidden_matches_filepath (Optional[str]): Path to the CSV file with forbidden matches.
        default_u (float): Default overall heat transfer coefficient (kW/m²K).
        default_cost_params (dict): Default cost parameters for matches.
        hot_streams (List[Dict]): List of hot stream data dictionaries.
        cold_streams (List[Dict]): List of cold stream data dictionaries.
        hot_utilities (List[Dict]): List of hot utility data dictionaries.
        cold_utilities (List[Dict]): List of cold utility data dictionaries.
        match_costs (Dict[Tuple[str, str], Dict]): Match-specific cost parameters.
        forbidden_matches (Set[Tuple[str, str]]): Set of forbidden (hot, cold) stream pairs.
        n_hot (int): Number of hot streams.
        n_cold (int): Number of cold streams.
        hot_ids (List[str]): List of hot stream names.
        cold_ids (List[str]): List of cold stream names.
        hot_tin (np.ndarray): Inlet temperatures of hot streams (°C).
        hot_tout (np.ndarray): Outlet temperatures of hot streams (°C).
        hot_fcp (np.ndarray): Heat capacity flow rates of hot streams (kW/°C).
        hot_h (np.ndarray): Heat transfer coefficients of hot streams (kW/m²K).
        cold_tin (np.ndarray): Inlet temperatures of cold streams (°C).
        cold_tout (np.ndarray): Outlet temperatures of cold streams (°C).
        cold_fcp (np.ndarray): Heat capacity flow rates of cold streams (kW/°C).
        cold_h (np.ndarray): Heat transfer coefficients of cold streams (kW/m²K).
        total_hot_duty (np.ndarray): Total heat duties of hot streams (kW).
        total_cold_duty (np.ndarray): Total heat duties of cold streams (kW).
        max_duty (float): Maximum duty across all streams (kW).
        max_temp (float): Maximum temperature across all streams (°C).
        min_temp (float): Minimum temperature across all streams (°C).
        temp_range (float): Temperature range (max_temp - min_temp) (°C).

    Expected CSV File Formats:
        - streams.csv:
            Columns: Name (str), Type (str, 'hot' or 'cold'), Tin (float, °C), Tout (float, °C),
                     Fcp (float, kW/°C), h (float, kW/m²K)
            Example:
                Name,Type,Tin,Tout,Fcp,h
                H1,hot,300.0,100.0,10.0,0.5
                C1,cold,50.0,200.0,15.0,0.6
        - utilities.csv:
            Columns: Name (str), Type (str, 'hot' or 'cold'), Tin (float, °C), Tout (float, °C),
                     Cost (float, $/kW), h (float, kW/m²K)
            Example:
                Name,Type,Tin,Tout,Cost,h
                HU1,hot,400.0,350.0,100.0,0.8
                CU1,cold,20.0,30.0,10.0,0.7
        - matches_cost.csv (optional):
            Columns: Hot_Stream (str), Cold_Stream (str), Fixed_Cost_Unit (float, $),
                     Area_Cost_Coeff (float, $/m²), Area_Cost_Exp (float)
            Example:
                Hot_Stream,Cold_Stream,Fixed_Cost_Unit,Area_Cost_Coeff,Area_Cost_Exp
                H1,C1,5000.0,1000.0,0.6
        - forbidden_matches.csv (optional):
            Columns: Hot_Stream (str), Cold_Stream (str)
            Example:
                Hot_Stream,Cold_Stream
                H1,C2
    """
    def __init__(
        self, 
        streams_filepath: str,
        utilities_filepath: str,
        matches_cost_filepath: Optional[str] = None,
        forbidden_matches_filepath: Optional[str] = None,
        default_u: float = 0.8,
        default_fixed_cost: float = 0.0,
        default_area_coeff: float = 1000.0,
        default_area_exp: float = 0.6,
        **kwargs
    ):
        self.streams_filepath = streams_filepath
        self.utilities_filepath = utilities_filepath
        self.matches_cost_filepath = matches_cost_filepath
        self.forbidden_matches_filepath = forbidden_matches_filepath
        
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
        
        self.hot_ids = [s['name'] for s in self.hot_streams]
        self.hot_tin = np.array([s['tin'] for s in self.hot_streams], dtype=np.float32)
        self.hot_tout = np.array([s['tout'] for s in self.hot_streams], dtype=np.float32)
        self.hot_fcp = np.array([s['fcp'] for s in self.hot_streams], dtype=np.float32)
        self.hot_h = np.array([s['h'] for s in self.hot_streams], dtype=np.float32)
        
        self.cold_ids = [s['name'] for s in self.cold_streams]
        self.cold_tin = np.array([s['tin'] for s in self.cold_streams], dtype=np.float32)
        self.cold_tout = np.array([s['tout'] for s in self.cold_streams], dtype=np.float32)
        self.cold_fcp = np.array([s['fcp'] for s in self.cold_streams], dtype=np.float32)
        self.cold_h = np.array([s['h'] for s in self.cold_streams], dtype=np.float32)
        
        self.total_hot_duty = (self.hot_tin - self.hot_tout) * self.hot_fcp
        self.total_cold_duty = (self.cold_tout - self.cold_tin) * self.cold_fcp
        
        self.max_duty = max(np.max(self.total_hot_duty), np.max(self.total_cold_duty)) if self.n_hot > 0 and self.n_cold > 0 else 1.0
        self.max_temp = np.max(self.hot_tin) if self.n_hot > 0 else 0
        self.min_temp = np.min(self.cold_tin) if self.n_cold > 0 else 0
        self.temp_range = self.max_temp - self.min_temp if self.max_temp > self.min_temp else 1.0

    def _validate_csv_file(self, filepath: str, required_columns: List[str], numeric_columns: List[str]) -> pd.DataFrame:
        """
        Validates a CSV file and loads it into a pandas DataFrame.

        Args:
            filepath (str): Path to the CSV file.
            required_columns (List[str]): List of required column names.
            numeric_columns (List[str]): List of columns that must be numeric.

        Returns:
            pd.DataFrame: Validated DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing or numeric columns are invalid.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        try:
            df = pd.read_csv(filepath, dtype={col: float for col in numeric_columns}, keep_default_na=False)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {filepath}: {str(e)}")

        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {filepath}: {missing_cols}")

        # Validate numeric columns
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} in {filepath} must be numeric")
            if df[col].isna().any():
                raise ValueError(f"Column {col} in {filepath} contains missing or invalid values")

        # Check for empty DataFrame
        if df.empty:
            raise ValueError(f"CSV file {filepath} is empty")

        return df

    def _load_data_from_files(self):
        """
        Loads and validates stream, utility, cost, and forbidden match data from CSV files
        using pandas for efficient parsing. Populates missing match costs with defaults.

        Raises:
            ValueError: If data validation fails or thermodynamic constraints are violated.
        """
        # --- Load and Validate Streams ---
        stream_cols = ['Name', 'Type', 'Tin', 'Tout', 'Fcp', 'h']
        stream_numeric_cols = ['Tin', 'Tout', 'Fcp', 'h']
        streams_df = self._validate_csv_file(self.streams_filepath, stream_cols, stream_numeric_cols)

        # Validate stream types and thermodynamics
        streams_df['Type'] = streams_df['Type'].str.lower()
        if not all(streams_df['Type'].isin(['hot', 'cold'])):
            raise ValueError(f"Invalid stream type in {self.streams_filepath}. Must be 'hot' or 'cold'")

        hot_streams_df = streams_df[streams_df['Type'] == 'hot']
        cold_streams_df = streams_df[streams_df['Type'] == 'cold']

        # Check thermodynamic consistency
        if (hot_streams_df['Tin'] < hot_streams_df['Tout']).any():
            raise ValueError(f"Hot streams in {self.streams_filepath} must have Tin >= Tout")
        if (cold_streams_df['Tin'] > cold_streams_df['Tout']).any():
            raise ValueError(f"Cold streams in {self.streams_filepath} must have Tin <= Tout")
        if (streams_df['Fcp'] <= 0).any():
            raise ValueError(f"Fcp in {self.streams_filepath} must be positive")
        if (streams_df['h'] <= 0).any():
            raise ValueError(f"h in {self.streams_filepath} must be positive")
        if streams_df['Name'].duplicated().any():
            raise ValueError(f"Duplicate stream names in {self.streams_filepath}")

        self.hot_streams = hot_streams_df[['Name', 'Tin', 'Tout', 'Fcp', 'h']].rename(
            columns={'Name': 'name', 'Tin': 'tin', 'Tout': 'tout', 'Fcp': 'fcp'}).to_dict('records')
        self.cold_streams = cold_streams_df[['Name', 'Tin', 'Tout', 'Fcp', 'h']].rename(
            columns={'Name': 'name', 'Tin': 'tin', 'Tout': 'tout', 'Fcp': 'fcp'}).to_dict('records')

        # --- Load and Validate Utilities ---
        util_cols = ['Name', 'Type', 'Tin', 'Tout', 'Cost', 'h']
        util_numeric_cols = ['Tin', 'Tout', 'Cost', 'h']
        utils_df = self._validate_csv_file(self.utilities_filepath, util_cols, util_numeric_cols)

        # Validate utility types and data
        utils_df['Type'] = utils_df['Type'].str.lower()
        if not all(utils_df['Type'].isin(['hot', 'cold'])):
            raise ValueError(f"Invalid utility type in {self.utilities_filepath}. Must be 'hot' or 'cold'")
        if (utils_df['Cost'] < 0).any():
            raise ValueError(f"Cost in {self.utilities_filepath} must be non-negative")
        if (utils_df['h'] <= 0).any():
            raise ValueError(f"h in {self.utilities_filepath} must be positive")
        if utils_df['Name'].duplicated().any():
            raise ValueError(f"Duplicate utility names in {self.utilities_filepath}")

        hot_utils_df = utils_df[utils_df['Type'] == 'hot']
        cold_utils_df = utils_df[utils_df['Type'] == 'cold']
        self.hot_utilities = hot_utils_df[['Name', 'Tin', 'Tout', 'Cost', 'h']].rename(
            columns={'Name': 'name', 'Tin': 'tin', 'Tout': 'tout', 'Cost': 'cost'}).to_dict('records')
        self.cold_utilities = cold_utils_df[['Name', 'Tin', 'Tout', 'Cost', 'h']].rename(
            columns={'Name': 'name', 'Tin': 'tin', 'Tout': 'tout', 'Cost': 'cost'}).to_dict('records')

        # --- Load and Validate Match-Specific Costs ---
        self.match_costs = {}
        if self.matches_cost_filepath and os.path.exists(self.matches_cost_filepath):
            cost_cols = ['Hot_Stream', 'Cold_Stream', 'Fixed_Cost_Unit', 'Area_Cost_Coeff', 'Area_Cost_Exp']
            cost_numeric_cols = ['Fixed_Cost_Unit', 'Area_Cost_Coeff', 'Area_Cost_Exp']
            costs_df = self._validate_csv_file(self.matches_cost_filepath, cost_cols, cost_numeric_cols)

            # Validate cost parameters
            if (costs_df['Fixed_Cost_Unit'] < 0).any() or (costs_df['Area_Cost_Coeff'] < 0).any():
                raise ValueError(f"Cost parameters in {self.matches_cost_filepath} must be non-negative")
            if (costs_df['Area_Cost_Exp'] <= 0).any():
                raise ValueError(f"Area_Cost_Exp in {self.matches_cost_filepath} must be positive")

            for _, row in costs_df.iterrows():
                key = (row['Hot_Stream'], row['Cold_Stream'])
                self.match_costs[key] = {
                    'fixed_cost': float(row['Fixed_Cost_Unit']),
                    'area_coeff': float(row['Area_Cost_Coeff']),
                    'area_exp': float(row['Area_Cost_Exp'])
                }

        # --- Load and Validate Forbidden Matches ---
        self.forbidden_matches = set()
        if self.forbidden_matches_filepath and os.path.exists(self.forbidden_matches_filepath):
            forbidden_cols = ['Hot_Stream', 'Cold_Stream']
            forbidden_df = self._validate_csv_file(self.forbidden_matches_filepath, forbidden_cols, [])

            for _, row in forbidden_df.iterrows():
                self.forbidden_matches.add((row['Hot_Stream'], row['Cold_Stream']))

        # --- Validate stream and utility names in match costs and forbidden matches ---
        valid_hot_ids = set([s['name'] for s in self.hot_streams] + [u['name'] for u in self.hot_utilities])
        valid_cold_ids = set([s['name'] for s in self.cold_streams] + [u['name'] for u in self.cold_utilities])
        for hot_id, cold_id in self.match_costs.keys():
            if hot_id not in valid_hot_ids or cold_id not in valid_cold_ids:
                raise ValueError(f"Invalid stream/utility names in {self.matches_cost_filepath}: ({hot_id}, {cold_id})")
        for hot_id, cold_id in self.forbidden_matches:
            if hot_id not in valid_hot_ids or cold_id not in valid_cold_ids:
                raise ValueError(f"Invalid stream/utility names in {self.forbidden_matches_filepath}: ({hot_id}, {cold_id})")

        # --- Populate missing match combinations with default values ---
        hot_stream_ids = [hs['name'] for hs in self.hot_streams]
        cold_stream_ids = [cs['name'] for cs in self.cold_streams]
        hot_utility_ids = [hu['name'] for hu in self.hot_utilities]
        cold_utility_ids = [cu['name'] for cu in self.cold_utilities]
        
        for h_id in hot_stream_ids:
            for c_id in cold_stream_ids + cold_utility_ids:
                if (h_id, c_id) not in self.match_costs:
                    self.match_costs[(h_id, c_id)] = self.default_cost_params
        for h_id in hot_utility_ids:
            for c_id in cold_stream_ids:
                if (h_id, c_id) not in self.match_costs:
                    self.match_costs[(h_id, c_id)] = self.default_cost_params

    def get_cost_params(self, hot_id: str, cold_id: str) -> dict:
        """
        Retrieves cost parameters for a given hot-cold stream match.

        Args:
            hot_id (str): Name of the hot stream or utility.
            cold_id (str): Name of the cold stream or utility.

        Returns:
            dict: Cost parameters (fixed_cost, area_coeff, area_exp).
        """
        return self.match_costs.get((hot_id, cold_id), self.default_cost_params)

    def get_u_value(self, h_hot: float, h_cold: float) -> float:
        """
        Calculates the overall heat transfer coefficient for a match.

        Args:
            h_hot (float): Heat transfer coefficient of the hot stream (kW/m²K).
            h_cold (float): Heat transfer coefficient of the cold stream (kW/m²K).

        Returns:
            float: Overall heat transfer coefficient (kW/m²K).
        """
        if h_hot <= 0 or h_cold <= 0:
            return self.default_u
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
        self.small_number = 1e-6

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
        
        _, opex, _, _ = self._calculate_tac()
        
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
        hot_duty_remaining = (final_hot_temps - self.problem.hot_tout) * self.problem.hot_fcp
        cold_duty_remaining = (self.problem.cold_tout - final_cold_temps) * self.problem.cold_fcp
        
        norm_hot_duty_rem = hot_duty_remaining / (self.problem.max_duty + self.small_number)
        norm_cold_duty_rem = cold_duty_remaining / (self.problem.max_duty + self.small_number)
        
        # Add feasible driving forces
        t_hot_in_stage = self.hot_temps_grid[:, obs_stage_idx]
        t_cold_in_stage = self.cold_temps_grid[:, obs_stage_idx + 1]
        hot_temps_matrix = t_hot_in_stage[:, np.newaxis]
        cold_temps_matrix = t_cold_in_stage[np.newaxis, :]
        feasible_driving_force = np.maximum(0, hot_temps_matrix - cold_temps_matrix - self.min_deltaT)
        norm_driving_forces = feasible_driving_force / (self.problem.temp_range + self.small_number)

        # Get normalized time feature
        time_feature = np.array([self.current_stage_idx / self.num_stages])
        # --- NEW: Added normalized num_stages feature ---
        norm_num_stages = np.array([self.num_stages / 10.0]) # Normalize by a reasonable max
        
        # --- NEW: Create and add forbidden match mask ---
        forbidden_mask = np.zeros((self.problem.n_hot, self.problem.n_cold))
        for i, h_id in enumerate(self.problem.hot_ids):
            for j, c_id in enumerate(self.problem.cold_ids):
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
        total_capex = 0
        for k, q_matrix in enumerate(self.network_design):
            t_hot_in, t_hot_out = self.hot_temps_grid[:, k], self.hot_temps_grid[:, k+1]
            t_cold_in, t_cold_out = self.cold_temps_grid[:, k+1], self.cold_temps_grid[:, k]
            for i in range(self.problem.n_hot):
                for j in range(self.problem.n_cold):
                    q_ij = q_matrix[i, j]
                    if q_ij > self.tolerance:
                        dT1, dT2 = t_hot_in[i] - t_cold_out[j], t_hot_out[i] - t_cold_in[j]
                        if dT1 < self.min_deltaT - self.tolerance or dT2 < self.min_deltaT - self.tolerance: return float('inf'), float('inf'), 0.0, 0.0
                        U = self.problem.get_u_value(self.problem.hot_h[i], self.problem.cold_h[j])
                        cost_params = self.problem.get_cost_params(self.problem.hot_ids[i], self.problem.cold_ids[j])
                        lmtd = calculate_lmtd(dT1, dT2)
                        area = q_ij / (U * lmtd + self.small_number)
                        total_capex += (cost_params['fixed_cost'] + cost_params['area_coeff'] * (area ** cost_params['area_exp']))
        
        opex = 0.0
        final_hot_temps, final_cold_temps = self.hot_temps_grid[:, -1], self.cold_temps_grid[:, 0]
        
        total_cold_utility_needed = 0.0
        for i in range(self.problem.n_hot):
            utility_capex, utility_opex = 0.0, 0.0
            duty_needed = (final_hot_temps[i] - self.problem.hot_tout[i]) * self.problem.hot_fcp[i]
            if duty_needed > self.tolerance:
                total_cold_utility_needed += duty_needed
                best_utility_tac = float('inf')
                for cu in self.problem.cold_utilities:
                    if (self.problem.hot_ids[i], cu['name']) in self.problem.forbidden_matches: continue
                    dT1, dT2 = final_hot_temps[i] - cu['tout'], self.problem.hot_tout[i] - cu['tin']
                    if dT1 >= self.min_deltaT and dT2 >= self.min_deltaT:
                        op_cost = cu['cost'] * duty_needed
                        lmtd = calculate_lmtd(dT1, dT2)
                        U = self.problem.get_u_value(self.problem.hot_h[i], cu['h'])
                        cost_params = self.problem.get_cost_params(self.problem.hot_ids[i], cu['name'])
                        area = duty_needed / (U * lmtd + self.small_number)
                        capex_cost = cost_params['fixed_cost'] + cost_params['area_coeff'] * (area ** cost_params['area_exp'])
                        if capex_cost + op_cost < best_utility_tac:
                            best_utility_tac = capex_cost + op_cost
                            utility_capex, utility_opex = capex_cost, op_cost
                if best_utility_tac == float('inf'): return float('inf'), float('inf'), 0.0, 0.0
                total_capex += utility_capex
                opex += utility_opex

        total_hot_utility_needed = 0.0
        for j in range(self.problem.n_cold):
            utility_capex, utility_opex = 0.0, 0.0
            duty_needed = (self.problem.cold_tout[j] - final_cold_temps[j]) * self.problem.cold_fcp[j]
            if duty_needed > self.tolerance:
                total_hot_utility_needed += duty_needed
                best_utility_tac = float('inf')
                for hu in self.problem.hot_utilities:
                    if (hu['name'], self.problem.cold_ids[j]) in self.problem.forbidden_matches: continue
                    dT1, dT2 = hu['tin'] - self.problem.cold_tout[j], hu['tout'] - final_cold_temps[j]
                    if dT1 >= self.min_deltaT and dT2 >= self.min_deltaT:
                        op_cost = hu['cost'] * duty_needed
                        lmtd = calculate_lmtd(dT1, dT2)
                        U = self.problem.get_u_value(hu['h'], self.problem.cold_h[j])
                        cost_params = self.problem.get_cost_params(hu['name'], self.problem.cold_ids[j])
                        area = duty_needed / (U * lmtd + self.small_number)
                        capex_cost = cost_params['fixed_cost'] + cost_params['area_coeff'] * (area ** cost_params['area_exp'])
                        if capex_cost + op_cost < best_utility_tac:
                            best_utility_tac = capex_cost + op_cost
                            utility_capex, utility_opex = capex_cost, op_cost
                if best_utility_tac == float('inf'): return float('inf'), float('inf'), 0.0, 0.0
                total_capex += utility_capex
                opex += utility_opex
        
        # --- Return all capital and operating cost ---
        return total_capex, opex, total_hot_utility_needed, total_cold_utility_needed

    def _solve_temperature_grid(self):
        """Recalculates the entire temperature grid based on the current network design."""
        self.hot_temps_grid[:, 0] = self.problem.hot_tin
        self.cold_temps_grid[:, -1] = self.problem.cold_tin
        for k in range(self.num_stages):
            self.hot_temps_grid[:, k + 1] = self.hot_temps_grid[:, k] - np.sum(self.network_design[k], axis=1) / self.problem.hot_fcp
        for k in range(self.num_stages - 1, -1, -1):
            self.cold_temps_grid[:, k] = self.cold_temps_grid[:, k + 1] + np.sum(self.network_design[k], axis=0) / self.problem.cold_fcp

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Processes the design for one stage."""
        
        # --- FIX: Calculate TAC based on the state *before* this action ---
        capex_before, opex_before, _, _= self._calculate_tac()
        tac_before = capex_before + opex_before
        
        # --- De-normalize the agent's action and update the network design ---
        q_matrix = (action.reshape(self.problem.n_hot, self.problem.n_cold)) * self.max_q_value
        
        # --- NEW: Check for forbidden matches before applying the action ---
        for i in range(self.problem.n_hot):
            for j in range(self.problem.n_cold):
                if q_matrix[i, j] > self.tolerance and (self.problem.hot_ids[i], self.problem.cold_ids[j]) in self.problem.forbidden_matches:
                    return self._get_observation(), -1e7, True, False, self._get_info()
        
        # --- Apply the action to the network design ---
        self.network_design[self.current_stage_idx] = q_matrix

        # --- Recalculate all temperatures based on the new design ---
        self._solve_temperature_grid()

        # --- FIX: Re-validate the entire network for thermodynamic feasibility ---
        capex_after, opex_after, _, _ = self._calculate_tac()
        tac_after = capex_after + opex_after

        if tac_after == float('inf'): # Check if the network is now infeasible
            return self._get_observation(), -10.0, True, False, self._get_info()

        # The reward is the reduction in total annual cost
        reward = (tac_before - tac_after) / (self.initial_tac + self.small_number) if self.initial_tac > 0 else 0
        
        # --- NEW: Early termination logic ---
        self.current_stage_idx += 1
        base_done = self.current_stage_idx >= self.num_stages
        
        final_hot_temps = self.hot_temps_grid[:, -1]
        final_cold_temps = self.cold_temps_grid[:, 0]
        hot_done = np.all(final_hot_temps <= self.problem.hot_tout + self.tolerance)
        cold_done = np.all(final_cold_temps >= self.problem.cold_tout - self.tolerance)
        early_done = hot_done and cold_done
        
        done = base_done or early_done
            
        return self._get_observation(), reward, done, False, self._get_info()
    
    def render(self, mode='human'):
        """Renders the current state or a final summary of the environment."""
        if mode == 'human':
            if self.current_stage_idx >= self.num_stages:
                head_str = "---       EPISODE FINISHED: FINAL NETWORK SUMMARY       ---"
                print("\n" + "="*len(head_str))
                print(head_str)
                print("="*len(head_str))
                
                total_capex, opex, hot_utility_needed, cold_utility_needed = self._calculate_tac()

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
                # --- UPDATE: Added utility usage to the summary ---
                print(f"  - Hot Utility Usage:      {hot_utility_needed:,.2f}")
                print(f"  - Cold Utility Usage:     {cold_utility_needed:,.2f}")
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
    

# ==============================================================================
#  main Function to test the GymEnv
# ==============================================================================
def main():
    # --- Create Dummy CSV files for demonstration ---
    from stable_baselines3.common.env_checker import check_env
    
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

if __name__ == '__main__':
    main()
    