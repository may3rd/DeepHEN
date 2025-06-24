from StageWiseHENGymEnv import generate_dataset_from_json

def main():
    """Main function to train and test the PPO agent."""
    
    json_filepath = "expert_solutions.json"

    # Generate the dataset
    generate_dataset_from_json(json_filepath, "hen_dataset.npz", verbose=True)

if __name__ == '__main__':
    main()