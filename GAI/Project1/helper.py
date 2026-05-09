import os
import re
import torch
import config
import random
import numpy as np
from typing import Tuple

#########################################################################
# Helper Functions for Agent File Management, Score Parsing and Seeding #
#########################################################################

def get_agent_file_prefix(agent_type_str:str) -> str:
    """Generates a standardized file prefix for saving/loading agent models.
    Args:
        agent_type_str (str): Type of the agent (e.g., "ppo", "genetic", "es").
    Returns:
        str: A standardized prefix for the agent model file.
    """
    return agent_type_str.replace("_", "-").lower()

def parse_score_from_filename(filename_basename:str, expected_prefix:str) -> (int | None):
    """Extracts the score from a filename that matches the expected pattern.
    Args:
        filename_basename (str): The basename of the file (without path).
        expected_prefix (str): The expected prefix for the agent type (e.g., "ppo-actor", "ppo-critic").
    Returns:
        int | None: The extracted score as an integer if the pattern matches, otherwise None.
    """
    pattern = re.compile(f"^{re.escape(expected_prefix)}_score_(\\d+)\\.pth$")
    match = pattern.match(filename_basename)
    if match:
        try: return int(match.group(1))
        except ValueError: return None
    return None

def find_best_existing_score(agent_prefix:str, model_dir:str) -> int:
    """Finds the highest score from existing model files in the specified directory.
    Args:
        agent_prefix (str): The prefix used for the agent type (e.g., "ppo-actor").
        model_dir (str): Directory where models are stored.
    Returns:
        int: The highest score found in the model files, or -1 if no valid scores
    """
    max_score = -1
    if not os.path.isdir(model_dir):
        try: os.makedirs(model_dir, exist_ok=True)
        except OSError:
            print(f"Warning: Model directory {model_dir} does not exist and could not be created.")
            return max_score
    for filename in os.listdir(model_dir):
        score = parse_score_from_filename(filename, agent_prefix)
        if score is not None and score > max_score:
            max_score = score
    return max_score

def find_latest_or_best_model_path(agent_type_str:str, model_dir:str) -> Tuple[str, str | None, None] | Tuple[str | None]:
    """Finds the latest or best model path for the specified agent type in the given directory.
    Args:
        agent_type_str (str): Type of the agent (e.g., "ppo", "genetic", "es").
        model_dir (str): Directory where models are stored.
    Returns:
        tuple([str, str]): Full paths to the actor and critic models if agent_type_str is "ppo",
                         or a single model path for other agent types.
                         Returns (None, None) if the directory does not exist or no models are found.
    """
    if not os.path.isdir(model_dir):
        print(f"Warning: Model directory {model_dir} does not exist.")
        return None

    agent_prefix = get_agent_file_prefix(agent_type_str)
    
    for filename in os.listdir(model_dir):
        if filename.startswith(agent_prefix) and filename.endswith(".pth"):
            return os.path.join(model_dir, filename)

def setup_seeds(seed:int = config.SEED) -> None:
    """Sets random seeds for reproducibility across numpy, torch, and random.
    Args:
        seed (int): The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seeds set to: {seed}")
