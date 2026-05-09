import torch
import random

from typing import Tuple

from .base_agent import BaseAgent
from src.tetris import Tetris

class RandomAgent(BaseAgent):
    """ A RandomAgent that selects actions randomly from the available next moves in Tetris."""
    def __init__(self, state_size, seed=0):
        """ Initializes the RandomAgent with a given state size and seed for reproducibility.
        Args:
            state_size (int): Size of the state representation.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(state_size)
        random.seed(seed)
        print("RandomAgent for Tetris initialized.")

    def select_action(self, state_features:torch.Tensor, tetris_game_instance:Tetris, epsilon_override=0.0) -> Tuple[Tuple[int, int], dict]:
        """ Selects a random action from the possible next moves.
        Args:
            state_features (torch.Tensor): The current state features of the Tetris game.
            tetris_game_instance (Tetris): The Tetris game instance to interact with.
            epsilon_override (float): Not used in this agent, but kept for compatibility.
        Returns:
            tuple: A tuple representing the chosen action (x, rotation).
            dict: An empty dictionary, as no additional information is needed for this agent.
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        possible_actions_tuples = list(next_steps_dict.keys())

        if not possible_actions_tuples:
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            return chosen_action_tuple, {}

        chosen_action_tuple = random.choice(possible_actions_tuples)

        return chosen_action_tuple, {}
    
    def reset(self) -> None:
        """ Reset method is not applicable for RandomAgent, but defined for interface compatibility. """
        pass
    
    def save(self, filepath=None) -> None:
        """ Save method is not applicable for RandomAgent, but defined for interface compatibility. """
        pass
    
    def load(self, filepath=None) -> None:
        """ Load method is not applicable for RandomAgent, but defined for interface compatibility. """
        pass
