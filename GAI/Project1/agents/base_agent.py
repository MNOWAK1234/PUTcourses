# tetris_rl_agents/agents/base_agent.py
from abc import ABC, abstractmethod
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.tetris import Tetris

class PolicyNetwork(nn.Module):
    """ A simple feedforward neural network for the policy in reinforcement learning. """
    def __init__(self, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, fc1_units=config.FC_UNITS, fc2_units=config.FC_UNITS, seed=config.SEED) -> None:
        """ Initializes the policy network.
        Args:
            state_size (int): Number of features in the state representation.
            action_size (int): Number of possible actions (default is 1 for Tetris).
            seed (int): Random seed for reproducibility.
            fc1_units (int): Number of units in the first fully connected layer.
            fc2_units (int): Number of units in the second fully connected layer.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self._create_weights()

    def _create_weights(self) -> None:
        """ Initializes the weights of the network using Xavier uniform initialization. """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_features) -> torch.Tensor:
        """ Forward pass through the network.
        Args:
            state_features (torch.Tensor): Input tensor representing the state features.
        Returns:
            torch.Tensor: Output tensor representing the action scores.
        """
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_weights_flat(self) -> np.ndarray:
        """ Flattens the weights of the network into a single numpy array."""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_weights_flat(self, flat_weights) -> None:
        """ Sets the weights of the network from a flattened numpy array.
        Args:
            flat_weights (np.ndarray): Flattened weights to set in the network.
        """
        offset = 0
        for param in self.parameters():
            num_elements = param.data.numel()
            param.data.copy_(torch.from_numpy(flat_weights[offset:offset + num_elements]).view(param.data.shape).to(config.DEVICE))
            offset += num_elements
        if offset != len(flat_weights):
            raise ValueError("Size mismatch in set_weights_flat!\n Expected {}, got {}".format(len(flat_weights), offset))


class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, state_size):
        """
        Args:
            state_size (int): The number of features in the state representation.
        """
        self.state_size = state_size
        self.last_loss = None

    @abstractmethod
    def select_action(self, current_board_features: torch.Tensor, tetris_game_instance: Tetris, epsilon_override: float = None) -> tuple:
        """
        Selects an action based on the current board features and game instance.

        Args:
            current_board_features (torch.Tensor): Features of the board *before* current piece placement.
            tetris_game_instance (Tetris): The current game instance to query for next possible states.
            epsilon_override (float, optional): If provided, overrides the agent's internal epsilon (for exploration).

        Returns:
            tuple: (action_tuple, aux_info_dict)
                action_tuple (tuple): (x_pos, rotation_idx) representing the chosen piece placement.
                aux_info_dict (dict): Auxiliary information dictionary that might contain:
                    'features_s_prime_chosen': Features of the board state *after* the chosen action.
                    'current_board_features_s_t': Echo back of the input features.
                    'log_prob': Log probability of the chosen action (for policy gradient methods).
                    'entropy': Entropy of the policy (for policy gradient methods).
                    'value_of_current_board': Value of the current board state (for actor-critic methods).
                    'all_available_s_prime_features': List of features for all possible next states.
                    'chosen_action_index': Index of the chosen action among possible next states.
        """
        pass

    # @abstractmethod
    def learn(self, state_features: torch.Tensor, action_tuple: tuple, reward: float, 
              next_state_features: torch.Tensor, done: bool, 
              game_instance_at_s = None, game_instance_at_s_prime = None, # For agents needing full game state
              aux_info: dict = None):
        """
        Primary learning step for agents that learn from individual transitions or need aux_info.
        Called by the training loop after each env.step().

        Args:
            state_features (torch.Tensor): Features of the board state S_t (before action_tuple).
            action_tuple (tuple): The action (placement) taken.
            reward (float): The reward R_{t+1} received after the action.
            next_state_features (torch.Tensor): Features of the board state S_{t+1} (after action and new piece appears).
            done (bool): True if the game ended after this transition.
            game_instance_at_s (Tetris, optional): Full Tetris game instance at state S_t.
            game_instance_at_s_prime (Tetris, optional): Full Tetris game instance at state S_{t+1}.
            aux_info (dict, optional): Auxiliary information dictionary returned by select_action.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Called at the start of each new game/episode by the training loop.
        Allows the agent to reset any internal episodic state.
        """
        pass

    @abstractmethod
    def save(self, filepath:str=None):
        """
        Saves the agent's model(s).
        PPO might use filename_primary for actor and filename_secondary for critic.
        Other agents might only use filename_primary.
        """
        pass

    @abstractmethod
    def load(self, filepath:str=None):
        """
        Loads the agent's model(s).
        """
        pass