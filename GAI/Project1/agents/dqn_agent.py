import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Tuple
from collections import deque, namedtuple

import config
from helper import *
from src.tetris import Tetris 
from .base_agent import BaseAgent, PolicyNetwork

DEVICE = config.DEVICE

Experience = namedtuple("Experience", field_names=["s_t_features", "s_prime_chosen_features", "reward", "done"])

class ReplayBuffer:
    """ A simple replay buffer to store experiences for DQN agent. """
    def __init__(self, buffer_size:int, batch_size:int) -> None:
        """ Initializes the replay buffer.
        Args:
            buffer_size (int): Maximum size of the buffer.
            batch_size (int): Size of the batches to sample from the buffer.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, s_t_features:torch.Tensor, s_prime_chosen_features:torch.Tensor, reward:int, done:bool) -> None:
        """ Adds a new experience to the replay buffer.
        Args:
            s_t_features (torch.Tensor): Features of the current state before taking the action.
            s_prime_chosen_features (torch.Tensor): Features of the state after taking the chosen action.
            reward (int): The reward received after taking the action.
            done (bool): Whether the episode has ended.
        """
        e = Experience(s_t_features, s_prime_chosen_features, reward, done)
        self.memory.append(e)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Samples a batch of experiences from the replay buffer.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - s_t_b: Tensor of current state features.
                - s_pc_b: Tensor of chosen state features after action.
                - r_b: Tensor of rewards.
                - d_b: Tensor of done flags.
        """
        actual_sample_size = min(len(self.memory), self.batch_size)
        if actual_sample_size == 0: return (torch.empty(0,device=DEVICE), torch.empty(0,device=DEVICE), torch.empty(0,device=DEVICE), torch.empty(0,device=DEVICE))
        
        experiences = random.sample(self.memory, k=actual_sample_size)
        valid_experiences = [e for e in experiences if e and e.s_t_features is not None and e.s_prime_chosen_features is not None]
        if not valid_experiences: return (torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE))
        
        s_t_b = torch.stack([e.s_t_features for e in valid_experiences]).float().to(DEVICE)
        s_pc_b = torch.stack([e.s_prime_chosen_features for e in valid_experiences]).float().to(DEVICE)
        r_b = torch.from_numpy(np.vstack([e.reward for e in valid_experiences])).float().to(DEVICE)
        d_b = torch.from_numpy(np.vstack([e.done for e in valid_experiences]).astype(np.uint8)).float().to(DEVICE)
        
        return (s_t_b, s_pc_b, r_b, d_b)
    
    def __len__(self): 
        """ Returns the current size of the replay buffer. """
        return len(self.memory)

class DQNAgent(BaseAgent):
    """ A DQN agent that learns to play Tetris using a value network and a replay buffer. """
    def __init__(self, state_size, seed:int=0):
        """ Initializes the DQN agent with a value network and replay buffer.
        Args:
            state_size (int): Size of the state representation.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(state_size)

        self.v_network = PolicyNetwork(state_size, seed=seed, fc1_units=config.DQN_FC1_UNITS, fc2_units=config.DQN_FC2_UNITS).to(DEVICE)
        self.optimizer = optim.Adam(self.v_network.parameters(), lr=config.DQN_LR)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(config.DQN_BUFFER_SIZE, config.DQN_BATCH_SIZE)
        
        self.learning_steps_done = 0
        self.total_pieces_placed_overall = 0

        self.epsilon = config.DQN_EPSILON_START
        self.last_loss = None 

        print("DQN Agent (V-Learning Style for Original Loop by train.py) initialized.")
        print(f"  Agent's internal epsilon starts at: {self.epsilon:.3f} (train.py will override during training steps)")

    def select_action(self, current_board_features_s_t: torch.Tensor, tetris_game_instance: Tetris, epsilon_override: float = None) -> tuple: 
        """ Selects an action based on the current board features and game instance.
        Args:
            current_board_features_s_t (torch.Tensor): Features of the current board state.
            tetris_game_instance (Tetris): The Tetris game instance.
            epsilon_override (float, optional): Override for epsilon value for exploration. Defaults to None.
        Returns:
            tuple: A tuple containing the chosen action and auxiliary information.
        """
        current_epsilon_for_decision = epsilon_override if None != epsilon_override else self.epsilon
        
        # Get the current board features
        next_steps_dict = tetris_game_instance.get_next_states()
        possible_actions_tuples = list(next_steps_dict.keys())
        possible_features = [features.to(DEVICE) for features in next_steps_dict.values()]
        
        if random.random() <= current_epsilon_for_decision:
            chosen_idx = random.randrange(len(possible_actions_tuples)) 
        else:
            self.v_network.eval()
            with torch.no_grad(): q_values = self.v_network(torch.stack(possible_features))
            self.v_network.train()
            chosen_idx = torch.argmax(q_values).item()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_chosen = possible_features[chosen_idx]
        
        return chosen_action_tuple, {
            'features_s_prime_chosen': features_chosen,
            'current_board_features_s_t': current_board_features_s_t.to(DEVICE)
        }

    def learn_from_ReplayBuffer(self, experiences: tuple, gamma: float) -> None:
        """ Learns from experiences stored in the replay buffer.
        Args:
            experiences (tuple): A tuple containing:
                - s_t_b: Tensor of current state features.
                - s_prime_chosen_b: Tensor of chosen state features after action.
                - rewards_b: Tensor of rewards.
                - dones_b: Tensor of done flags.
            gamma (float): Discount factor for future rewards.
        """
        s_t_b, s_prime_chosen_b, rewards_b, dones_b = experiences
        
        if s_t_b.nelement() == 0: self.last_loss = None; return
        
        v_expected = self.v_network(s_t_b)
        with torch.no_grad(): v_of_s_prime_chosen = self.v_network(s_prime_chosen_b)
        
        # Calculate the expected value for the next state
        v_targets = rewards_b + (gamma * v_of_s_prime_chosen * (1 - dones_b))
        loss = self.criterion(v_expected, v_targets.detach())
        
        # Update the learning steps done
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()
        self.last_loss = loss.item()

    def expand_memory(self, reward:int, done:bool, state_info:dict[torch.Tensor,torch.Tensor]=None) -> None:
        """ Adds a new experience to the replay buffer.
        Args:
            reward (int): The reward received after taking the action.
            done (bool): Whether the episode has ended.
            state_info (dict, optional): Auxiliary information containing:
                - 'current_board_features_s_t': Features of the current board state before action.
                - 'features_s_prime_chosen': Features of the board state after the chosen action.
        """
        if state_info is None: print("Warning: DQNAgent.learn() called without state_info. No memory update will occur."); return
        
        board_features = state_info.get('current_board_features_s_t')
        chosen_features = state_info.get('features_s_prime_chosen')
        
        if board_features is None or chosen_features is None: print("Warning: DQNAgent.learn() called without valid features in aux_info. No memory update will occur."); return
        
        self.memory.add(board_features.cpu(), chosen_features.cpu(), reward, done)
        self.total_pieces_placed_overall += 1

    def reset(self) -> None:
        """ Resets the agent's internal state at the start of a new game/episode."""
        self.last_loss = None 

    def save(self, filename:str=None) -> None:
        """ Saves the current state of the DQN agent.
        Args:
            filename (str, optional): The path to save the model. If None, uses the default path from config.
        """
        path = filename or config.DQN_MODEL_PATH 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'v_network_state_dict': self.v_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_steps_done': self.learning_steps_done,
            'total_pieces_placed_overall': self.total_pieces_placed_overall,
            'epsilon_agent_internal': self.epsilon 
        }, path)
        print(f"DQN Agent (V-Learning, train.py controlled epsilon) saved to {path}")

    def load(self, filename:str=None) -> None:
        """ Loads the DQN agent's model from a file.
        Args:
            filename (str, optional): The path to load the model from. If None, uses the default path from config.
        """
        path = filename or config.DQN_MODEL_PATH
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.v_network.load_state_dict(checkpoint['v_network_state_dict'])
            if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_steps_done = checkpoint.get('learning_steps_done', 0)
            self.total_pieces_placed_overall = checkpoint.get('total_pieces_placed_overall', 0)
            self.epsilon = checkpoint.get('epsilon_agent_internal', config.DQN_EPSILON_START)
            self.v_network.train()
            print(f"DQN Agent (V-Learning, train.py controlled epsilon) loaded from {path}. "
                  f"Loaded learning steps: {self.learning_steps_done}, Agent internal epsilon: {self.epsilon:.4f}")
        else:
            print(f"ERROR: No DQN V-Learning model (train.py controlled epsilon) found at {path}.")
