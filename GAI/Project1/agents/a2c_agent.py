import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import config
from helper import *
from typing import Tuple
from src.tetris import Tetris
from .base_agent import BaseAgent, PolicyNetwork

DEVICE = config.DEVICE

class ValueNetwork(nn.Module):
    """ Value Network for the critic in A2C algorithm. """
    def __init__(self, state_size:int=config.STATE_SIZE, seed:int=config.SEED, hidden_size:int=64) -> None:
        """ Initialize the Value Network for the critic.
        Args:
            state_size (int): Size of the state representation.
            seed (int): Random seed for reproducibility.
            hidden_size (int): Size of the hidden layers.
        """
        torch.manual_seed(seed)
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ Forward pass through the network.
        Args:
            x (Tensor): Input tensor representing the state.
        Returns:
            Tensor: Output value for the state.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class A2CAgent(BaseAgent):
    """ A2C Agent implementing the Advantage Actor-Critic algorithm. """
    def __init__(self, state_size:int, seed:int=config.SEED) -> None:
        """ Initialize the A2C agent with actor and critic networks.
        Args:
            state_size (int): Size of the state representation.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(state_size)
        self.actor = PolicyNetwork(state_size, seed=seed).to(DEVICE)
        self.critic = ValueNetwork(state_size, seed=seed+1).to(DEVICE)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.A2C_LEARNING_RATE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.A2C_LEARNING_RATE)

        self.last_loss = None

    def select_action(self, current_board_features:torch.Tensor|torch.FloatTensor, env:Tetris, epsilon_override=None) -> Tuple[Tuple[int,int], dict[str, any]]:
        """ Select an action based on the current board features and the environment.
        Args:
            current_board_features (Tensor): Features of the current board state.
            env (Tetris): The Tetris environment instance.
        Returns:
            tuple: A tuple containing the chosen action and auxiliary information.
        """
        next_states = env.get_next_states()
        action_tuples = list(next_states.keys())
        state_features = [next_states[a].to(DEVICE) for a in action_tuples]
        stacked = torch.stack(state_features)  # (N, state_size)

        logits = self.actor(stacked).squeeze(-1)  # (N,)
        probs = F.softmax(logits, dim=0)
        dist = Categorical(probs)
        idx = dist.sample().item()
        chosen_action = action_tuples[idx]

        return chosen_action, {
            "log_prob": dist.log_prob(torch.tensor(idx, device=DEVICE)),
            "entropy": dist.entropy(),
            "value_of_current_board": self.critic(current_board_features),
            "features_s_prime_chosen": state_features[idx],
        }

    def learn(self, reward:int, next_state_features:torch.FloatTensor|torch.Tensor, done:bool, aux_info:dict) -> None:
        """
        Update the actor and critic networks based on the reward and next state features.
        Args:
            reward(int): Reward received from the environment.           
            next_state_features(Tensor): Features of the next state after taking the action.        
            done(bool): Boolean indicating if the episode has ended.      
            aux_info(dict): Auxiliary information containing log probability, entropy, and value of the current board.
        """
        log_prob = aux_info["log_prob"]
        entropy = aux_info["entropy"]
        value_s = aux_info["value_of_current_board"]
        value_s_prime = self.critic(next_state_features).detach()

        # Calculate target and advantage
        target = reward + (config.A2C_GAMMA * value_s_prime * (1 - int(done)))
        advantage = target - value_s

        # Calculate losses
        critic_loss = advantage.pow(2)
        actor_loss = -log_prob * advantage.detach() - config.A2C_ENTROPY_COEFF * entropy

        # Update actor network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.last_loss = (actor_loss.item(), critic_loss.item())

    def reset(self) -> None:
        """ Reset the agent's state for a new episode. """
        self.last_loss = None

    def save(self, filepath:str=None) -> None:
        """ Save the actor and critic networks to a file.
        Args:
            filepath (str, optional): Path to save the model. Defaults to config.A2C_MODEL_PATH.
        """
        path = filepath or config.A2C_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
        print(f"A2C Agent saved to {path}")

    def load(self, filepath:str=None) -> None:
        """ Load the actor and critic networks from a file.
        Args:
            filepath (str, optional): Path to load the model from. Defaults to config.A2C_MODEL_PATH.
        """
        path = filepath or config.A2C_MODEL_PATH
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            print(f"A2C Agent loaded from {path}")
        else:
            print(f"Error: Model not found at {path}")
