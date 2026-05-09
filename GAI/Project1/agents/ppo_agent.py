import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple

import config
from src.tetris import Tetris
from .base_agent import BaseAgent, PolicyNetwork


DEVICE = config.DEVICE

class PPOAgent(BaseAgent):
    """ Proximal Policy Optimization (PPO) Agent for Tetris"""
    
    def __init__(self, state_size, seed=0):
        """ Initializes the PPO Agent with the given state size and seed.
        Args:
            state_size (int): Size of the state representation.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(state_size)
        
        # Hyperparameters from config
        self.gamma = config.PPO_GAMMA
        self.gae_lambda = config.PPO_GAE_LAMBDA
        self.ppo_epochs = config.PPO_EPOCHS_PER_UPDATE
        self.ppo_clip = config.PPO_CLIP_EPSILON
        self.entropy_coeff = config.PPO_ENTROPY_COEFF
        self.value_loss_coeff = config.PPO_VALUE_LOSS_COEFF
        self.batch_size = config.PPO_BATCH_SIZE
        self.update_horizon = config.PPO_UPDATE_HORIZON

        # Actor and Critic networks
        self.actor = PolicyNetwork(state_size, seed=seed).to(DEVICE)
        self.critic = PolicyNetwork(state_size, seed=seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.PPO_ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.PPO_CRITIC_LR)

        # Buffer to store trajectory data
        self.memory = {
            "s_t_features": [],
            "all_s_prime_features_lists": [],
            "chosen_action_indices": [],
            "log_probs_old": [],
            "rewards": [],
            "dones": [],
            "values_s_t": [],
        }
        self.last_loss = {}

        print(f"PPO Agent initialized. Update Horizon: {self.update_horizon}, Device: {DEVICE}")

    def select_action(self, current_board_features_s_t:torch.Tensor, tetris_game_instance:Tetris, epsilon_override:float=None) -> Tuple[Tuple[int, int], dict]:
        """ Selects an action based on the current state features and the Tetris game instance.
        Args:
            current_board_features_s_t (torch.Tensor): Features of the current board state S_t.
            tetris_game_instance (Tetris): The current Tetris game instance.
            epsilon_override (float, optional): Not used.
        Returns:
            Tuple: A tuple containing the chosen action (as coordinates) and auxiliary information.
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict: return (tetris_game_instance.width // 2, 0), {}  # Return default action, empty aux_info

        all_s_prime_features_list = [s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()]
        all_s_prime_features_batch = torch.stack(all_s_prime_features_list)

        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            action_scores = self.actor(all_s_prime_features_batch).squeeze(-1)
            value_s_t = self.critic(current_board_features_s_t.unsqueeze(0).to(DEVICE))
            
        self.actor.train()
        self.critic.train()

        action_probs = F.softmax(action_scores, dim=0)
        dist = Categorical(action_probs)
        chosen_action_index_tensor = dist.sample()
        chosen_action_index = chosen_action_index_tensor.item()

        log_prob_tensor = dist.log_prob(chosen_action_index_tensor)
        entropy_tensor = dist.entropy()

        chosen_action_tuple = list(next_steps_dict.keys())[chosen_action_index]

        aux_info = {
            "log_prob": log_prob_tensor,
            "entropy": entropy_tensor,
            "value_s_t": value_s_t.squeeze(),
            "all_available_s_prime_features": all_s_prime_features_list,
            "chosen_action_index": chosen_action_index,
        }
        return chosen_action_tuple, aux_info

    def learn(self, state_features:torch.Tensor, reward:int, next_state_features:torch.Tensor, done:bool, aux_info:dict) -> None:
        """ Learn from the current state, reward, next state, and auxiliary information.
        Args:
            state_features (torch.Tensor): Features of the current state S_t.
            reward (int): Reward received from the environment.
            next_state_features (torch.Tensor): Features of the next state S_{t+1}.
            done (bool): Whether the episode has ended.
            aux_info (dict): Auxiliary information containing log probabilities, entropy, and values.
        """
        if not aux_info: return

        self.memory["s_t_features"].append(state_features.cpu())
        self.memory["all_s_prime_features_lists"].append([f.cpu() for f in aux_info["all_available_s_prime_features"]])
        self.memory["chosen_action_indices"].append(aux_info["chosen_action_index"])
        self.memory["log_probs_old"].append(aux_info["log_prob"].cpu())
        self.memory["values_s_t"].append(aux_info["value_s_t"].cpu())
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)

        # If buffer is full, trigger the update
        if len(self.memory["rewards"]) >= self.update_horizon: self.learn_from_memory(next_state_features)

    def learn_on_episode_end(self) -> None:
        """ Learn from the memory at the end of an episode."""
        if len(self.memory["rewards"]) > 0: self.learn_from_memory(last_s_t_plus_1_features=None)

    def learn_from_memory(self, last_s_t_plus_1_features:torch.Tensor=None) -> None:
        """ Learn from the collected memory.
        Args:
            last_s_t_plus_1_features (torch.Tensor, optional): Features of the next state S_{t+1}. If None, assumes the episode ended.
        """
        if len(self.memory["rewards"]) == 0:  # Nothing to learn from
            self.last_loss = {}
            return

        # --- 1. Calculate Advantages and Returns (Targets for Critic) ---
        rewards = torch.tensor(self.memory["rewards"], dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(self.memory["dones"], dtype=torch.float32).to(DEVICE)
        
        if not self.memory["values_s_t"]:
            print("Warning: PPO memory for values_s_t is empty during learn_from_memory.")
            self.clear_memory()
            return

        values_cpu = self.memory["values_s_t"]
        if all(isinstance(v, torch.Tensor) for v in values_cpu): values = (torch.stack(values_cpu).squeeze().to(DEVICE))
        else: values = torch.tensor(values_cpu, dtype=torch.float32).to(DEVICE)

        # Ensure values tensor is not empty and has correct dimensions
        if (values.nelement() == 0 or values.dim() == 0):
            if (len(rewards) == 1 and values.nelement() == 1 and values.dim() == 0): 
                values = values.unsqueeze(0)
            else:
                print(f"Warning: PPO values tensor is problematic. Shape: {values.shape}. Memory length: {len(rewards)}")
                self.clear_memory()
                return

        # Ensure rewards, dones, values have the same length (number of trajectory steps)
        if not (len(rewards) == len(dones) == len(values)):
            print(f"Warning: Mismatch in PPO memory lengths. R:{len(rewards)}, D:{len(dones)}, V:{len(values)}")
            self.clear_memory()
            return

        advantages = []
        gae = 0.0

        with torch.no_grad():
            if last_s_t_plus_1_features is not None and not self.memory["dones"][-1]:
                last_value = self.critic(last_s_t_plus_1_features.unsqueeze(0).to(DEVICE)).squeeze()
            else: 
                last_value = torch.tensor(0.0, device=DEVICE)

        for i in reversed(range(len(rewards))):
            next_val_bootstrap = values[i + 1] if i + 1 < len(values) else last_value

            delta = (rewards[i] + self.gamma * next_val_bootstrap * (1 - dones[i]) - values[i])
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        if not advantages:
            self.clear_memory()
            return

        advantages = torch.stack(advantages)
        returns_for_critic = advantages + values

        # Normalize advantages
        if advantages.numel() > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        elif (advantages.numel() == 1): advantages = torch.zeros_like(advantages)

        # --- 2. Convert other memory lists to tensors ---
        old_log_probs_cpu = self.memory["log_probs_old"]
        if not all(isinstance(lp, torch.Tensor) for lp in old_log_probs_cpu):
            old_log_probs = torch.tensor(old_log_probs_cpu, dtype=torch.float32).to(DEVICE)
        else:
            old_log_probs = torch.stack(old_log_probs_cpu).to(DEVICE)

        old_s_t_features = torch.stack(self.memory["s_t_features"]).to(DEVICE)
        old_all_s_prime_lists_cpu = self.memory["all_s_prime_features_lists"]
        old_chosen_action_indices = torch.tensor(self.memory["chosen_action_indices"], dtype=torch.long).to(DEVICE)

        num_samples = len(rewards)
        sample_indices = np.arange(num_samples)

        actor_loss_epoch, critic_loss_epoch, entropy_epoch = (0, 0, 0)

        # --- 3. Perform PPO Update for K Epochs ---
        for k_epoch_num in range(self.ppo_epochs):
            np.random.shuffle(sample_indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = sample_indices[start:end]

                mb_s_t = old_s_t_features[batch_idx]
                mb_all_s_prime_lists_cpu = [old_all_s_prime_lists_cpu[i] for i in batch_idx]
                mb_chosen_indices = old_chosen_action_indices[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns_for_critic[batch_idx]

                new_log_probs_list, entropy_list = [], []
                valid_batch_item_count = 0
                for i in range(len(batch_idx)):  # Iterate over items in the minibatch
                    s_prime_features_for_this_s_t_cpu_list = mb_all_s_prime_lists_cpu[i]

                    if not s_prime_features_for_this_s_t_cpu_list: continue

                    s_prime_features_batch_for_item = torch.stack(s_prime_features_for_this_s_t_cpu_list).to(DEVICE)

                    action_scores = self.actor(s_prime_features_batch_for_item).squeeze(-1)

                    # Check for NaNs in scores
                    if torch.isnan(action_scores).any():
                        print(f"Warning: NaN detected in action_scores for PPO update. Epoch {k_epoch_num}, Batch start {start}, Item index {i}")

                    dist = Categorical(logits=action_scores)

                    new_log_probs_list.append(dist.log_prob(mb_chosen_indices[i]))
                    entropy_list.append(dist.entropy())
                    valid_batch_item_count += 1

                if valid_batch_item_count == 0:
                    continue

                new_log_probs = torch.stack(new_log_probs_list)
                entropy = torch.stack(entropy_list).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs[:valid_batch_item_count])

                # Slice advantages and returns to match valid items
                mb_adv_valid = mb_advantages[:valid_batch_item_count]
                mb_ret_valid = mb_returns[:valid_batch_item_count]

                surr1 = ratio * mb_adv_valid
                surr2 = (torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv_valid)
                actor_loss = -torch.min(surr1, surr2).mean()

                new_values = self.critic(mb_s_t[:valid_batch_item_count]).squeeze()
                critic_loss = F.mse_loss(new_values, mb_ret_valid)

                total_loss = (actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.item()

        num_update_loops = self.ppo_epochs * (num_samples // self.batch_size + int(num_samples % self.batch_size != 0))
        if num_update_loops == 0: num_update_loops = 1

        self.last_loss = {
            "actor": actor_loss_epoch / num_update_loops,
            "critic": critic_loss_epoch / num_update_loops,
            "entropy": entropy_epoch / num_update_loops,
        }
        self.clear_memory()

    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        for key in self.memory:
            self.memory[key] = []

    def reset(self) -> None:
        """Reset the agent's memory and last loss."""
        self.clear_memory()
        self.last_loss = {}
    
    def save(self, filepath:str=None) -> None:
        """ Save the actor and critic networks to a file.
        Args:
            filepath (str, optional): Path to save the model. Defaults to config.PPO_MODEL_PATH.
        """
        path = filepath or config.PPO_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
        print(f"PPO Agent saved to {path}")

    def load(self, filepath:str=None) -> None:
        """ Load the actor and critic networks from a file.
        Args:
            filepath (str, optional): Path to load the model from. Defaults to config.PPO_MODEL_PATH.
        """
        path = filepath or config.PPO_MODEL_PATH
        print(f"Loading PPO Agent from {path}...")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            print(f"PPO Agent loaded from {path}")
        else:
            print(f"Error: Model not found at {path}")
