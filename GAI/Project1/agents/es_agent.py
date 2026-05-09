import os
import torch
import numpy as np
from typing import Tuple

import config
from src.tetris import Tetris
from .base_agent import BaseAgent, PolicyNetwork

DEVICE = config.DEVICE

class ESAgent(BaseAgent):
    """Evolution Strategy (ES) Agent for Tetris."""
    def __init__(self, state_size,
        population_size=config.ES_POPULATION_SIZE,
        sigma=config.ES_SIGMA,
        learning_rate=config.ES_LEARNING_RATE,
        eval_games_per_param=config.ES_EVAL_GAMES_PER_PARAM,
        max_pieces_per_eval_game=config.ES_MAX_PIECES_PER_ES_EVAL_GAME,
        seed:int=0
    ) -> None:
        """ Initializes the ES Agent with the given parameters.
        Args:
            state_size (int): Size of the state representation.
            population_size (int): Number of candidate solutions in the population.
            sigma (float): Standard deviation for perturbations.
            learning_rate (float): Learning rate for updating the policy.
            eval_games_per_param (int): Number of games to evaluate each candidate parameter set.
            max_pieces_per_eval_game (int): Maximum pieces per evaluation game.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(state_size)

        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.eval_games_per_param = eval_games_per_param
        self.max_pieces_per_eval_game = max_pieces_per_eval_game

        # Central policy network whose parameters are evolved
        self.central_policy_net = PolicyNetwork(state_size, action_size=1, seed=seed, fc1_units=config.ES_FC1_UNITS, fc2_units=config.ES_FC2_UNITS).to(DEVICE)
        self.num_params = self.central_policy_net.get_weights_flat().size
        self.current_best_fitness = -float('inf')
        self.current_best_weights = self.central_policy_net.get_weights_flat().copy()

        print(f"ES Agent (Controller) initialized. Pop_Size: {self.population_size}, Sigma: {self.sigma}, LR: {self.learning_rate}")

    def _evaluate_parameters(self, flat_weights_candidate:np.ndarray, eval_env_template:Tetris) -> float:
        """ Evaluates a candidate set of parameters by running multiple games.
        Args:
            flat_weights_candidate (np.ndarray): Flattened weights of the policy network to evaluate.
            eval_env_template (Tetris): A template environment to evaluate the policy.
        Returns:
            float: The average score across multiple games with the candidate parameters.
        """
        temp_policy_net = PolicyNetwork(self.state_size, action_size=1, fc1_units=config.ES_FC1_UNITS, fc2_units=config.ES_FC2_UNITS).to(DEVICE) # seed for structure only
        temp_policy_net.set_weights_flat(flat_weights_candidate)
        temp_policy_net.eval()

        total_rewards = []
        for _ in range(self.eval_games_per_param):
            
            eval_env = Tetris() # eval_env_template
            
            current_board_features = eval_env.reset()
            if DEVICE.type == 'cuda': current_board_features = current_board_features.cuda()
            
            game_score = 0
            game_over = False
            pieces_this_game = 0

            while not game_over and pieces_this_game < self.max_pieces_per_eval_game:
                next_steps_dict = eval_env.get_next_states()
                if not next_steps_dict: break

                possible_actions_tuples = list(next_steps_dict.keys())
                feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

                with torch.no_grad(): action_scores = temp_policy_net(feature_tensors_for_next_steps)
                
                chosen_idx = torch.argmax(action_scores.squeeze()).item()
                chosen_action_tuple = possible_actions_tuples[chosen_idx]

                reward, game_over = eval_env.step(chosen_action_tuple, render=False)
                game_score += reward
                pieces_this_game +=1
                
                if not game_over:
                    current_board_features = eval_env.get_state_properties(eval_env.board)
                    if DEVICE.type == 'cuda': current_board_features = current_board_features.cuda()
            total_rewards.append(game_score)
        return np.mean(total_rewards) if total_rewards else -float('inf')

    def learn(self, eval_env_template: Tetris) -> Tuple[float, float, float]:
        """ Perform one iteration of the Evolution Strategy learning process.
        Args:
            eval_env_template (Tetris): A template environment to evaluate the policy.
        Returns:
            Tuple[float, float, float]: Mean fitness of the population, max fitness in this generation, and fitness of the central policy.
        """
        
        current_central_weights = self.central_policy_net.get_weights_flat()
        noise_samples = np.random.randn(self.population_size, self.num_params)
        
        fitness_scores = np.zeros(self.population_size)
        for i in range(self.population_size):
            perturbed_weights = current_central_weights + self.sigma * noise_samples[i]
            fitness_scores[i] = self._evaluate_parameters(perturbed_weights, eval_env_template)
        
        # Standardize fitness scores
        if np.std(fitness_scores) > 1e-6:
            standardized_fitness = (fitness_scores - np.mean(fitness_scores)) / np.std(fitness_scores)
        else:
            standardized_fitness = np.zeros_like(fitness_scores)

        update_direction = np.dot(noise_samples.T, standardized_fitness)
        new_central_weights = current_central_weights + (self.learning_rate / (self.population_size * self.sigma)) * update_direction
        self.central_policy_net.set_weights_flat(new_central_weights)

        # Evaluate the new central policy to track its own fitness (and update best_overall)
        current_eval_of_central = self._evaluate_parameters(new_central_weights, eval_env_template)
        if current_eval_of_central > self.current_best_fitness:
            self.current_best_fitness = current_eval_of_central
            self.current_best_weights = new_central_weights.copy()
            print(f"==> ES: New best central policy fitness: {self.current_best_fitness:.2f}")
        
        return np.mean(fitness_scores), np.max(fitness_scores), current_eval_of_central

    def select_action(self, current_board_features: torch.Tensor, tetris_game_instance: Tetris, epsilon_override=None) -> Tuple[Tuple[int, int],dict[str, torch.Tensor | None]]:
        """ Selects an action based on the current board features using the central policy network.
        Args:
            current_board_features (torch.Tensor): The current state of the Tetris board.
            tetris_game_instance (Tetris): The current Tetris game instance.
            epsilon_override (float, optional): Override for exploration rate. Defaults to None.
        Returns:
            Tuple[Tuple[int, int], dict[str, torch.Tensor | None]]: The chosen action as a tuple (x, rotation) and auxiliary information.
        """
        self.central_policy_net.set_weights_flat(self.current_best_weights)
        self.central_policy_net.eval()

        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict: return (tetris_game_instance.width // 2, 0), {}

        possible_actions_tuples = list(next_steps_dict.keys())
        feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

        with torch.no_grad(): action_scores = self.central_policy_net(feature_tensors_for_next_steps)
        chosen_idx = torch.argmax(action_scores.squeeze()).item()
        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        
        return chosen_action_tuple, {'features_after_chosen_action': feature_tensors_for_next_steps[chosen_idx]}

    def reset(self) -> None:
        """ Reset method is not applicable for ESAgent, but defined for interface compatibility. """
        pass
    
    def save(self, filepath:str=None) -> None:
        """ Save the ES agent's policy network to a file.
        Args:
            filepath (str, optional): Path to save the model file. If None, uses the global config path.
        """
        path = filepath or config.ES_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.central_policy_net.set_weights_flat(self.current_best_weights)
        torch.save(self.central_policy_net.state_dict(), path)
        print(f"ES Agent (best policy) saved to: {path}")

    def load(self, filepath:str=None) -> None:
        """ Load the ES agent's policy network from a file.
        Args:
            filepath (str, optional): Path to the model file. If None, uses the global config path.
        """
        path = filepath or config.ES_MODEL_PATH
        if os.path.exists(path):
            self.central_policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
            self.central_policy_net.eval()
            self.current_best_weights = self.central_policy_net.get_weights_flat().copy()
            print(f"ES Agent (policy) loaded from {path}. Fitness needs re-evaluation.")
        else:
            print(f"Error: ES model file not found at {path}")
            raise FileNotFoundError(f"ES model not found: {path}")