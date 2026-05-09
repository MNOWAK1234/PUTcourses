import os
import copy
import torch
import random
import numpy as np

from typing import Tuple

import config
from src.tetris import Tetris
from .base_agent import BaseAgent, PolicyNetwork


DEVICE = config.DEVICE

class GeneticAlgorithmController:
    """ Genetic Algorithm Controller for Tetris, managing the evolution of policy networks. """
    def __init__(self, state_size,
        population_size=config.GA_POPULATION_SIZE,
        mutation_rate=config.GA_MUTATION_RATE,
        mutation_strength=config.GA_MUTATION_STRENGTH,
        crossover_rate=config.GA_CROSSOVER_RATE,
        tournament_size=config.GA_TOURNAMENT_SIZE,
        elitism_count=config.GA_ELITISM_COUNT,
        eval_games_per_individual=config.GA_EVAL_GAMES_PER_INDIVIDUAL,
        max_pieces_per_eval_game=config.GA_MAX_PIECES_PER_GA_EVAL_GAME
    ) -> None:
        """ Initializes the Genetic Algorithm Controller for Tetris.
        Args:
            state_size (int): Size of the state representation.
            population_size (int): Number of individuals in the population.
            mutation_rate (float): Probability of mutation for each weight.
            mutation_strength (float): Standard deviation of the Gaussian noise added during mutation.
            crossover_rate (float): Probability of performing crossover between two parents.
            tournament_size (int): Number of individuals to select for tournament selection.
            elitism_count (int): Number of top individuals to carry over to the next generation without modification.
            eval_games_per_individual (int): Number of games to evaluate each individual.
            max_pieces_per_eval_game (int): Maximum number of pieces to play in each evaluation game.
        """
        print(f"GA Controller initialized. Population: {population_size}")
        self.state_size = state_size
        self.action_size = 1  # Tetris has a single action space (x, rotation)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

        self.eval_games_per_individual = eval_games_per_individual
        self.max_pieces_per_eval_game = max_pieces_per_eval_game

        self.population = self._initialize_population()
        self.best_individual_network = None  # Stores the best PolicyNetwork found
        self.best_fitness = -float("inf")

    def _initialize_population(self) -> list:
        """Initializes the population with random PolicyNetworks."""
        population = []
        for i in range(self.population_size):
            individual_seed = config.SEED + i
            policy_net = PolicyNetwork(self.state_size, self.action_size, seed=individual_seed, fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(DEVICE)
            population.append(policy_net)
        return population

    def _evaluate_fitness(self, individual_policy_net:PolicyNetwork, eval_env_template:Tetris) -> float:
        """Evaluates fitness by playing a few full games.
        Args:
            individual_policy_net (PolicyNetwork): The policy network to evaluate.
            eval_env_template (Tetris): A template Tetris environment to create new game instances.
        Returns:
            float: The average score across multiple games played with the individual policy network.
        """
        total_rewards_for_individual = []
        individual_policy_net.eval()  # Set to evaluation mode

        for _ in range(self.eval_games_per_individual):
            # Create a fresh game environment for each evaluation game
            eval_env = Tetris() # eval_env_template

            current_board_features = eval_env.reset()
            if DEVICE.type == "cuda": current_board_features = current_board_features.cuda()

            game_score = 0
            game_over = False
            pieces_this_game = 0

            while not game_over and pieces_this_game < self.max_pieces_per_eval_game:
                next_steps_dict = eval_env.get_next_states()

                possible_actions_tuples = list(next_steps_dict.keys())
                feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

                with torch.no_grad(): action_scores = individual_policy_net(feature_tensors_for_next_steps)

                chosen_idx = torch.argmax(action_scores.squeeze()).item()
                chosen_action_tuple = possible_actions_tuples[chosen_idx]

                reward, game_over = eval_env.step(chosen_action_tuple, render=False)
                game_score += reward
                pieces_this_game += 1

                if not game_over:
                    current_board_features = eval_env.get_state_properties(eval_env.board)
                    if DEVICE.type == "cuda": current_board_features = current_board_features.cuda()

            total_rewards_for_individual.append(game_score)

        return (np.mean(total_rewards_for_individual) if total_rewards_for_individual else -float("inf"))

    def _selection(self, fitnesses:list[float]) -> list[PolicyNetwork]:
        """Selects parents for the next generation using tournament selection.
        Args:
            fitnesses (list): List of fitness scores for the current population.
        Returns:
            list: Selected parents for the next generation.
        """
        selected_parents = []
        for _ in range(self.population_size - self.elitism_count):
            tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index_in_tournament = np.argmax(tournament_fitnesses)
            selected_parents.append(self.population[tournament_indices[winner_index_in_tournament]])
        return selected_parents

    def _crossover(self, parent1_net: PolicyNetwork, parent2_net: PolicyNetwork) -> Tuple[PolicyNetwork, PolicyNetwork]:
        """Performs crossover between two parent networks to create two child networks.
        Args:
            parent1_net (PolicyNetwork): First parent network.
            parent2_net (PolicyNetwork): Second parent network.
        Returns:
            Tuple[PolicyNetwork, PolicyNetwork]: Two child networks created from the parents.
        """
        child1_net = PolicyNetwork(self.state_size, self.action_size, seed=random.randint(0, 1000000), fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(DEVICE)
        child2_net = PolicyNetwork(self.state_size, self.action_size, seed=random.randint(0, 1000000), fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(DEVICE)

        p1_weights = parent1_net.get_weights_flat()
        p2_weights = parent2_net.get_weights_flat()

        if random.random() < self.crossover_rate:
            # Averaging crossover
            child1_w = (p1_weights + p2_weights) / 2.0
            child2_w = (p1_weights + p2_weights) / 2.0

            child1_net.set_weights_flat(child1_w)
            child2_net.set_weights_flat(child2_w)
        else:
            child1_net.set_weights_flat(p1_weights)  # Clone
            child2_net.set_weights_flat(p2_weights)  # Clone
        return child1_net, child2_net

    def _mutate(self, individual_net: PolicyNetwork) -> PolicyNetwork:
        """Mutates an individual network by adding Gaussian noise to its weights.
        Args:
            individual_net (PolicyNetwork): The network to mutate.
        Returns:
            PolicyNetwork: The mutated network.
        """
        weights = individual_net.get_weights_flat()
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.mutation_strength)
                weights[i] += noise
        individual_net.set_weights_flat(weights)
        return individual_net

    def evolve_population(self, eval_env_template: Tetris) -> Tuple[float, float]:
        """Evolves the population by evaluating fitness, selecting parents, performing crossover, and applying mutation.
        Args:
            eval_env_template (Tetris): A template Tetris environment to create new game instances for evaluation.
        Returns:
            Tuple[float, float]: Mean and maximum fitness of the current generation.
        """
        fitnesses = [self._evaluate_fitness(ind, eval_env_template) for ind in self.population]

        current_gen_best_idx = np.argmax(fitnesses)
        if fitnesses[current_gen_best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[current_gen_best_idx]
            self.best_individual_network = copy.deepcopy(self.population[current_gen_best_idx])
            self.best_individual_network.to(DEVICE)
            print(f"    GA: New best overall fitness: {self.best_fitness:.2f}")

        new_population = []
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending sort

        for i in range(self.elitism_count):
            elite_individual = copy.deepcopy(self.population[sorted_indices[i]])
            new_population.append(elite_individual.to(DEVICE))

        parents = self._selection(fitnesses)

        num_offspring_needed = self.population_size - self.elitism_count
        offspring_generated = 0
        parent_idx = 0
        while offspring_generated < num_offspring_needed:
            p1 = parents[parent_idx % len(parents)]
            p2 = parents[(parent_idx + 1 + random.randint(0, len(parents) - 2)) % len(parents)]
            parent_idx += 1

            child1, child2 = self._crossover(p1, p2)
            new_population.append(self._mutate(child1))
            offspring_generated += 1
            if offspring_generated < num_offspring_needed:
                new_population.append(self._mutate(child2))
                offspring_generated += 1

        self.population = new_population[: self.population_size]
        return np.mean(fitnesses), np.max(fitnesses)

    def get_best_policy_network(self) -> PolicyNetwork:
        """Returns the best individual network found so far."""
        if self.best_individual_network is None and self.population:
            print("Warning: Best individual network not yet identified. Returning first in population.")
            return self.population[0]
        return self.best_individual_network

    def save_best_individual(self, filename:str=None) -> None:
        """Saves the best individual network to a file.
        Args:
            filename (str, optional): Path to save the best individual network. Defaults to config.GA_MODEL_PATH.
        """
        path = filename or config.GA_MODEL_PATH
        if self.best_individual_network:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.best_individual_network.state_dict(), path)
            print(f"Best GA Individual saved to: {path}")
        else:
            print("No best GA individual to save yet.")

    def load_best_individual(self, filename:str=None, state_size_override:int=None) -> bool:
        """Loads the best individual network from a file.
        Args:
            filename (str, optional): Path to load the best individual network from. Defaults to config.GA_MODEL_PATH.
            state_size_override (int, optional): Override for the state size if needed. Defaults to None.
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        path = filename or config.GA_MODEL_PATH
        _state_size = (state_size_override or self.state_size)
        if not _state_size:
            print("Error: Cannot load GA model without state_size.")
            return False

        if os.path.exists(path):
            loaded_net = PolicyNetwork(_state_size, self.action_size, seed=0, fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(DEVICE)
            loaded_net.load_state_dict(torch.load(path, map_location=DEVICE))
            loaded_net.eval()
            self.best_individual_network = loaded_net
            print(f"Best GA Individual loaded from {path}. Fitness needs re-evaluation.")
            return True
        else:
            print(f"Error: GA model file not found at {path}")
            return False

class GeneticAgent(BaseAgent):
    """Agent wrapper that uses the best policy found by GA Controller."""

    def __init__(self, state_size:int, seed:int=0, policy_network_instance:PolicyNetwork=None) -> None:
        super().__init__(state_size)
        self.policy_network = policy_network_instance
        if self.policy_network is None:
            print("GeneticAgent initialized without a policy network. Will use a default one (likely untrained).")
            self.policy_network = PolicyNetwork(state_size, action_size=1, seed=seed, fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(DEVICE)
        self.policy_network.eval()

    def select_action(self, current_board_features:torch.Tensor, tetris_game_instance:Tetris, epsilon_override:float=None) -> Tuple[Tuple[int, int], dict]:
        """Selects an action based on the current board features using the policy network.
        Args:
            current_board_features (torch.Tensor): The current state of the Tetris board.
            tetris_game_instance (Tetris): The current Tetris game instance.
            epsilon_override (float, optional): Override for exploration rate. Defaults to None.
        Returns:
            Tuple[Tuple[int, int], dict]: The chosen action as a tuple (x, rotation) and auxiliary information.
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict:
            return (tetris_game_instance.width // 2, 0), {}  # Fallback

        possible_actions_tuples = list(next_steps_dict.keys())
        feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

        with torch.no_grad(): action_scores = self.policy_network(feature_tensors_for_next_steps)

        chosen_idx = torch.argmax(action_scores.squeeze()).item()
        chosen_action_tuple = possible_actions_tuples[chosen_idx]

        # Aux_info can be minimal for GA during inference
        return chosen_action_tuple, {"features_after_chosen_action": feature_tensors_for_next_steps[chosen_idx]}

    def set_policy_network(self, policy_net:PolicyNetwork) -> None:
        """Sets the policy network for this agent.
        Args:
            policy_net (PolicyNetwork): The policy network to set.
        """
        self.policy_network = policy_net
        if self.policy_network:
            self.policy_network.to(DEVICE).eval()
    
    def reset(self) -> None:
        """ Reset method is not applicable for GeneticAgent, but defined for interface compatibility. """
        pass
    
    def save(self, filepath:str=None):
        path = (filepath or config.GA_MODEL_PATH)
        if self.policy_network:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.policy_network.state_dict(), path)
            print(f"GeneticAgent's current policy saved to: {path}")
        else:
            print("GeneticAgent has no policy network to save.")

    def load(self, filepath:str=None):
        path = filepath or config.GA_MODEL_PATH
        if os.path.exists(path):
            if self.policy_network is None:
                self.policy_network = PolicyNetwork(self.state_size, action_size=1, seed=0, fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(DEVICE)
            self.policy_network.load_state_dict(torch.load(path, map_location=DEVICE))
            self.policy_network.eval()
            print(f"GeneticAgent policy loaded from: {path}")
        else:
            print(f"Error: GA model file not found at {path} for GeneticAgent instance.")
            raise FileNotFoundError(f"GA model not found for GeneticAgent: {path}")
