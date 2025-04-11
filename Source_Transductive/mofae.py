from autograd import numpy as np
from autograd import elementwise_grad
from collections import defaultdict
import math
import json
import os
import sys
import pickle
import utility
import cvxpy as cp
import main_functions
from sklearn.linear_model import LinearRegression
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import random



########STEP 1: Initialization of Population#########
def initialize_population(X, W_sg_list, W_lime_list):
    '''
    args: 
        fn: function to be explained
        X: input data
        sg_dir: directory containing smoothgrad explanations
        lime_dir: directory containing
    returns:
        population: list of explanations
    # '''

    population = []
    N, D = X.shape
    for i in range(100): ##commenting this out for images
        print(f"W_sg_list[{i}] shape: {W_sg_list[i].shape}")
        assert W_sg_list[i].shape == (N, D), "W_sg has incorrect shape"
        population.append(W_sg_list[i])
    for i in range(3):
        assert W_lime_list[i].shape == (N, D), "W_lime has incorrect shape"
        population.append(W_lime_list[i])
    return population



#########STEP 2: Evaluation on each objective #########
def evaluate(X, fn, W_E, kernel_matrix_inv_negated, gradient_set, prop_combination):
    """
    Evaluate the objectives for a single explanation.

    Args:
        X (N, D): Input data.
        fn (callable): Function to explain.
        W_E (N, D): Explanation weights.
        kernel_matrix (N, N):  matrix of kernel_matrix.
        gradient_set (N, D): Gradient.

    Returns:
        tuple: Robustness and faithfulness losses.
    """
    N, D = gradient_set.shape
    assert W_E.shape == (N, D), "W_E has incorrect shape"
    
    if prop_combination == 'GA': 
        robustness_loss = utility.compute_robustness_loss(W_E, kernel_matrix_inv_negated)
        faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W_E)
        complexity_loss = None
    elif prop_combination == 'FA':
        robustness_loss = utility.compute_robustness_loss(W_E, kernel_matrix_inv_negated)
        faithfulness_loss = utility.compute_fn_matching_faithfulness_loss(W_E, fn, X, N, D)
        complexity_loss = None
    elif prop_combination == "GAC":
        robustness_loss = utility.compute_robustness_loss(W_E, kernel_matrix_inv_negated)
        faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W_E)
        complexity_loss = utility.compute_complexity_loss(W_E)
    else: 
        robustness_loss = None
        faithfulness_loss = None
        complexity_loss = None
        
    ### if None don't return the value
    if robustness_loss is None:
        return faithfulness_loss, complexity_loss
    elif faithfulness_loss is None:
        return robustness_loss, complexity_loss
    elif complexity_loss is None:
        return robustness_loss, faithfulness_loss
    return robustness_loss, faithfulness_loss, complexity_loss


#########STEP 3: Reproduction#########
def generate_new_population(population):
    ''' 
    Args:
        population (list): Current population of explanations, each with shape (N, D).
        reproduction_strategy (callable): Function defining the reproduction strategy.

    Returns:
        list: A new population of explanations (offspring).
    """
    '''
    print('population type', type(population))
    new_population = []
    for _ in range(len(population)):
        # Select two parent explanations randomly
        parent1, parent2 = random.sample(population, 2)
        N, D = parent1.shape
        # Generate offspring using the reproduction strategy
        print('parent 1 shape',parent1.shape)
        print('parent 2 shape', parent2.shape)
        offspring = reproduction_strategy(parent1, parent2)
        print(f"Expected shape: ({N}, {D})")
        print(f"Actual offspring shape: {offspring.shape}")
        assert offspring.shape == (N, D), "offspring has incorrect shape"

        new_population.append(offspring)
    print('new population shape', new_population[0].shape)

    return new_population

#########STEP 3.1: crossover#########

def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create an offspring.

    Args:
        parent1 (np.ndarray): First parent explanation (N, D).
        parent2 (np.ndarray): Second parent explanation (N, D).

    Returns:
        np.ndarray: The offspring explanation (N, D).
    """
    # Combine parents' weights (50-50 crossover)
    result = (parent1 + parent2) / 2
    assert result.shape == parent1.shape
    return result

#########STEP 3.2: mutate#########

def mutate(offspring, mutation_rate=0.5):
    """
    Apply mutation to the offspring explanation.

    Args:
        offspring (np.ndarray): The offspring explanation (N, D).
        mutation_rate (float): Standard deviation of the noise to add during mutation.

    Returns:
        np.ndarray: Mutated offspring explanation (N, D).
    """
    # Add random noise to the offspring
    noise = np.random.normal(scale=mutation_rate, size=offspring.shape)
    return offspring + noise

#########STEP 3.3: Reproduction Strategy#########
def reproduction_strategy(parent1, parent2):
    """
    Reproduce a new explanation using crossover(callable) and mutation(callable).

    Args:
        parent1 (np.ndarray): First parent explanation (N, D).
        parent2 (np.ndarray): Second parent explanation (N, D).

    Returns:
        np.ndarray: New offspring explanation (N, D).
    """
    # Perform crossover
    offspring = crossover(parent1, parent2)
    # Apply mutations
    offspring = mutate(offspring, mutation_rate=0.01)
    print('offspring shape', offspring.shape)
    return offspring

#########STEP 4: Selection Strategy#########

def select_by_reference_point(front, remaining_slots, combined_metrics, reference_points):
    """
    Select solutions from a Pareto front using reference points.

    Args:
        front (list): Indices of solutions in the Pareto front.
        remaining_slots (int): Number of slots remaining in the next generation.
        combined_metrics (list): Metrics for all solutions in the population.
        reference_points (np.ndarray): Predefined reference points in the objective space.

    Returns:
        list: Indices of selected solutions from the Pareto front.
    """
    # Extract the metrics of solutions in the Pareto front
    front_metrics = np.array([combined_metrics[i] for i in front])
    # Compute distances from all solutions to all reference points (vectorized)
    distances = np.linalg.norm(front_metrics[:, np.newaxis] - reference_points, axis=2)
    
    # Find the closest reference point for each solution
    closest_references = np.argmin(distances, axis=1)
    
    # Group solutions by reference points using defaultdict for efficiency
    reference_to_solutions = defaultdict(list)
    for solution_idx, reference_idx in zip(front, closest_references):
        reference_to_solutions[reference_idx].append(solution_idx)

    # Select solutions while maintaining diversity
    selected_solutions = []
    while len(selected_solutions) < remaining_slots:
        for solutions in reference_to_solutions.values():
            if solutions:
                selected_solutions.append(solutions.pop(0))
                if len(selected_solutions) == remaining_slots:
                    return selected_solutions
    print("selected solutions")
    return selected_solutions


#########STEP 4.1: Dominance Check#########
def dominates(metric_i, metric_j):
    """
    Check if one solution dominates another.
    Args:
        metric1 (tuple): Metrics of the first solution (robustness_loss, faithfulness_loss).
        metric2 (tuple): Metrics of the second solution (robustness_loss, faithfulness_loss).

    Returns:
        bool: True if metric1 dominates metric2.
    """
    at_least_one_better = any(m1 < m2 for m1, m2 in zip(metric_i, metric_j))
    
    # make sure no metric is worse
    no_worse_metrics = all(m1 <= m2 for m1, m2 in zip(metric_i, metric_j))
    
    return at_least_one_better and no_worse_metrics
#########STEP 4.2: Non Dominated Sorting#########

def non_dominated_sorting(metrics):
    """
    Perform non-dominated sorting of solutions based on metrics.

    Args:
        metrics (list): List of tuples (robustness_loss, faithfulness_loss).

    Returns:
        list: List of Pareto fronts, each containing indices of solutions.
    """
    
    num_solutions = len(metrics)
    pareto_fronts = []
    # dominated_counts = [0] * len(metrics)
    # dominance_set = [[] for _ in range(len(metrics))]
    dominated_counts = np.zeros(num_solutions, dtype=int) # Number of times a solution is dominated
    dominance_set = [[] for _ in range(num_solutions)]  # Solutions each solution dominates
    
    
    for i, metric_i in enumerate(metrics):
        for j, metric_j in enumerate(metrics):
            if i != j:
                if dominates(metric_i, metric_j):
                    dominance_set[i].append(j)  # i dominates j
                elif dominates(metric_j, metric_i):
                    dominated_counts[i] += 1  # i is dominated by j

    # First Pareto front: solutions that are **not** dominated
    first_front = [i for i in range(num_solutions) if dominated_counts[i] == 0]
    if first_front:
        pareto_fronts.append(first_front)

    # Build additional Pareto fronts
    current_front = 0
    while len(pareto_fronts[current_front]) > 0:
        next_front = []
        for i in pareto_fronts[current_front]:
            for j in dominance_set[i]:
                dominated_counts[j] -= 1
                if dominated_counts[j] == 0:
                    next_front.append(j)
        pareto_fronts.append(next_front)
        current_front += 1

    return pareto_fronts


#########STEP 4.3: Environment selection#########

def generate_reference_points(num_objectives, num_divisions):
    """
    Generate evenly distributed reference points in the objective space.

    Args:
        num_objectives (int): Number of objectives.
        num_divisions (int): Number of divisions per objective.

    Returns:
        np.ndarray: Array of reference points.
    """
    points = np.linspace(0, 1, num_divisions)
    reference_points = np.array(np.meshgrid(*[points] * num_objectives)).T.reshape(-1, num_objectives)
    reference_points = reference_points[np.sum(reference_points, axis=1) == 1]  
    print("generated reference points")
    return reference_points

def environment_selection(parents, offspring, kernel_matrix_inv_negated, gradient_set, max_population_size, reference_points,  prop_combination, fn, X):
    """
    Perform environment selection to create the next generation.

    Args:
        parents (list): Current population of explanations (N, D).
        offspring (list): Newly generated population of explanations (N, D).
        kernel_matrix_inv_negated (np.ndarray): Kernel matrix for evaluation (N, N).
        gradient_set (np.ndarray): Gradient set for evaluation (N, D).
        max_population_size (int): Maximum size of the next generation.

    Returns:
        list: Selected explanations for the next generation.
    """
    # Step 1: Combine parent and offspring populations
    N, D = gradient_set.shape
    combined_population = parents + offspring
    # Step 2: Evaluate metrics for all solutions in the combined population
    combined_metrics = [] 
    for i  in range(len(combined_population) - 1):
        W_E = combined_population[i]
        assert W_E.shape == (N, D), f"W_E has incorrect shape,shape is {W_E.shape}"
        combined_metrics.append(evaluate(X, fn, W_E, kernel_matrix_inv_negated, gradient_set, prop_combination))
    print(' combined_metrics 1', combined_metrics[0])

    # Step 3: non-dominated sorting
    pareto_fronts = non_dominated_sorting(combined_metrics)

    # Step 4: Select solutions from Pareto fronts
    next_generation = []
    for front in pareto_fronts:
        if len(next_generation) + len(front) <= max_population_size:
            # Add all solutions in the current front if space allows
            next_generation.extend([combined_population[i] for i in front])  # Map indices to solutions
        else:
            # Fill remaining slots using reference points for diversity
            remaining_slots = max_population_size - len(next_generation)
            selected_solutions = select_by_reference_point(front, remaining_slots, combined_metrics, reference_points)
            next_generation.extend([combined_population[i] for i in selected_solutions])  # Map back to solutions
            break

    # check if the next generation is a list of NumPy arrays
    assert isinstance(next_generation, list), "next_generation must be a list"
    assert all(isinstance(W_E, np.ndarray) for W_E in next_generation), \
        "next_generation must contain only NumPy arrays"
    return next_generation

    
######### STEP 5: MOFAE Algorithm #########

def compute_mofae(X, fn, gradient_set, kernel_matrix_inv_negated, sg_list, lime_list,num_objectives, prop_combination):
    """
    Full NSGA-III algorithm for optimizing feature explanations.

    Args:
        X (np.ndarray): Dataset (N, D) used for generating explanations.
        fn_cubed (callable): Function to explain.
        gradient_set (np.ndarray): Gradient set for evaluating faithfulness.
        kernel_matrix_inv_negated (np.ndarray): a matrix (N, N).
        sg_list (list): List of SmoothGrad explanations.
        lime_list (list): List of LIME explanations.
        num_objectives (int): Number of objectives.
        prop_combination (str): Combination of properties to optimize ('GA', 'FA', 'GAC').
    Returns:
        list: Final Pareto-optimal solutions (explanations).
    """
    
    N, D = X.shape
    print('shape of X', X.shape)
    num_divisions = 100
    max_generations = 500
    max_population_size = num_divisions + 1
    population = initialize_population(X, sg_list, lime_list) 
    assert type(population) == list, "population must be a list"
    assert all(isinstance(W_E, np.ndarray) for W_E in population), "population must contain only NumPy arrays"
    print("initialized population")
    reference_points = generate_reference_points(num_objectives, num_divisions)
    
    
    for generation in range(max_generations):
        print(f"Generation {generation + 1}/{max_generations}")
        offspring = generate_new_population(population)
        print("generated offspring")

        population = environment_selection(population, offspring, kernel_matrix_inv_negated, gradient_set, max_population_size, reference_points,  prop_combination, fn, X)
        print("selected population")
        print('population len', len(population))
    return population

def compute_mofae_for_images(X, fn, gradient_set,  kernel_matrix_inv_negated, sg_list, lime_list,num_objectives, prop_combination):
    """
    Full NSGA-III algorithm for optimizing feature explanations.

    Args:
        X (np.ndarray): Dataset (N, D) used for generating explanations.
        fn_cubed (callable): Function to explain.
        kernel_matrix_inv_negated (np.ndarray): a matrix (N, N).
        gradient_set (np.ndarray): Gradient set for evaluating faithfulness.
        population_size (int): Size of the population (τ).
        max_generations (int): Maximum number of generations.

    Returns:
        list: Final Pareto-optimal solutions (explanations).
    """
    N, D = X.shape
    print('shape of X', X.shape)

    num_divisions = 10
    max_generations = 50
    max_population_size = num_divisions + 1
    # max_population_size = 50

    population = initialize_population(X, sg_list, lime_list) 
    assert type(population) == list, "population must be a list"
    assert all(isinstance(W_E, np.ndarray) for W_E in population), "population must contain only NumPy arrays"
    print("initialized population")
    reference_points = generate_reference_points(num_objectives, num_divisions)

    for generation in range(max_generations):
        print(f"Generation {generation + 1}/{max_generations}")
        offspring = generate_new_population(population)
        print("generated offspring")
        population = environment_selection(population, offspring, kernel_matrix_inv_negated, gradient_set, max_population_size, reference_points,  prop_combination, fn, X)
        print("selected population")
        print('population len', len(population))
    return population

