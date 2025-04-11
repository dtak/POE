import utility
import time 
import os
import torch
import pickle
import cvxpy as cp
import autograd.numpy as np
from autograd import grad, elementwise_grad
import pandas as pd
from sklearn.linear_model import LinearRegression
import itertools
import scipy.special
from scipy.special import comb
from scipy.special import binom
from scipy import linalg


def compute_agg_explanations_new(W_lime_list, W_sg_list, gradient_set, lambda_value,
                              kernel_matrix_inv_negated, fn, prop_combination, X):
    """
    Note: this is a memory-optimized version of compute_agg_explanations for high-dimensional settings like images.
    Aggregates explanations using weighted combinations of W_lime_list and W_sg_list,
    optimized via convex combination. This version is memory-optimized for high-dimensional settings.

    Returns:
        dict with optimized alpha values and aggregated explanation matrix W (N, D)
    """
    N, D = X.shape
    num_lime = len(W_lime_list)
    num_sg = len(W_sg_list)

    # Stack explanation matrices for efficiency
    W_lime_array = np.stack(W_lime_list, axis=0)  # shape: (num_lime, N, D)
    W_sg_array = np.stack(W_sg_list, axis=0)      # shape: (num_sg, N, D)

    # Flatten to 2D for matrix-vector multiplication
    W_lime_flat = W_lime_array.reshape(num_lime, -1)  # shape: (num_lime, N*D)
    W_sg_flat = W_sg_array.reshape(num_sg, -1)        # shape: (num_sg, N*D)
    alpha_lime = cp.Variable(num_lime, nonneg=True)
    alpha_sg = cp.Variable(num_sg, nonneg=True)
    W_lime = cp.reshape(alpha_lime @ W_lime_flat, (N, D))
    W_sg = cp.reshape(alpha_sg @ W_sg_flat, (N, D))
    w = W_lime + W_sg


    if prop_combination == 'GA':
        faithfulness_loss = utility.compute_faithfulness_loss_cv(gradient_set, w)
        robustness_loss = utility.compute_robustness_loss_cv(w, kernel_matrix_inv_negated)

    elif prop_combination == 'FA':
        faithfulness_loss = utility.compute_fn_matching_faithfulness_loss_cv(w, fn, X, N, D)
        robustness_loss = utility.compute_robustness_loss_cv(w, kernel_matrix_inv_negated)

    else:
        raise ValueError(f"Unknown prop_combination: {prop_combination}")


    objective = cp.Minimize(lambda_value * faithfulness_loss + (1 - lambda_value) * robustness_loss)


    constraints = [
        cp.sum(alpha_lime) + cp.sum(alpha_sg) == 1
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=True)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: solver did not find an optimal solution!")


    results = {
        "alpha_lime": alpha_lime.value,
        "alpha_sg": alpha_sg.value,
        "agg_explanation": w.value  # shape: (N, D)
    }

    return results

def compute_agg_explanations(W_lime_list, W_sg_list, gradient_set, lambda_value, kernel_matrix_inv_negated, fn, prop_combination, X):
    
    """
    Aggregates explanations using weighted combinations of W_lime_list and W_sg_list.

    Args:
        W_lime_list: List of W_lime matrices.
        W_sg_list: List of W_sg matrices.
        kernel_matrix_inv_negated: (N, N) matrix, the inverse of the kernel matrix negated.
        gradient_set: The gradient set used for computing faithfulness loss.
        lambda_value: Trade-off parameter.
        fn: Function to explain.

    Returns:
        A dictionary containing the optimized alpha values and aggregated explanation matrix.
    """
    N, D = X .shape
    num_lime = len(W_lime_list)
    num_sg = len(W_sg_list)


    alpha_lime = cp.Variable(num_lime, nonneg=True)
    alpha_sg = cp.Variable(num_sg, nonneg=True)


    W_lime = sum(alpha_lime[i] * W_lime_list[i] for i in range(num_lime))
    W_sg = sum(alpha_sg[j] * W_sg_list[j] for j in range(num_sg))


    w = W_lime + W_sg


    print(f"W_lime_list length: {num_lime}")
    print(f"W_sg_list length: {num_sg}")
    print(f"gradient_set shape: {gradient_set.shape}")
    print(f"w shape: {w.shape}")

    if prop_combination == 'GA':
        faithfulness_loss = utility.compute_faithfulness_loss_cv(gradient_set, w)
        print(f"compute faithfulness loss for agg's objective function")
        robustness_loss = utility.compute_robustness_loss_cv(w, kernel_matrix_inv_negated)

    elif prop_combination =='FA':
        faithfulness_loss = utility.compute_fn_matching_faithfulness_loss_cv(w, fn, X, N, D)
        print(f"compute faithfulness loss for agg's objective function")
        robustness_loss = utility.compute_robustness_loss_cv(w, kernel_matrix_inv_negated)
    print(f"compute robustness loss for agg's objective function")
    objective = cp.Minimize(lambda_value * faithfulness_loss + (1 - lambda_value) * robustness_loss)

    constraints = [
        cp.sum(alpha_lime) + cp.sum(alpha_sg) == 1,  
    ]
    problem = cp.Problem(objective, constraints)
    # problem.solve(solver=cp.SCS, verbose=True, eps=1e-6) ECOS
    problem.solve(solver=cp.SCS, verbose=True)
    # Check solver status
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: Solver did not find an optimal solution!")
    # problem.solve(solver=cp.SCS, verbose=True)
    print("solved the agg objective function")

    results = {
        "alpha_lime": alpha_lime.value,
        "alpha_sg": alpha_sg.value,
        "agg_explanation": w.value
    }

    return results

def compute_agg_explanations_w_complexity(W_lime_list, W_sg_list, gradient_set, lambda_, kernel_matrix_inv_negated, fn):
    """
    Aggregates explanations using weighted combinations of W_lime_list and W_sg_list.

    Args:
        W_lime_list: List of W_lime matrices.
        W_sg_list: List of W_sg matrices.
        kernel_matrix_inv_negated: (N, N) matrix, the inverse of the kernel matrix negated.
        gradient_set: The gradient set used for computing faithfulness loss.
        lambda_value: Trade-off parameter.
        fn: Function to explain.

    Returns:
        A dictionary containing the optimized alpha values and aggregated explanation matrix.
    """
    lambda_c , lambda_f , lambda_r = lambda_

    num_lime = len(W_lime_list)
    num_sg = len(W_sg_list)
    
    alpha_lime = cp.Variable(num_lime, nonneg=True)
    alpha_sg = cp.Variable(num_sg, nonneg=True) 
    
    W_lime = sum(alpha_lime[i] * W_lime_list[i] for i in range(num_lime))
    W_sg = sum(alpha_sg[j] * W_sg_list[j] for j in range(num_sg)) 
    
    # Convert alpha into a column vector
    w = W_lime + W_sg
    
    print(f"W_lime_list length: {num_lime}")
    print(f"W_sg_list length: {num_sg}")
    print(f"gradient_set shape: {gradient_set.shape}")
    print(f"w shape: {w.shape}")
    faithfulness_loss = utility.compute_faithfulness_loss_cv(gradient_set, w)
    robustness_loss = utility.compute_robustness_loss_cv(w, kernel_matrix_inv_negated)
    complexity_loss = utility.compute_complexity_loss_cv(w)

    objective = cp.Minimize(
    lambda_f * faithfulness_loss + 
    lambda_r * robustness_loss + 
    lambda_c * complexity_loss
)

    constraints = [
        cp.sum(alpha_lime) + cp.sum(alpha_sg) == 1,  
        alpha_lime >= 0,
        alpha_sg >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)
    # Check solver status
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: Solver did not find an optimal solution!")
    results = {
        "alpha_lime": alpha_lime.value,
        "alpha_sg": alpha_sg.value,
        "agg_explanation": w.value
    }

    return results

def compute_objective_function(A_f, b_f, A_r, b_r, lambda_value):
    """
    compute_objective_function : computes the objective function of our framework
    input : coefficent of faithfulness matricies (A_f, b_f), computed by compute_faithfulness_matricies function
      : coefficent of robustness matricies (A_r, b_r), computed by compute_robustness_matricies function
      : lambda_value, the hyperparameter of our framework to manage trade-off
    output: solution, the explanation solved algebrically
    """
    # Compute A and B
    A = (1 - lambda_value) * A_f + lambda_value * A_r
    B = (1 - lambda_value) * b_f + lambda_value * b_r
    determinant = np.linalg.det(A)
    if np.isclose(determinant, 0):
        print("Matrix is singular. Using pseudoinverse instead.")
        A_inv = np.linalg.pinv(A)
        solution = np.dot(A_inv, -B)
    else:
        solution = np.linalg.solve(A, -B)
        print("Solved without pinv.")
    return solution
    
    
def compute_objective_function_cvxpy(A_f, b_f, A_r, b_r, lambda_value):
    """
    Solves the objective function using convex optimization (cvxpy).

    Parameters:
        A_f, b_f : Faithfulness matrices
        A_r, b_r : Robustness matrices
        lambda_value : Trade-off hyperparameter

    Returns:
        solution : The optimal explanation vector (W)
    """
    N, D = b_f.shape

    # Define CVXPY variable
    W_E = cp.Variable((N, D))

    # Objective components
    term_f = (1 - lambda_value) * cp.sum_squares(A_f @ W_E + b_f)
    term_r = lambda_value * cp.sum_squares(A_r @ W_E + b_r)

    # Full objective
    objective = cp.Minimize(term_f + term_r)
    problem = cp.Problem(objective)

    problem.solve(solver=cp.MOSEK, verbose=True)

    return W_E.value

def compute_sg_for_images(smoothed_gradients):
    
    # N = X.shape[0]
    # X_perturbed, smoothed_gradients = utility.create_perturbations_for_all(N, delta)
    smoothed_gradients = smoothed_gradients.mean(dim=0).detach().numpy()
    return smoothed_gradients



def compute_smoothgrad(X_perturbed, fn):
    """
    compute_smoothgrad: computes smoothgrad explanations
    input: X_perturbed(S, N, D), the perturbed input is computed by another function called generate_X_perturbed()
             : f, the function to explain
    output: W_sg(N,D), an explanation
    """
    S = X_perturbed.shape[0]
    N = X_perturbed.shape[1]
    D = X_perturbed.shape[2]

    assert X_perturbed.shape == (S, N, D)

    # grad_f = elementwise_grad(f)
    # gradient = grad_f(X_perturbed)
    grad_f = elementwise_grad(fn)
    gradients = np.zeros((S, N, D))

    for s in range(S):
        gradients[s] = grad_f(X_perturbed[s])

    assert gradients.shape == (S, N, D)
    W_sg = np.mean(gradients, axis=0)
    assert W_sg.shape == (N, D)

    return W_sg


def compute_smoothgrad_cv(X_perturbed, fn, fn_name):
    """
    compute_smoothgrad: computes smoothgrad explanations
    input: X_perturbed(S, N, D), the perturbed input is computed by another function called generate_X_perturbed()
             : f, the function to explain
    output: W_sg(N,D), an explanation
    """
    S = X_perturbed.shape[0]
    N = X_perturbed.shape[1]
    D = X_perturbed.shape[2]

    assert X_perturbed.shape == (S, N, D)

    # grad_f = elementwise_grad(f)
    # gradient = grad_f(X_perturbed)
    grad_f = elementwise_grad(fn)
    gradients = np.zeros((S, N, D))

    for s in range(S):
        gradients[s] = grad_f(X_perturbed[s])

    assert gradients.shape == (S, N, D)
    W_sg = cp.mean(gradients, axis=0)
    assert W_sg.shape == (N, D)

    return W_sg


def compute_lime_for_images( X_perturbed, model):
    """
    Computes LIME explanations using shared perturbed inputs.

    Args:
        X_perturbed (np.ndarray or torch.Tensor): shape (S, N, D)
        model (transformers.PreTrainedModel): Hugging Face model


    Returns:
        explanations (np.ndarray): shape (N, D)
    """

    
    # X_perturbed, _ = utility.create_perturbations_for_all(X, delta)

    S, N, D = X_perturbed.shape
    
    model.eval()
    explanations = []

    for n in range(N):
        X_local = X_perturbed[:, n, :]  # shape: (S, D)
        def model_fn(X_batch_flat):
            X_batch_tensor = torch.tensor(X_batch_flat, dtype=torch.float32)
            # Assume input is image-like: reshape to (S, 3, 224, 224)
            X_batch_tensor = X_batch_tensor.view(-1, 3, 224, 224)
            with torch.no_grad():
                logits = model(pixel_values=X_batch_tensor).logits
                scores = logits.max(dim=-1).values  # shape: (S,)
                return scores.cpu().numpy()

        y_local = model_fn(X_local)  # shape: (S,)
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X_local.cpu().numpy(), y_local)
        explanations.append(reg.coef_)  # shape: (D,)

    return np.stack(explanations, axis=0)  # shape: (N, D)


def compute_Lime(X_perturbed, fn):
    
    """
    compute_Lime: computes lime explanations
    input: X_perturbed(S, N, D):  the perturbed input computed by generate_perturbed_input function
             : f - the function to explain
    output: Explanations(N, D)
    """

    assert len(X_perturbed.shape) == 3
    S = X_perturbed.shape[0]
    N = X_perturbed.shape[1]
    D = X_perturbed.shape[2]

    explanations = np.zeros((N, D))
    reg = LinearRegression()

    for n in range(N):
        x = X_perturbed[:, n, :]
        y_s = fn(x)
        if len(y_s.shape) == 3:
            y_s = y_s.reshape(-1)
        else:
            y_s = y_s
        reg.fit(x, y_s)
        explanations[n] = reg.coef_
    return explanations


def powerset(iterable):

    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def shapley_kernel(M, s):
    
    if s == 0 or s == M:
        return 10000
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


def compute_kernel_shap(fn, X, reference):
    """
    Adopted from the original SHAP library:
    Compute SHAP values for a batch of inputs using Kernel SHAP.

    Parameters:
        fn : function to explain
            The model to explain. Takes input of shape (N, D) and returns (N,)
        X : np.ndarray
            Input data of shape (N, D)
        reference : np.ndarray
            Baseline input of shape (D,)
    
    Returns:
        W : np.ndarray
            SHAP values of shape (N, D)
    """
    N, D = X.shape              
    W = np.zeros((N, D))         # Initialize SHAP value matrix to store results

    for n in range(N):
        x = X[n]                 # Current input sample of shape (M,)
        
        X_design = np.zeros((2**D, D + 1))  # Binary mask design matrix + bias term
        X_design[:, -1] = 1                 # Set intercept term to 1
        weights = np.zeros(2**D)            # SHAP kernel weights
        V = np.zeros((2**D, D))             # Perturbed inputs initialized to reference

        # Fill all rows of V with the baseline (fully masked input)
        for i in range(2**D):
            V[i, :] = reference

        # Loop through all subsets of features (2^M total)
        for i, s in enumerate(powerset(range(D))):
            s = list(s)                     # s = indices of features included in the subset
            V[i, s] = x[s]                  # Replace reference values with actual input for selected features
            X_design[i, s] = 1              # Mark those features as active in the binary mask
            weights[i] = shapley_kernel(D, len(s))  # Compute kernel weight for this subset

        y = fn(V)                            # Evaluate the model on all perturbed inputs
        wsq = np.sqrt(weights)             # Square root of weights for weighted regression

        # Fit weighted linear regression: solve for feature contributions
        result = np.linalg.lstsq(wsq[:, None] * X_design, wsq * y, rcond=None)[0]

        W[n] = result[:-1]                 # Store SHAP values (exclude bias term)

    return W  # Return SHAP values for all input samples





def quadratic_function_with_interaction(X, seed):
    """
    quadratic_function_with_interaction: a function that computes quadratic function with interaction
    input: X(N, D), dataset
         : seed, a scalar to control randomness of the coefficent
    output: final_result(N,)
    """

    N = X.shape[0]
    D = X.shape[1]
    np.random.seed(seed=seed)
    c_d = np.random.uniform(-1, 1, D)
    c_d_prime = np.random.uniform(-1, 1, (D, D))

    results = {}
    final_result = np.zeros(N)
    linear_term = np.zeros(N)
    quadratic_term = np.zeros(N)

    # linear_term = np.sum([c_d[d] * np.sum(X[:, d]) for d in range(D)])
    for d in range(D):
        linear_term += c_d[d] * X[:, d]

    for d in range(D):
        for d_prime in range(D):
            quadratic_term += c_d_prime[d, d_prime] * X[:, d] * X[:, d_prime]

    final_result = linear_term + quadratic_term
    results["final_result"] = final_result
    results["c_d"] = c_d
    results["c_d_prime"] = c_d_prime
    return results


def cubic_function_with_interaction(X, seed):
    """
    cubic_function_with_interaction: a function that computes cubic function with interaction
    input: X(N, D), dataset
             : seed, a scalar to control randomness of the coefficent
    output: final_result(N,)
    """

    N = X.shape[0]
    D = X.shape[1]
    np.random.seed(seed=seed)
    c_d = np.random.uniform(-1, 1, D)
    c_d_prime = np.random.uniform(-1, 1, (D, D))
    c_d_prime_prime = np.random.uniform(-1, 1, (D, D, D))

    final_result = np.zeros(N)
    linear_term = np.zeros(N)
    quadratic_term = np.zeros(N)
    cubic_term = np.zeros(N)

    linear_term = np.sum([c_d[d] * np.sum(X[:, d]) for d in range(D)])

    for d in range(D):
        for d_prime in range(D):
            quadratic_term += c_d_prime[d, d_prime] * X[:, d] * X[:, d_prime]

    for d in range(D):
        for d_prime in range(D):
            for d_prime_prime in range(D):
                cubic_term += (
                    c_d_prime_prime[d, d_prime, d_prime_prime]
                    * X[:, d]
                    * X[:, d_prime]
                    * X[:, d_prime_prime]
                )
    final_result = linear_term + quadratic_term + cubic_term
    results = {}
    results["final_result"] = final_result
    results["c_d"] = c_d
    results["c_d_prime"] = c_d_prime
    results["c_d_prime_prime"] = c_d_prime_prime

    return results


def quartic_function_with_interaction(X, seed):
    """
    quartic_function_with_interaction: a function that computes quartic function with interaction
    input: X(N, D), dataset
             : seed, a scalar to control randomness of the coefficent
    output: final_result(N,)
    """
    N = X.shape[0]
    D = X.shape[1]
    np.random.seed(seed=seed)
    c_d = np.random.uniform(-1, 1, D)
    c_d_prime = np.random.uniform(-1, 1, (D, D))
    c_d_prime_prime = np.random.uniform(-1, 1, (D, D, D))
    c_d_prime_prime_prime = np.random.uniform(-1, 1, (D, D, D, D))

    final_result = np.zeros(N)
    linear_term = np.zeros(N)
    quadratic_term = np.zeros(N)
    cubic_term = np.zeros(N)
    quartic_term = np.zeros(N)

    linear_term = np.sum([c_d[d] * np.sum(X[:, d]) for d in range(D)])

    for d in range(D):
        for d_prime in range(D):
            quadratic_term += c_d_prime[d, d_prime] * X[:, d] * X[:, d_prime]

    for d in range(D):
        for d_prime in range(D):
            for d_prime_prime in range(D):
                cubic_term += (
                    c_d_prime_prime[d, d_prime, d_prime_prime]
                    * X[:, d]
                    * X[:, d_prime]
                    * X[:, d_prime_prime]
                )

    for d in range(D):
        for d_prime in range(D):
            for d_prime_prime in range(D):
                for d_prime_prime_prime in range(D):
                    quartic_term += (
                        c_d_prime_prime_prime[
                            d, d_prime, d_prime_prime, d_prime_prime_prime
                        ]
                        * X[:, d]
                        * X[:, d_prime]
                        * X[:, d_prime_prime]
                        * X[:, d_prime]
                        * X[:, d_prime_prime_prime]
                    )
    final_result = linear_term + quadratic_term + cubic_term + quartic_term
    results = {}
    results["final_result"] = final_result
    results["c_d"] = c_d
    results["c_d_prime"] = c_d_prime
    results["c_d_prime_prime"] = c_d_prime_prime
    results["c_d_prime_prime_prime"] = c_d_prime_prime_prime
    return results


def exponential_function_with_sine(X, seed):
    """
    exponential_function_with_sine: a function that computes exponential function with sine
    input: X(N, D), dataset
             : seed, a scalar to control randomness of the coefficent
    output: final_result(N,)
    """
    N = X.shape[0]
    D = X.shape[1]
    np.random.seed(seed=seed)
    c_d = np.random.uniform(-1, 1, D)
    c_sine_d = np.random.uniform(-1, 1, D)
    exponential_term = np.zeros(N)
    sine_term = np.zeros(N)
    final_term = np.zeros(N)
    for d in range(D):
        exponential_term += c_d[d] * np.exp(X[:, d])
        sine_term += c_sine_d[d] * np.sin(X[:, d])
        final_result = exponential_term + sine_term
    results = {}
    results["final_result"] = final_result
    results["c_d"] = c_d
    results["c_sine_d"] = c_sine_d

    return results


def exponential_function_with_quadratic(X, seed):
    """
    exponential_function_with_sine: a function that computes exponential function with quadratic function
    input: X(N, D), dataset
             : seed, a scalar to control randomness of the coefficent
    output: final_result(N,)
    """
    N = X.shape[0]
    D = X.shape[1]
    np.random.seed(seed=seed)
    c_exp_d = np.random.uniform(-1, 1, D)
    c_d = np.random.uniform(-1, 1, D)
    c_d_prime = np.random.uniform(-1, 1, (D, D))

    linear_term = np.zeros(N)
    exponential_term = np.zeros(N)
    quadratic_term = np.zeros(N)
    final_term = np.zeros(N)

    for d in range(D):
        linear_term += c_d[d] * X[:, d]
        exponential_term += c_exp_d[d] * np.exp(X[:, d])

    for d in range(D):
        for d_prime in range(D):
            quadratic_term += c_d_prime[d, d_prime] * X[:, d] * X[:, d_prime]

    final_result = exponential_term + linear_term + quadratic_term
    results = {}
    results["final_result"] = final_result
    results["c_d"] = c_d
    results["c_d_prime"] = c_d_prime
    results["c_exp_d"] = c_exp_d
    return results


def exponential_function_with_cubic(X, seed):
    """
    exponential_function_with_sine: a function that computes exponential function with quadratic function
    input: X(N, D), dataset
             : seed, a scalar to control randomness of the coefficent
    output: final_result(N,)
    """
    N = X.shape[0]
    D = X.shape[1]
    np.random.seed(seed=seed)
    c_exp_d = np.random.uniform(-1, 1, D)
    c_d = np.random.uniform(-1, 1, D)
    c_d_prime = np.random.uniform(-1, 1, (D, D))
    c_d_prime_prime = np.random.uniform(-1, 1, (D, D, D))

    linear_term = np.zeros(N)
    exponential_term = np.zeros(N)
    quadratic_term = np.zeros(N)
    cubic_term = np.zeros(N)
    final_result = np.zeros(N)

    for d in range(D):
        linear_term += c_d[d] * X[:, d]
        exponential_term += c_exp_d[d] * np.exp(X[:, d])

    for d in range(D):
        for d_prime in range(D):
            quadratic_term = c_d_prime[d, d_prime] * X[:, d] * X[:, d_prime]

    for d in range(D):
        for d_prime in range(D):
            for d_prime_prime in range(D):
                cubic_term += (
                    c_d_prime_prime[d, d_prime, d_prime_prime]
                    * X[:, d]
                    * X[:, d_prime]
                    * X[:, d_prime_prime]
                )
    final_result = exponential_term + linear_term + quadratic_term + cubic_term
    results = {}
    results["final_result"] = final_result
    results["c_d"] = c_d
    results["c_d_prime"] = c_d_prime
    results["c_d_prime_prime"] = c_d_prime_prime
    results["c_exp_d"] = c_exp_d
    return results





def load_pickles_from_directory(directory):
    '''
     Function to load pickle files from a directory
     '''
    data_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pickle"):
                file_path = os.path.join(root, file)
                print(f"Loading: {file_path}")
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    data_list.append(data)
    return data_list