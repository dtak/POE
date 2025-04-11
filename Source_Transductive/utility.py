import autograd.numpy as np
from autograd import grad, elementwise_grad
import hashlib
import json
import os 
import cvxpy as cp
import tqdm
from mpl_toolkits.mplot3d import Axes3D
# from ucimlrepo import fetch_ucimlrepo 
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
from PIL import Image
from itertools import islice
from tqdm import trange
import pickle


def load_W_lime_files(folder_path):
    W_lime_list = []
    for fname in sorted(os.listdir(folder_path))[:3]:
        if fname.endswith(".pickle"):
            full_path = os.path.join(folder_path, fname)
            with open(full_path, "rb") as f:
                W_lime = pickle.load(f)["W"]
                W_lime_list.append(W_lime)
    return W_lime_list


def load_W_sg_files(folder_path):
    W_sg_list = []
    for fname in sorted(os.listdir(folder_path))[:3]:
        if fname.endswith(".pickle"):
            full_path = os.path.join(folder_path, fname)
            with open(full_path, "rb") as f:
                W_sg = pickle.load(f)["W"]
                W_sg_list.append(W_sg)
    return W_sg_list

def create_perturbations_for_all(
    N: int,
    delta: float,
    dataset_N:list,
    model=None,
    processor=None,
    # path_dataset: str = "imagenet-1k",
    path_model: str = "resnet50_saved_model_new",
    seed: int = 42,
    S: int = 100,
    BATCH: int = 16
):
    """
    Generate perturbed inputs (S+1, N, D) and gradients (S+1, N, D) for N images.
    Will reuse dataset/model if provided to avoid repeat downloads.
    """
    #Step 1: assert the length of the dataset is atleast equal to  than N
    assert len(dataset_N) >= N, f"Not enough images in loaded dataset"
    # Step 2: Load model and processor if not passed in
    if processor is None:
        processor = AutoImageProcessor.from_pretrained(path_model)

    if model is None:
        model = AutoModelForImageClassification.from_pretrained(path_model)
        model.eval()

    all_inputs = []
    all_grads = []

    for n in tqdm.trange(N, desc="Generating perturbations"):
        image = dataset_N[n]['image'].convert("RGB")
        _input = processor(image, return_tensors='pt')['pixel_values']  # (1, 3, 224, 224)
        _range = _input.max() - _input.min()

        # Generate perturbations
        perturbed = [_input.clone().detach()]
        with torch.random.fork_rng():
            torch.manual_seed(seed + n)
            for _ in range(S):
                noise = delta * _range * torch.randn_like(_input)
                perturbed.append(_input.clone().detach() + noise)

        perturbed = torch.cat(perturbed, dim=0)  # (S+1, 3, 224, 224)
        perturbed = torch.nn.Parameter(perturbed)
        perturbed.requires_grad_(True)

        # Compute gradients
        for i in range(len(perturbed) // BATCH):
            outputs = model(pixel_values=perturbed[i*BATCH:(i+1)*BATCH])
            outputs['logits'].max(dim=-1).values.sum().backward()

        if len(perturbed) % BATCH != 0:
            i_last = len(perturbed) // BATCH
            outputs = model(pixel_values=perturbed[i_last*BATCH:])
            outputs['logits'].max(dim=-1).values.sum().backward()

        grad = perturbed.grad.detach()

        all_inputs.append(perturbed.detach().view(S + 1, -1))  # (S+1, D)
        all_grads.append(grad.view(S + 1, -1))                 # (S+1, D)

    raw_x = torch.stack(all_inputs, dim=1)  # (S+1, N, D)
    raw_y = torch.stack(all_grads, dim=1)  # (S+1, N, D)

    return raw_x, raw_y






def compute_gradient(fn, X):
    '''
    compute_gradient: computes the gradient of the function to explain wrt x using autograd
    input: f - the function to explain
    output: W_G(N,D), the gradient of the function
    '''
    f_grad = elementwise_grad(fn)
    W_G = f_grad(X)
    assert W_G.shape == X.shape

    return W_G
    
def new_make_grid(lower, upper, num_points, num_dimensions):
    # Iterate over potential grid sizes
    for grid_size in range(2, num_points+1):
        total_points = grid_size ** num_dimensions
        if total_points >= num_points:
            break

    x = np.linspace(lower, upper, grid_size)
    arrs = np.meshgrid(*([x] * num_dimensions))
    grid = np.concatenate([x.flatten()[..., None] for x in arrs], axis=-1)

    # If the grid has more points than desired, truncate it
    if grid.shape[0] > num_points:
        grid = grid[:num_points]

    return grid

def sample_sphere(radius, size):
    assert size is not None
    '''
    Adapted from:
    https://stats.stackexchange.com/questions/481715/generating-uniform-points-inside-an-m-dimensional-bal
    sample_sphere: a function that generate points in  a sphere 
    input : radius of a sphere 
          : size refers to the shape of the points sampled with in the sphere 
    output: X - points with in a sphere 
    '''
    z = np.random.normal(size=size)
    z_norm = np.sqrt(np.sum(z ** 2.0, axis=-1, keepdims=True))
    assert z_norm.shape[:-1] == z.shape[:-1]

    u = np.random.uniform(size=size[:-1] + (1,))
    
    X = radius * (u ** (1.0 / float(size[-1]))) * (z / z_norm)
    assert (np.sum(X ** 2.0, axis=-1) <= radius ** 2.0).all()
    
    return X



def compute_kernel_matrix(X, kernel_type, delta):
    '''
    compute_kernel_matrix: Computes the distance between every combination of points in a given dataset 
    input : X(N,D) the dataset
          : kernel_type: a string to indicate a kernel name for e.g, threshold, rbf, or periodic
          : delta: the length scale for rbf or periodic kernel
          : period: the period for the periodic kernel
    output: kernel_matrix(N,N) a matrix representing the distance between every point in the dataset
    '''
    N = X.shape[0]
    D = X.shape[1]
    assert X.ndim == 2
    
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    assert diff.shape == (N, N, D)

    dist = np.linalg.norm(diff, axis=2)
    assert dist.shape == (N, N)
    
    if kernel_type == "threshold":
        kernel_matrix = (dist <= delta).astype(int)
    elif kernel_type == "rbf":
        gamma = 1.0/(2.0 * delta ** 2)
        kernel_matrix = rbf_kernel(X, gamma=gamma)
    else:
        raise Exception('Kernel type {} not implemented.'.format(kernel_type))
    return kernel_matrix

# def compute_k_tilde(X, delta):
def compute_k_tilde(X, kernel_matrix):
    '''
    compute_k_tilde: compute the k_tilde matrix which is a diagonal matrix with the diagnoal equal to the sum of K(x1, xn), sum of K(x2, xn)... sum     of k(x_N, x_n)
    input: X(N,D) is the input dataset, is generated by a function called make_grid() 
          : delta: is standard deviation of the distribution if we want to sample noise from a normal distribution or the radius of the sphere if we     want to sample from a spherical distribution
    output: K_tilde(N, N)
    '''
    N = X.shape[0]
    K_tilde = np.eye(N) 
    for n in range(N):
        # K_tilde[n,n] = (compute_kernel_matrix(X, "threshold", delta)[n]).sum()
        K_tilde[n,n] = (kernel_matrix[n]).sum()
    return K_tilde 


def compute_faithfulness_matrcies(W_G,X):
    '''
    compute_faithfulness_matricies(W_G): computes the coeffiecent of faithfulness matrices by comparing the outputs/gradients
    We defined the faithfullness as the gradient matching.
    The faithfulness objective is sum_n||(grad(f) - grad(np.dot(WE_n.T, x_n)||^2 -> f is the function to explain and np.dot(WE_n.T,x_n) is the          explaination at x_n
    Then grad of f returns W_G and grad(np.dot(WE_n.T, x_n))  returns WE_n so the objective becomes the sum_n||WG_n - WE_n||^2 here WG_n and WE_n       are DX1 vectors
    The gradient of faithfulness objective wrt WE_n becomes -2*WG_n + 2*WE_n -> this is for a single n and WG_n and WE_n
    The gradient of faithfulness objective for n = N : -2W_G(N,D) + 2*I(N,N)*W_E(N,D)
    input: W_G(NXD) is the gradient of a function f
    output: A_f(N,N) = 2 *I(NXN), b_f = -2*W_G(NXD) where A_f is the coefficent of the W_E
    '''
    # W_G = compute_gradient(fn, X) # commented out for resnet50
    N = W_G.shape[0]
    A_f = 2 * np.eye(N)
    b_f = -2 * W_G
    return A_f, b_f
    
'''
compute_robustness_matricies:computes the coeffiecent of robustness matrices by comparing explanation pairs
The robustness objective is sum_n(sum_n'||(WE_n - WE_n')||^2 )*k(xn, xn'))),K(xn,xn') is a scalar that measures the distance between xn and xn'
WE_n and WE_n' represents the explanation at x_n and x_n'.
The robustness objective then becomes 2*sum_n'||(WE_n - W_E_n')||^2 )*k(xn, xn')
                                     -> when accounting for the duplicates when n =n'

Then we take the gradient wrt WE_n: 4(sum_n'(WE_n - WE_n')*k(xn,xn'))
                                     -> Note this is for a single n and the dimension for WE_n and WE_n' is (1,D) and k(x_n, x_n') is a scalar
Then when we distrbute the sum: 4*sum_n'(WE_n*k(xn,xn')) - 4*sum_n'(WE_n'*k(xn,xn'))
We vectorize the above the sum's above separately as follows:
4*sum_n'(WE_n*k(xn,xn')) = 4(WE_n.T*k_tilde), 
                        where k_tilde is a diagnoal matrix with the diagonal as follows: sum(k(x1, xn)), sum(k(x2, xn)), ... etc. 
4*sum_n'(WE_n'*k(xn,xn') = (k(xn,.)WE)
When written together: 4(WE_n.T*sum_n(k_tilde(xn, .))  - ((k(xn,.)WE))^T
                                    -> Here WE_n is (1,D) and k(xn,.) is (1,N) measuring the distance between a fixed xn and all other points
                                    -> WE is a (N,D) matrix
Then we write the objective For all N points by stacking the vectors of WEn's to in a matrix of WE
 4(WE_n.T*sum(k_tilde(x_n, .))  becomes 4((WE.T *K_tilde(.,.))
  k(xn,.)WE) becomes 4(WE.T*K(.,.))
-> WE is(N,D) matrix
-> K_tilde and K(.,.) are (N,N) matricies 

For N points  the gradient of robustness obective is: 4((WE.T *K(.,.)*1_N) - 4(WE.T*K(.,.))

To make the dimensionality match we take the transpose of (WE.T *K(.,.)*1_N) so that we get a (D,N): 4*1_N*K^T(.,.)*WE
The shape of the about objective is DXN however we want NXD so we take the transpose of the whole objective : 4*1_N*K^T(.,.)*WE - 4*K(.,.)WE
                                               > WE is now a matrix of shape  (N,D)
Then Taking WE out of the objective to get it's coefficent we will get : (4*1_N)*K^T(NXN) - 4*K(NXN))WE(NXD)

In short:
input: X(NXD), the entrire datapoints
       compute_kernel: a function that computes the distance between two points and returns the distance
       this is used an input to compute_kernel_matrix
output: the coefficents of the robustness matrices  A_r(NXN)-> coeffiencent of W_E and b_r(NXD)
Note: here is where the negative sign is added.
'''

def compute_robustness_matricies(kernel_matrix, X):
    N = X.shape[0]
    D = X.shape[1]
    A_r = - ((4 * compute_k_tilde(X, kernel_matrix)).T -  (4*kernel_matrix.T ))/N  
    b_r = -(np.zeros((N,D)))/N  
    return A_r, b_r  

def compute_faithfulness_loss(W_G, W_E):
    '''
    compute_faithfulness_loss : computes the loss for the faithfulness objective
    input : W_G(N, D), the gradient of the function to explain 
          : W_E(N,D), an explantion of the function using a certain explanation method, which could be smoothgrad, lime, our method, etc
    output: summation: A sum of the difference between the W_G and W_E
    '''

    diff_matrix = W_G - W_E
    norm_matrix = np.power(diff_matrix,2)
    summation = np.sum(norm_matrix)
    return summation


def compute_robustness_loss(W_E, kernel_matrix_inv_negated):
    '''
    compute_robustness_loss: computes the loss for the robustness objective
    input : W_E: the explanation, 
            kernel_matrix_inv_negated: contains the distance between every point in a given dataset 
    output: the sum of the difference between pair of points n and n_prime (W_E_n and W_E_n_prime)
    '''
    
    # Compute the pairwise differences between each point
    difference = W_E[:, np.newaxis, :] - W_E[np.newaxis, :, :]
    
    # Compute the squared norm for each difference
    norm_squared = np.sum(difference ** 2, axis=2)
    
    # Compute the weighted norms using the kernel_matrix
    # weighted_norm = norm_squared #* kernel_matrix
    weighted_norm = norm_squared * kernel_matrix_inv_negated

    # Sum all the weighted norms to get the final robustness loss
    summation = (np.sum(weighted_norm))
    # summation = np.sum(weighted_norm)
    return summation

def compute_robustness_loss_with_constraint(W_E):
    '''
    Compute the robustness loss using CVXPY with a constraint: d_n >= ||W_E_n - W_E_n'||^2_2.
    
    input: W_E (N, D)
    
    output: The sum of maximum distance between points
    '''
    N, D = W_E.shape
    d_n = np.zeros(N)
    max_distance = 0 
    pairwise_distances = np.sum((W_E[:, np.newaxis] - W_E[np.newaxis, :])**2, axis=2)
     # Set the diagonal elements to a very small value (since we don't compare a point with itself)
    np.fill_diagonal(pairwise_distances, -np.inf)
    d_n = np.max(pairwise_distances, axis=1)
    robustness_loss = np.sum(d_n)
    return robustness_loss

def compute_complexity_loss(W_E):
    '''
    complexity: Compute the complexity metric as the sum of the L1 norm of each row of W_E.
    input: W_E, explanation 
    output: sum of L1 norm of each row 
    '''
    # Compute the L1 norm of each row of W_E
    l1_norms = np.sum(np.abs(W_E), axis=1)
    total_complexity = np.sum(l1_norms)
    
    return total_complexity
def compute_complexity_loss_cv(W_E):
    '''
    complexity: Compute the complexity metric as the sum of the L1 norm of each row of W_E.
    input: W_E, explanation 
    output: sum of L1 norm of each row 
    '''
    # Compute the L1 norm of each row of W_E
    l1_norms = cp.sum(cp.abs(W_E), axis=1)
    

    total_complexity = cp.sum(l1_norms)
    
    return total_complexity

def compute_objective_loss(faithfulness_loss, robustness_loss, lambda_value):
    '''
    compute_objective_loss:compute the objective loss by taking in faithfulness, robustness loss and lambda
    input : faithfulness_loss, computed by compute_faithfulness_loss function 
          : robustness_loss, computed by compute_robustness_loss function 
          : lambda_value: the hyperparameter for our framework that ranges from [0, 1] that manages the trade-off between properties
    output: loss which is computed as follows: ((1 - lambda_value)*faithfulness_loss) + ((lambda_value) * robustness_loss ))
    '''
    
    loss = ((1 - lambda_value)*faithfulness_loss) + ((lambda_value) * robustness_loss )
    return loss
    
def compute_objective_loss_with_robustness_constraint(lambda_value, gradient_set,  N, D):
    '''
    compute_objective_loss:compute the objective loss by taking in faithfulness, robustness loss and lambda
    input : faithfulness_loss, computed by compute_faithfulness_loss function 
          : robustness_loss_with_constraint, computed by compute_robustness_loss with constraint function 
          : lambda_value: the hyperparameter for our framework that ranges from [0, 1] that manages the trade-off between properties
    output: loss which is computed using QPQC 
    '''
    # Variables
    W_E = cp.Variable((N, D)) 
    d = cp.Variable(N) 
    
    faithfulness_loss = cp.sum(cp.norm(W_E - gradient_set, axis=1)**2)
    objective = cp.Minimize((1 - lambda_value) * faithfulness_loss + lambda_value *  cp.sum(d))
    constraints = []
    for n in range(N):
        for n_prime in range(N):
            if n != n_prime:
                constraints += [d[n] >= cp.norm(W_E[n, :] - W_E[n_prime, :], 2)**2]
    

    problem = cp.Problem(objective, constraints)
    # problem.solve(solver=cp.SCS, verbose=True) ##commennted out to see if it causing unoptimal results
    problem.solve(solver=cp.MOSEK, verbose=True)

    return W_E.value
    
def compute_objective_loss_with_complexity(X, gradient_set, N, D, lambda_):
    '''
    Compute the objective function based on faithfulness, robustness, and complexity.
    
    Parameters:
    X: Dataset on which to compute the kernel matrix.
    gradient_set: (N, D) matrix representing gradients (∇f_n).
    N: Number of rows (data points).
    D: Number of columns (features).
    lambda_: a tuple containing different lambda values, lambda_f, lambda_r, and lambda_c 
    
    Returns:
    The minimized objective value and optimized explanation matrix W_E.
    '''

    lambda_c , lambda_f , lambda_r = lambda_
    # Create the W_E and d_n variables
    W_E = cp.Variable((N, D))  
    d_n = cp.Variable(N)      

    faithfulness_term = (lambda_f) * cp.sum([cp.norm(W_E[n] - gradient_set[n], 2)**2 for n in range(N)])
    robustness_term = lambda_r * cp.sum([cp.norm(W_E[n] - W_E[n_prime], 2)**2 for n in range(N) for n_prime in range(N)]) 
    complexity_term = (lambda_c ) * cp.sum([cp.norm(W_E[n], 1) for n in range(N)])
    
    total_loss = faithfulness_term + robustness_term + complexity_term

    # Constraints: d_n >= ||W_E_n - W_E_n' ||_2^2 for all pairs n, n'
    constraints = []

    for n in range(N):
        for n_prime in range(N):
            if n != n_prime:
               constraints += [d_n[n] >= cp.norm(W_E[n] - W_E[n_prime], 2)**2]
    
   

    # Define the optimization problem (minimize the total loss)
    problem = cp.Problem(cp.Minimize(total_loss), constraints)
    problem.solve(solver=cp.SCS, verbose=True)

    return  W_E.value

def hyperparameters_to_results_filename(exp_dir, exp_kwargs):  
    '''
    hyperparameters_to_results_filename: hashing function to give file a unique file name based on exp_kwargs(parameters for each experiment) 
    input: exp_dir, exp_kwargs
    output: a file name 
    '''
    hex_code = hashlib.md5(json.dumps(exp_kwargs).encode('utf-8')).hexdigest()
    return os.path.join(exp_dir, '{}.pickle'.format(hex_code))



def compute_objective_with_fn_matching_faithfulness_and_max_sensitivty(X, fn, N, D, lambda_value):

    W_E = cp.Variable((N, D))
    N, D = W_E.shape
    robustness_term = 0
    for n in range(N):
        # Create a list of expressions for pairwise distances (excluding diagonal elements)
        pairwise_distances = [
            cp.norm(W_E[n] - W_E[n_prime], 2)**2
            for n_prime in range(N) if n != n_prime
        ]
        # Combine the list of expressions using cp.vstack and take the maximum
        max_distance = cp.max(cp.vstack(pairwise_distances))
        robustness_term += max_distance
    
    faithfulness_term_list = []
    for n in range(N):

        W_E_Xn = W_E[n] @ X[n] 
        abs_difference = cp.abs(fn(X[n]) - W_E_Xn)
        squared_difference = abs_difference**2  
        faithfulness_term_list.append(squared_difference)
        
    faithfulness_term = cp.sum(faithfulness_term_list)
    objective = cp.Minimize((1- lambda_value) * faithfulness_term + (lambda_value) * robustness_term)
    problem = cp.Problem(objective) 
    problem.solve(solver=cp.SCS, verbose=True)
    
    return W_E.value


def compute_objective_with_fn_matching_faithfulness(X, fn, N, D, lambda_value):

    W_E = cp.Variable((N, D))
    
    robustness_term = cp.sum(
        [cp.norm(W_E[n] - W_E[n_prime], 2)**2 for n in range(N) for n_prime in range(N)]
    )
    faithfulness_term_list = []
    for n in range(N):

        W_E_Xn = W_E[n] @ X[n] 
        abs_difference = cp.abs(fn(X[n]) - W_E_Xn)
        squared_difference = abs_difference**2  
        faithfulness_term_list.append(squared_difference)
    faithfulness_term = cp.sum(faithfulness_term_list)
    objective = cp.Minimize((1- lambda_value) * faithfulness_term + (lambda_value) * robustness_term)
    problem = cp.Problem(objective) 
    problem.solve(solver=cp.SCS, verbose=True)
    
    return W_E.value

def compute_fn_matching_faithfulness_loss(W_E, fn, X, N, D):
    '''
    Note: this is only for fn_mathcing faithfulness matching 
    '''
    faithfulness_term_list = []
    
    for n in range(N):

        W_E_Xn = W_E[n] @ X[n]
        abs_difference = np.abs(fn(X[n]) - W_E_Xn)
        squared_difference = abs_difference**2
        faithfulness_term_list.append(squared_difference)
    faithfulness_term = np.sum(faithfulness_term_list)
    
    return faithfulness_term


def compute_fn_matching_faithfulness_loss_cv(W_E, fn, X, N, D):
    '''
    Note: this is only for fn_mathcing faithfulness matching 
    '''
    faithfulness_term_list = []
    
    for n in range(N):

        W_E_Xn = W_E[n] @ X[n]
        abs_difference = np.abs(fn(X[n]) - W_E_Xn)
        squared_difference = abs_difference**2
        faithfulness_term_list.append(squared_difference)
    faithfulness_term = cp.sum(faithfulness_term_list)
    
    return faithfulness_term

def generate_perturbed_X(X, delta, sampling, S):
    ###Note: seed was removed from the input space 
    # np.random.seed(seed=seed)
    N = X.shape[0]
    D = X.shape[1]
    
    X_expanded = X[np.newaxis, :, :]
    if sampling == 'sphere':
         epsilon_s = sample_sphere(delta, size=(S, N, D))
         X_perturbed =  X_expanded + epsilon_s
    elif sampling == 'normal':
         epsilon_s = np.random.normal(0 , np.abs(delta), size=(S, N, D))
         X_perturbed =  X_expanded + epsilon_s
    else:
        raise  Exception('Sampling {} not defined,'.format(exp_kwargs['sampling']))

    return X_perturbed


def compute_faithfulness_loss_cv(W_G, W_E):
    diff_matrix = W_G - W_E
    norm_matrix = cp.square(diff_matrix)
    summation = cp.sum(norm_matrix)
    return summation

def compute_robustness_loss_cv(W_E, kernel_matrix):
    """
    Computes the robustness loss using a double for loop.

    Args:
        W_E: NumPy array of shape (N, D), representing the explanation matrix.
        kernel_matrix: NumPy array of shape (N, N), containing the pairwise distances or weights.

    Returns:
        summation: The robustness loss, computed as the sum of weighted pairwise squared differences.
    """
    N, D = W_E.shape
    summation = 0  # 

    
    for i in range(N):
        for j in range(N):

            squared_norm = cp.sum((W_E[i] - W_E[j]) ** 2)


            weighted_norm = squared_norm  #* kernel_matrix[i, j]

            summation += weighted_norm

    return summation
