import math
import json
import os
import mofae
import sys
import pickle
import utility
import dill 
import time
import main_functions
import torch
import autograd.numpy as np
from autograd import elementwise_grad
from ucimlrepo import fetch_ucirepo
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.model_selection import train_test_split
from collections import Counter
from shap import KernelExplainer
from nn import wrapper_for_ds

DRYRUN = True # Set to True jobs wont be submitted to the Cluster. 
RUN_RESNET = False # Set to True to run the resnet model.
RUN_NN = False
OUTPUT_DIR = "output"

##Hyperparameters for each experiment 
DELTA = list(np.linspace(1e-5, 3.0, 1))
delta_kernel_resnet = np.array([650.0])
delta_kernel_synthetic = np.array([3.0])
SAMPLING = ["normal"]# "sphere"]
S = 10000
log_part = np.logspace(np.log10(0.889), np.log10(1),1) # 500)
lin_part = np.linspace(0.0001,  0.889,0) #30)
lambda_values = np.concatenate([lin_part, log_part])

##LAMBDA is only used for GAC Experiments inplace of the lambda values
LAMBDA = []
for lambda_group in [0.00, 0.25, 0.50, 0.75, 1.00]: # small, maximum five values
    for lambda_linecoeff in np.linspace(0.001, 0.999, 400): # large, lots of points for smooth line
        
        LAMBDA.append((lambda_group, (1 - lambda_group) * lambda_linecoeff, (1 - lambda_group) * (1 - lambda_linecoeff)))
        LAMBDA.append(((1 - lambda_group) * lambda_linecoeff, lambda_group, (1 - lambda_group) * (1 - lambda_linecoeff)))
        LAMBDA.append(((1 - lambda_group) * lambda_linecoeff, (1 - lambda_group) * (1 - lambda_linecoeff), lambda_group))
    
prop_combination = ["GA"]  #'GA', 'FA', 'GM', 'GAC'] ->
'''
GA is solved is using Gradient Matching Faithfulness and Average Robustness.
FA is solved using Function Matching Faithfulness and Average Robustness, 
GM is solved using Gradient Matching Faithfulness and Maximum SENS Robustness.
GAC is solved using Function Matching Faithfulness, Average Robustness and Complexity.
'''


##Dataset IDs for NN 
data_set_ids = [89]
rand_state = [0, 1, 2, 3, 4, 5]
targets_dict = {
    89: 10,
}

FN_NAMES = {
    "cubed": lambda x: np.sum(x**3.0, axis=-1),##**
    # # "quadratic": lambda x: np.sum(x**2.0, axis=-1),
    # # # # 'polynomial': lambda x: np.sum(x ** 7.0,  axis=-1),
    # # "sine": lambda x: np.sum(np.sin(x), axis=-1),##**
    # 'sine_exponent': lambda x: np.sum(np.sin(np.exp(x)), axis=-1),##**
    # # # # 'sine_added_with_polynomial': lambda x: (np.sum(np.sin(np.exp(x)), axis=1) + x[:, -1]**3),
    # 'quasi': lambda x: np.sum(x + np.sin(3 * x), axis=-1),##**
    # # 'quasi_2':  lambda x:  np.sum(x + np.sin(np.exp(x)), axis=-1),
    # # # 'cubed_quasi': lambda x: np.sum((x ** 3) + np.sin(10 * x), axis=-1) ,
    # 'quadratic_quasi': lambda x: np.sum(((x ** 2)/10) + np.sin(3 * x), axis=-1) , ##**
    # 'exponential': lambda x: np.sum(np.exp(x), axis=-1),##**
    # 'quasi': lambda x: np.sum(x + np.sin(3 * x), axis=-1),##**
    # 'quasi_2':  lambda x:  np.sum(x + np.sin(np.exp(x)), axis=-1),
    # # 'cubed_quasi': lambda x: np.sum((x ** 3) + np.sin(10 * x), axis=-1) ,
    # 'quadratic_quasi': lambda x: np.sum(((x ** 2)/10) + np.sin(3 * x), axis=-1) , ##**
    # 'exponential': lambda x: np.sum(np.exp(x), axis=-1),##**
    # 'degree_four': lambda x: np.sum(x ** 4.0, axis=-1),
    # cubic_interaction:lambda x: main_functions.cubic_function_with_interaction(x, 1)
}


"""
List of all experiments you want to run.
For each experiment, list all combinations of parameters to use.
When explaining the neural network, make sure to uncomment the dataset `ids` and update `fn_names` accordingly.
"""
fn_list = list(FN_NAMES.keys())
if RUN_RESNET:
    fn_list+=['RESNET50']
    delta_kernel = delta_kernel_resnet
elif RUN_NN:
    fn_list+=['neural_network']
    delta_kernel = delta_kernel_synthetic
else:
    delta_kernel = delta_kernel_synthetic
    
QUEUE = [

    	# ('shap', dict(
        #     delta_kernel=delta_kernel,
        #     fn_name =fn_list,
        #     sampling=SAMPLING,
        #     prop_combination=prop_combination,
        #     # rand_state=rand_state,
        #     # ids=data_set_ids,

        # )),
    # (
    #     "lime",
    #     dict(
    #         delta=DELTA,
    #         fn_name=fn_list,
    #         delta_kernel=delta_kernel_synthetic,
    #         sampling=SAMPLING,
    #         prop_combination=prop_combination,
    #         # rand_state=rand_state,
    #         # ids=data_set_ids,
    #     ),
    # ),

    # (
    #     "smooth_grad",
    #     dict(
    #         delta=DELTA,
    #         fn_name=fn_list,
    #         delta_kernel=delta_kernel_synthetic,
    #         sampling=SAMPLING,
    #         prop_combination=prop_combination,
    #         ##rand_state=rand_state,
    #         # #ids=data_set_ids,


    #     ),
    # ), 
    # (
    #     "us",
    #     dict(
    #         lambda_value=lambda_values,
    #         fn_name=fn_list,
    #         delta_kernel=delta_kernel_synthetic,
    #         # LAMBDA=LAMBDA,
    #         sampling=SAMPLING,
    #         prop_combination=prop_combination,
    #         # rand_state=rand_state,
    #         # ids=data_set_ids,
    #     ),
    # ),
    # (
    # "mofae",
    #     dict(
    #     fn_name =fn_list,
    #     delta_kernel=delta_kernel_synthetic,
    #     sampling=SAMPLING,
    #     prop_combination = prop_combination,
    #     # rand_state=rand_state,
    #     # ids=data_set_ids,
        
    # )
    # ),
    ( 
        "agg",
        dict(
            fn_name =fn_list,
            lambda_value=lambda_values,
            delta_kernel=delta_kernel_synthetic,
            # LAMBDA=LAMBDA,
            prop_combination=prop_combination,
            # rand_state=rand_state,
            # ids=data_set_ids,

        )
    )
]


def run(exp_dir, exp_name, exp_kwargs):
    """'
    This is the function that will actually execute the job.
    To use it, here's what you need to do:
    1. Create directory 'exp_dir' as a function of 'exp_kwarg'.
    This is so that each set of experiment+hyperparameters get their own directory.
    2. Get your experiment's parameters from 'exp_kwargs'
    3. Run your experiment
    4. Store the results however you see fit in 'exp_dir'
    """
    print("Running experiment {}:".format(exp_name))
    print("Results are stored in:", exp_dir)
    print("with hyperparameters", exp_kwargs)
    print("\n")
    
    exp_kwargs["exp_name"] = exp_name
    fn_name = exp_kwargs["fn_name"]
    
    if exp_name in ['smooth_grad', 'lime']:
        delta = exp_kwargs['delta']
        prop_combination = exp_kwargs['prop_combination']

    elif exp_name in ['us', 'agg']:
        delta_kernel = exp_kwargs['delta_kernel']
        prop_combination = exp_kwargs['prop_combination']

        if prop_combination == 'GAC':
            lambda_value = exp_kwargs['LAMBDA']
        else:
            lambda_value = exp_kwargs['lambda_value']
    elif exp_name in ['mofae', 'shap','random']:
        delta_kernel = exp_kwargs['delta_kernel']
        prop_combination = exp_kwargs['prop_combination']
    else:
        pass
    if exp_kwargs['fn_name'] == 'RESNET50':
        with open("imagenet_subset.pkl", "rb") as f:
            dataset_N = pickle.load(f)
        model = AutoModelForImageClassification.from_pretrained("path_to_saved_model", local_files_only=True)
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50",  local_files_only=True)
        processor.save_pretrained("path_to_saved_model", local_files_only=True)
        X, gradient_set = np.load("X.npy"), np.load("path_to_gradiens_saved")
        FN_NAMES["RESNET50"] = lambda x: model(pixel_values=x)

    elif exp_kwargs['fn_name'] == 'neural_network':
        print('Running neural network') 
        rand_state = exp_kwargs['rand_state']
        uci_id = exp_kwargs['ids']
        target_index = targets_dict[uci_id]
        hidden_layers = 2
        X, y, neural_network_callable = wrapper_for_ds(uci_id, target_index, hidden_layers, rand_state)
    else:
        N = 100 
        D = 3
        if D == 3:
            X = utility.new_make_grid(-5.0, 5.0, N, D)
        elif D > 4:
            seed = 42
            np.random.seed(seed)
            X =  np.random.uniform(-5.0, 5.0, (N, D))
        fn = FN_NAMES[exp_kwargs['fn_name']]
        gradient_set = utility.compute_gradient(fn, X)  
    
    fn = FN_NAMES[exp_kwargs['fn_name']]
    kernel_matrix = utility.compute_kernel_matrix(X, "rbf",exp_kwargs['delta_kernel'])
    kernel_matrix_inv_negated = -1 * np.linalg.pinv(kernel_matrix)
    
    
    if exp_name in {"smooth_grad", "lime"}:
        sampling = exp_kwargs.get("sampling")
        if sampling not in {"normal", "sphere"}:
            raise Exception(f"Sampling '{sampling}' not implemented.")
        if fn_name == "RESNET50":
            X_perturbed, smoothed_gradients, dataset_N, model =  utility.create_perturbations_for_all(10, delta,dataset_N)
            W = main_functions.compute_sg_for_images(smoothed_gradients) if exp_name == "smooth_grad" else main_functions.compute_lime_for_images(X_perturbed, model)
        else:
            X_perturbed = utility.generate_perturbed_X(X, exp_kwargs["delta"], sampling, S)
            W = main_functions.compute_smoothgrad(X_perturbed, fn) if exp_name == "smooth_grad" else main_functions.compute_Lime(X_perturbed, fn)
    
    elif exp_name == "us":
            
        ### NOTE:
        '''
        GA is solved is using Gradient Matching Faithfulness and Average Robustness.
        GAC is solved using Gradient Matching Faithfulness and Average Robustness and Complexity. 
        FA is solved using Function Matching Faithfulness and Average Robustness. 
        FC is solved using Function Matching Faithfulness, Average Robustness and Complexity. 
        '''

        if  prop_combination == "GA":
            kernel_matrix_inv = np.linalg.inv(kernel_matrix)
            A_r, b_r = utility.compute_robustness_matricies(kernel_matrix_inv, X)
            A_f, b_f = utility.compute_faithfulness_matrcies(gradient_set, X)
            W = main_functions.compute_objective_function(
                A_f, b_f, A_r, b_r, exp_kwargs["lambda_value"],
            )
        elif prop_combination == "FA":
            W = utility.compute_objective_with_fn_matching_faithfulness(X, fn, N, D, exp_kwargs['lambda_value'])
        elif prop_combination == "GAC":
            W = utility.compute_objective_loss_with_complexity(X, gradient_set, N, D, exp_kwargs['LAMBDA'])
        elif prop_combination == "GM":
            N, D = gradient_set.shape
            W = utility.compute_objective_loss_with_robustness_constraint(lambda_value, gradient_set,  N, D)
        else:
            raise Exception("Unknown prop_combination {}".format(prop_combination))

    elif exp_name == "shap":
        reference = np.mean(X, axis=0)
        reference = np.reshape(reference, (1, len(reference)))
        # reference = np.zeros(D)
        if D >10:
            W = np.array([KernelExplainer(fn, reference).shap_values(X[i]) for i in range(N)])
        else:
            W = main_functions.compute_kernel_shap(fn, X,  reference)
    
    elif exp_name == "mofae":
        
        if (exp_kwargs['prop_combination'] == 'GA') or (exp_kwargs['prop_combination'] == 'FA'):
            num_objectives = 2
        elif exp_kwargs['prop_combination'] == 'GAC':
            num_objectives = 3
            
        if fn_name == "RESNET50":
            # W_sg_list = utility.load_W_sg_files("path_to_precomputed_sg")
            # W_lime_list = utility.load_W_lime_files("path_to_precomputed_lime")
            W = mofae.compute_mofae_for_images(X, fn, gradient_set, kernel_matrix_inv_negated, W_sg_list, W_lime_list, num_objectives, exp_kwargs['prop_combination'])
        else:
            W_lime_list = []
            for delta in np.linspace(1e-5, 3, 3):
                W_lime_list.append(main_functions.compute_Lime(utility.generate_perturbed_X(X, delta, exp_kwargs['sampling'], S), fn))
            W_sg_list = []     
            for delta in np.linspace(1e-5, 3, 100):
                W_sg_list.append(main_functions.compute_smoothgrad(utility.generate_perturbed_X(X, delta,  exp_kwargs['sampling'], S), fn))
            W = mofae.compute_mofae(X, fn, gradient_set, kernel_matrix_inv_negated, W_sg_list, W_lime_list, num_objectives, exp_kwargs['prop_combination'])
    
    elif exp_name == "agg":
        if fn_name == "RESNET50":
            W_sg_list = utility.load_W_sg_files("path_to_precomputed_sg")
            W_lime_list = utility.load_W_lime_files("path_to_precomputed_lime")
        else: 
            W_lime_list = [] 
            W_sg_list = []
            for delta in np.linspace(1e-5, 3, 3):
                W_sg = main_functions.compute_smoothgrad(utility.generate_perturbed_X(X, delta, "normal", S), fn)
                print('computed smoothgrad')
                W_lime = main_functions.compute_Lime(utility.generate_perturbed_X(X, delta, "normal", S), fn)
                print('computed lime')
                W_sg_list.append(W_sg)
                W_lime_list.append(W_lime)   
        if exp_kwargs['prop_combination'] == 'GA':
            W = main_functions.compute_agg_explanations(W_lime_list,
                                                    W_sg_list, 
                                                    gradient_set, 
                                                    exp_kwargs['lambda_value'], 
                                                    kernel_matrix_inv_negated,
                                                    fn, exp_kwargs['prop_combination'] , X)['agg_explanation']

        elif exp_kwargs['prop_combination'] == 'GAC':
            W = main_functions.compute_agg_explanations_w_complexity(W_lime_list,
                                                        W_sg_list, 
                                                        gradient_set, 
                                                        exp_kwargs['LAMBDA'], 
                                                        kernel_matrix_inv_negated, fn,)['agg_explanation']
        elif exp_kwargs['prop_combination'] == 'FA':
            W = main_functions.compute_agg_explanations(W_lime_list,
                                    W_sg_list, 
                                    gradient_set, 
                                    exp_kwargs['lambda_value'], 
                                    kernel_matrix_inv_negated, fn, exp_kwargs['prop_combination'] , X)['agg_explanation']       

        else:
            raise Exception("Unknown prop_combination {}".format(prop_combination))

    ####Evaluate the explanations for each experiment with respect to prop combination. 
    if exp_name == "agg":
        if exp_kwargs['prop_combination'] == 'GAC':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated)
            faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W)
            complexity_loss = utility.compute_complexity_loss_cv(W)
        elif exp_kwargs['prop_combination'] == 'GA':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated)
            faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W)
            complexity_loss = None
        elif exp_kwargs['prop_combination'] == 'FA':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated)      
            faithfulness_loss = utility.compute_fn_matching_faithfulness_loss(W, fn, X, N, D)
            complexity_loss = None
        else:
            raise Exception("Unknown prop_combination {}".format(prop_combination))
        
    elif exp_name == "mofae":
        if exp_kwargs['prop_combination']== 'GA':
            robustness_loss = []
            faithfulness_loss = []
            complexity_loss = []
            for i in range(len(W)):
                # robustness_loss.append(utility.compute_robustness_loss_with_constraint(W[i]))
                robustness_loss.append(utility.compute_robustness_loss(W[i], kernel_matrix_inv_negated))
                faithfulness_loss.append(utility.compute_faithfulness_loss(W[i], gradient_set))
                complexity_loss = None
        elif exp_kwargs['prop_combination'] == 'GAC':
            robustness_loss = []
            faithfulness_loss = []
            complexity_loss = []
            for i in range(len(W)):
                robustness_loss.append(utility.compute_robustness_loss(W[i], kernel_matrix_inv_negated))
                faithfulness_loss.append(utility.compute_faithfulness_loss(W[i], gradient_set))
                complexity_loss.append(utility.compute_complexity_loss(W[i]))
        elif exp_kwargs['prop_combination'] == 'FA':
            robustness_loss = []
            faithfulness_loss = []
            complexity_loss = []
            for i in range(len(W)):
                robustness_loss.append(utility.compute_robustness_loss(W[i], kernel_matrix_inv_negated))
                faithfulness_loss.append(utility.compute_fn_matching_faithfulness_loss(W[i], fn, X, N, D))
                complexity_loss = None
        else:
            raise Exception("Unknown prop_combination {}".format(prop_combination))
        
    elif exp_name in {'smooth_grad', 'lime', 'shap'}:

        if exp_kwargs["prop_combination"] == 'GAC':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated)
            faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W)
            complexity_loss = utility.compute_complexity_loss(W)
        elif exp_kwargs["prop_combination"] == 'GA':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated) 
            faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W)
            complexity_loss = None
        elif exp_kwargs["prop_combination"] =='FA':
            robustness_loss = utility.compute_robustness_loss_with_constraint(W)
            faithfulness_loss = utility.compute_fn_matching_faithfulness_loss(W, fn, X, N, D)
        elif exp_kwargs["prop_combination"] =='GM':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated)
            faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W)
            complexity_loss = None
    elif exp_name == 'us':
        if exp_kwargs['prop_combination'] == 'GAC':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated)
            faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W)
            complexity_loss = utility.compute_complexity_loss(W)
        elif exp_kwargs['prop_combination'] == 'GA':
            robustness_loss = utility.compute_robustness_loss(W, kernel_matrix_inv_negated)
            faithfulness_loss = utility.compute_faithfulness_loss(gradient_set, W)
            complexity_loss = None
        elif exp_kwargs['prop_combination'] == 'FA':
            robustness_loss = utility.compute_robustness_loss_with_constraint(W)
            faithfulness_loss = utility.compute_fn_matching_faithfulness_loss(W, fn, X, N, D)
            complexity_loss = None
    else:
        raise Exception("Unknown experiment {}".format(exp_name))
    
    ### Feature importance ranking computation
    if exp_name == "mofae":
        for w in W:
            importance_ranking = np.zeros_like(w, dtype=int)
            # Loop through each row in W
            for n in range(w.shape[0]):
                # sort in descending order
                w = np.abs(w)
                sorted_indices = np.argsort(w[n])[::-1]
                for rank, index in enumerate(sorted_indices):
                    importance_ranking[n, index] = w.shape[1] - rank
    else: 
        #most important feature is the one with the highest rank
        importance_ranking = np.zeros_like(W, dtype=int)
        # Loop through each row in W
        for n in range(W.shape[0]):
            # sort in descending order
            W = np.abs(W)
            sorted_indices = np.argsort(W[n])[::-1]
            for rank, index in enumerate(sorted_indices):
                importance_ranking[n, index] = W.shape[1] - rank
    #################################################################

    if "lambda_value" not in exp_kwargs:
        objective_loss = None
    else:
        objective_loss = utility.compute_objective_loss(
            faithfulness_loss, robustness_loss, exp_kwargs["lambda_value"]
        )
    """
    Write results to file:
    - For each experiment, a subfolder is created inside the 'output' directory.
    - The subfolder is named after the experiment being run.
    - Inside each subfolder, files are written corresponding to the varied parameters.
    - For example, if we are varying `delta` and there are 10 different `delta` values, 
    10 separate files will be created, one for each value.
    """

    results = dict(
        W=W,
        robustness_loss=robustness_loss,
        faithfulness_loss=faithfulness_loss,
        complexity_loss=complexity_loss,
        objective_loss=objective_loss,
        gradient_set=gradient_set,
        feature_importance=importance_ranking,
        **exp_kwargs,
    )
    fname = utility.hyperparameters_to_results_filename(exp_dir, exp_kwargs)
    with open(fname, "wb") as f:
        pickle.dump(results, f)


def main():
    assert len(sys.argv) > 2

    exp_dir = sys.argv[1]
    exp_name = sys.argv[2]
    exp_kwargs = json.loads(sys.argv[3])
    run(exp_dir, exp_name, exp_kwargs)

if __name__ == "__main__":
    main()
