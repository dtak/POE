

from autograd import grad
from autograd import numpy as np
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import utility
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import argparse
import pickle
import dill
import os 
from random import choice


class Feedforward:
    def __init__(self, architecture, random=None, weights=None):
        self.params = {
            "H": architecture["width"],
            "L": architecture["hidden_layers"],
            "D_in": architecture["input_dim"],
            "D_out": architecture["output_dim"],
            "activation_type": architecture["activation_fn_type"],
            "activation_params": architecture["activation_fn_params"],
        }

        self.D = (
            (architecture["input_dim"] * architecture["width"] + architecture["width"])
            + (
                architecture["output_dim"] * architecture["width"]
                + architecture["output_dim"]
            )
            + (architecture["hidden_layers"] - 1)
            * (architecture["width"] ** 2 + architecture["width"])
        )

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture["activation_fn"]

        self.weights = self.random.rand(1, self.D) * 2 - 1 
        # if weights is None:
        #     self.weights = self.random.normal(0, 1, size=(1, self.D))
        # else:
        #     self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))

    def forward(self, weights, x):
        """Forward pass given weights and input"""
        from autograd import numpy as np
        H = self.params["H"]
        D_in = self.params["D_in"]
        D_out = self.params["D_out"]

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            if x.shape[0] == D_in:
                x = x.reshape((1, D_in, -1))
            else:
                x = x.T
                assert x.shape[0] == D_in
        else:
            if x.shape[-1] == D_in:
                x = np.moveaxis(x, -1, -2)
            else:
                assert x.shape[1] == D_in
        weights = weights.T

        # input to first hidden layer
        W = weights[: H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in : H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = H * D_in + H

        assert input.shape[1] == H

        # additional hidden layers
        for _ in range(self.params["L"] - 1):
            before = index
            W = weights[index : index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index : index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        # output layer
        W = weights[index : index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out :].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params["D_out"]

        return output

    def make_objective(self, x_train, y_train, reg_param):

        def objective(W, t):
            squared_error = (
                np.linalg.norm(y_train - self.forward(W, x_train), axis=1) ** 2
            )
            if reg_param is None:
                sum_error = np.sum(squared_error)
                return sum_error
            else:
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

        return objective, grad(objective)

    def fit(self, x_train, y_train, params, reg_param=0.01):
        print(f"x_train.shape[0]: {x_train.shape[0]}")
        print(f"self.params['D_in']: {self.params['D_in']}")
        assert x_train.shape[0] == self.params["D_in"]
        assert y_train.shape[0] == self.params["D_out"]

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param=0.01)

        ### set up optimization
        step_size = 0.01
        max_iteration = 10
        check_point = 4
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = "adam"
        random_restarts = 5

        if "step_size" in params.keys():
            step_size = params["step_size"]
        if "max_iteration" in params.keys():
            max_iteration = params["max_iteration"]
        if "check_point" in params.keys():
            self.check_point = params["check_point"]
        if "init" in params.keys():
            weights_init = params["init"]
        if "call_back" in params.keys():
            call_back = params["call_back"]
        if "mass" in params.keys():
            mass = params["mass"]
        if "optimizer" in params.keys():
            optimizer = params["optimizer"]
        if "random_restarts" in params.keys():
            random_restarts = params["random_restarts"]

        def call_back(weights, iteration, g):
            """Actions per optimization step"""
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % self.check_point == 0:
                print(
                    "Iteration {} lower bound {}; gradient mag: {}".format(
                        iteration,
                        objective,
                        np.linalg.norm(self.gradient(weights, iteration)),
                    )
                )

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights
        max_norm = 5.0
        for i in range(random_restarts):
            if optimizer == "adam":
                adam(
                    self.gradient,
                    weights_init,
                    step_size=step_size,
                    num_iters=max_iteration,
                    callback=call_back,
            )
            # gradients = self.gradient(weights_init)  # Store
            # grad_norm = np.linalg.norm(gradients)
            # if grad_norm > max_norm:
            #     gradients = gradients * (max_norm / grad_norm) # Rescale
            local_opt = np.min(self.objective_trace[-100:])
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]
        



def wrapper_for_ds(ds_id, target_index, hidden_layers, rand_state):
    """
    wrapper_for_ds: a function that download ds using a dataset id and target_index
    features is a list of strings, each string is a feature name
    """

    ######### DATA PROCESSSING
    ##################################################################
    #### GET DATASET FROM UCI
    dataset = fetch_ucirepo(id=ds_id)
    num_targets = dataset.data.targets.shape[1]
    X = dataset.data.features.values
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])[:1000]
    X = X[indices]

    # If no target_index is provided or it's out of bounds, select a valid index
    if target_index is None or target_index >= num_targets:
        print(
            f"Warning: Target index {target_index} is out of bounds or not provided for dataset ID {ds_id}."
        )
        target_index = choice(range(num_targets))
        print(f"Choosing target index {target_index} instead.")

    y = dataset.data.targets.iloc[:, target_index].values.reshape((-1, 1))

    # Select the first 100 columns (before filtering for numerical types)
    X = pd.DataFrame(dataset.data.features, columns=dataset.data.features.columns)

    print(f"X shape before filtering: {X.shape}")

    # Filter only numerical columns from the selected 100
    numerical_cols = X.select_dtypes(include=["number"]).columns
    X = X[numerical_cols]

# Select the first 3 numerical columns (or fewer if not enough exist)
    X = X[numerical_cols].iloc[:, :3]

    # Limit to the first 100 rows
    X = X.iloc[:100, :]
    y = y[:100, :]  

    print(f"X shape after filtering: {X.shape}")
    D = X.shape[1]
    #### SCALE THE DATA
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
	#### SPLIT THE DATA
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state
    )
    # breakpoint()
    print("Processed data")
    ##################################################################

    ######### TRAINING NN
    ##################################################################

    ###relu activation
    activation_fn_type = "relu"
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)

    ###neural network model design choices
    width = 50
    hidden_layers = 1
    input_dim = D
    output_dim = 1

    architecture = {
        "width": width,
        "hidden_layers": hidden_layers,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "activation_fn_type": "relu",
        "activation_fn_params": "rate=1",
        "activation_fn": activation_fn,
    }

    # set random state to make the experiments replicable
    rand_state = rand_state
    random = np.random.RandomState(seed=rand_state)

    # instantiate a Feedforward neural network object
    nn = Feedforward(architecture, random=random)

    # print("Instantiated Neural Network")

    ###define design choices in gradient descent
    step_size = 5e-3
    # max_iteration = 5000
    max_iteration = 5000
    random_restarts = 5
    #random_restarts = 5
    
    check_point = 100001

    params = {
        "step_size": step_size,
        "max_iteration": max_iteration,
        "check_point": check_point,
        "random_restarts": random_restarts,
    }


    # fit my neural network to minimize MSE on the given data
    nn.fit(x_train.T, y_train.T, params)
    
    

    y_pred_test = nn.forward(nn.weights, x_test.T)
    y_pred_train = nn.forward(nn.weights, x_train.T)

    print(
        "mean_squared_error for test",
        mean_squared_error(y_test.flatten(), y_pred_test.flatten()),
    )
    print(
        "mean_squared_error for train",
        mean_squared_error(y_train.flatten(), y_pred_train.flatten()),
    )

    ##################################################################

    # plt.plot(
    #     np.arange(max_iteration),
    #     nn.objective_trace,
    #     color="red",
    #     label="training loss vs iterations",
    # )
    # plt.legend()
    # plt.title("Loss vs. training iterations")
    # plt.ylabel("Loss")
    # plt.savefig(f"losses_{ds_id}.png")
    # plt.close()

    return X, y, lambda x: nn.forward(nn.weights, x)

def save_results(uci_id, rand_state):
    """Saves the dataset and model function in a pickle file."""
    subdirectory = "nn_results"
    os.makedirs(subdirectory, exist_ok=True)
    filename = os.path.join(subdirectory, f"dataset{uci_id}_{rand_state}.pickle")
    hidden_layers = 1
    target_index = 10 
    rand_state = int(rand_state)
    X, y, neural_network_callable = wrapper_for_ds(uci_id, target_index, hidden_layers, rand_state)

    with open(filename, "wb") as f:
        dill.dump({"X": X, "y": y, "model_fn": neural_network_callable}, f)
    print(f"Saved results to subdirectory_{filename}")







