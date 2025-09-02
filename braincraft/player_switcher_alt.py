# Braincraft challenge entry — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# by Alain Dutech
"""
Example and evaluation of the performances of a handcrafter switcher player.
"""

import numpy as np
from bot import Bot
from environment_1 import Environment

# # ** activation functions ******************************************************
def identity(x):
    return x

# ** define a model ***********************************************************
def switcher_player():
    """Optimal heuristic driving always left and going through the 'tunnel' once it encountered reward."""

    bot = Bot()
    max_energy= 1.030

    # network hyperparameters
    leak = 0.95
    def act(x):
        x = np.tanh(x)
        return np.where(x > 0, x, 0)

    # model size
    n_cam = bot.camera.resolution
    n_inp = n_cam + 3
    n_rec = 1000 

    n_min = n_cam - 1  # only using every n_min'th sensor node

    model_values = 1, 0.77, 0.79, 0.98, 0.82, 0.01, 0.1

    # input parameters
    exc_input = model_values[0]
    thresh_inp = model_values[1]

    thresh_inp_2 = 2 * model_values[2]

    # energy triggers
    energy_level_1 = max_energy * model_values[3]
    energy_level_2 = max_energy * model_values[4]

    # steering parameters
    bias_left_static = model_values[5]
    bias_left_triggered = model_values[6]  

    # construct the model
    W_in = np.zeros((n_rec, n_inp))
    W = np.zeros((n_rec, n_rec))
    W_out = np.zeros((1, n_rec))

    model = W_in, W, W_out, 0, leak, act, identity

    # input weights for connecting sensor to steering pop
    for i in range(0, n_cam, n_min):  W_in[i, i] = exc_input
    W_in[:n_cam:n_min, -1] = -thresh_inp

    # 2nd pop for opposite steering behaviour (draws agent towards walls)
    for i in range(0, n_cam, n_min):  W_in[i+n_cam, i] = exc_input
    W_in[n_cam:2*n_cam:n_min, -1] = -thresh_inp_2

    # reward sensor - input
    W_in[2*n_cam+1, n_cam+1] = 1
    W_in[2*n_cam+2, n_cam+1] = -1
    # energy threshold - fast reward (triggers on left reward)
    W_in[2*n_cam+1, n_cam+2] = -energy_level_1
    # energy threshold - delayed reward (triggers on right reward)
    W_in[2*n_cam+2, n_cam+2] = energy_level_2

    # persistent activity once triggered
    W[2*n_cam+1, 2*n_cam+1] = 1000
    W[2*n_cam+2, 2*n_cam+2] = 1000
    # inhibition of delayed reward trigger
    W[2*n_cam+2, 2*n_cam+1] = -1000

    # activation of second steering pop
    W[n_cam:2*n_cam:n_min, 2*n_cam+1] = 1
    W[n_cam:2*n_cam:n_min, 2*n_cam+2] = 1

    # constant left bias
    W_in[2*n_cam+3, n_cam+2] = 1

    # output weights
    W_out[0, :n_cam:n_min] = n_min * np.linspace(-1, 1, len(W_out[0, :n_cam:n_min]))
    W_out[0, n_cam:2*n_cam:n_min] = bias_left_triggered * n_min * np.linspace(1, -1, len(W_out[0, :n_cam:n_min]))
    W_out[0, 2*n_cam+1] = bias_left_triggered
    W_out[0, 2*n_cam+2] = bias_left_triggered
    W_out[0, 2*n_cam+3] = bias_left_static

    yield model

    return model

def generate_info_json():
    """Generate information for 'debug_plot', json format.
    """
    import json

    bot = Bot()
    n_cam = bot.camera.resolution
    dim_input = n_cam + 3
    # indexes for some input and X neurones
    i_left = 0
    i_right = n_cam - 1
    i_nrj = dim_input - 2
    i_fast_reward = 2*n_cam+1
    i_del_reward = 2*n_cam+2

    info = {
        "player": {
            "module": "player_switcher_alt",
            "func":   "switcher_player"
        },
        "neurons": {
            "nni": {
                "indices": [i_left, i_right, i_nrj],
                "labels":  ["L", "R", "nrj"],
                "lim":     [-0.05, 1.05]
            },
            "nnx": {
                "indices": [i_fast_reward, i_del_reward],
                "labels":  ["fr", "dr"],
                "lim":     [-0.05, 1.05]
            },
            "nno": {
                "indices": [0],
                "labels":  ["out"],
                "lim":     [-6.1, 6.1]
            }
        }
    }

    print( "info for using debug_plot" )
    print( json.dumps(info) )

if __name__ == "__main__":
    import time
    from challenge import train, evaluate

    seed = 78
    np.random.seed(seed)

    # comment if not needed
    generate_info_json()

    # Training (100 seconds)
    timeout_train = 100
    print(f"Starting training for {timeout_train} seconds (user time)")
    model = train(switcher_player, timeout=timeout_train)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, runs=10, debug=False, seed=seed )
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")

