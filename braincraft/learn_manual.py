# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Evaluation of the performance of a manual (human) player.
"""
import sys
import numpy as np


def asVecStr( x ):
    """Inline vec or first col"""
    return np.array2string( x[:,0], precision=2, separator=",",
                            suppress_small=True)
def nice_nonzero_1d( x ):
    idr = np.nonzero(x)
    for r in idr[0]:
        print( f"  ({r},): {x[r]}" )
def nice_nonzero_2d( x ):
    idr, idc = np.nonzero(x)
    for r,c in zip(idr, idc):
        print( f"  ({r}, {c}): {x[r,c]}" )

def identity(x):
    return x
def ReLU(x):
    return np.clip(x, a_min=0, a_max=None)

def on_press(event):
    """ Change bot direction with lefT/right arrows """
    global move
    global environment, bot
    global step_simu
    global mem_time, mem_output, mem_x0, mem_x1

    sys.stdout.flush()
    if move:
        if event.key == 'left':
            bot.direction += np.radians(5)
        elif event.key == 'right':
            bot.direction -= np.radians(5)
        move = False

    if event.key == ' ':
        step_simu = True

    # listen to keys to change variables
    # print( f"event=[{event.key}]" )
    # reset environment
    if event.key == 'r':
        environment = Environment()
        bot = Bot()
        time_simu = 0
        step_sime = False
        mem_time = []
        mem_output = []
        mem_x0 = []
        mem_x1 = []

def switcher_player():
    """An agent with 3 "behaviors", in a kind of subsumption way.
    - go away from things on the right => leads to circling outer perimeter anti-clockwise
    - goaway from close things on the left => avoid abstacle too close
    - if rewarded, take a turn left in the "central corridor"

    NO LEARNING.
    """
    leak = 1.0
    warmup = 1

    bot = Bot()
    dim_cam = bot.camera.resolution
    dim_input = dim_cam + 3
    dim_res = 1000
    # indexes for the some input and X neurones
    i_left = 0
    i_right = dim_cam - 1
    i_center_right = dim_cam - 31
    i_nrj = dim_input - 2
    i_bias = dim_input - 1
    i_filter = i_nrj + 10
    i_turn = i_filter + 10
    i_turn_mid = i_turn+10

    # weights
    W_in = np.zeros((dim_res, dim_input)  )
    W = np.zeros((dim_res, dim_res))
    W_out = np.zeros((1,dim_res))

    # TurnLeft (away from SensorRight)
    W_in[1, i_right] = 1.0
    W_in[1, i_bias] = -0.79
    W_out[0, 1] = 100

    # TurnRight (avoid close SensorLeft)
    W_in[i_turn+1, i_left] = 1.0
    W_in[i_turn+1, i_bias] = -0.88
    W_out[0, i_turn+1] = -50

    # detect NRJ increase.
    W_in[i_nrj, i_nrj] = 1.0
    W[i_nrj+1, i_nrj] = 1.0
    W[i_nrj+2, i_nrj] = 1.0
    W[i_nrj+2, i_nrj+1] = - 1.0
    # W[i_nrj+2, i_nrj+2] = 1.0
    # need a band-filter so as to only detect "small" NRJ bumps
    # (and not the starting bump with value '1')
    W[i_filter, i_nrj+2] = 1.0
    W[i_filter, i_bias]  = -0.004
    W[i_filter+1, i_nrj+2] = 1.0
    # i_filter+2 accumulate and "memorise" reward increase
    W[i_filter+2, i_filter]   = -2.0
    W[i_filter+2, i_filter+1] = 1.0
    W[i_filter+2, i_filter+2] = 1.05
    W[i_filter+2, i_turn_mid+4] = -1000.0  # shut off when end_midturn
    # i_filter+3 is a NOT(i_filter+2)
    W[i_filter+3, i_filter+2] = -1.0
    W[i_filter+3, i_bias] = +0.07

    # "mid_turn" are started when a far away obstacle on i_left has been seen
    # enough time since some time (hence the * 1.05 feedback on i_turn_mid+1)
    # cumulate i_left < 0.40 when rewarded
    W_in[i_turn_mid, i_left] = -1.0
    W_in[i_turn_mid, i_bias] = 0.45
    W[i_turn_mid, i_filter+3] = -10 # do not detect if no reward received yet
    # accumulate diff
    W[i_turn_mid+1, i_turn_mid] = 1.6
    W[i_turn_mid+1, i_turn_mid+1] = 1.07
    W[i_turn_mid+1, i_turn_mid+4] = -1000  # shut off at end of mid_turn
    # check when to start (threshold i_turn_mid+2)
    W[i_turn_mid+2, i_turn_mid+1] = 1.0
    W[i_turn_mid+2, i_bias] = -1.3

    # when to stop "mid_turn"
    # need to know xhen NOT(mid_turn)
    W[i_turn_mid+3, i_turn_mid+2] = -1.0
    W_in[i_turn_mid+3, i_bias] = 1.3
    # detect if center_right < 43 AND turning with i_turn_mid+2 (i.e i_turn_mid+2 > 0)
    W_in[i_turn_mid+4, i_center_right] = -1
    W_in[i_turn_mid+4, i_bias] = 0.43
    W[i_turn_mid+4, i_turn_mid+3] = -1   # no while NOT mid_turn

    # finaly, apply to output
    W_out[0, i_turn_mid+2] = 100

    f = lambda x: ReLU( x )
    g = identity
    model = W_in, W, W_out, warmup, leak, f, g

    return model

# ******************************************************************************
# ****************************************************************** NN_env_simu
# ******************************************************************************
class NN_env_simu:
    """Simulate the NN and move the bot."""

    def __init__(self, model, bot, environment):
        """Create the NN memory (self.X), time, distance, has_moved."""
        self.model = model
        self.bot = bot
        self.environment = environment
        # nn memory
        self.X = None
        self.time = 0

        # distance traveled
        self.distance = 0
        # agent just moved ?
        self.has_moved = False

        self.reset()

    def reset(self):
        """Reset NN, suppose that environment has been reseted before ?"""
        # NN
        self.X = np.zeros( (1000,1) )

        self.time = 0
        self.distance = 0
        self.has_moved = False

        self.bot.camera.render(self.bot.position, self.bot.direction,
                               self.environment.world,
                               self.environment.colormap)

    def step(self):
        """Step the NN, update self.X, set self.distance, self.has_moved.

        Returns: time, I, X, O.
        """
        W_in, W, W_out, warmup, leak, f, g = self.model
        n = self.bot.camera.resolution
        I = np.zeros((n+3,1))

        # The higher, the closer
        I[:n,0] = 1 - self.bot.camera.depths
        I[n:,0] = self.bot.hit, self.bot.energy, 1.0
        self.X = (1-leak)*self.X + leak*f(np.dot(W_in, I) + np.dot(W, self.X))
        O = np.dot(W_out, g(self.X))

        if self.time > warmup:
            position = self.bot.position
            self.bot.forward( O, self.environment, debug=False )  # no 3D
            self.distance += np.linalg.norm(position - self.bot.position)
            self.has_moved = True

        self.time += 1
        return self.time, I, self.X, O
# ******************************************************************************

# ******************************************************************************
# ***************************************************************** GraphUpdater
# ******************************************************************************
class GraphUpdater:
    def __init__(self, nn_model, info ):
        """
        nn_model = W_in, W, W_out, warmup, leak, f, g
        info = { layer_name: {ax, indices, labels, lim} }
        """
        self.nn = nn_model
        self.info = info

        self.mem_time = []
        self.mem_I = None
        self.mem_X = None

        for _, info_layer in self.info.items():
            self.gen_matplot_lines( info_layer )

    def gen_matplot_lines(self, info_layer ):
        idx, labels, lim = info_layer["indices"], info_layer["labels"], info_layer["lim"]
        ax = info_layer["ax"]

        nb_lines = len(idx)
        x_data = np.zeros( (1,) )
        y_data = np.zeros( (1, nb_lines) )
        ax.set_ylim( lim[0], lim[1] )
        info_layer["lines"] = ax.plot( x_data, y_data, label=labels )

    def store_and_plot_data(self, mem, data, info_layer ):
        ax = info_layer["ax"]
        lines = info_layer["lines"]
        idx, labels, lim = info_layer["indices"], info_layer["labels"], info_layer["lim"]

        if mem is None:
            mem = data[idx, :].transpose()
        else:
            mem = np.concatenate( (mem, data[idx, :].transpose()) )

        for (i, line) in enumerate( lines ):
            line.set_data( self.mem_time, mem[:,i] )

        rescale = False
        if np.max(mem) > lim[1]:
            lim[1] = 1.1 * np.max(mem)
            rescale = rescale or True
        if np.min(mem) < lim[0]:
            lim[O] = 1.1 * np.min(mem)
            rescale = rescale or True
        if rescale:
            ax.set_ylim( bottom=lim[0], top=lim[1] )
            info_layer["lim"] = lim

        return mem


    def update(self, frame=0, nn=None):
        """ Update display, bot and stats. """
        
        global anim, graphics, bot, environment, step_simu

        if not step_simu:
            return

        # print( f"frame={frame}, time={self.nn.time}, step={step_simu}" )
        step_simu = False

        energy = bot.energy
        time_simu, I, X, O = self.nn.step()

        # print( f"X_shape={X.shape}" )
        # print( f"memX = {X[self.idX, :]}, shape:{X[self.idX, :].shape}" )

        self.mem_time.append(time_simu)
        self.mem_I = self.store_and_plot_data( self.mem_I, I, self.info["input"])
        self.mem_X = self.store_and_plot_data( self.mem_X, X, self.info["nx"])
        self.mem_O = self.store_and_plot_data( self.mem_O, O, self.info["output"])

        graphics["rays"].set_segments(bot.camera.rays)
        graphics["hits"].set_offsets(bot.camera.rays[:,1,:])
        graphics["bot"].set_center(bot.position)

        graphics["ax2"].set_xlim( left=-5, right=time_simu+10 )
        for _, info_layer in self.info.items():
            info_layer["ax"].legend( loc='best' )

        if energy < bot.energy:
            graphics["energy"].set_color( ("black", "white", "C2") )
        else:
            graphics["energy"].set_color( ("black", "white", "C1") )

        if bot.energy > 0:
            ratio = bot.energy/bot.energy_max
            graphics["energy"].set_segments([[(0.1, 0.05),(0.9, 0.05)],
                                             [(0.1, 0.05),(0.9, 0.05)],
                                             [(0.1, 0.05),(0.1 + ratio*0.8, 0.05)]])
        else:
            graphics["energy"].set_segments([[(0.1, 0.05),(0.9, 0.05)],
                                             [(0.1, 0.05),(0.9, 0.05)]])
            anim.event_source.stop()
# ******************************************************************************

if __name__ == "__main__":
    from bot import Bot
    from environment_1 import Environment

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    environment = Environment()
    bot         = Bot()
    world = environment.world
    world_rgb = environment.world_rgb

    move = True
    step_simu = False

    model = switcher_player()
    simu = NN_env_simu( model, bot, environment )
    n = bot.camera.resolution
    i_left = 0
    i_right = n - 1
    i_center_right = n - 31
    i_nrj = n + 1
    i_filter = i_nrj + 10
    i_turn = i_filter + 10
    i_turn_mid = i_turn+10

    fig = plt.figure(figsize=(6,2.5))
    ax1 = plt.axes([0.0,0.0,1/2,1.0], aspect=1, frameon=False)
    ax1.set_xlim(0,1), ax1.set_ylim(0,1), ax1.set_axis_off()
    # ax2 for plotting X and O values
    ax2 = plt.axes([1/2+0.05, 1/3.0, 1/2-0.1, 2/3.0-0.05-0.1], aspect='auto',frameon=True)
    # ax3 for plotting I values
    ax3 = plt.axes([1/2+0.05, 0.05, 1/2-0.1, 1/3-0.05], aspect='auto', frameon=True,
                   sharex=ax2)
    # ax4 for ploting O values
    ax4 = plt.axes([1/2+0.05, 0.85, 1/2-0.1, 0.1], aspect='auto', frameon=True,
                   sharex=ax2)

    info_graph = {
        "input":  { "ax":      ax3,
                    "indices": [i_left, i_right, i_center_right, i_nrj],
                    "labels":  ["L", "R", "C", "nrj"],
                    "lim":     [-0.05, 1.05]},
        "nx":     { "ax":      ax2,
                    "indices": [i_filter+2, i_turn_mid+1],
                    "labels":  ["X_f", "X_t"],
                    "lim":     [-0.05, 1.05] },
        "output": { "ax":      ax4,
                    "indices": [0],
                    "labels":  ["out"],
                    "lim":     [-5.1, 5.1] },
        }
    gu = GraphUpdater( simu, info_graph )

    graphics = {
        "topview" : ax1.imshow(environment.world_rgb, interpolation="nearest", origin="lower",
                               extent = [0.0, world.shape[1]/max(world.shape),
                                         0.0, world.shape[0]/max(world.shape)]),
        "bot" : ax1.add_artist(Circle((0,0), 0.05,
                                      zorder=50, facecolor="white", edgecolor="black")),
        "rays" : ax1.add_collection(LineCollection([], color="C1", linewidth=0.5, zorder=30)),
        "hits" :  ax1.scatter([], [], s=1, linewidth=0, color="black", zorder=40),
        # "camera" : ax2.imshow(np.zeros((1,1,3)), interpolation="nearest",
        #                       origin="lower", extent = [0.0, 1.0, 0.0, 1.0]),
        "energy" : ax1.add_collection(
            LineCollection([[(0.1, 0.05),(0.9, 0.05)],
                            [(0.1, 0.05),(0.9, 0.05)],
                            [(0.1, 0.05),(0.9, 0.05)]],
                           color=("black", "white", "C1"), linewidth=(20,18,12),
                           capstyle="round", zorder=150)),
        "ax2": ax2,
        "ax3": ax3,
        "ax4": ax4,
    }

    print("The manual player is controlled by left and right arrows")
    print("Since the human player (e.g. you) has a bird eye view, this serves as an ideal reference")

    fig.canvas.mpl_connect('key_press_event', on_press)
    anim = FuncAnimation(fig, gu.update,
                         frames=3600, interval=60, repeat=False)
    plt.show()
    print(f"Final score: {simu.distance:.2f} (single run)")
    sys.stdout.flush()
