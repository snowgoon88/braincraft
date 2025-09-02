# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
#
# Evaluate and "visualize" the behavior of a player.

import importlib
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np
import sys

from bot import Bot
from environment_1 import Environment

desc_msg = """
Evaluate and "visualize" the behavior of a player.

Agent moves in debug mode (only the top-down 2D view) and the activity
of some neurones is plotted.

Keyboard can be used:
 - q/Q: quit
 - SPACE: play/pause simulation
 - a:     Advance ONE step of simulation
 - left/right arriws: change agent orientation
"""

def usage_msg( prog_name ):
    return f"""
usage: {prog_name} <player_description.json>

    where <player_description.json> is a JSON formatted file with information
    on the 'player' to test and on the neurons to visualize.

"""

# ******************************************************************************
def on_press(event):
    """Listen to keyboard events.

    - left/right arrows => change agent orientation.
    - q: quit
    - s: advance ONE step
    - SPACE: play/pause simulation
    """

    global step_simu, run_simu, allow_arrow_move

    sys.stdout.flush()
    if allow_arrow_move:
        if event.key == 'left':
            bot.direction += np.radians(5)
        elif event.key == 'right':
            bot.direction -= np.radians(5)
        allow_arrow_move = False

    if event.key == ' ':
        run_simu = not run_simu

    if event.key == 'a':
        step_simu = True

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
        self.reset()

    def reset(self):
        """Reset NN, suppose that environment has been reseted before ?"""
        # NN
        self.X = np.zeros( (1000, 1) )

        self.time = 0
        self.distance = 0

        self.bot.camera.render(self.bot.position, self.bot.direction,
                               self.environment.world,
                               self.environment.colormap)

    def step(self):
        """Step the NN, update self.X, set self.distance, self.has_moved.

        Returns: time, I, X, O.
        """
        W_in, W, W_out, warmup, leak, f, g = self.model
        n = self.bot.camera.resolution
        I = np.zeros((n+3, 1))

        # The higher, the closer
        I[:n, 0] = 1 - self.bot.camera.depths
        I[n:, 0] = self.bot.hit, self.bot.energy, 1.0
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
    def __init__(self, nn_simu, info ):
        """
        nn_model = W_in, W, W_out, warmup, leak, f, g
        info = { ...,
                 neurons: {layer_name: {ax, indices, labels, lim} }},
                 ax_env, ax_nni, ax_nnx, ax_nno,
               }
        """
        self.nn = nn_simu
        self.info = info

        self.mem_time = []
        self.mem_I = None
        self.mem_X = None
        self.mem_O = None

        for k, info_layer in self.info["neurons"].items():
            self._gen_matplot_lines( info["ax_"+k], info_layer )

    def _gen_matplot_lines(self, ax, info_layer ):
        idx, labels, lim = info_layer["indices"], info_layer["labels"], info_layer["lim"]

        nb_lines = len(idx)
        x_data = np.zeros( (1,) )
        y_data = np.zeros( (1, nb_lines) )
        ax.set_ylim( lim[0], lim[1] )
        info_layer["lines"] = ax.plot( x_data, y_data, label=labels )

    def store_and_plot_data(self, mem, data, ax, info_layer,
                            update_lims=True ):
        # ax = info_layer["ax"]
        lines = info_layer["lines"]
        idx, lim = info_layer["indices"], info_layer["lim"]

        if mem is None:
            mem = data[idx, :].transpose()
        else:
            mem = np.concatenate( (mem, data[idx, :].transpose()) )

        for (i, line) in enumerate( lines ):
            line.set_data( self.mem_time, mem[:, i] )

        if update_lims:
            rescale = False
            if np.max(mem) > lim[1]:
                lim[1] = 1.1 * np.max(mem)
                rescale = rescale or True
            if np.min(mem) < lim[0]:
                lim[0] = 1.1 * np.min(mem)
                rescale = rescale or True
            if rescale:
                ax.set_ylim( bottom=lim[0], top=lim[1] )
                info_layer["lim"] = lim

        return mem


    def update(self, frame=0):
        """ Update display, bot and stats. """

        global anim, bot, environment
        global step_simu, run_simu, allow_arrow_move

        if not step_simu and not run_simu:
            return

        # print( f"frame={frame}, time={self.nn.time}, step={step_simu}" )
        step_simu = False
        allow_arrow_move = True

        energy = bot.energy
        time_simu, I, X, O = self.nn.step()

        # print( f"X_shape={X.shape}" )
        # print( f"memX = {X[self.idX, :]}, shape:{X[self.idX, :].shape}" )

        self.mem_time.append(time_simu)
        self.mem_I = self.store_and_plot_data( self.mem_I, I,
                                               self.info["ax_nni"],
                                               self.info["neurons"]["nni"])
        self.mem_X = self.store_and_plot_data( self.mem_X, X,
                                               self.info["ax_nnx"],
                                               self.info["neurons"]["nnx"])
        self.mem_O = self.store_and_plot_data( self.mem_O, O,
                                               self.info["ax_nno"],
                                               self.info["neurons"]["nno"],
                                               update_lims=False)

        self.info["rays"].set_segments(bot.camera.rays)
        self.info["hits"].set_offsets(bot.camera.rays[:, 1, :])
        self.info["bot"].set_center(bot.position)

        self.info["ax_nnx"].set_xlim( left=-5, right=time_simu+10 )
        for k, _ in self.info["neurons"].items():
            self.info["ax_"+k].legend( loc='best' )

        if energy < bot.energy:
            self.info["energy"].set_color( ("black", "white", "C2") )
        else:
            self.info["energy"].set_color( ("black", "white", "C1") )

        if bot.energy > 0:
            ratio = bot.energy/bot.energy_max
            self.info["energy"].set_segments([[(0.1, 0.05), (0.9, 0.05)],
                                             [(0.1, 0.05), (0.9, 0.05)],
                                             [(0.1, 0.05), (0.1 + ratio*0.8, 0.05)]])
        else:
            self.info["energy"].set_segments([[(0.1, 0.05), (0.9, 0.05)],
                                             [(0.1, 0.05), (0.9, 0.05)]])
            anim.event_source.stop()

# ******************************************************************************
# ************************************************************************* main
# ******************************************************************************
if __name__ == "__main__":

    # check args and load info from json
    if len(sys.argv) < 2:
        print( "Error: not enough command line args." )
        print( usage_msg( sys.argv[0] ) )
        sys.exit(1)

    with open(sys.argv[1]) as json_file:
        info = json.load( json_file )

    environment = Environment()
    bot         = Bot()
    world = environment.world
    world_rgb = environment.world_rgb

    # can use key <- and -> to change bot orientation
    allow_arrow_move = True
    # ask for ONE step of simulation
    step_simu = False
    # run simulation non-stop
    run_simu = False

    # get the model from the player "module"
    module_player = importlib.import_module( info["player"]["module"] )
    function_player = getattr( module_player, info["player"]["func"] )
    for res in function_player():
        model = res
    simu = NN_env_simu( model, bot, environment )

    fig = plt.figure(figsize=(6, 2.5))
    ax_env = plt.axes([0.0, 0.0, 1/2, 1.0], aspect=1, frameon=False)
    ax_env.set_xlim(0, 1), ax_env.set_ylim(0, 1), ax_env.set_axis_off()
    # axes for plotting I, X and O values
    ax_nnx = plt.axes([1/2+0.05, 1/3.0, 1/2-0.1, 2/3.0-0.05-0.1],
                      aspect='auto', frameon=True)
    ax_nni = plt.axes([1/2+0.05, 0.05, 1/2-0.1, 1/3-0.05],
                      aspect='auto', frameon=True,
                      sharex=ax_nnx)
    ax_nno = plt.axes([1/2+0.05, 0.85, 1/2-0.1, 0.1],
                      aspect='auto', frameon=True,
                      sharex=ax_nnx)

    info["topview"] = ax_env.imshow( environment.world_rgb,
                                     interpolation="nearest",
                                     origin="lower",
                                     extent=[0.0, world.shape[1]/max(world.shape),
                                             0.0, world.shape[0]/max(world.shape)])
    info["bot"] = ax_env.add_artist(Circle((0, 0), 0.05,
                                           zorder=50,
                                           facecolor="white",
                                           edgecolor="black"))
    info["rays"] = ax_env.add_collection(LineCollection([],
                                                        color="C1",
                                                        linewidth=0.5,
                                                        zorder=30))
    info["hits"] = ax_env.scatter([], [], s=1, linewidth=0,
                                  color="black", zorder=40)

    info["energy"] = ax_env.add_collection(
        LineCollection([[(0.1, 0.05), (0.9, 0.05)],
                        [(0.1, 0.05), (0.9, 0.05)],
                        [(0.1, 0.05), (0.9, 0.05)]],
                       color=("black", "white", "C1"),
                       linewidth=(20, 18, 12),
                       capstyle="round", zorder=150))
    info["ax_env"] = ax_env
    info["ax_nni"] = ax_nni
    info["ax_nnx"] = ax_nnx
    info["ax_nno"] = ax_nno
    gu = GraphUpdater( simu, info )

    print( desc_msg )

    fig.canvas.mpl_connect('key_press_event', on_press)
    anim = FuncAnimation(fig, gu.update,
                         frames=3600, interval=60, repeat=False)
    plt.show()
    print(f"Final score: {simu.distance:.2f} (single run)")
    sys.stdout.flush()
