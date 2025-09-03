# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
#
# Generate a Graph (using graphviz/dot) of a model returned by a 'playuer_*'

import json
import importlib
import numpy as np
import pydot
import sys


def usage_msg( prog_name ):
    return f"""
usage: {prog_name} <player_description.json>

    where <player_description.json> is a JSON formatted file with information
    on the 'player' from which to get a graph of neuons.
"""

def get_model(info):
    """Get model from a player_ module.

    Param:
    - info: { ..., player{ module, func }, ... }
    """
    module_player = importlib.import_module( info["player"]["module"] )
    function_player = getattr( module_player, info["player"]["func"] )
    for res in function_player():
        model = res

    return model

# ******************************************************************************
# format name of node and edges
def I( id ):
    return "I_"+str(id)
def X( id ):
    return "X_"+str(id)
def O( id ):
    return "O_"+str(id)
def Label( v ):
    return f"{v:.3f}"

# ******************************************************************************
def get_info( w_matrix ):
    """Get Nodes_info and Edge_info from weight_matrix.

    Returns: ([idx of from_nodes], [idx of to_nodes],
              [(i_row, i_col, val) of Edges)]
    """

    # look for nodes
    coord = np.nonzero( w_matrix )
    to_neur_idx = np.unique( coord[0] )
    from_neur_idx = np.unique( coord[1] )

    edges = [(id_from, id_to, w_matrix[id_to, id_from])
             for id_to, id_from in zip(coord[0], coord[1])]

    return (from_neur_idx, to_neur_idx, edges)

def make_graph( model ):
    """Make a Graphiz graph from model information."""

    W_in, W, W_out, _, _, _, _ = model

    id_input, id_x, edge_i_x = get_info( W_in )
    tmp_x_in, tmp_x_out, edge_x_x = get_info( W )
    tmp_x_o, id_out, edge_x_o = get_info( W_out )

    gp = pydot.Dot("braincraft", graph_type="digraph")
    # nodes
    for i in id_input:
        gp.add_node(pydot.Node( I(i), label=I(i), shape="box" ))
    for i in np.unique( np.concat( (id_x, tmp_x_in, tmp_x_out, tmp_x_o) ) ):
        gp.add_node(pydot.Node( X(i), label=X(i), shape="circle" ))
    for i in id_out:
        gp.add_node(pydot.Node( O(i), label=O(i), shape="cds" ))

    # edges
    for i_from, i_to, v in edge_i_x:
        from_n = I(i_from)
        to_n = X(i_to)
        val = Label(v)
        col = "blue" if v > 0 else "red"
        gp.add_edge(pydot.Edge( from_n, to_n, label=val,
                                fontcolor=col, color=col ))
    for i_from, i_to, v in edge_x_x:
        from_n = X(i_from)
        to_n = X(i_to)
        val = Label(v)
        col = "blue" if v > 0 else "red"
        gp.add_edge(pydot.Edge( from_n, to_n, label=val,
                                fontcolor=col, color=col ))
    for i_from, i_to, v in edge_x_o:
        from_n = X(i_from)
        to_n = O(i_to)
        val = Label(v)
        col = "blue" if v > 0 else "red"
        gp.add_edge(pydot.Edge( from_n, to_n, label=val,
                                fontcolor=col, color=col ))
    return gp

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

    model = get_model( info )
    graph = make_graph( model )
    graph.write_png( "tmp_graph.png" )
