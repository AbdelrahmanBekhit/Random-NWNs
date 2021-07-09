#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to evolve nanowire networks over time.
# See: https://doi.org/10.1038/nature06932
# 
# Author: Marcus Kasdorf
# Date:   June 18, 2021

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

from numbers import Number
from typing import Callable, List, Union, Tuple
from scipy.integrate._ivp.ivp import OdeResult

from .calculations import solve_network


def resist_func(
    NWN: nx.Graph,
    w: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    The HP group's resistance function in nondimensionalized form.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    w : ndarray or scalar
        Nondimensionalized state variable of the memristor element(s).

    Returns
    -------
    R : ndarray or scalar
        Resistance of the memristor element(s).

    """
    Roff_Ron = NWN.graph["units"]["Roff_Ron"]
    R = w * (1 - Roff_Ron) + Roff_Ron
    return R


def _deriv(
    t: float, 
    w: np.ndarray,
    NWN: nx.Graph,
    source_node: tuple, 
    drain_node: tuple,
    voltage_func: Callable,
    edge_list: list,
    window_func: Callable,
    solver: str = "spsolve",
    kwargs: dict = None
) -> np.ndarray:
    """
    Derivative of the nondimensionalized state variables `w`.

    """
    if kwargs is None:
        kwargs = dict()

    # Solve for and set resistances
    R = resist_func(NWN, w)
    attrs = {
        edge: {"conductance": 1 / R[i]} for i, edge in enumerate(edge_list)   
    }
    nx.set_edge_attributes(NWN, attrs)

    # Find applied voltage at the current time
    applied_V = voltage_func(t)

    # Solve for voltage at each node
    *V, I = solve_network(
        NWN, source_node, drain_node, applied_V, 
        "voltage", solver, **kwargs
    )
    V = np.array(V)

    # Find voltage differences
    v0, v1 = np.zeros_like(w), np.zeros_like(w)
    for i, edge in enumerate(edge_list):
        v0_indx = NWN.graph["node_indices"][edge[0]]
        v1_indx = NWN.graph["node_indices"][edge[1]]
        v0[i] = V[v0_indx] 
        v1[i] = V[v1_indx]
    V_delta = np.abs(v0 - v1) * np.sign(applied_V)
        
    # Find dw/dt
    dwdt = V_delta / R * window_func(w)

    return dwdt


def solve_evolution(
    NWN: nx.Graph, 
    t_eval: np.ndarray,
    source_node: tuple, 
    drain_node: tuple, 
    voltage_func: Callable,
    window_func: Callable = None,
    tol: float = 1e-12,
    solver: str = "spsolve",
    **kwargs
) -> Tuple[OdeResult, List[Tuple]]:
    """
    Solve for the state variables `w` of the junctions of the given nanowire
    network at various points in time with an applied voltage.
    
    Parameters
    ----------
    NWN: Graph
        Nanowire network.

    t_eval : ndarray
        Time points to evaluate the nanowire network at. These should have
        units of `t0`.

    source_node : tuple
        Voltage/current source node.

    drain_node : tuple
        Grounded output node.

    voltage_func : Callable
        The applied voltage with the calling signature `func(t)`. The voltage 
        should have units of `v0`.

    window_func : Callable, optional
        The window function used in the derivative of `w`. The calling
        signature is `f(w)` where w is the array of state variables.
        The default window function is `f(w) = 1`.

    tol : float, optional
        Tolerance of `scipy.integrate.solve_ivp`. Defaults to 1e-12.

    solver : str, optional
        Name of sparse matrix solving algorithm to use. Default: "spsolve".

    **kwargs
        Keyword arguments passed to the solver.
    
    Returns
    -------
    sol : OdeResult
        Output from `scipy.intergrate.solve_ivp`. See the SciPy documentation
        for information on this output's formatting.

    edge_list : list of tuples
        List of the edges corresponding with each `w`.

    """
    # Default window function
    if window_func is None:
        window_func = lambda x: 1

    # Get list of junction edges and the time bounds
    t_span = (t_eval[0], t_eval[-1])
    edge_list, w0 = map(list, zip(*[
        ((u, v), w) for u, v, w in NWN.edges.data("w") if w is not None]
    ))

    # Solve the system of ODEs
    sol = solve_ivp(
        _deriv, t_span, w0, "DOP853", t_eval, 
        atol = tol, 
        rtol = tol,
        args = (NWN, source_node, drain_node, voltage_func, edge_list, 
            window_func, solver, kwargs)
    )
    final_w = sol.y[:, -1]

    # Update the w value of each edge junction
    attrs = {edge: {"w": final_w[i]} for i, edge in enumerate(edge_list)}
    nx.set_edge_attributes(NWN, attrs)

    return sol, edge_list


def set_state_variables(
    NWN: nx.Graph, 
    w: Union[float, np.ndarray], 
    edge_list: list = None
):
    """
    Sets the given nanowire network's state variable. One can either pass
    a list of state variable values, as well as the a list of edges, or one
    can simply pass a single value and all junctions will be set to that value.

    This also updates the conductances accordingly.

    Parameters
    ----------
    NWN: Graph
        Nanowire network. 

    w : float or ndarray
        Either a single value or an array of values. If an array is passed,
        `edge_list` also needs to be passed as the order will be dependent
        on that list.

    edge_list : list of tuples, optional
        The corresponding edge to each `w` value. Only used if `w` is an array.
    
    """
    R = resist_func(NWN, w)

    if isinstance(w, Number):
        attrs = {
            edge: {
                "w": w, "conductance": 1 / R
            } for edge in NWN.edges if NWN.edges[edge]["type"] == "junction"
        }
        nx.set_edge_attributes(NWN, attrs)
    elif isinstance(w, np.ndarray):
        attrs = {
            edge: {
                "w": w[i], "conductance": 1 / R[i]
            } for i, edge in enumerate(edge_list)
        }
        nx.set_edge_attributes(NWN, attrs)
    else:
        raise ValueError("Parameter w must be a number or an ndarray.")


def get_evolution_values(
    NWN: nx.Graph, 
    sol: OdeResult, 
    edge_list: List[Tuple], 
    source_node: Tuple, 
    drain_node: Tuple, 
    voltage_func: Callable,
    scaled: bool = False,
    solver: str = "spsolve",
    **kwargs
) -> Tuple[np.ndarray]:
    """
    To be used in conjunction with `solve_evolution`. Takes the output from
    `solve_evolution` and finds the voltage, sheet resistance, and current of
    the given nanowire network for the set of `w` values for each time step.

    The appropriate parameters passed should be the same as `solve_evolution`.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    sol : OdeResult
        Output from `solve_evolution`.

    edge_list : list of tuples
        Output from `solve_evolution`.

    source_node : tuple
        Voltage/current source node.

    drain_node : tuple
        Grounded output node.

    voltage_func : Callable
        The applied voltage with the calling signature `func(t)`. The voltage 
        should have units of `v0`.

    scaled : bool, optional
        Scale the output by the characteristic values. Default: False.

    solver : str, optional
        Name of sparse matrix solving algorithm to use. Default: "spsolve".

    **kwargs
        Keyword arguments passed to the solver.

    Returns
    -------
    voltage_array : ndarray
        Array containing the voltage difference between the source and drain
        at each time step.

    resistance_array: ndarray
        Array containing the resistance between the source and drain at each
        time step.

    current_array: ndarray
        Array containing the current flow between the source and drain at each
        time step.

    """
    # Preallocate output
    voltage_array = np.zeros_like(sol.t)
    resistance_array = np.zeros_like(sol.t)
    current_array = np.zeros_like(sol.t)

    source_index = NWN.graph["node_indices"][source_node]
    drain_index = NWN.graph["node_indices"][drain_node]

    # Loop through each time step
    for i in range(len(sol.t)):
        # Set state variables and solve network 
        input_V = voltage_func(sol.t[i])
        set_state_variables(NWN, sol.y.T[i], edge_list)
        out = solve_network(
            NWN, source_node, drain_node, input_V, "voltage", solver, **kwargs
        )

        # Find voltage, sheet resistance, and current
        # between the source and drain nodes
        V = out[source_index] - out[drain_index]
        I = out[-1]
        voltage_array[i] = V
        current_array[i] = I

        if I != 0:
            resistance_array[i] = V / I
        else: 
            resistance_array[i] = np.nan

    # Scale the output if desired
    if scaled:
        units = NWN.graph["units"]
        voltage_array *= units["v0"]
        resistance_array *= units["Ron"]
        current_array *= units["i0"]
    
    return voltage_array, resistance_array, current_array
