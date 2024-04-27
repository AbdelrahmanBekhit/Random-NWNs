#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This is a dynamic program interface that will simplify creating graphs for non-coders
# 
# Author: Abdelrahman Bekhit
# Date:   August 13, 2023

from randomnwn import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL, messagebox
import numpy as np
from scipy import signal
import math
import copy
# If there is a need to change the values [sigma, theta, a, w0, tau0, epi0] of check out parameter_info in parameter_sliders()

# List to store electrode information
electrodes = [
    ["left", 2, 1, [-0.5, 0.5]],
    ["right", 2, 1, [-0.5, 0.5]]
]

# List to store electrode types
electrode_types = ["left", "right", "top", "bottom"]

# Dictionary to store graph options data
graph_options_data = {
    "wire_length": 7.0 / 7,
    "size": 50.0 / 7,
    "density": 0.3 * 7 ** 2,
    "conductance": 0.1 / 0.1,
    "capacitance": 1000,
    "diameter": 50.0 / 50.0,
    "resistivity": 22.6 / 22.6,
    "tolerance":1e-12,
    "max_time":2000,
    "period":666
}

sliders = {
    "Sigma" : 0.03,
    "Theta" : 0.01,
    "a" : 0.75,
    "w" : 0.055,
    "Tau" : 20,
    "Epsilon" : 0.055
}
  
cached_params = None
cached_results = None

def create_and_plot_graph():
    """
    Creates and plots graphs based on selected options and parameters.

    This function utilizes various parameters and settings to create and visualize graphs
    related to the Chen memristor model and nanowire network.

    Returns:
    None
    """
    global cached_params, cached_results, sliders
    units = get_units()  # Get units
    font = 14
    advanced = {}
    
    graph_type = graph_type_var.get()
       
    # check if seed is valid
    try:
        seed = int(seed_var.get())
    except:
        messagebox.showwarning("Value error", "Please enter a valid number for the seed")
        return
    
    # Make sure that the values are convertable to a float 
    for key, value in graph_options_data.items():
        try:
            if(value != ""):
                advanced[key] = float(value)
            else:
                messagebox.showwarning("Value error", f"{key} is empty")
                return
        except ValueError:
            messagebox.showwarning("Float error",f"{key} is not a possible decimal number")
            return
    
    # Make sure that tolerance is more than 1x10^-5
    if(advanced["tolerance"] >= 1e-5):
        messagebox.showwarning("Tolerance error","Tolerance Must be less than 1e-5")
        return

    def ver():
        """
        Checks if a user needs to recalculate the Solutions or not.

        Returns:
        Bool: True if the user needs to recalculate values, False otherwise
        """
        global cached_params
        
        if cached_params is not None:
            if cached_params == (advanced, sliders, electrodes, seed, cached_params[4], graph_mode_var.get()):
                if graph_type == "parabolic wave":
                    if cached_params[4] == "square wave" or cached_params[4] == "triangular wave":
                        return True
                    cached_params = (advanced, sliders, electrodes, seed, "parabolic wave", graph_mode_var.get())
                elif graph_type == "square wave":
                    if cached_params[4] == "parabolic wave" or cached_params[4] == "triangular wave":
                        return True
                    cached_params = (advanced, sliders, electrodes, seed, "square wave", graph_mode_var.get())
                elif graph_type == "triangular wave":
                    if cached_params[4] == "square wave" or cached_params[4] == "parabolic wave":
                        return True
                    cached_params = (advanced, sliders, electrodes, seed, "triangular wave", graph_mode_var.get())
                return False
            else:
                return True
        else:
            return True
           
    if (ver() == False):
        sol, V, I, R, NWN, electrode_1, electrode_2, electrode_3, electrode_4 = cached_results         
    else:             
        # Create Nanowire Network (NWN)
        NWN = create_NWN(seed=seed, wire_length=advanced["wire_length"], size=advanced["size"],
                        density=advanced["density"], conductance=advanced["conductance"],
                        capacitance=advanced["capacitance"], diameter=advanced["diameter"],
                        resistivity=advanced["resistivity"])   
                
        # Create and position electrodes
        electrode_1, electrode_2, electrode_3, electrode_4 = add_electrodes(
            NWN, [electrodes[0][0], 2, 1, electrodes[0][3]], [electrodes[1][0], 2, 1, electrodes[1][3]])
        
        # Define voltage function
        def voltage_func(t):
            V0 = 20
            T = int(advanced["period"])
            f = 1 / T
            if(graph_type == "square wave"):
                return V0 * (signal.square(2*np.pi*f*t) + 1) / 2
            elif(graph_type == "parabolic wave"):
                return V0 * (math.pow(2*np.pi*f*t,2) + 1) / 2
            elif(graph_type == "triangular wave"):
                return V0 * (1 - np.abs(2 * np.mod(f * t, 1) - 1))
            else:
                return V0 * (signal.square(2*np.pi*f*t) + 1) / 2
        
        # Define window function
        def window_func(w):
            return w * (1 - w)

        # Set Chen memristor model parameters
        set_chen_params(NWN, sliders["Sigma"], sliders["Theta"], sliders["a"]) 

        # Set state variables for memristor model
        set_state_variables(NWN, sliders["w"], sliders["Tau"], sliders["Epsilon"])

        # Time points for evaluation
        t_eval = np.linspace(start=0, stop=advanced["max_time"], num=1000)
        
        # Solve memristor evolution and retrieve data
        sol, edge_list = solve_evolution(
            NWN, t_eval, [electrode_1, electrode_2], [electrode_3, electrode_4],
            voltage_func, window_func, tol=advanced["tolerance"], model=graph_mode_var.get())

        V = [voltage_func(t) for t in sol.t]
        I = get_evolution_current(NWN, sol, edge_list, electrode_1, electrode_4, voltage_func, scaled=True)
        R = V / I
                
        # print the Time required for the volts to go from one electrode to the other
        print("Bottom left -> Bottom right: {:e}\nBottom left -> Top right: {:e}".format(
            *solve_drain_current(NWN, electrode_2, [electrode_4, electrode_3], 10.0, scaled=True)
        ))
        print("Top left -> Bottom right: {:e}\nTop left -> Top right: {:e}\n".format(
            *solve_drain_current(NWN, electrode_1, [electrode_4, electrode_3], 10.0, scaled=True)
        ))
        if(graph_type == "parabolic wave"):
            cached_params = (advanced, sliders, electrodes, seed, "parabolic wave", graph_mode_var.get())
        elif(graph_type == "triangular wave"):
            cached_params = (advanced, sliders, electrodes, seed, "triangular wave", graph_mode_var.get())
        else:
            cached_params = (advanced, sliders, electrodes, seed, "square wave", graph_mode_var.get())
        cached_params = copy.deepcopy(cached_params)
        cached_results = sol, V, I, R, NWN, electrode_1, electrode_2, electrode_3, electrode_4
                
    if graph_type == "Nanowire Network":
        plot_NWN(NWN) 
    
    if (graph_type in ["square wave", "parabolic wave", "triangular wave"]):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        ax.plot(sol.t, V, "red")
        ax.set_ylabel("Voltage (arb. units)", color="red", fontsize=font)
        ax.set_xlabel("Time (arb. units)", fontsize=font)
        ax.tick_params(labelsize=font, which="both")

        ax2 = ax.twinx()
        ax2.plot(sol.t, I, "blue")
        ax2.set_ylabel("Current (arb. units)", color="blue", fontsize=font)
        ax2.tick_params(labelsize=font, which="both")

        ax.grid(alpha=0.5)
    elif (graph_type == "w,tau,epsilon"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        for ax in axes:
            ax.set_xlabel("Time (sec)")
            ax.grid(alpha=0.5)

        axes[0].set_ylabel("w")
        axes[1].set_ylabel("tau")
        axes[2].set_ylabel("epsilon")

        w_list, tau_list, eps_list = np.split(sol.y, 3)
        for w in w_list:
            axes[0].plot(sol.t * NWN.graph["units"]["t0"] * 1e-6, w)
        for tau in tau_list:
            axes[1].plot(sol.t * NWN.graph["units"]["t0"] * 1e-6, tau)
        for eps in eps_list:
            axes[2].plot(sol.t * NWN.graph["units"]["t0"] * 1e-6, eps)
    elif graph_type == "Voltage vs. Current":       
        plt.figure(figsize=(10, 6))
        plt.scatter(V, I, color='blue', edgecolors='#6495ED', label='Current vs. Voltage')
        plt.title('Voltage vs. Current in Nanowire Networks')
        plt.xlabel('Voltage')
        plt.ylabel('Current')
        plt.grid(True)
                    
    plt.show()

def parameter_sliders(parameters_frame):
    """
    Creates sliders for adjusting parameters in the Chen memristor model.

    Parameters:
    parameters_frame (ttk.Frame): The frame where the sliders will be placed.

    Returns:
    tuple: A tuple containing the created sliders for various parameters.
    """
        
    # List of parameter information: (label_text, from_value, to_value, resolution, default_value,tooltip)
    parameter_info = [
        ("Sigma:", 0.01, 0.1, 0.001, 0.03 ,"Represents the decay rate of the memristor elements in the Chen model"),
        ("Theta:", 0.01, 1, 0.001, 0.01, "Controls the threshold for memristor switching in the Chen model"),
        ("a:", 0.5, 1, 0.01, 0.75, "Determines the saturation level in the Chen model and represents the maximum value that 'w' can attain"),
        ("w:", 0.01, 0.1, 0.001, 0.055,"Represents the initial state variable 'w' of the memristor elements"),
        ("Tau:", 1, 30, 1, 20,"Determines the characteristic time constant in the decay term of the memristor elements in the decay model"),
        ("Epsilon:", 0.001, 1, 0.001, 0.055,"Represents an additional variable in the Chen model")
    ]
    
    # code below will create sliders for each of the paramters listed above
    for row, (label_text, from_value, to_value, resolution, default_value, tooltip) in enumerate(parameter_info, start=1):
        # Create and configure the label
        label = ttk.Label(parameters_frame, text=label_text)
        label.grid(row=row, column=0)

        # Create and configure the slider
        default_slider = tk.DoubleVar(value=default_value)
        slider = Scale(parameters_frame, from_=from_value, to=to_value, length=200, resolution=resolution, orient=HORIZONTAL, variable=default_slider)
        slider.grid(row=row, column=1)

        # Add tooltip to the label
        bind_tooltip(label, tooltip)
        
        # Attach a callback function to update the value in the 'sliders' dictionary
        parameter_name = label_text.strip(":")
        slider.config(command=lambda value, parameter_name=parameter_name: slider_callback(parameter_name, value))
        
    def slider_callback(parameter_name, value):
        '''
        This is a function to update the value of the sliders dynamically
        '''
        sliders[parameter_name] = float(value)
    
    return
    
def advanced_options(options_frame):
    '''Creates advanced options widgets for electrode configuration and graph parameters.

    Parameters:
    options_frame (ttk.Frame): The frame where the advanced options widgets will be placed.

    Returns:
    dict: A dictionary containing the updated graph options data excluding the electrodes.'''
    
    electrode_widgets = []  # List to store electrode widgets
    var_dict = {}  # Create a dictionary to hold StringVar objects
    graph_options = {
        "wire_length": "Length of each nanowire. Given in units of l0 ",
        "size":"The size of the nanowire network given in units of l0 ",
        "density":"Density of nanowires in the area determined by the width. Given in units of (l0)^-2 ",
        "conductance":"The junction conductance of the nanowires where they intersect. Given in units of (Ron)^-1 ",
        "capacitance":"The junction capacitance of the nanowires where they intersect. Given in microfarads (unused) ",
        "diameter":"The diameter of each nanowire. Given in units of D0 ",
        "resistivity":"The resistivity of each nanowire. Given in units of rho0 ",
        "tolerance":" The tolerance of the Nanowire network. Defaults to 1e-12. can't be larger than 1e-5",
        "max_time": "Maximum time alotted to calculate the values",
        "period":"The period of the wave graphs, the number inputed will be 1/T"
    }
    
    def validate_input(P):
        """
        A function that only allows a user to add integers, ".", and "-" 

        Parameters:
        P (str): The value being input.

        Returns:
        bool: True if the input is valid, False otherwise.
        """
        return all(c.isdigit() or c in ".-e" for c in P)

    # Iterate over left_graph_data and create corresponding widgets
    for row, (param_name, default_value) in enumerate(graph_options_data.items()):
        label = ttk.Label(options_frame, text=f"{param_name}:")
        
        if row < 5:
            label.grid(row=row, column=0, sticky="e", padx=(10, 0))
        else:
            label.grid(row=row - 5, column=2, sticky="e", padx=(10, 0))
        
        default_var = tk.StringVar(value=str(default_value))
        value_entry = ttk.Entry(options_frame, textvariable=default_var)
        validate_cmd = (value_entry.register(validate_input), "%P")
        value_entry.config(validate="key", validatecommand=validate_cmd)

        # Use tooltip from graph_options dictionary
        tooltip = graph_options[param_name]
        bind_tooltip(value_entry, tooltip)
        
        if row < 5:
            value_entry.grid(row=row, column=1, sticky="w", padx=(0, 10))
        else:
            value_entry.grid(row=row - 5, column=3, sticky="w", padx=(0, 10))
        
        var_dict[param_name] = default_var  # Store StringVar in the dictionary

        def update_parameter_data(var=default_var, param=param_name):
            """
            Updates the graph options data when the value of an entry changes.

            Parameters:
            var (tk.StringVar): The StringVar associated with the entry widget.
            param (str): The parameter name corresponding to the entry widget.
            """
            graph_options_data[param] = var.get()
        
        # Create a lambda function to update the data and associate it with the StringVar
        update_parameter_data_lambda = lambda *args, var=default_var, param=param_name: update_parameter_data(var, param)
        default_var.trace("w", update_parameter_data_lambda)
        update_parameter_data_lambda()

        # Configure uniform weight for rows to achieve even spacing
        options_frame.grid_rowconfigure(row, weight=1)
        
    def create_electrode_widget(index, electrode_info):
        """
        Creates widgets for configuring an electrode's properties.

        Parameters:
        index (int): Index of the electrode.
        electrode_info (list): Information about the electrode.

        Returns:
        None
        """
        row = index + 6
        electrode_number = index + 1
        label = ttk.Label(options_frame, text=f"Electrode {electrode_number}:", width=15)
        label.grid(row=row, column=0)

        electrode_menu = ttk.Combobox(options_frame, values=electrode_types, state="readonly", width=15)
        electrode_menu.grid(row=row, column=1)
        electrode_menu.set(electrode_info[0])  # Set initial value from electrodes
        bind_tooltip(electrode_menu, f"Position of electrode in the Nanowire network")

        # Create entry widgets for length, width, x-coordinate, and y-coordinate of the electrode
        length_var = tk.StringVar(value=electrode_info[1])
        length = ttk.Entry(options_frame, textvariable=length_var, validate="key", width=5)
        length.grid(row=row, column=2, padx=(10, 10))
        length_cmd = (length.register(validate_input), "%P")
        length.config(validate="key", validatecommand=length_cmd)
        bind_tooltip(length, f"length of an electrode")

        width_var = tk.StringVar(value=electrode_info[2])
        width = ttk.Entry(options_frame, textvariable=width_var, validate="key", width=5)
        width.grid(row=row, column=3, padx=(0, 10))
        width_cmd = (width.register(validate_input), "%P")
        width.config(validate="key", validatecommand=width_cmd)
        bind_tooltip(width, f"the width of an electrode")

        x_var = tk.StringVar(value=electrode_info[3][0])
        x = ttk.Entry(options_frame, textvariable=x_var, validate="key", width=5)
        x.grid(row=row, column=4, padx=(0, 10))
        x_cmd = (x.register(validate_input), "%P")
        x.config(validate="key", validatecommand=x_cmd)
        bind_tooltip(x, f"the x-coordinate of electrode {electrode_number}")

        y_var = tk.StringVar(value=electrode_info[3][1])
        y = ttk.Entry(options_frame, textvariable=y_var, validate="key", width=5)
        y.grid(row=row, column=5, padx=(0, 10))
        y_cmd = (y.register(validate_input), "%P")
        y.config(validate="key", validatecommand=y_cmd)
        bind_tooltip(y, f"the y-coordinate of electrode {electrode_number}")

        def update_data(*args):
            """
            Updates the electrode data when the values of the entry widgets change.
            """
            new_electrode_type = electrode_menu.get()
            duplicate_electrode = False

            for i, widget_data in enumerate(electrode_widgets):
                if i != index and widget_data[0].get() == new_electrode_type:
                    duplicate_electrode = True
                    break

            if duplicate_electrode:
                messagebox.showwarning("Duplicate Electrode", "An electrode of this type already exists.")
                electrode_menu.set(electrode_info[0])  # Reset to the previous value
            else:  
                if(length_var.get() != '' and width_var.get() != '' and x_var.get() != '' and y_var.get() != ''):
                    try:
                        electrodes[index] = [
                            electrode_menu.get(),
                            float(length_var.get()),
                            float(width_var.get()),
                            [float(x_var.get()), float(y_var.get())]
                    ]
                    except:
                        messagebox.showwarning("Electrode value", "Please enter a valid decimal number")
                
                    
        # Bind functions to events for updating data when entry values change
        electrode_menu.bind("<<ComboboxSelected>>", update_data)
        length_var.trace_add("write", update_data)
        width_var.trace_add("write", update_data)
        x_var.trace_add("write", update_data)
        y_var.trace_add("write", update_data)

        electrode_widgets.append((electrode_menu, length_var, width_var, x_var, y_var))

    # Create electrode widgets based on the provided electrode information
    for row, electrode_info in enumerate(electrodes):
        create_electrode_widget(row, electrode_info)

    
    return graph_options_data  # Return the updated graph options data

def reset_widget_area(frame, widget_count):
    """
    Reset the weights of rows in a frame.

    Parameters:
    frame (ttk.Frame): The frame whose rows' weights will be reset.
    widget_count (int): The number of rows in the frame.

    Returns:
    None
    """
    for i in range(widget_count):
        frame.rowconfigure(i, weight=0)

def toggle_widgets(widget_frame, button, label_show, label_hide):
    """
    Toggle the visibility of a widget frame and update the button text accordingly.

    Parameters:
    widget_frame (ttk.Frame): The frame containing the widgets to be toggled.
    button (ttk.Button): The button that triggers the toggle action.
    label_show (str): The label to display when the widgets are hidden.
    label_hide (str): The label to display when the widgets are visible.

    Returns:
    None
    """
    if button["text"] == label_show:
        button["text"] = label_hide
        advanced_options(widget_frame)  # Call the electrode_gui function to show the widgets
        widget_frame.grid()
    else:
        button["text"] = label_show
        widget_frame.grid_remove()

def bind_tooltip(widget, text):
    """
    Binds a tooltip to a widget to display additional information when hovered.

    Parameters:
    widget (tk.Widget): The widget to which the tooltip will be bound.
    text (str): The text to be displayed in the tooltip.

    Returns:
    None
    """
    tooltip = [None]

    def show_tooltip(event):
        """Displays the tooltip when the widget is hovered over."""
        nonlocal tooltip
        if tooltip[0] is None:
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            tooltip[0] = tk.Toplevel(widget)
            tooltip[0].wm_overrideredirect(True)
            tooltip[0].wm_geometry(f"+{x}+{y}")

            label = tk.Label(tooltip[0], text=text, background="lightyellow", relief="solid", borderwidth=1)
            label.pack()

    def hide_tooltip():
        """Hides and destroys the tooltip when the widget is no longer hovered over."""
        nonlocal tooltip
        if tooltip[0] is not None and tooltip[0].winfo_exists():
            tooltip[0].destroy()
            tooltip[0] = None

    def on_enter(event):
        """Event handler for when the mouse enters the widget."""
        show_tooltip(event)

    def on_leave(event):
        """Event handler for when the mouse leaves the widget."""
        hide_tooltip()

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)
        
def setup_gui():
    """
    Set up the GUI for the NWN Parameter Graph Plotter.

    This function initializes the main GUI window, creates frames, widgets, and configures their layout.

    Returns:
    None
    """
    global root, seed_var
    global options_frame, graph_mode_var, graph_type_var

    # Initialize the main GUI window
    root = tk.Tk()
    root.title("NWN Parameter Graph Plotter")
    root.resizable(False, False)

    # Create parameters frame for sliders
    parameters_frame = ttk.Frame(root, padding=(20, 10))
    parameters_frame.grid(row=0, column=0, sticky="nsew")

    for i in range(10):
        parameters_frame.rowconfigure(i, weight=1)

    # Initialize seed variable and parameter sliders
    seed_label = ttk.Label(parameters_frame, text="Seed:")
    seed_label.grid(row=0, column=0)
    seed_var = tk.StringVar(value="123")
    seed_entry = ttk.Entry(parameters_frame, textvariable=seed_var)
    seed_entry.grid(row=0, column=1)
    bind_tooltip(seed_entry, " Seed for random nanowire generation ")  
    
    parameter_sliders(parameters_frame)

    # Create options frame for advanced settings
    options_frame = ttk.Frame(root, padding=(20, 10))
    options_frame.grid(row=0, column=1, sticky="nsew")
    
    for i in range(5):
        options_frame.rowconfigure(i, weight=1)

    # Create button to toggle advanced options
    options_button = ttk.Button(parameters_frame, text="Advanced Options", command=lambda: toggle_widgets(options_frame, options_button, "Advanced Options", "Hide Advanced Options"))
    options_button.grid(row=7, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    # Create dropdown for graph mode selection
    graph_mode_label = ttk.Label(parameters_frame, text="Graph Mode:")
    graph_mode_label.grid(row=8, column=0, pady=(10, 0))
    graph_mode_var = tk.StringVar(value='chen')
    graph_mode_dropdown = ttk.OptionMenu(parameters_frame, graph_mode_var, 'chen', 'default', 'decay', 'chen')
    graph_mode_dropdown.grid(row=8, column=1, pady=(10, 0), sticky="ew")
    
    # Create dropdown for graph type selection
    graph_type_label = ttk.Label(parameters_frame, text="Graph Type:")
    graph_type_label.grid(row=9, column=0, pady=(10, 0))
    graph_type_var = tk.StringVar(value='Nanowire Network')
    graph_type_dropdown = ttk.OptionMenu(parameters_frame, graph_type_var, 'Nanowire Network', 'Nanowire Network', 'square wave', 'parabolic wave', 'triangular wave','Voltage vs. Current','w,tau,epsilon')
    graph_type_dropdown.grid(row=9, column=1, pady=(10, 0), sticky="ew")

    # Create button to plot the graph
    plot_button = ttk.Button(root, text="Plot Graph", command=lambda: create_and_plot_graph())
    plot_button.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    # Configure column weights
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)

    # Start the main GUI event loop
    root.mainloop()

if __name__ == "__main__":
    setup_gui()