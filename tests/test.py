import randomnwn as rnwn
import matplotlib as plt
import numpy as np
from scipy import signal
from randomnwn import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

units = get_units()
#from examples.monitor import Runtime

NWN = rnwn.create_NWN(seed=4290, density=5.5)
#NWN = rnwn.create_NWN(seed=5)

left_l, left_2, top_1, top_2 = add_electrodes(
    NWN, ["left", 2, 1, [-0.5, 0.5]], ["top", 2, 1, [-0.5, 0.5]])


max_time = 2000
def voltage_func(t):
    V0 = 20
    T = 666
    f = 1 / T
    return V0 * (signal.square(2*np.pi*f*t) + 1) / 2

plot_NWN(NWN)
def window_func(w):
    return w * (1 - w)

sigma, theta, a = 0.03, 0.01, 0.75
set_chen_params(NWN, sigma, theta, a) 

w0, tau0, epi0 = 0.055, 20, 0.055 
set_state_variables(NWN, w0, tau0, epi0)

t_eval = np.linspace(0, max_time, 1000)

sol, edge_list = solve_evolution(
        NWN, t_eval, [left_l, left_2], [top_1, top_2],voltage_func, window_func, 1e-7, "chen")

V = [voltage_func(t) for t in sol.t]
I = get_evolution_current(NWN, sol, edge_list, left_l, top_2, voltage_func, scaled=True)
R = V / I

font = 14
fig, ax1 = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax1.plot(sol.t, V, "red")
ax1.set_ylabel("Voltage (arb. units)", color="red", fontsize=font)
ax1.set_xlabel("Time (arb. units)", fontsize=font)
ax1.tick_params(labelsize=font, which="both")

ax2 = ax1.twinx()
ax2.plot(sol.t, I, "blue")
ax2.set_ylabel("Current (arb. units)", color="blue", fontsize=font)
ax2.tick_params(labelsize=font, which="both")

ax1.grid(alpha=0.5)

print("Bottom left -> Bottom right: {:e}\nBottom left -> Top right: {:e}".format(
    *solve_drain_current(NWN, left_l, [left_2, top_2], 10.0, scaled=True)
))

print("Top left -> Bottom right: {:e}\nTop left -> Top right: {:e}".format(
    *solve_drain_current(NWN, top_1, [top_1, left_2], 10.0, scaled=True)
))

# fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
# for ax in axes:
#     ax.set_xlabel("Time (sec)")
#     ax.grid(alpha=0.5)

# axes[0].set_ylabel("w")
# axes[1].set_ylabel("tau")
# axes[2].set_ylabel("epsilon")

# w_list, tau_list, eps_list = np.split(sol.y, 3)
# for w in w_list:
#     axes[0].plot(sol.t * NWN.graph["units"]["t0"] * 1e-6, w)
# for tau in tau_list:
#     axes[1].plot(sol.t * NWN.graph["units"]["t0"] * 1e-6, tau)
# for eps in eps_list:
#     axes[2].plot(sol.t * NWN.graph["units"]["t0"] * 1e-6, eps)
