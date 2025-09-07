# Copyright (c) 2025 [HAN, Wei]
# This code is licensed under the MIT License.
# See the LICENSE file for more details.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("Brownian Motion and Free Energy Simulation")

# --- Sidebar for User Inputs ---
st.sidebar.header("Simulation Controls")
# Use number inputs to set the initial particle distribution
n_left = st.sidebar.number_input("Initial particles on left (x<0)", 0, 500, 50)
n_right = st.sidebar.number_input("Initial particles on right (x>0)", 0, 500, 0)
n_balls = n_left + n_right

delta = st.sidebar.slider("Max Step Size (delta)", 0.1, 1.0, 0.2, 0.05)
beta = 1 / 0.58

# --- Core Simulation Functions (Unchanged) ---
def potential_energy(q):
    """Calculates the potential energy for a given position q."""
    return 0.5 * ((q - 1.5)**2 * (q + 1.5)**2 - 1.5 * q)

def free_energy(h):
    """Calculates the free energy f(h) as a function of the fraction h."""
    epsilon = 1e-9
    h_safe = np.clip(h, epsilon, 1 - epsilon)
    term1 = h_safe * np.log(h_safe / (1 - h_safe))
    term2 = np.log(1 - h_safe)
    return -1.5 * h_safe + 0.58 * (term1 + term2)

# --- State Initialization ---
# st.session_state is used to store variables between reruns
if 'initialized' not in st.session_state or st.session_state.n_balls != n_balls:
    st.session_state.initialized = True
    st.session_state.n_balls = n_balls
    # Initialize particle positions based on user input
    q_positions = np.full(n_balls, 1.5)
    if n_left > 0:
        q_positions[:n_left] = -1.5
    st.session_state.q_positions = q_positions
    st.session_state.u_energies = potential_energy(st.session_state.q_positions)

# --- App Layout and Placeholders ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Particle Simulation")
    plot_placeholder = st.empty()
with col2:
    st.subheader("Free Energy f(h)")
    free_energy_placeholder = st.empty()

# Add a button to start/stop the simulation loop
if 'running' not in st.session_state:
    st.session_state.running = False

if st.sidebar.button("Start/Stop Simulation"):
    st.session_state.running = not st.session_state.running
    if not st.session_state.running:
        st.sidebar.warning("Simulation stopped.")
    else:
        st.sidebar.success("Simulation running...")

# --- Main Animation Loop ---
# This loop replaces FuncAnimation and runs as long as 'running' is True [3].
while st.session_state.running:
    q_positions = st.session_state.q_positions
    u_energies = st.session_state.u_energies

    # Update particle positions via Monte Carlo step
    for i in range(n_balls):
        q_new = q_positions[i] + random.uniform(-delta, delta)
        u_new = potential_energy(q_new)
        if math.exp(-beta * (u_new - u_energies[i])) > random.uniform(0, 1):
            q_positions[i] = q_new
            u_energies[i] = u_new

    # Store the updated state
    st.session_state.q_positions = q_positions
    st.session_state.u_energies = u_energies

    # --- Redraw Plots in Each Iteration ---
    # Create the first plot (Particle Simulation)
    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-2, 5)
    ax1.set_title("Particle Simulation")
    ax1.set_xlabel("Reaction Coordinate (q)")
    ax1.set_ylabel("Energy (kcal/mol)")
    ax1.grid(True)
    q_path = np.linspace(-3, 3, 200)
    ax1.plot(q_path, potential_energy(q_path), 'g--', lw=1, alpha=0.5)

    color_indices = np.arange(n_balls)
    ax1.scatter(q_positions, u_energies, s=80, c=color_indices, cmap='viridis')
    
    right_count = np.sum(q_positions > 0)
    left_count = n_balls - right_count
    right_percentage = (right_count / n_balls) * 100 if n_balls > 0 else 0
    
    counter_text = (f'Left (x<0): {left_count}\n'
                    f'Right (x>0): {right_count} ({right_percentage:.1f}%)')
    ax1.text(0.05, 0.95, counter_text, transform=ax1.transAxes, ha='left', va='top', 
             fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    # Display the plot in the placeholder [4]
    plot_placeholder.pyplot(fig1)
    plt.close(fig1) # Close the figure to free up memory

    # Create the second plot (Free Energy)
    fig2, ax2 = plt.subplots()
    ax2.set_title("Free Energy as a Function of Fraction (h)")
    ax2.set_xlabel("Fraction of particles with x>0 (h)")
    ax2.set_ylabel("Free Energy f(h)")
    ax2.grid(True)
    
    h_curve = np.linspace(0, 1, 200)
    ax2.plot(h_curve, free_energy(h_curve), 'b-')

    current_h = right_count / n_balls if n_balls > 0 else 0
    current_f_h = free_energy(current_h)
    ax2.plot(current_h, current_f_h, '+', c='red', markersize=15, markeredgewidth=3)
    
    free_energy_placeholder.pyplot(fig2)
    plt.close(fig2)

    # Control animation speed and allow for Stop button to be responsive
    time.sleep(0.05)
