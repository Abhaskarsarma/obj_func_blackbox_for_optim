from mealpy import FloatVar, BBO, PSO, DE, ABC, SRSR, GWO, WOA, CSA, SHO, GA, BA, SSpiderA, SRSR, DO, GOA, QSA, BSA, OOA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plot

import viswanath_sir_obj_func as obj

## Define your own problems
f1 = obj.obj_funct


# Define bounds for the variables
# Adjust these based on the expected feasible range of your parameters
lower_bounds = [100.0, 1.0]
upper_bounds = [300, 21] # These are from your p1 definition

p1 = {
    "bounds": FloatVar(lb=lower_bounds, ub=upper_bounds),
    "obj_func": f1,
    "minmax": "max",
    "name": "F5",
    "log_to": "console",
    "save_population": True # Keep this True to track solutions
}

# **List of optimizers**
optimizers = {
    "PSO": PSO.AIW_PSO(epoch=500, pop_size=200),
    "BBO": BBO.OriginalBBO(epoch=500, pop_size=200),
    #"DE": DE.OriginalDE(epoch=500, pop_size=200),
    #"ABC": ABC.OriginalABC(epoch=500, pop_size=200),
    #"WOA": WOA.OriginalWOA(epoch=500, pop_size=200),
    #"CSA": CSA.OriginalCSA(epoch=500, pop_size=200),
    "GA": GA.BaseGA(epoch=500, pop_size=200),
    #"BA": BA.OriginalBA(epoch=500, pop_size=200),
    #"SSpiderA": SSpiderA.OriginalSSpiderA(epoch=500, pop_size=200),
    #"DO": DO.OriginalDO(epoch=500, pop_size=200),
    #"GWO": GWO.OriginalGWO(epoch=500, pop_size=200),
    #"QSA": QSA.OriginalQSA(epoch=500, pop_size=200),
    #"BSA": BSA.OriginalBSA(epoch=500, pop_size=200),
    #"SHO": SHO.OriginalSHO(epoch=500, pop_size=200),
    #"SRSR": SRSR.OriginalSRSR(epoch=500, pop_size=200),
   
}

term_dict = {
    "max_early_stop": 15    # 15 epochs if the best fitness is not getting better we stop the program
}
# --- Plotting Setup for Objective Function Background (kept for reference, not used in final plots) ---
x0_range = np.linspace(lower_bounds[0], upper_bounds[0], 100)
x1_range = np.linspace(lower_bounds[1], upper_bounds[1], 100)
X0, X1 = np.meshgrid(x0_range, x1_range)

Z_obj = np.zeros(X0.shape)
for i in range(X0.shape[0]):
    for j in range(X0.shape[1]):
        current_x = [X0[i, j], X1[i, j]]
        val = obj.obj_funct(current_x)
        
        if val > 0:
            Z_obj[i, j] = val
        else:
            Z_obj[i, j] = np.nan   # hide loss region

# Replace inf/NaN values with a large finite number for better plotting
max_finite_z = np.max(Z_obj[np.isfinite(Z_obj)]) if np.any(np.isfinite(Z_obj)) else 1e10
Z_obj[np.isinf(Z_obj)] = max_finite_z
Z_obj[np.isnan(Z_obj)] = max_finite_z

# **Solve the problem for each optimizer and collect path data**
history_data = {}
optimization_paths = {} # To store the [x0, x1] path of the global best solution

for name, model in optimizers.items():
    print(f"\n--- Running optimization with {name} ---")
    model.solve(p1, seed=10, termination=term_dict) # Run the optimization
    history_data[name] = model.history

    # Extract the optimization path for the best solution (global_best)
    path_x0 = []
    path_x1 = []
    path_fitness = []
    for epoch_best_agent in model.history.list_global_best:
        path_x0.append(epoch_best_agent.solution[0])
        path_x1.append(epoch_best_agent.solution[1])
        path_fitness.append(epoch_best_agent.target.fitness)

    optimization_paths[name] = {
        'x0': np.array(path_x0),
        'x1': np.array(path_x1),
        'fitness': np.array(path_fitness), # Store fitness along the path for 3D plot
        'final_x': model.g_best.solution,
        'final_fitness': model.g_best.target.fitness
    }
    print(f"  Final x: {model.g_best.solution}, Final Objective: {model.g_best.target.fitness:.4f}")

# --- Individual 3D Plots with Optimization Paths ---
for name, path_data in optimization_paths.items():
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Plot the surface of the objective function
    surf = ax_3d.plot_surface(X0, X1, Z_obj, cmap='jet', alpha=0.5, rstride=10, cstride=10)
    
    # Get the Z values for the path
    path_z = np.array([obj.obj_funct(np.array([x0_val, x1_val])) for x0_val, x1_val in zip(path_data['x0'], path_data['x1'])])
    
    # Plot the final point
    ax_3d.plot([path_data['final_x'][0]], [path_data['final_x'][1]], [path_data['final_fitness']])

    ax_3d.set_title(f'3D Optimization Path for {name}')
    ax_3d.set_xlabel('x[0] (Parameter 1)')
    ax_3d.set_ylabel('x[1] (Parameter 2)')
    ax_3d.set_zlabel('Objective Function Value')
    
    ax_3d.invert_xaxis()
    ax_3d.invert_yaxis()
   
    ax_3d.view_init(elev=30, azim=45) # Adjust viewing angle for better perspective
    ax_3d.grid(False)
    ax_3d.legend()
    plt.show()


# --- MODIFIED: Individual 3D Plots showing ONLY Optimization Paths ---
print("\n--- Generating 3D Optimization Path Plots (without full landscape) ---")
for name, path_data in optimization_paths.items():
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    #Plot the optimization path without label
    ax_3d.plot(path_data['x0'], path_data['x1'], path_data['fitness'], '*:', markersize=6, linewidth=2, color='magenta')
    
    # Plot the final point
    ax_3d.plot([path_data['final_x'][0]], [path_data['final_x'][1]], [path_data['final_fitness']], 'X', markersize=10, color='red', markeredgecolor='black',)

    ax_3d.set_title(f'3D Path of Best Solution for {name}')
    ax_3d.set_xlabel('x[0] (Parameter 1)')
    ax_3d.set_ylabel('x[1] (Parameter 2)')
    ax_3d.set_zlabel('Objective Function Value')
    ax_3d.view_init(elev=30, azim=45) # Adjust viewing angle for better perspective
    ax_3d.grid(False)
    ax_3d.legend()
    plt.show()

# --- NEW: 2D Plots of individual parameters (x[0] and x[1]) over Epochs ---
print("\n--- Generating 2D Plots for Evolution of x[0] and x[1] over Epochs ---")
epochs = np.arange(1, optimizers["PSO"].epoch + 1) # Assuming all optimizers run for the same number of epochs

for name, path_data in optimization_paths.items():
    # Plot x[0] vs Epoch
    fig_x0 = plt.figure(figsize=(10, 6))
    plt.plot(epochs[:len(path_data['x0'])], path_data['x0'], 'D-.', label=f'{name}', markersize=6, linewidth=2, color='maroon')
    plt.xlabel('Epoch')
    plt.ylabel('x[0] Value')
    plt.title(f'Evolution of Best x[0] for {name}')
    plt.grid(False)
    plt.legend()
    plt.show()

    # Plot x[1] vs Epoch
    fig_x1 = plt.figure(figsize=(10, 6))
    plt.plot(epochs[:len(path_data['x1'])], path_data['x1'], 'D-.', label=f'{name}', markersize=6, linewidth=2, color='maroon')
    plt.xlabel('Epoch')
    plt.ylabel('x[1] Value')
    plt.title(f'Evolution of Best x[1] for {name}')
    plt.grid(False)
    plt.legend()
    plt.show()


# --- Combined Convergence Plot (from previous response, still useful) ---
print("\n--- Generating Combined Convergence Plot ---")
fig_convergence = plt.figure(figsize=(10, 6))
for name, history in history_data.items():
    fitness_values = history.list_global_best_fit
    plt.plot(fitness_values, label=f'{name} Cost')

plt.title('Global Best Objective Function Value per Epoch (All Optimizers)')
plt.xlabel('Epoch')
plt.ylabel('Objective Function Value')
plt.legend()
plt.grid(False)
plt.yscale('log') # Use log scale if fitness values vary greatly
plt.show()

# --- Your original mealpy plots (modified for clarity) ---
print("\n--- Generating Mealpy's Default Performance Plots ---")
plot_attributes_fitness = [
    ("list_current_best_fit", "Current Best Fitness"),
    ("list_global_best_fit", "Global Best Fitness"),
]

for attr_name, title in plot_attributes_fitness:
    plt.figure(figsize=(10, 5))
    for name, history in history_data.items():
        values = getattr(history, attr_name, None)
        if values is not None and len(values) > 0:
            plt.plot(values, label=name)

    plt.xlabel("Iteration (Epoch)")
    plt.ylabel(title)
    plt.title(f"Convergence of {title}")
    plt.legend()
    plt.grid(False)
    if "Fitness" in title: # Apply log scale for fitness plots
        plt.yscale('log')
    plt.show()

# **Plot Diversity**
plt.figure(figsize=(10, 5))
for name, history in history_data.items():
    diversity_values = getattr(history, "list_diversity", None)
    if diversity_values is not None and len(diversity_values) > 0:
        plt.plot(diversity_values, label=name)

plt.xlabel("Iteration (Epoch)")
plt.ylabel("Diversity")
plt.title("Population Diversity Over Epochs")
plt.legend()
plt.grid(False)
plt.show()


# **Plot Exploration vs Exploitation**
plt.figure(figsize=(10, 5))
for name, history in history_data.items():
    exploration_values = getattr(history, "list_exploration", None)
    exploitation_values = getattr(history, "list_exploitation", None)

    if exploration_values is not None and len(exploration_values) > 0:
        plt.plot(exploration_values, 'D', label=f"{name} - Exploration", linestyle="--")

    if exploitation_values is not None and len(exploitation_values) > 0:
        plt.plot(exploitation_values, '8', label=f"{name} - Exploitation", linestyle=":")

plt.xlabel("Iteration (Epoch)")
plt.ylabel("Exploration vs Exploitation Metric")
plt.title("Exploration vs Exploitation Comparison")
plt.legend()
plt.grid(False)
plt.show()

print("\n--- Final Optimization Results Summary ---")
for name, path_data in optimization_paths.items():
    print(f"Optimizer: {name}")
    print(f"  Optimal x: {path_data['final_x']}")
    print(f"  Max Objective Value: {path_data['final_fitness']:.4f}")
    print(f"  Iterations (Epochs): {len(path_data['x0'])}\n")
    