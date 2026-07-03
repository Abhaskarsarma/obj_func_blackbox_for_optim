# obj_func_blackbox_for_optim

A Python toolbox for visualizing and comparing metaheuristic optimization algorithms using the Mealpy framework.

The toolbox allows researchers to optimize any two-variable objective function while simultaneously visualizing

- Optimization trajectory
- 3D search path
- Convergence curves
- Diversity
- Exploration vs Exploitation
- Best solution evolution

---

## Features

✔ Multiple optimization algorithms

- Particle Swarm Optimization (PSO)
- Biogeography Based Optimization (BBO)
- Genetic Algorithm (GA)

Additional Mealpy algorithms can be enabled with one line.

---

## Visualization

The toolbox automatically generates

- 3D optimization landscape
- Optimization path
- Parameter evolution
- Convergence plot
- Diversity plot
- Exploration vs Exploitation

---

## Installation

```bash
git clone https://github.com/<username>/Optimization-Toolbox.git

cd Optimization-Toolbox

pip install -r requirements.txt
```

---

## Requirements

Python 3.10+

Main libraries

- mealpy
- numpy
- matplotlib

---

## Usage

Define your objective function

```python
def obj_funct(solution):
    ...
    return fitness
```

Import it

```python
import objective_functions as obj
```

Run

```bash
python Optimization_Toolbox.py
```

---

## Applications

- Hyperparameter optimization
- Engineering optimization
- Machine learning
- Medical image optimization
- Controller tuning
- Research benchmarking

---

## Future Improvements

- GUI
- Plotly interactive visualization
- Animation
- CSV export
- Benchmark functions
- Parallel optimization
- Multi-objective optimization
- Constraint handling

---

## Citation

If you use this software in your research, please cite

```bibtex
@software{Sarma2026OptimizationToolbox,
  author = {Bhaskar Sarma},
  title = {Optimization Toolbox},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Abhaskarsarma/obj_func_blackbox_for_optim}
}

---
