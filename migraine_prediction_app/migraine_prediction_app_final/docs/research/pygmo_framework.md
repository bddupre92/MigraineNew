# PyGMO Optimization Framework Research

## Overview
PyGMO (Python Parallel Global Multiobjective Optimizer) is a scientific library for massively parallel optimization. It provides a unified interface to optimization algorithms and problems, with a focus on evolutionary and meta-heuristic approaches.

## Key Components

### 1. Problem Definition
- Problems are defined by creating a class with:
  - `fitness(x)`: Returns objective function value(s)
  - `get_bounds()`: Returns bounds for decision variables
  - `get_nobj()`: (Optional) Returns number of objectives for multi-objective problems
  - `get_name()`: (Optional) Returns problem name
  - `get_extra_info()`: (Optional) Returns additional information

### 2. Algorithms
PyGMO implements several evolutionary algorithms that are relevant for our migraine prediction app:

#### Differential Evolution (DE)
- Parameters:
  - `gen`: Number of generations
  - `F`: Weight coefficient (default 0.8)
  - `CR`: Crossover probability (default 0.9)
  - `variant`: Mutation variant (default 2: /rand/1/exp)
  - `ftol`, `xtol`: Stopping criteria tolerances
- Best for: Expert hyperparameter optimization (as specified in PRD)

#### Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Parameters:
  - `gen`: Number of generations
  - `cc`: Learning rate for covariance matrix
  - `sigma0`: Initial step size
  - `ftol`, `xtol`: Stopping criteria tolerances
- Best for: Expert hyperparameter optimization (as specified in PRD)

#### Particle Swarm Optimization (PSO)
- Parameters:
  - `gen`: Number of generations
  - `omega`: Inertia weight
  - `eta1`, `eta2`: Social and cognitive learning rates
  - `max_vel`: Maximum particle velocity
  - `variant`: Algorithm variant (1-6)
  - `neighb_type`: Swarm topology
- Best for: Gating hyperparameter optimization (as specified in PRD)

#### Ant Colony Optimization (GACO)
- Parameters:
  - `gen`: Number of generations
  - `ker`: Kernel size
  - `q`: Convergence speed parameter
  - `oracle`: Oracle parameter
  - `acc`: Accuracy parameter
- Best for: Gating hyperparameter optimization (as specified in PRD)

### 3. Population
- Represents a collection of candidate solutions
- Created with `pygmo.population(problem, size=N)`
- Provides methods to access solutions and their fitness

### 4. Archipelago
- Enables parallel optimization across multiple "islands"
- Each island evolves a population independently
- Periodically exchanges solutions between islands
- Ideal for distributed hyperparameter optimization

## Integration with FuseMoE

For our migraine prediction app, we'll need to:

1. **Define Custom Problem Classes**:
   - `ExpertHPO`: For optimizing expert hyperparameters
   - `GatingHPO`: For optimizing gating network hyperparameters
   - `EndToEndHPO`: For joint optimization

2. **Implement Fitness Functions**:
   - Convert hyperparameters to FuseMoE configurations
   - Train models with these configurations
   - Return validation metrics (negative accuracy, inference time)

3. **Multi-Phase Optimization**:
   - Phase 1: Use DE/CMA-ES for expert hyperparameters
   - Phase 2: Use PSO/ACO for gating hyperparameters
   - Phase 3: Use mixed algorithms for end-to-end optimization

## Example Workflow

```python
# 1. Define a problem for hyperparameter optimization
class FuseMoEHPO:
    def __init__(self, search_space):
        self.search_space = search_space
        
    def fitness(self, x):
        # Convert x to hyperparameters
        config = self._map_to_config(x)
        
        # Train FuseMoE with these hyperparameters
        results = self._train_and_evaluate(config)
        
        # Return negative metrics (for minimization)
        return [-results["val_accuracy"], results["inference_time"]]
        
    def get_bounds(self):
        # Return lower and upper bounds for each hyperparameter
        return self.search_space["lower_bounds"], self.search_space["upper_bounds"]
        
    def get_nobj(self):
        # Multi-objective optimization (accuracy and inference time)
        return 2
        
    def _map_to_config(self, x):
        # Convert normalized parameters to actual hyperparameter values
        # ...
        
    def _train_and_evaluate(self, config):
        # Train FuseMoE with the given config
        # Return validation metrics
        # ...

# 2. Create PyGMO problem
prob = pygmo.problem(FuseMoEHPO(search_space))

# 3. Select algorithm based on optimization phase
if phase == "expert":
    algo = pygmo.algorithm(pygmo.de(gen=50, F=0.8, CR=0.9))
elif phase == "gating":
    algo = pygmo.algorithm(pygmo.pso(gen=50, omega=0.7, eta1=2.0, eta2=2.0))
else:  # end-to-end
    algo = pygmo.algorithm(pygmo.cmaes(gen=50, sigma0=0.5))

# 4. Create archipelago for parallel optimization
archi = pygmo.archipelago(n=8, algo=algo, prob=prob, pop_size=20)

# 5. Evolve and wait for results
archi.evolve()
archi.wait()

# 6. Extract best solution
best_island = min(archi, key=lambda isl: isl.get_population().champion_f[0])
best_hyperparams = best_island.get_population().champion_x
best_metrics = best_island.get_population().champion_f
```

## Considerations for Migraine Prediction

1. **Hyperparameter Ranges**: As specified in PRD section 3.4
2. **Fitness Evaluation**: Will need to balance accuracy vs. inference time
3. **Computational Resources**: Archipelago allows distributed optimization
4. **Convergence Criteria**: Can use improvement plateau < 0.005 over 10 generations
