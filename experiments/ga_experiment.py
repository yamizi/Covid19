import sys
sys.path.append("./")

import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
import matplotlib.pyplot as plt
from pymoo.performance_indicator.hv import Hypervolume
import json
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from ga.problem import ScheduleProblem
from pymoo.visualization.scatter import Scatter

#covid19.healthdata/org
countryname = "Luxembourg"
capacity = 42

#countryname, capacity = "France", 1980

problem = ScheduleProblem(country_name=countryname, critical_capacity=capacity)


algorithm = NSGA2(
    pop_size=100,
    n_offsprings=50,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 200)
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)


np.save("./experiments/ga_{}.npy".format(countryname),np.array(res))

# get the pareto-set and pareto-front for plotting
#ps = problem.pareto_set(use_cache=False, flatten=False)
#pf = problem.pareto_front(use_cache=False, flatten=False)


# Objective Space
plot = Scatter(title = "Objective Space")
plot.add(res.F)
#plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.show()

# create the performance indicator object with reference point (4,4)
metric = Hypervolume(ref_point=np.array([1.0, 1.0]))

# collect the population in each generation
pop_each_gen = [a.pop for a in res.history]

with open("./experiments/ga_{}_lastpop.json".format(countryname), 'w') as f:
    json.dump( [e.to_dict() for e in problem.last], f)

#with open("./experiments/ga_{}_lastpop.json".format(countryname), "w") as fp:


# receive the population in each generation
obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]

# calculate for each generation the HV metric
hv = [metric.calc(f) for f in obj_and_feasible_each_gen]

# function evaluations at each snapshot
n_evals = np.array([a.evaluator.n_eval for a in res.history])

# visualize the convergence curve
plt.plot(n_evals, hv, '-o')
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()
