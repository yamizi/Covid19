
import sys, getopt
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




def run(countryname, capacity):
    problem = ScheduleProblem(country_name=countryname, critical_capacity=capacity, record_all=True)


    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=0.9, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 100)
    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                pf=problem.pareto_front(use_cache=False),
                save_history=True,
                verbose=True)


    #np.save("./experiments/ga_{}.npy".format(countryname),np.array(res))

    # get the pareto-set and pareto-front for plotting
    #ps = problem.pareto_set(use_cache=False, flatten=False)
    #pf = problem.pareto_front(use_cache=False, flatten=False)

    # create the performance indicator object with reference point (4,4)
    metric = Hypervolume(ref_point=np.array([1.0, 1.0]))

    # collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]

    with open("./experiments/ga_{}_lastpop.json".format(countryname), 'w') as f:
        json.dump( {"df":[e.to_dict() for e in problem.last[0]],"x":problem.last[1].tolist()}, f)
        #{"df":[e.to_dict() for e in self.last[0]],"x":self.last[1]}, f)

    with open("./experiments/ga_{}_lastobj.json".format(countryname), 'w') as f:
        json.dump( {"deaths": problem.last_objectives[0].tolist(), "activity":problem.last_objectives[1].tolist()} , f)



    # Objective Space
    fig = plt.figure()
    plot = Scatter(title = "Objective Space")
    plot.add(res.F)
    plt.savefig("./experiments/ga_{}_objective.png".format(countryname))


    #with open("./experiments/ga_{}_lastpop.json".format(countryname), "w") as fp:


    # receive the population in each generation
    obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]

    # calculate for each generation the HV metric
    hv = [metric.calc(f) for f in obj_and_feasible_each_gen]

    # function evaluations at each snapshot
    n_evals = np.array([a.evaluator.n_eval for a in res.history])

    # visualize the convergence curve
    fig = plt.figure()
    plt.plot(n_evals, hv, '-o')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig("./experiments/ga_{}_hypervolume.png".format(countryname))
    plt.show()


def main(argv):
   country = 'luxembourg'
   try:
      opts, args = getopt.getopt(argv,"c:",["country="])
   except getopt.GetoptError:
      print ('ga_experiment.py -c <country> -h <critical hospital capacity>')
      sys.exit(2)
   for opt, arg in opts:
      if opt in ("-c", "--country"):
         country = arg
   

    #covid19.healthdata/org
   if country=="luxembourg":
       run("Luxembourg", 42)
   elif country=="france":
       run("France", 1980)
   elif country=="japan":
       run("Japan", 2054)
   elif country=="italy":
       run("Italy", 1822)
   
   #countryname, capacity = "Luxembourg", 42

    #countryname, capacity = "France", 1980
   #countryname, capacity = "Japan", 2054


if __name__ == "__main__":
   main(sys.argv[1:])