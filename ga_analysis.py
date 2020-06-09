import sys
sys.path.append("./")

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from model_run import simulate
from sklearn.metrics import auc

import matplotlib.pyplot as plt

countries = ["Italy","Japan","Luxembourg"]
criticals = [2054, 1822, 42]
country = "Italy"


features = ["workplace",
            "transit_stations",
            "retail/recreation", "parks"]

begin_date="2020-04-30"
end_date="2020-09-30"

version="v2_1"
folder = "./data"

step = 14

dates = np.arange(np.datetime64(begin_date), np.datetime64(end_date), timedelta(days=step)).astype(datetime)
measures_dates = np.array([(f, d) for f in features for d in dates])
end_date = pd.to_datetime(end_date)

merged = pd.read_csv("{}/features.csv".format(folder), parse_dates=["Date"])

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


for k, country in enumerate(countries):
    country_df = merged[merged["CountryName"] == country]
    
    objectives = json.load(open("./experiments/ga2/ga_{}_lastobj.json".format(country)))
    deaths = np.array(objectives.get("deaths"),np.int32)
    activity = np.array(objectives.get("activity"),np.float)

    min_death = deaths > country_df["ConfirmedDeaths"].max()

    
    costs =np.column_stack((deaths, activity)) #np.concatenate([[deaths], [activity]], axis=0)
    is_pareto = is_pareto_efficient_simple(costs)

    pareto_inputs = costs[is_pareto & min_death]
    plt.figure()
    plt.scatter(pareto_inputs[:,1], pareto_inputs[:,0])
    plt.title("Pareto scenarios (lockdown vs deaths) at last iteration for {}".format(country))
    plt.xlabel = "Total deaths"
    plt.ylabel = "Average activity"

    population = json.load(open("./experiments/ga2/ga_{}_lastpop.json".format(country)))
    population_df = population.get("df")
    population_x = population.get("x")
    populations = pd.DataFrame({"scenario":[],"R":[],"SimulationCritical":[],"SimulationDeaths":[]})
    
    print(country)
        
    for i, p in enumerate(population_df):
        if not min_death[i]:
            continue

        if not is_pareto[i]:
            continue

        x = np.array(population_x[i])
        x_mean = x.reshape(4,-1).mean(axis=1)
        
        measures_to_lift = measures_dates[:,0]
        dates = measures_dates[:,1]
        lift_date_values = [pd.to_datetime(d) for d in dates]

        res = simulate(country_df, [measures_to_lift], 0, end_date, None, [], [], None, None,
                           measure_values=[x], base_folder=None, lift_date_values=lift_date_values,
                           seed="", return_measures=True)

        R = pd.Series(p.get("R_max"))
        SimulationCritical = pd.Series(p.get("SimulationCritical_max"))
        SimulationDeaths = pd.Series(p.get("SimulationDeaths_max"))
        pop = pd.DataFrame({"scenario":[i]*len(R),"R":R,"SimulationCritical":SimulationCritical,"SimulationDeaths":SimulationDeaths}, index=R.index)
        
        size = len(R)
        features_period = res[features].tail(size)
        metrics = []
        for f in features:
            val_auc = auc(np.arange(size), features_period[f])
            metrics.append(val_auc)

        metrics = [str(np.mean(metrics))]
        metrics.append(str(deaths[i]))
        metrics.append(str(criticals[k]))    
        print(" & ".join(metrics))
            
        pop.index = pop.index.astype(int)
        pop.sort_index(inplace=True)
        populations = populations.append(pop)

    selection = populations["scenario"]<100 
    plt.figure()
    populations[selection].groupby("scenario")["R"].plot(legend=True, title="R for scenarios at last iteration for {}".format(country))
    
    plt.figure()
    populations[selection].groupby("scenario")["SimulationDeaths"].plot(legend=True, title="Deaths for scenarios at last iteration for {}".format(country))


plt.show()


