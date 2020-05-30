import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt

df_objectives = pd.DataFrame(data=[], columns=["country","deaths","activity"])

countries = ["Italy","Japan","Luxembourg"]
country = "Italy"




for country in countries:
    objectives = json.load(open("./experiments/ga1/ga_{}_lastobj.json".format(country)))
    deaths = np.array(objectives.get("deaths"),np.int32)
    activity = np.array(objectives.get("activity"),np.float)
    df_objectives = df_objectives.append(pd.DataFrame({"country":[country]*len(deaths),"deaths":deaths,"activity":activity}))

    plt.figure()
    df_country = df_objectives[df_objectives["country"]==country]
    plt.scatter(df_country["activity"], df_country["deaths"])
    plt.title("Best scenarios (lockdown vs deaths) at last iteration for {}".format(country))
    plt.xlabel = "Total deaths"
    plt.ylabel = "Average activity"

    population = json.load(open("./experiments/ga1/ga_{}_lastpop.json".format(country)))
    populations = pd.DataFrame({"scenario":[],"R":[],"SimulationCritical":[],"SimulationDeaths":[]})
    for i, p in enumerate(population):
        R = pd.Series(p.get("R"))
        SimulationCritical = pd.Series(p.get("SimulationCritical_max"))
        SimulationDeaths = pd.Series(p.get("SimulationDeaths_max"))
        pop = pd.DataFrame({"scenario":[i]*len(R),"R":R,"SimulationCritical":SimulationCritical,"SimulationDeaths":SimulationDeaths}, index=R.index)
        pop.index = pop.index.astype(int)
        pop.sort_index(inplace=True)
        populations = populations.append(pop)

    selection = populations["scenario"]<5 
    plt.figure()
    populations[selection].groupby("scenario")["R"].plot(legend=True, title="R for scenarios at last iteration for {}".format(country))
    
    plt.figure()
    populations[selection].groupby("scenario")["SimulationDeaths"].plot(legend=True, title="Deaths for scenarios at last iteration for {}".format(country))
    #plt.figure()
    #populations[populations["scenario"]==0]["SimulationDeaths"].plot()
    #print(populations[populations["scenario"]==0]["SimulationDeaths"].head())

plt.show()


