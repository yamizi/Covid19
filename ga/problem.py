import numpy as np
from pymoo.model.problem import Problem
import pandas as pd
import json, time, os
from datetime import datetime, timedelta
from sklearn.externals import joblib
from model_run import simulate

class ScheduleProblem(Problem):

    def __init__(self, begin_date="2020-04-30",end_date="2020-09-30",country_name = "Luxembourg", critical_capacity=90, step=14, record_all=False):

        folder = "data"
        self.country_name = country_name
        merged = pd.read_csv("{}/features.csv".format(folder), parse_dates=["Date"])
        country_df = merged[merged["CountryName"] == country_name]

        self.initial_deaths = country_df["ConfirmedDeaths"].tail(1).values[0]

        if country_df.shape[0] ==0:
            raise ValueError()
        country_sub = country_df[country_df["Date"]<=begin_date]

        if country_sub.shape[0] ==0:
            country_sub = country_df

        self.df = country_sub

        features = ["workplace",
            "parks",
            "transit_stations",
            "retail/recreation"]

        dates = np.arange(np.datetime64(begin_date), np.datetime64(end_date), timedelta(days=step)).astype(datetime)
        self.end_date = pd.to_datetime(end_date)
        self.critical_capacity = critical_capacity

        self.measures_dates = np.array([(f, d) for f in features for d in dates])
        nb_var = len(self.measures_dates)

        self.last = []
        self.last_objectives = []
        self.iteration = 0
        self.record_all = record_all

        if record_all:
            self.record_path = "./experiments/ga_steps/{}".format(country_name)
            os.makedirs(self.record_path,exist_ok=True)

        super().__init__(n_var=nb_var,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([-100]*nb_var),
                         xu=np.array([0]*nb_var))

    def _evaluate(self, x, out, *args, **kwargs):

        measures_to_lift = self.measures_dates[:,0]
        dates = self.measures_dates[:,1]

        lift_date_values = [pd.to_datetime(d) for d in dates]

        begin= time.time()
        res = simulate(self.df.copy(), [measures_to_lift]*len(x), 0, self.end_date, None,
                           measure_values=x, lift_date_values=lift_date_values, seed="")
        end = time.time()

        # f1 minimize the deaths
        # f2 maximize the activity (maximize x values => minimize the absolute value of f2)
        f1 = np.array([e["SimulationDeaths_max"].tail(1).values[0] for e in res])
        f2 = np.abs(x.mean(axis=1))
        self.last = [res, x]
        self.last_objectives = [f1, f2]
        #print(end-begin,f1, f2)

        with open("./experiments/ga_{}_temppop.json".format(self.country_name), 'w') as f:
            json.dump( {"df":[e.to_dict() for e in res],"x":x.tolist()}, f)

        with open("./experiments/ga_{}_tempobj.json".format(self.country_name), 'w') as f:
            json.dump( {"deaths": f1.tolist(), "activity":f2.tolist()} , f)


        if self.record_all:

            with open("{}/pop_{}.json".format(self.record_path, self.iteration), 'w') as f:
                json.dump( {"df":[e.to_dict() for e in res],"x":x.tolist()}, f)

            with open("{}/obj_{}.json".format(self.record_path, self.iteration), 'w') as f:
                json.dump( {"deaths": f1.tolist(), "activity":f2.tolist()} , f)



        g1 = np.array([e["SimulationCritical_max"].max() for e in res]) - self.critical_capacity

        f1 = (f1-self.initial_deaths)/self.initial_deaths
        f2 = f2/100
        out["F"] = np.column_stack([np.abs(f1), f2])
        out["G"] = np.column_stack([g1])

        self.iteration = self.iteration +1


    # --------------------------------------------------
    # Pareto-front - not necessary but could be used for plotting
    # --------------------------------------------------
    def _calc_pareto_front(self, flatten=True, **kwargs):
        pass

    # --------------------------------------------------
    # Pareto-set - not necessary but could be used for plotting
    # --------------------------------------------------
    def _calc_pareto_set(self, flatten=True, **kwargs):
        pass

