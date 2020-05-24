import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem

import pandas as pd
import json, time
from datetime import datetime, timedelta
from sklearn.externals import joblib
from simulations import simulate

class ScheduleProblem(Problem):

    def __init__(self, version="v2_1",begin_date="2020-04-30",end_date="2020-09-30",country_name = "Luxembourg", critical_capacity=90, step=14):

        folder = "./models/seirhcd/{}".format(version)
        self.scaler = joblib.load("{}/scaler.save".format(folder))
        self.mlp_clf = joblib.load("{}/mlp.save".format(folder))
        merged = pd.read_csv("{}/features.csv".format(folder), parse_dates=["Date"])
        country_df = merged[merged["CountryName"] == country_name]

        self.initial_deaths = country_df["ConfirmedDeaths"].tail(1).values[0]

        if country_df.shape[0] ==0:
            raise ValueError()
        country_sub = country_df[country_df["Date"]<=begin_date]

        if country_sub.shape[0] ==0:
            country_sub = country_df

        self.df = country_sub

        with open('{}/metrics.json'.format(folder)) as fp:
            metrics = json.load(fp)
            self.yvar = np.power(metrics["std_test"], 0.5)
            self.columns = metrics["columns"]

        features = ["workplace",
            "parks",
            "transit_stations",
            "retail/recreation"]

        dates = np.arange(np.datetime64(begin_date), np.datetime64(end_date), timedelta(days=step)).astype(datetime)
        self.end_date = pd.to_datetime(end_date)
        self.critical_capacity = critical_capacity

        self.measures_dates = np.array([(f, d) for f in features for d in dates])
        nb_var = len(self.measures_dates)

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
        res = simulate(self.df.copy(), [measures_to_lift]*len(x), 0, self.end_date, None, self.columns, self.yvar, self.mlp_clf, self.scaler,
                           measure_values=x, base_folder=None, lift_date_values=lift_date_values,
                           seed="", filter_output=["SimulationCritical","SimulationDeaths"], confidence_interval=False)
        end = time.time()

        # f1 minimize the deaths
        # f2 maximize the activity (maximize x values => minimize the absolute value of f2)
        f1 = np.array([e["SimulationDeaths"].tail(1).values[0] for e in res])
        f2 = np.abs(x.mean(axis=1))
        print(end-begin,f1, f2)

        g1 = np.array([e["SimulationCritical"].max() for e in res]) - self.critical_capacity

        out["F"] = np.column_stack([np.abs((f1-self.initial_deaths)/self.initial_deaths), f2/100])
        out["G"] = np.column_stack([g1])


    # --------------------------------------------------
    # Pareto-front - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_front(self, flatten=True, **kwargs):
        f1_a = np.linspace(0.1**2, 0.4**2, 100)
        f2_a = (np.sqrt(f1_a) - 1)**2

        f1_b = np.linspace(0.6**2, 0.9**2, 100)
        f2_b = (np.sqrt(f1_b) - 1)**2

        a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
        return stack(a, b, flatten=flatten)

    # --------------------------------------------------
    # Pareto-set - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_set(self, flatten=True, **kwargs):
        x1_a = np.linspace(0.1, 0.4, 50)
        x1_b = np.linspace(0.6, 0.9, 50)
        x2 = np.zeros(50)

        a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
        return stack(a,b, flatten=flatten)