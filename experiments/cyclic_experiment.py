import pandas as pd
import json, time
from datetime import datetime, timedelta
from sklearn.externals import joblib
from simulations import simulate




def run(version="v2_1",begin_date="2020-04-30",end_date="2020-09-30",country_name = "Luxembourg", critical_capacity=90, step=14):

        folder = "./models/seirhcd/{}".format(version)
        scaler = joblib.load("{}/scaler.save".format(folder))
        mlp_clf = joblib.load("{}/mlp.save".format(folder))
        merged = pd.read_csv("{}/features.csv".format(folder), parse_dates=["Date"])
        country_df = merged[merged["CountryName"] == country_name]

        initial_deaths = country_df["ConfirmedDeaths"].tail(1).values[0]

        if country_df.shape[0] ==0:
            raise ValueError()
        country_sub = country_df[country_df["Date"]<=begin_date]

        if country_sub.shape[0] ==0:
            country_sub = country_df

        df = country_sub

        with open('{}/metrics.json'.format(folder)) as fp:
            metrics = json.load(fp)
            yvar = np.power(metrics["std_test"], 0.5)
            columns = metrics["columns"]

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


        def simulation():
            measures_to_lift = self.measures_dates[:,0]
            dates = self.measures_dates[:,1]

            lift_date_values = [pd.to_datetime(d) for d in dates]

            begin= time.time()
            res = simulate(self.df.copy(), [measures_to_lift]*len(x), 0, self.end_date, None, self.columns, self.yvar, self.mlp_clf, self.scaler,
                            measure_values=x, base_folder=None, lift_date_values=lift_date_values,
                            seed="", filter_output=["R_max","R","SimulationCritical_max","SimulationDeaths_max"], confidence_interval=True)

            f1 = np.array([e["SimulationDeaths_max"].tail(1).values[0] for e in res])
            f2 = np.abs(x.mean(axis=1))