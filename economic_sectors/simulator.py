import json, utils, joblib
from helpers import seir
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from utils import load_luxembourg_dataset
from typing import List
from datetime import timedelta, datetime
from utils import extend_features_with_means
import joblib, json

class EconomicSimulator(object):

    possibile_inputs = {
          "b_be": ["open", "close"],
          "b_fr": ["open", "close"],
          "b_de": ["open", "close"],
          "schools_m" : ["open", "partial", "preventive_measure", "close"],
          "public_gath":["yes", "no"],
          "social_dist": ["yes", "no"],
          "private_gath":[1000,0,5,10,20],
          "parks_m":["yes","no"],
          "travel_m":["yes", "no"],
          "activity_restr":["open", "close", "mixed"],
        }


    def __init__(self,country="luxembourg",  initial_dataframe=None, initial_inputs=None):
        # HERE ADDING FILE CONTAINING POPULATION IN SECTORS + MIN REQUIREMENT PER SECTORS
        self.base_path = './data/{}/economy'.format(country)
        self.workers_borders = pd.read_excel('{}/HZD971_Covid19_staticCountry.xlsx'.format(self.base_path))

        cases =  pd.read_excel('{}/PJG397_Covid19_daily_workSector_Residents_IGSS.xlsx'.format(self.base_path),"Daily_Infections_by_Sector")
        cases.index = pd.to_datetime(pd.to_datetime(cases['Dates\Age Range']))
        cases = cases.drop(['Dates\Age Range'], axis=1).dropna()
        cases["ALL"] = cases.sum(axis=1)
        self.cases = cases.cumsum()

        population = pd.read_excel(
            '{}/PJG397_Covid19_daily_workSector_Residents_IGSS.xlsx'.format(self.base_path),"Total amount of workers",usecols=[0,1,2], names=["sector","label", "pop"])
        self.sectors_population = population.append({"sector": "ALL", "label":"Toutes activités","pop": population["pop"].sum()}, ignore_index=True)


        if initial_dataframe is None and country=="luxembourg":
            initial_dataframe = load_luxembourg_dataset()

        self.initial_df, self.ml_outputs = initial_dataframe
        self.ml_inputs = self.initial_df.columns
        self.ml_outputs = self.ml_outputs.values


        if initial_inputs is None:
            for k,v in self.possibile_inputs.items():
                self.initial_df[k] = v[0]
        else:
            for k, v in initial_inputs.items():
                self.initial_df[k] = v

    def build_df(self, measures_to_change: List[dict], end_date: str,
                 measure_values: List[dict] = None, change_date_values: List[str] = None, start_date:str=None) -> pd.DataFrame:

        df = self.initial_df
        columns = [e for e in df.columns if "_" not in e]
        if "Date" not in columns:
            df["Date"] = df.index

        if start_date is not None:
            start_date = pd.to_datetime(start_date, yearfirst=True)
            df = df[df["Date"] > start_date]

        end_date = pd.to_datetime(end_date, yearfirst=True)
        change_dates = [pd.to_datetime(e, yearfirst=True) for e in change_date_values]

        init_date = df["Date"].tail(1).dt.date.values[0]
        for k, measure_to_change in enumerate(measures_to_change):
            country_change = df.copy()
            current_date = init_date

            while current_date < end_date:
                current_date = current_date + timedelta(days=1)
                obj = country_change.tail(1).to_dict('records')[0]
                obj["Date"] = current_date
                vals = measure_values[k] if measure_values is not None and len(measure_values) > 1 else measure_values[0]

                for i, measure in enumerate(measure_to_change):
                    change = current_date >= change_dates[i]
                    if change:
                        obj[measure] = vals[i]

                obj = self.update_ML_params(obj)
                country_change = country_change.append(obj, ignore_index=True)

            all_columns_extended = country_change.drop(self.possibile_inputs,axis=1).fillna(method="pad").fillna(method="backfill")
            smoothing_days = [5, 10, 15]
            all_columns_extended = extend_features_with_means(all_columns_extended, columns, smoothing_days)
            #country_change = pd.get_dummies(country_change, prefix="day_of_week", columns=["day_of_week"])
            #country_change = update_mean(country_change.fillna(method="pad")).fillna(method="bfill")

            return all_columns_extended, init_date


    def update_population(self, b_be='open', b_fr='open', b_de='open', activity_restr="open"):
        population = self.workers_borders

        if activity_restr == 'close':
            population = population['luxembourg'].sum() + population['min_be'].sum() + population['min_de'].sum() + \
                         population['min_fr'].sum()

        if b_be == 'open' and b_fr == 'open' and b_de == 'open':
            population = population['total'].sum()  # normal population
        else:
            if b_be == 'open':
                be_worker = population['belgium'].sum()
            else:
                be_worker = population['min_be'].sum()

            if b_fr == 'open':
                fr_worker = population['france'].sum()
            else:
                fr_worker = population['min_fr'].sum()

            if b_de == 'open':
                de_worker = population['germany'].sum()
            else:
                de_worker = population['min_de'].sum()

            population = population['luxembourg'].sum() + be_worker + de_worker + fr_worker

        return population


    def update_ML_params(self,obj):
        # Activity Resctrictions

        obj["population"] = self.update_population(obj["b_be"],obj["b_fr"],obj["b_de"],obj["activity_restr"])

        if obj["activity_restr"] == 'close':
            obj["retail/recreation"] = self.initial_df['retail/recreation'].min()
            #obj["residential"] = self.initial_df['residential'].max()
            obj["C2"] = 1
            obj["C6"] = 1
            obj["C7"] = 1
        elif obj["activity_restr"] == 'mixed':
            obj["retail/recreation"] = self.initial_df['retail/recreation'].mean()
            #obj["residential"] = self.initial_df['residential'].mean()
            obj["C2"] = 0.5
            obj["C6"] = 0.5
            obj["C7"] = 0.5
        else:
            obj["retail/recreation"] = max(self.initial_df['retail/recreation'])
            #obj["residential"] = min(self.initial_df['residential'])
            obj["C2"] = 0
            obj["C6"] = 0
            obj["C7"] = 0

        # Borders
        if obj["b_be"] == 'close' or obj["b_fr"] == 'close' or obj["b_de"] == 'close':
            obj["C8"] = 0
        else:
            obj["C8"] = 1

        # Travel

        if obj["travel_m"] == 'no':
            obj["C8"] = 0
        else:
            obj["C8"] = 1

        # Parks
        if obj["parks_m"] == 'yes':
            obj["parks_m"] = max(self.initial_df['parks'])
        else:
            obj["parks_m"] = min(self.initial_df['parks'])

        # Schools
        if obj["schools_m"] == 'open':
            obj["C1"] = 3
        elif obj["schools_m"] == 'close':
            obj["C1"] = 0
        elif obj["schools_m"] == 'partial':
            obj["C1"] = 1
        else:
            obj["C1"] = 2

        # Gatherings
        if obj["public_gath"] == 'no':
            obj["C3"] = 1
        else:
            obj["C3"] = 0

        if obj["private_gath"] == 0:
            obj["C4"] = 1
        elif obj["private_gath"] == 1000:
            obj["C4"] = 0

        return obj

    def simulate(self, Rt_sector, population_total, deaths_per_sectors=None, init_date=None ):
        simulations = {}
        sectors = Rt_sector.columns[:-1]
        sectors_workers = self.sectors_population[["sector","pop"]]

        for sector in sectors:
            sector_df = Rt_sector[["Date", sector]]
            ## Normaliser le Rt avec la distribution nationale
            sector_df.columns = ["Date", "R"]
            sector_population = sectors_workers[sectors_workers["sector"]==sector].values[0][1]
            susceptible_factor = population_total["population"].min() / population_total["population"].max()

            cases =  pd.DataFrame(data={"ConfirmedCases":self.cases[sector], "Date":self.cases.index}, index=self.cases.index)
            cases["Date"] = pd.to_datetime(cases["Date"])
            sector_df["Date"] = pd.to_datetime(sector_df["Date"])
            sector_df = pd.merge(sector_df,cases, on="Date", how="left").fillna(0)
            sector_df["ConfirmedDeaths"] = 0 if deaths_per_sectors is None else deaths_per_sectors[sector]

            dates = sector_df["Date"].to_list()
            if init_date is None:
                init_date = dates[15]

            simulation = self.update_seir(sector_df, active_date=init_date, e_date=dates[-1],
                                     population=sector_population*susceptible_factor)
            simulation = simulation.drop(["Date"], axis=1)
            simulations[sector] = simulation

        simulations = self.simulate_economic_impact(simulations)
        return simulations

    def simulate_economic_impact(self, df):
        return df

    def update_seir(self, df, active_date, e_date, population):
        active_date = pd.to_datetime(active_date)
        cols = list(df.columns)
        # t_hosp=7, t_crit=14, m_a=0.7, c_a=0.1, f_a=0.3
        params = [7, 10, 0.8, 0.3, 0.3]
        params.append(True)

        data = df[df["Date"] >= active_date]
        ref_data = df[df["Date"] == pd.to_datetime(active_date + timedelta(days=-7))]
        inf_data = df[df["Date"] == pd.to_datetime(active_date + timedelta(days=-14))]

        # Herd immunity is assumed at 70%
        N = population
        n_infected = ref_data['ConfirmedCases'].iloc[0] - inf_data['ConfirmedCases'].iloc[
            0]  # data['InfectiousCases'].iloc[0]
        n_exposed = data['ConfirmedCases'].iloc[0] - ref_data['ConfirmedCases'].iloc[
            0]  # data['ExposedCases'].iloc[0]
        n_hospitalized = (1 - params[2]) * n_exposed  # data['HospitalizedCases'].iloc[0]*1.5
        n_exposed = params[2] * n_exposed
        n_critical = (params[3]) * n_hospitalized  # data['CriticalCases'].iloc[0]*1.5
        n_recovered = inf_data['ConfirmedCases'].iloc[0] - inf_data['ConfirmedDeaths'].iloc[
            0]  # data['RecoveredCases'].iloc[0]
        n_deaths = data['ConfirmedDeaths'].iloc[0]
        # S, E, I, R, H, C, D
        initial_state = [(N - n_infected) / N, n_exposed / N, n_infected / N, n_recovered / N, n_hospitalized / N,
                         n_critical / N, n_deaths / N]



        R_t = data['R'].values

        def time_varying_reproduction(t):
            index = np.min((int(t), len(R_t) - 1))
            return R_t[index]

        args = (time_varying_reproduction, 5.6, 2.9, *params)

        init_date = datetime.combine(active_date, datetime.min.time())
        max_days = (pd.Timestamp(e_date) - pd.Timestamp(init_date)) / np.timedelta64(1, 'D')
        max_days = int(max_days)
        sol = solve_ivp(seir.model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
        sus, exp, inf, rec, hosp, crit, deaths = sol.y

        y_pred_cases = np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population
        y_pred_critic = np.clip(crit, 0, np.inf) * population
        y_pred_hosp = np.clip(hosp, 0, np.inf) * population
        y_pred_deaths = np.clip(deaths, 0, np.inf) * population
        y_pred_infectious = np.clip(inf, 0, np.inf) * population

        dates = data["Date"].iloc[1:]
        l = len(dates)

        smoothing_columns = ["SimulationCases", "SimulationHospital", "SimulationCritical", "SimulationDeaths"]

        simulations = pd.DataFrame({"Date": dates, "SimulationCases": y_pred_cases.astype(int),
                                    "SimulationHospital": y_pred_hosp.astype(int) + y_pred_critic.astype(int),
                                    "SimulationCritical": y_pred_critic.astype(int),
                                    "SimulationDeaths": y_pred_deaths.astype(int),
                                    "SimulationInfectious": y_pred_infectious.astype(int)})

        simulations["R"] = data['R'].iloc[1:].values

        for e in smoothing_columns:
            simulations[e] = simulations[e].rolling(3, 2, center=True).mean().astype(int)

        simulations = simulations.dropna()

        return simulations


def run():
    sim = EconomicSimulator()
    dates = ["2020-08-15", "2020-08-30"]
    measures, values = [["b_be","b_fr"]], [["close","close"]]
    end_date = "2020-10-15"
    df, init_date = sim.build_df(measures, end_date, values, dates)

    mlp_clf= joblib.load("./models/mlp_economic_sectors.save")
    scaler =joblib.load("./models/scaler_economic_sectors.save")
    metrics = json.load(open("./models/metrics_economic_sectors.json"))
    columns = metrics["x_columns"]
    X_lift = scaler.transform(df[columns])
    y_lift = mlp_clf.predict(X_lift)
    Rt = pd.DataFrame(data=y_lift, index=df.index, columns=sim.ml_outputs)
    Rt["Date"] = df["Date"]
    population_total = pd.DataFrame(data={"population":df["population"],"date":df["Date"]}, index=df.index)

    simulation = sim.simulate(Rt, population_total, deaths_per_sectors=None, init_date=init_date )
    print(sim.ml_inputs)

run()
