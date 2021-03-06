import json, utils, joblib

from pandas.core.reshape import merge
from seir_helpers import seir
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from utils import load_luxembourg_dataset
from typing import List, Union
from datetime import timedelta, datetime
from utils import extend_features_with_means
import joblib, json
import json
from matplotlib import pyplot as plt
import seaborn as sns

class EconomicSimulator(object):

    possibile_inputs = {
          "b_be": ["open", "close"],
          "b_fr": ["open", "close"],
          "b_de": ["open", "close"],
          "schools_m" : ["open", "partial", "preventive_measure", "close"],
          "public_gath":["yes", "no"],
          "resp_gov_measure": ["yes", "no"],
          "private_gath":[1000,0,5,10,20],
          "parks_m":["yes","no"],
          "travel_m":["yes", "no"],
          "activity_restr":["open", "close", "mixed"],
          "social_dist": ["yes", "no"],
          "vaccinated_peer_week":[0, 30000]
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
        self.sectors_population = population.append({"sector": "ALL", "label":"Toutes activités",
                                                    "pop": population["pop"].sum()}, ignore_index=True)

        if initial_dataframe is None and country=="luxembourg":
            initial_dataframe = load_luxembourg_dataset(get_past_rt_as_features=True)

        self.initial_df, self.ml_outputs = initial_dataframe
        self.ml_inputs = self.initial_df.columns
        self.ml_outputs = self.ml_outputs.values

        if initial_inputs is None:
            for k,v in self.possibile_inputs.items():
                self.initial_df[k] = v[0]
        else:
            for k, v in initial_inputs.items():
                self.initial_df[k] = v

        self.load_rt_models()
        self.load_economic_models()

    def load_rt_models(self):
        self.mlp_clf = joblib.load("./models/mlp_economic_sectors.save")
        self.scaler = joblib.load("./models/scaler_economic_sectors.save")
        self.metrics = json.load(open("./models/metrics_economic_sectors.json"))

    def load_economic_models(self):
        self.model_export = joblib.load("./models/economic_impact/model_export.save")
        self.model_unemployment = joblib.load("./models/economic_impact/model_unemployment.save")
        self.model_inflation = joblib.load("./models/economic_impact/model_inflation.save")
        self.model_ipcn = joblib.load("./models/economic_impact/model_ipcn.save")

        # self.economic_scaler = joblib.load("./models/economic_impact/scaler_econ.save")

    def get_limit_dates(self) -> Union[datetime.date, datetime.date]:
        """To Define the maximum and minimum date the end-user can ask.

        Returns:
            Union[datetime.date, datetime.date]: The maximum and minimum dataset Date.
        """

        min_date = self.initial_df.head(1).index.date[0]
        max_date  = self.initial_df.tail(1).index.date[0]

        return min_date, max_date
        

    def build_df(self, measures_to_change: List[dict], end_date: str,
                 measure_values: List[dict] = None, change_date_values: List[str] = None, 
                 start_date:str=None, init_date_p:str=None) -> pd.DataFrame:

        df = self.initial_df

        columns = [e for e in df.columns if "_" not in e]
        if "Date" not in columns:
            df["Date"] = df.index

        # There is some duplicates in the dataset, 
        # (e.g : `2020-05-16`) (A strange thing is, for the same day, the value of `N` moves...)
        # Here I will keep the last sample.
        df = df.drop_duplicates(subset='Date', keep='last')

        if start_date is not None:
            start_date = pd.to_datetime(start_date, yearfirst=True)
            df = df[df["Date"] > start_date]

        end_date = pd.to_datetime(end_date, yearfirst=True)

        if change_date_values is not None and len(change_date_values) == len(measures_to_change[0]):
            change_dates = [pd.to_datetime(e, yearfirst=True) for e in change_date_values]
        else:
            change_dates = [df["Date"].head(1).dt.date.values[0]] * len(measures_to_change[0])
            change_dates = [pd.to_datetime(e, yearfirst=True) for e in change_dates]

        if init_date_p is None:
            init_date = df["Date"].tail(1).dt.date.values[0]
        else:
            init_date = pd.to_datetime(init_date_p)

        for k, measure_to_change in enumerate(measures_to_change):
            # Initialise the dataset, 
            # `update_seir` needs 14 days before `init_date` to do his computations
            # (initial conditions)
            seir_day_offset = 14 

            df_begin_date = init_date - timedelta(days=seir_day_offset)
            country_change = df.loc[df_begin_date: init_date]

            current_date = pd.to_datetime(init_date)

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

        elif b_be == 'open' and b_fr == 'open' and b_de == 'open':
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


    def update_ML_params(self,obj:dict) -> dict:
        """Update inputs for the Rt model
        This function uses 

        Args:
            obj (dict): A row of our initial dataframe transformed into a dictionnary

        Returns:
            dict: the updated row.
        """
        # Activity Resctrictions
        obj["population"] = self.update_population(obj["b_be"],obj["b_fr"],obj["b_de"],obj["activity_restr"])

        sim_date = obj['Date']
        date_shift =  sim_date - timedelta(days=120)
        initial_df = self.initial_df.loc[date_shift:sim_date]
            
        if obj["resp_gov_measure"] == 'yes':
            obj["H1"] = 2
            obj["H2"] = 3
            obj["H3"] = 2
        else:
            obj["H1"] = 0
            obj["H2"] = 0
            obj["H3"] = 0

        # Borders
        if obj["b_be"] == 'close' or obj["b_fr"] == 'close' or obj["b_de"] == 'close':
            obj["C8"] = initial_df['C8'].min()
            obj["C5"] = initial_df['C5'].min()
            obj["transit"] = initial_df['transit'].min()

        else:
            obj["C8"] = 1

        # Travel
        if obj["travel_m"] == 'no':
            obj["C8"] = initial_df['C8'].min()
            obj["C5"] = initial_df['C5'].min()
            obj["transit"] = initial_df['transit'].min()
        else:
            obj["C8"] = 1
            obj["C5"] = 1
            obj["transit"] = initial_df['transit'].max()

        # Parks
        if obj["parks_m"] == 'yes':
            obj["parks"] = max(initial_df['parks'])
        else:
            obj["parks"] = min(initial_df['parks'])

        # Schools
        # Is Preventive measure missing ? 
        if obj["schools_m"] == 'open':
            obj["C1"] = initial_df['C1'].max()
        elif obj["schools_m"] == 'close':
            obj["C1"] = initial_df['C1'].min()
        elif obj["schools_m"] == 'partial':
            obj["C1"] = initial_df['C1'].mean()
        else:
            obj["C1"] = 2

        # Gatherings
        if obj["public_gath"] == 'no':
            obj["C3"] = initial_df['C3'].min()
        else:
            obj["C3"] = initial_df['C3'].max()


        # Where are conditions for {10, 20} ? 
        if obj["private_gath"] == 0:
            obj["C4"] = initial_df['C4'].max()
        elif obj["private_gath"] == 1000:
            obj["C4"] = initial_df['C4'].min()

        if obj["activity_restr"] == 'close':
            obj["retail/recreation"] = initial_df['retail/recreation'].min()
            obj["grocery/pharmacy"] = initial_df['grocery/pharmacy'].min()
            obj["workplaces"] = initial_df['workplaces'].min()
            obj["C2"] = initial_df['C2'].min()
            obj["C6"] = initial_df['C6'].min()
            obj["C7"] = initial_df['C7'].min()
            obj["transit"] = initial_df["transit"].min()
            obj["driving"] = initial_df["driving"].min()
            obj["public_transport"] = initial_df["public_transport"].min()
            obj["walking"] = initial_df["walking"].min() 

        elif obj["activity_restr"] == 'mixed':
            obj["retail/recreation"] = initial_df['retail/recreation'].mean()
            obj["grocery/pharmacy"] = initial_df['grocery/pharmacy'].mean()
            obj["workplaces"] = initial_df['workplaces'].mean()
            obj["C2"] = initial_df['C2'].mean()
            obj["C6"] = initial_df['C6'].mean()
            obj["C7"] = initial_df['C7'].mean()
        else:
            obj["retail/recreation"] = max(initial_df['retail/recreation'])
            obj["grocery/pharmacy"] = initial_df['grocery/pharmacy'].max()
            obj["workplaces"] = initial_df['workplaces'].max()
            obj["C2"] = initial_df['C2'].max()
            obj["C6"] = initial_df['C6'].max()
            obj["C7"] = initial_df['C7'].max()
        return obj

    def simulate(self, Rt_sector, population_total, deaths_per_sectors=None, init_date=None, vaccinated_peer_day=None):
        simulations = {}
        sectors = Rt_sector.columns[:-1]
        
        sectors_workers = self.sectors_population[["sector","pop"]]

        total_pop = sectors_workers[sectors_workers["sector"]=='ALL'].values[0][1]

        for sector in sectors:

            # print('[+]',sector)
            # print('[+]',sectors_workers[sectors_workers["sector"]==sector])

            sector_df = Rt_sector[["Date", sector]]
            # Normaliser le Rt avec la distribution nationale
            sector_df.columns = ["Date", "R"]

            sector_population = sectors_workers[sectors_workers["sector"]==sector].values[0][1]
            susceptible_factor = population_total["population"].min() / population_total["population"].max()

            cases =  pd.DataFrame(data={"ConfirmedCases":self.cases[sector], "Date":self.cases.index}, 
                                  index=self.cases.index)

            cases["Date"] = pd.to_datetime(cases["Date"])
            sector_df["Date"] = pd.to_datetime(sector_df["Date"])
            sector_df = pd.merge(sector_df,cases, on="Date", how="left").fillna(0)
            sector_df["ConfirmedDeaths"] = 0 if deaths_per_sectors is None else deaths_per_sectors[sector]


            


            dates = sector_df["Date"].to_list()
            if init_date is None:
                init_date = dates[15]

            # Add the vaccination  measure, This measure influence th R0 value
            if vaccinated_peer_day is not None:
                n_day = sector_df[sector_df['Date'] >= init_date].shape[0]
                # Compute the weight of vaccine in a given sector
                # It assumes that the vacine is fairly distributed.
                vaccine_sector = vaccinated_peer_day * sector_population / total_pop
                vaccine_sector = int(np.floor(vaccine_sector))

                rt_factor = self.compute_vaccinated_weight(n_day, vaccine_sector, sector_population*susceptible_factor)

                sector_df.index = sector_df['Date']

                r_array = sector_df.loc[init_date:,'R'].values
                r_array = r_array * rt_factor

                sector_df.loc[init_date:,'R'] = r_array

                sector_df.reset_index(drop=True, inplace=True)

            simulation = self.update_seir(sector_df, active_date=init_date, e_date=dates[-1],
                                          population=sector_population*susceptible_factor)

            simulations[sector] = simulation

        return simulations


    def update_seir(self, df, active_date, e_date, population):
        active_date = pd.to_datetime(active_date)

        # cols = list(df.columns)
        # t_hosp=7, t_crit=14, m_a=0.7, c_a=0.1, f_a=0.3
        params = [7, 10, 0.8, 0.3, 0.1]
        
        # Decay values
        params.append(False)

        data = df[df["Date"] >= active_date]
        ref_data = df[df["Date"] == pd.to_datetime(active_date + timedelta(days=-7))]
        inf_data = df[df["Date"] == pd.to_datetime(active_date + timedelta(days=-14))]

        N = population 
        n_infected = ref_data['ConfirmedCases'].iloc[0] - inf_data['ConfirmedCases'].iloc[0] # data['InfectiousCases'].iloc[0]
        n_exposed = data['ConfirmedCases'].iloc[0] - ref_data['ConfirmedCases'].iloc[0] # data['ExposedCases'].iloc[0]
        n_hospitalized = (1 - params[2]) * n_exposed  # data['HospitalizedCases'].iloc[0]*1.5
        n_exposed = params[2] * n_exposed
        n_critical = (params[3]) * n_hospitalized  # data['CriticalCases'].iloc[0]*1.5

        n_recovered = inf_data['ConfirmedCases'].iloc[0] - inf_data['ConfirmedDeaths'].iloc[0]  # data['RecoveredCases'].iloc[0]
        n_deaths = data['ConfirmedDeaths'].iloc[0]

        # S, E, I, R, H, C, D
        initial_state = [(N - n_infected) / N, n_exposed / N, n_infected / N, n_recovered / N, 
                          n_hospitalized / N, n_critical / N, n_deaths / N]

        R_t = data['R'].values

        def time_varying_reproduction(t):
            index = np.min((int(t), len(R_t) - 1))
            return R_t[index]

        args = (time_varying_reproduction, 5., 2.9, *params)

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

        smoothing_columns = ["SimulationCases", "SimulationHospital", "SimulationCritical", "SimulationDeaths"]

        simulations = pd.DataFrame({"Date": dates, 
                                    "SimulationCases": y_pred_cases.astype(int),
                                    "SimulationHospital": y_pred_hosp.astype(int) + y_pred_critic.astype(int),
                                    "SimulationCritical": y_pred_critic.astype(int),
                                    "SimulationDeaths": y_pred_deaths.astype(int),
                                    "SimulationInfectious": y_pred_infectious.astype(int)})

        simulations["R"] = data['R'].iloc[1:].values

        for e in smoothing_columns:
            simulations[e] = simulations[e].rolling(3, 2, center=True).mean().astype(int)

        simulations = simulations.dropna()

        return simulations

    def predict_economic(self, data):
        df = data.drop(["Date"], axis=1)

        cols = df.columns.to_list()

        cols = list(filter(lambda c: not 'Critical' in c, cols))
        cols = list(filter(lambda c: not 'Deaths' in c, cols))
        cols = list(filter(lambda c: not 'Hospital' in c, cols))
        cols = list(filter(lambda c: not '_min' in c, cols))
        cols = list(filter(lambda c: not '_max' in c, cols))
        cols = list(filter(lambda c: not 'population' in c, cols))

        X = df[cols].values

        # Scaler is not used beceause our economic model show better performances 
        # without scaler.
        # X = self.economic_scaler.transform(df)   
        inflation = self.model_inflation.predict(X)
        unemploy = self.model_unemployment.predict(X)
        export = self.model_export.predict(X)

        cols.append('population')
        X = df[cols].values
        ipcn = self.model_ipcn.predict(X)

        data["inflation"] = inflation
        data["ipcn"] = ipcn
        data["unemploy"] = unemploy
        data["export"] = export

        return data


    def run_all_simulation(self, Rt, population_total, init_date, df, vaccinated_peer_day=None):
        """This funciton run all simulations. Added this function when I wanted to add confidance interval.

        Args:
            Rt (pd.DataFrame): The prediction of our Rt model wrapped to a DataFrame
            population_total ([type]): The population dataframe, which is a columns of the result of `self.build_df` function
            init_date (datetime.date): The begining date of the simulation.
            df (pd.Dataframe): The dataframe generated by `self.build_df` function
            vaccinated_peer_day (int, optional): The number of people vaccined peer day. Defaults to None.

        Returns:
            pd.DataFrame: The simulation dataframe.
        """

        simulation = self.simulate(Rt, population_total, deaths_per_sectors=None, init_date=init_date, vaccinated_peer_day=vaccinated_peer_day)
        
        merged = pd.merge(simulation["ALL"], simulation["A"], suffixes=["_ALL", "_A"], on="Date")

        for key in simulation.keys():
            if key!="ALL" and key!="A":
                simu_key = simulation[key]
                renamed_cols = [e + '_{}'.format(key) for e in simu_key.columns[1:]]
                rename_dict = dict(zip(simu_key.columns[1:], renamed_cols))
                simu_key = simu_key.rename(columns=rename_dict)
                
                merged = pd.merge(merged, simu_key, on="Date")

        df["Date"] = pd.to_datetime(df["Date"])

        merged_ = pd.merge(merged, df[['Date','population']], on="Date")

        merged_ = self.predict_economic(merged_)
        merged_ = merged_.drop(["population"], axis=1)

        merged_final = pd.merge(merged_, df, on="Date")

        merged_final.index = merged_final["Date"]

        return merged_final


    def compute_vaccinated_weight(self, n_day:int, people_vaccined_per_day:int, n_pop:int):
        r_weight = []
        for d in range(n_day):
            vaccinated_person = people_vaccined_per_day * d
            ratio = vaccinated_person / n_pop
            r_weight.append(np.abs(-1/(ratio+1)))

        return np.asarray(r_weight)

    def run(self, dates, measures, values, end_date, init_date=None):

        df, init_date = self.build_df(measures, end_date, values, dates, init_date_p=init_date)

        rt_ref = self.initial_df.loc[init_date]

        df['ALL_past'] =rt_ref['ALL_past']
        df['ALL_past_5'] = rt_ref['ALL_past_5']
        df['ALL_past_10'] = rt_ref['ALL_past_10']
        df['ALL_past_15'] = rt_ref['ALL_past_15']

        columns = self.metrics["x_columns"]

        X_lift = self.scaler.transform(df[columns])
        y_lift = self.mlp_clf.predict(X_lift)
        # y_lift = np.clip(y_lift, 0, 10)

        Rt = pd.DataFrame(data=y_lift, index=df.index, columns=self.ml_outputs)


        # Add the vaccination measure here.
        if 'vaccinated_peer_week' in measures[0]:
            idx_vaccinated_value = measures[0].index('vaccinated_peer_week')
            percentage_vaccinated = values[0][idx_vaccinated_value]  # The value from the user interface is a purcentage.
            vaccinated_peer_day = self.possibile_inputs['vaccinated_peer_week'][-1]*percentage_vaccinated/700  
            vaccinated_peer_day = int(np.floor(vaccinated_peer_day))
        else:
            vaccinated_peer_day = None

        # Copy DataFrame aiming to include the minimal and maximal.
        Rt_max, Rt_min = Rt.copy(), Rt.copy()
        
        # include the confidance interval using Rt
        std_peer_features = self.metrics['std_test']  
        for model_output_feature in std_peer_features.keys():
            yvar = np.power(std_peer_features[model_output_feature],0.5) 
            Rt_max[model_output_feature] = np.clip(Rt[model_output_feature] + yvar.mean() / 2, 0, 10)
            Rt_min[model_output_feature] = np.clip(Rt[model_output_feature] - yvar.mean() / 2, 0, 10)
            

        # Passing dates to dataframe
        Rt["Date"] = df["Date"]
        Rt_min['Date'] = df['Date']
        Rt_max['Date'] = df['Date']

        # Make a simple dataframe for the polulation 
        population_total = pd.DataFrame(data={"population": df["population"], "date": df["Date"]}, 
                                        index=df.index)

        simulation_merged = self.run_all_simulation(Rt, population_total, init_date, df, vaccinated_peer_day)
        simulation_merged_min = self.run_all_simulation(Rt_min, population_total, init_date, df, vaccinated_peer_day)
        simulation_merged_max = self.run_all_simulation(Rt_max, population_total, init_date, df, vaccinated_peer_day)

        simulation_merged_max = simulation_merged_max.drop(columns=['Date'])

        simulation_merged = pd.merge(simulation_merged, simulation_merged_max, suffixes=['','_max'], 
                                     left_index=True, right_index=True)
        simulation_merged = pd.merge(simulation_merged, simulation_merged_min, suffixes=['','_min'], 
                                     left_index=True, right_index=True)


        return simulation_merged


# def run(dates=None,measures=None,values=None):
#     sim = EconomicSimulator()
#     if dates is None:
#         dates = ["2020-08-15", "2020-08-30"]
#     if measures is None:
#         measures = [["b_be","b_fr"]]
#     if values is None:
#         values = [["close","close"]]

#     end_date = sim.initial_df.index[-1]

#     simulation = sim.run(dates, measures, values, end_date)
#     #simulation["O"].plot(title="Sector O")
#     #simulation["ALL"].plot(title="All sectors")

#     return simulation

# #run()
# #run(measures = [["activity_restr"]],values = [["close"]])

# #plt.show()
# dt = ["2020-04-15"]
# scenarios = []
# measures_possibles = EconomicSimulator.possibile_inputs
# nb_values = len(measures_possibles.keys())

# combinations = []
# values = []
# max_scenarios = 50000

# def iter(k, last_combinations=[], last_values=[]):
#     for i in range(k, nb_values):
#         ms, vs = list(measures_possibles.keys())[i], list(measures_possibles.values())[i]
#         for v in vs:
#             if len(combinations) > max_scenarios:
#                 return

#             last_c = last_combinations.copy()
#             last_c.append(ms)
#             last_v =  last_values.copy()
#             last_v.append(v)

#             if i+1<nb_values:
#                 iter(i + 1, last_c, last_v)
#             else:
#                 if len(last_c)==nb_values:
#                     dates = dt * len(last_v)
#                     df = run(dates=dates, measures=[last_c], values=[last_v])
#                     measures = {"dates": dates, "inputs": last_c, "values": last_v}
#                     combinations.append(last_c)
#                     values.append(last_v)
#                     json.dump(measures, open("scenarios/{}.json".format(len(combinations)), "w"))
#                     df.to_csv("scenarios/{}.csv".format(len(combinations)))
#     return

# iter(0)
