import json, utils, joblib
from helpers import seir
import pandas as pd
import numpy as np
from utils import load_luxembourg_dataset
from typing import List
from datetime import timedelta, datetime
from scipy.integrate import solve_ivp
from utils import extend_features_with_means

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
        self.workers_sectors = pd.read_excel('{}/HZD971_Covid19_staticCountry.xlsx'.format(self.base_path))

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
                 measure_values: List[dict] = None, change_date_values: List[str] = None) -> pd.DataFrame:

        end_date = pd.to_datetime(end_date, yearfirst=True)
        change_dates = [pd.to_datetime(e, yearfirst=True) for e in change_date_values]

        df = self.initial_df
        columns = [e for e in df.columns if "_" not in e]
        if "Date" not in columns:
            df["Date"] = df.index

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

            all_columns_extended = country_change.drop(self.possibile_inputs,axis=1).fillna(method="pad")
            smoothing_days = [5, 10, 15]
            all_columns_extended = extend_features_with_means(all_columns_extended, columns, smoothing_days)
            #country_change = pd.get_dummies(country_change, prefix="day_of_week", columns=["day_of_week"])
            #country_change = update_mean(country_change.fillna(method="pad")).fillna(method="bfill")

            return all_columns_extended


    def update_population(self, b_be='open', b_fr='open', b_de='open', activity_restr="open"):
        population = self.workers_sectors

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


def run():
    sim = EconomicSimulator()
    dates = ["2020-08-15", "2020-08-30"]
    measures, values = [["b_be","b_fr"]], [["close","close"]]
    end_date = "2020-10-15"
    df = sim.build_df(measures, end_date, values, dates)
    print(sim.ml_inputs)
    #dates = [["2020/"]]

run()
