from dateutil.relativedelta import relativedelta
from pandas.core.indexes.base import InvalidIndexError

from economic_sectors.simulator import EconomicSimulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import json
sns.set()

import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

dt = ["2020-05-01"]
scenarios = []
measures_possibles = EconomicSimulator().possibile_inputs
nb_values = len(measures_possibles.keys())
combinations = []
values = []
max_scenarios = 50000

DATE_PAST_SHIFT = 7
SEIR_SHIFT = 14
DATE_FUTURE_SHIFT = 30


def run_simple_simulation(Rt, population_total, df, eco_sim):
    simulation = eco_sim.simulate(Rt, population_total, deaths_per_sectors=None, init_date=None, vaccinated_peer_day=None)
    merged = pd.merge(simulation["ALL"], simulation["A"], suffixes=["_ALL", "_A"], on="Date")
    for key in simulation.keys():
        if key!="ALL" and key!="A":
            merged = pd.merge(merged, simulation[key], suffixes=["", "_{}".format(key)], on="Date")

    df["Date"] = pd.to_datetime(df["Date"])
    return merged

def make_simulation_for(start_date, end_date):
    eco_sim = EconomicSimulator()
    df = eco_sim.initial_df
    
    df['population'] = eco_sim.update_population()
    df = df.loc[start_date: end_date]
    df = df.reset_index()

    columns = eco_sim.metrics["x_columns"]

    X_lift = eco_sim.scaler.transform(df[columns])
    y_lift = eco_sim.mlp_clf.predict(X_lift)
    # y_lift = np.clip(y_lift/2,0,10)
    Rt = pd.DataFrame(data=y_lift, index=df.index, columns=eco_sim.ml_outputs)

    # Copy DataFrame aiming to include the minimal and maximal.
    Rt_max, Rt_min = Rt.copy(), Rt.copy()

    # include the confidance interval using Rt
    std_peer_features = eco_sim.metrics['std_test']  
    for model_output_feature in std_peer_features.keys():
        yvar = np.power(std_peer_features[model_output_feature],0.5) 
        Rt_max[model_output_feature] = np.clip(Rt[model_output_feature] + yvar.mean() / 2, 0, 10)
        Rt_min[model_output_feature] = np.clip(Rt[model_output_feature] - yvar.mean() / 2, 0, 10)

    # Passing dates to dataframe
    Rt["Date"] = df["Date"]
    Rt_min['Date'] = df["Date"]
    Rt_max['Date'] = df["Date"]

    # Make a simple dataframe for the population 
    population_total = df[['population', 'Date']]

    # Run all simulations 
    simulation_merged = run_simple_simulation(Rt, population_total, df, eco_sim)

    simulation_merged_min = run_simple_simulation(Rt_min, population_total, df, eco_sim)
    simulation_merged_max = run_simple_simulation(Rt_max, population_total, df, eco_sim)
    
    simulation_merged_max = simulation_merged_max.drop(columns=['Date'])
    simulation_merged = pd.merge(simulation_merged, simulation_merged_max, suffixes=['','_max'], 
                                 left_index=True, right_index=True)
    simulation_merged = pd.merge(simulation_merged, simulation_merged_min, suffixes=['','_min'], 
                                 left_index=True, right_index=True)

    return simulation_merged


def export_simulations_on_real_data():
    real_rt = pd.read_csv('data/luxembourg/luxembourg_allsectors_rt.csv')
    real_rt = real_rt.rename(columns={'Unnamed: 0':'Date'})
    real_rt['Date'] = pd.to_datetime(real_rt['Date'])
    all_cases =  EconomicSimulator().cases

    total_sim = None

    initial_df = EconomicSimulator().initial_df
    start_date = initial_df.index[0]
    end_date = initial_df.index[-1]
    date_shift = relativedelta(days=DATE_FUTURE_SHIFT)

    while start_date + date_shift <= end_date:
        end_simulation_date = start_date + date_shift

        sim = make_simulation_for(start_date, end_simulation_date)

        if total_sim is None:
            total_sim = sim
        else:
            total_sim = total_sim.append(sim)

        start_date = end_simulation_date - relativedelta(days=DATE_PAST_SHIFT)

    total_sim = total_sim.set_index(['Date'])
    real_rt = real_rt.set_index(['Date'])

    real_rt = real_rt.loc[total_sim.index[0]: total_sim.index[-1]]
    all_cases = all_cases.loc[total_sim.index[0]: total_sim.index[-1]]

    total_sim.to_csv('datasets/simulation_from_{}_to_{}_from_real_data.csv'.format(str(initial_df.index[0].date()), str(initial_df.index[-1].date())))

    total_sim['SimulationCases_ALL'].plot()
    all_cases['ALL'].plot()
    plt.legend()
    plt.show()

    return total_sim

def run(measures,values):
    sim = EconomicSimulator()
    start_date = datetime.strptime("2020-05-15", '%Y-%m-%d')
    df_end_date = sim.initial_df.index[-1]

    simulation = None

    while start_date + timedelta(days=DATE_FUTURE_SHIFT) < df_end_date:

        dates = [str(start_date.date())]
        init_date = datetime.strptime(dates[0], '%Y-%m-%d') - timedelta(days=SEIR_SHIFT) 
        end_date =  datetime.strptime(dates[0], '%Y-%m-%d') + timedelta(days=DATE_FUTURE_SHIFT) 

        simu = EconomicSimulator().run([init_date], measures, values, end_date, init_date=init_date)

        if simulation is None:
            simulation = simu
        else:
            simulation = simulation.append(simu)

        start_date = end_date + timedelta(days=SEIR_SHIFT) 

    return simulation


def iter(k, last_combinations=[], last_values=[]):
    for i in range(k, nb_values):
        ms, vs = list(measures_possibles.keys())[i], list(measures_possibles.values())[i]
        for v in vs:
            if len(combinations) > max_scenarios:
                return

            last_c = last_combinations.copy()
            last_c.append(ms)
            last_v =  last_values.copy()
            last_v.append(v)

            if i+1<nb_values:
                iter(i + 1, last_c, last_v)
            else:
                if len(last_c)==nb_values:
                    print([last_c], [last_v])
                    df = run(measures=[last_c], values=[last_v])
                    df['SimulationCases_ALL'].plot()
                    plt.show()
                    exit()
                    measures = {"inputs": last_c, "values": last_v}
                    combinations.append(last_c)
                    values.append(last_v)
                    json.dump(measures, open("scenarios/{}.json".format(len(combinations)), "w"))
                    df.to_csv("scenarios/{}.csv".format(len(combinations)))
    return


def make_simulations_on_reborn_inputs():
    iter(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-data", help="Make simulations from train data.", action="store_true")
    parser.add_argument("-g", "--measure-data", help="Make simulations from measures. This will itterate over 50 000 combinaisons and make simulations over them.", action="store_true")
    args = parser.parse_args()

    print(args)
    if(args.train_data or (not args.train_data and not args.measure_data)):
        print('[+] make simulations on train data.')
        export_simulations_on_real_data()

    elif(args.measure_data):
        print('[+] Make simulations over generated user inputs.')
        make_simulations_on_reborn_inputs()
        