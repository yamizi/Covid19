from operator import index

from seaborn import palettes
from economic_sectors.simulator import EconomicSimulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()


def run_simple_simulation(Rt, population_total, df):
    simulation = eco_sim.simulate(Rt, population_total, deaths_per_sectors=None, init_date=None, vaccinated_peer_day=None)
        
    merged = pd.merge(simulation["ALL"], simulation["A"], suffixes=["_ALL", "_A"], on="Date")

    for key in simulation.keys():
        if key!="ALL" and key!="A":
            merged = pd.merge(merged, simulation[key], suffixes=["", "_{}".format(key)], on="Date")

    df["Date"] = pd.to_datetime(df["Date"])
    merged_final = pd.merge(merged, df, on="Date")

    return merged_final

if __name__ == '__main__':

    eco_sim = EconomicSimulator()
    df = eco_sim.initial_df
    
    df['population'] = eco_sim.update_population()
    df = df.reset_index()

    print(df.columns.to_list())
    
    columns = eco_sim.metrics["x_columns"]
    for i in range(0, len(columns), 5):
        df[columns[i:i+5]].plot()
        plt.legend()
    
    plt.show()

    X_lift = eco_sim.scaler.transform(df[columns])
    y_lift = eco_sim.mlp_clf.predict(X_lift)
    y_lift = np.clip(y_lift/2,0,10)

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

    # Make a simple dataframe for the polulation 
    population_total = df[['population', 'Date']]

    # Run all simulations 
    simulation_merged = run_simple_simulation(Rt, population_total, df)
    simulation_merged_min = run_simple_simulation(Rt_min, population_total, df)
    simulation_merged_max = run_simple_simulation(Rt_max, population_total, df)

    simulation_merged_max = simulation_merged_max.drop(columns=['Date'])

    simulation_merged = pd.merge(simulation_merged, simulation_merged_max, suffixes=['','_max'], 
                                 left_index=True, right_index=True)
    simulation_merged = pd.merge(simulation_merged, simulation_merged_min, suffixes=['','_min'], 
                                 left_index=True, right_index=True)

    print(simulation_merged)

    simulation_merged.to_csv('datasets/simulation_merged.csv')
    
        