from json import load

from joblib.numpy_pickle_compat import load_compatibility
from numpy.lib.type_check import real
from economic_sectors.simulator import EconomicSimulator
from utils import load_luxembourg_dataset
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()



if __name__ == "__main__":


    econ_sector = EconomicSimulator()
    initial_dataset, columns = load_luxembourg_dataset(get_past_rt_as_features=True) 

    real_rt = pd.read_csv('data/luxembourg/luxembourg_allsectors_rt.csv')
    real_rt = real_rt.rename(columns={'Unnamed: 0':'Date'})
    real_rt['Date'] = pd.to_datetime(real_rt['Date'])


    real_data = initial_dataset[econ_sector.metrics["x_columns"]]
    
    X_lift = econ_sector.scaler.transform(real_data)

    
    y_lift = econ_sector.mlp_clf.predict(X_lift)

    Rt = pd.DataFrame(data=y_lift, index=real_data.index, columns=econ_sector.ml_outputs)
    Rt = Rt.reset_index()
    
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=real_rt, x='Date', y='ALL', label='Rt_ALL True')
    sns.lineplot(data=Rt, x='Date', y='ALL', label='Rt_ALL Predicted')
    plt.legend()
    plt.show()

