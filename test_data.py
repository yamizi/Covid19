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
    initial_dataset, columns = load_luxembourg_dataset() 

    cases = pd.read_excel('C:/snt/test reborn/PJG397_Covid19_daily_workSector_Residents_IGSS.xlsx',  'Daily_Infections_by_Sector')

    cases = cases.rename(columns={'Dates\Age Range': 'Date'})
    cases['Date'] = pd.to_datetime(cases['Date'])
    cases = cases.set_index(['Date'])
    
    cases['ALL'] = cases.sum(axis=1)
    cases = cases.reset_index()

    cases1 = cases[cases['Date'] > pd.to_datetime('2020-08-07')]
    cases1 = cases1[cases1['Date'] < pd.to_datetime('2020-09-07')]

    cases2 = cases[cases['Date'] > pd.to_datetime('2020-07-07')]
    cases2 = cases2[cases2['Date'] < pd.to_datetime('2020-08-07')]

    plt.figure(figsize=(10, 8))
    sns.lineplot(data=cases1, x='Date', y='ALL', label='Cases1')
    sns.lineplot(data=cases2, x='Date', y='ALL', label='Cases2')
    plt.legend()
    plt.show()

