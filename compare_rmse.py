import pandas as pd
import numpy as np
import os , json , random, sys
from datetime import date
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.externals import joblib
from model_run import simulate, update_seir

from scipy.stats import wilcoxon
sys.path.append('./')

import matplotlib.pyplot as plt

#the SEIR is fitted for 2 months. starting from 2020-02-24
end_fit_date = pd.to_datetime(date(2020,4,23))

#We have mobility data until 2020-05-04
end_evaluation_date = pd.to_datetime(date(2020,4,30))

def r2_rmse( g, predicted, actual ):
    r2 = r2_score( g[actual], g[predicted] )
    rmse = np.sqrt( mean_squared_error( g[actual], g[predicted] ) )
    return pd.Series( dict(  r2 = r2, rmse = rmse ) )


def get_mlp_data():
    # I can call the train and simulate with the right data
    if not os.path.exists('models/featues_rmse.csv'):
        # I wanted to do a loc on the data to only have the date up to end_fit_date
        pass
    
    #this one is trained on the entire dataset but what is supposed to be here?
    #the thing is that we do not simulate later for the remaining days
    merged = pd.read_csv('models/features_random.csv', parse_dates=['Date'])

    with open('models/metrics_random.json') as fp:
        metrics = json.load(fp)
        yvar = np.power(metrics['std_test'],0.5)


    return merged, yvar

merged, yvar = get_mlp_data()

countries = merged['CountryName'].unique()

all_countries = pd.DataFrame([],
                             columns=['Date', 'CountryName', 'ConfirmedCases_x', 'ConfirmedCases_y', 'SimulationCases',
                                      'SimulationCases_max', 'SimulationCases_min'])

print('nb_countries',len(countries))
faulty = 0
for i, country_name in enumerate(countries):

    try:
        print('country', country_name)
        country_seir = merged[merged['CountryName'] == country_name]
        country_seir = country_seir[country_seir['Date']<=end_evaluation_date]
        country_seir['R_min'] = np.clip(country_seir['R'] - yvar.mean()/2, 0, 10)
        country_seir['R_max'] = np.clip(country_seir['R'] + yvar.mean()/2, 0, 10)
        
        #here I thought we would have to call simulate for the remaining days
        country_ml = update_seir(country_seir, end_fit_date, end_evaluation_date, None)

        country_df = pd.merge(country_seir[['Date','CountryName','ConfirmedCases_x','ConfirmedCases_y']],
                              country_ml[['Date','SimulationCases','SimulationCases_max','SimulationCases_min']], on='Date')

        all_countries = all_countries.append(country_df)
    except Exception as e:
        print('not enough data')
        faulty = faulty+1

print('nb_faulty',faulty)

metrics_seir_cases = all_countries.groupby( 'CountryName' ).apply( r2_rmse, 'ConfirmedCases_x','ConfirmedCases_y').reset_index()
metrics_seir_ml_cases = all_countries.groupby( 'CountryName' ).apply( r2_rmse, 'SimulationCases','ConfirmedCases_y').reset_index()
metrics_seir_ml_cases_min = all_countries.groupby( 'CountryName' ).apply( r2_rmse, 'SimulationCases_min','ConfirmedCases_y').reset_index()
metrics_seir_ml_cases_max = all_countries.groupby( 'CountryName' ).apply( r2_rmse, 'SimulationCases_max','ConfirmedCases_y').reset_index()

concat = pd.merge(metrics_seir_cases,metrics_seir_ml_cases, on='CountryName', suffixes=['_seir','_seir_ml'])
concat = pd.merge(concat,metrics_seir_ml_cases_min, on='CountryName')
concat = pd.merge(concat,metrics_seir_ml_cases_max, on='CountryName')

concat = concat.rename(columns={'r2_x':'r2_seir_min','r2_y':'r2_seir_max','rmse_x':'rmse_seir_min','rmse_y':'rmse_seir_max'})

rmse_plots = concat[['rmse_seir','rmse_seir_ml','rmse_seir_min','rmse_seir_max']]
rmse_plots.columns = ['SEIR', 'DN-SEIR', 'DN-SEIR min', 'DN-SEIR max']

full_plots = concat[['CountryName', 'rmse_seir','rmse_seir_ml','rmse_seir_min','rmse_seir_max']]
full_plots.columns = ['CountryName', 'SEIR', 'DN-SEIR', 'DN-SEIR min', 'DN-SEIR max']

full_plots.to_csv('./rmse.csv')

x = full_plots['SEIR']
y = full_plots['DN-SEIR']
rank_test = wilcoxon(x,y)
print(rank_test)

rmse_plots.boxplot(figsize=(20,15),fontsize=14)
plt.show()