import os , json , random, sys, utils
import pandas as pd
import numpy as np

from datetime import date
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.externals import joblib
from model_taining import train_mlp
from model_run import simulate, update_seir

from scipy.stats import wilcoxon
sys.path.append('./')

import matplotlib.pyplot as plt

END_FIT_DATE = pd.to_datetime(date(2020,4,23))
END_EVALUATION_DATE = pd.to_datetime(date(2020,4,30))

def r2_rmse( g, predicted, actual ):
    r2 = r2_score( g[actual], g[predicted] )
    rmse = np.sqrt( mean_squared_error( g[actual], g[predicted] ) )
    return pd.Series( dict(  r2 = r2, rmse = rmse ) )


def train_model():
    if not os.path.exists('models/featues_rmse.csv') or not os.path.exists('models/featues_rmse.csv'):
        data = utils.load_dataset()
        data.loc['Date' <= END_FIT_DATE]
        train_mlp(data, output_suffix='rmse')
    

def get_features():
    features = pd.read_csv('models/features_random.csv', parse_dates=['Date'])

    with open('models/metrics_random.json') as fp:
        y_var = np.power(json.load(fp)['std_test'],0.5)

    return features, y_var


def compute_regression_prediction(data, country):
    country_seir = data.loc[data['CountryName'] == country]
    country_seir = country_seir[country_seir['Date'] <= END_EVALUATION_DATE]
    country_seir['R_min'] = np.clip(country_seir['R'] - yvar.mean()/2, 0, 10)
    country_seir['R_max'] = np.clip(country_seir['R'] + yvar.mean()/2, 0, 10)

    return country_seir


def compute_mlp_prediction(data, country):
    train_model()
    return update_seir(data, END_FIT_DATE, END_EVALUATION_DATE, None)


def compute_rmse(data):
    faulty = 0
    columns = ['Date', 'CountryName', 'ConfirmedCases_x', 'ConfirmedCases_y', 'SimulationCases','SimulationCases_max', 'SimulationCases_min']
    predictions = pd.DataFrame([], columns=columns)
    
    for i, country in enumerate(data['CountryName'].unique()):
        try:
            print('Compute rmse values for {}'.format(country))

            prediction_regression = compute_regression_prediction(data, country)
            prediction_mlp = compute_mlp_prediction(data, country)

            prediction_country = pd.merge(prediction_regression[['Date','CountryName','ConfirmedCases_x','ConfirmedCases_y']],
                                prediction_mlp[['Date','SimulationCases','SimulationCases_max','SimulationCases_min']], on='Date')

            predictions = predictions.append(prediction_country)
        except Exception:
            print('not enough data')
            faulty = faulty+1
    
    rmse_seir_regression_cases = predictions.groupby( 'CountryName' ).apply( r2_rmse, 'ConfirmedCases_x','ConfirmedCases_y').reset_index()
    rmse_seir_ml_cases = predictions.groupby( 'CountryName' ).apply( r2_rmse, 'SimulationCases','ConfirmedCases_y').reset_index()
    rmse_seir_ml_cases_min = predictions.groupby( 'CountryName' ).apply( r2_rmse, 'SimulationCases_min','ConfirmedCases_y').reset_index()
    rmse_seir_ml_cases_max = predictions.groupby( 'CountryName' ).apply( r2_rmse, 'SimulationCases_max','ConfirmedCases_y').reset_index()

    rmse = pd.merge(rmse_seir_regression_cases, rmse_seir_ml_cases, on='CountryName', suffixes=['_seir','_seir_ml'])
    rmse = pd.merge(rmse, rmse_seir_ml_cases_min, on='CountryName')
    rmse = pd.merge(rmse, rmse_seir_ml_cases_max, on='CountryName')

    rmse = rmse.rename(columns={'r2_x':'r2_seir_min','r2_y':'r2_seir_max','rmse_x':'rmse_seir_min','rmse_y':'rmse_seir_max'})

    return rmse, faulty


def output_rmse(rmse):
    rmse_plots = rmse[['rmse_seir','rmse_seir_ml','rmse_seir_min','rmse_seir_max']]
    rmse_plots.columns = ['SEIR', 'DN-SEIR', 'DN-SEIR min', 'DN-SEIR max']

    full_plots = rmse[['CountryName', 'rmse_seir','rmse_seir_ml','rmse_seir_min','rmse_seir_max']]
    full_plots.columns = ['CountryName', 'SEIR', 'DN-SEIR', 'DN-SEIR min', 'DN-SEIR max']

    full_plots.to_csv('./rmse.csv')

    x = full_plots['SEIR']
    y = full_plots['DN-SEIR']
    rank_test = wilcoxon(x,y)
    print(rank_test)

    rmse_plots.boxplot(figsize=(20,15),fontsize=14)
    plt.show()