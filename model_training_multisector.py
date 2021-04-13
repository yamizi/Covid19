import json, os, time, utils, joblib
from copy import deepcopy
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt 


def split_data(data, by="random", on=None, feature_columns=None, target_columns = 'R'):
    feature_columns = utils.get_feature_columns() if feature_columns is None else feature_columns

    if by == 'country':
        country_names = on
        country = data[data["CountryName"].isin(country_names)]
        non_country = data[~data["CountryName"].isin(country_names)]

        X_train, y_train = non_country[feature_columns], non_country["R"]
        X_test, y_test = country[feature_columns], country["R"]
    elif by == 'random':
        X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data[target_columns], test_size=0.2)

    return X_train, y_train, X_test, y_test


def scale_data(X_train, X_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled


def find_best_model(X_train, y_train, X_test, y_test):
    best_perf = -1
    best_model = None

    for i in range(5):
        print("Iter search {}".format(i))
        parameter_space = {
            'hidden_layer_sizes': [(1000,50),], #[(1000,50),(50, 100, 50), (50, 100, 100), (50, 500, 50)],
            'alpha': [0.0001, 0.05]
        }


        mlp = MLPRegressor((1000,50),max_iter=1500, verbose=True, solver="adam")
        mlp_clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=True)
        mlp_clf.fit(X_train,y_train.values)

        reports, __ = utils.metrics_report(X_test, y_test, mlp_clf), utils.metrics_report(X_train, y_train, mlp_clf)

        if reports["r2_score"] > best_perf:
            best_model = deepcopy(mlp_clf.best_estimator_ if mlp!=mlp_clf else mlp)
            best_perf =  reports["r2_score"]

    return best_model, reports


def find_best_bayesian_ridge(X_train, y_train):
    reg = BayesianRidge(compute_score=True, tol=1e-5)
    parameters = {'alpha_init':(0.2, 0.5, 1, 1.5), 'lambda_init':[1e-3, 1e-4, 1e-5,1e-6]}
    srch = GridSearchCV(reg, parameters)
    srch.fit(X_train, y_train)
    params = srch.get_params()

    reg.set_params(alpha_init=params["estimator__alpha_init"], lambda_init=params["estimator__lambda_init"])
    reg.fit(X_train, y_train)

    return reg, params


def get_output_name(folder, name, suffix, extension):
    return '{}/{}.{}'.format(folder, name, extension) if suffix == '' else '{}/{}_{}.{}'.format(folder, name, suffix, extension)


def save_model(model, reports, std_peer_features, scaler, x_columns, data, suffix):
    folder = './models'
    os.makedirs(folder, exist_ok=True)

    joblib.dump(model, get_output_name(folder, 'mlp', suffix, 'save'))
    joblib.dump(scaler, get_output_name(folder, 'scaler', suffix, 'save'))

    with open(get_output_name(folder, 'metrics', suffix, 'json'), 'w') as fp:
        json.dump({'perf':reports, 'std_test': std_peer_features, 'x_columns':x_columns, 'hidden_layer_sizes': model.hidden_layer_sizes}, fp)

    data.to_csv(get_output_name(folder, 'features', suffix, 'csv'))


def train_mlp(data, target_columns, split_by='random', split_on=None, output_suffix=''):

    all_columns = data.columns
    feature_columns = [x for x in all_columns if x not in target_columns]

    X_train, y_train, X_test, y_test = split_data(data, by=split_by, on=split_on, feature_columns=feature_columns, 
                                                  target_columns = target_columns)
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model, reports = find_best_model(X_train_scaled, y_train, X_test_scaled, y_test)


    print('[+] Compute y_std for each features')
    features_std = {}
    for target_column in tqdm(target_columns):
        reg, _ = find_best_bayesian_ridge(X_train, y_train['ALL'])
        _, y_std = reg.predict(X_test, return_std=True)
        features_std[target_column] = list(y_std.values)

    output_suffix = split_by if output_suffix == '' else output_suffix
    save_model(model, reports, features_std, scaler, list(X_train.columns), data, output_suffix)

    return model, reports


if __name__ == '__main__':
    t_start = time.perf_counter()
    data, y = utils.load_luxembourg_dataset(get_past_rt_as_features=True)

    model, reports = train_mlp(data, y, output_suffix="economic_sectors")

    print(reports)

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------") 
