# -*- coding: utf-8 -*-
"""seir-hcd-model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dBYFOx5BVeFjPb3fugPeWrCV0CjWphc4
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from pathlib import Path
import json, os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from datetime import date
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

from helpers import metrics_report

country_name = "Luxembourg"


"""# Fitting the model to data
There are certain variables that we can play with to fit the model to real data:
* Average incubation period, `t_inc`
* Average infection period, `t_inf`
* Average hospitalization period, `t_hosp`
* Average critital period, `t_crit`
* The fraction of mild/asymptomatic cases, `m_a`
* The fraction of severe cases that turn critical, `c_a`
* The fraction of critical cases that result in a fatality, `f_a`
* Reproduction number, `R_0` or `R_t`
"""

current_dataset_date = date(2020,4,23).strftime("%Y_%m_%d")
dataset= pd.read_csv("./datasets/{}_google.csv".format(current_dataset_date), parse_dates=['Date'])
dataset = dataset.drop(["Unnamed: 0"],axis=1)

dataset.tail(1)

"""## ML to predict Reproduction Rate"""

#current_dataset_date = date(2020,4,15).strftime("%Y_%m_%d")
all_countries= pd.read_csv("./datasets/{}_seirhcd.csv".format(current_dataset_date), parse_dates=['Date'])

oxford_raw = pd.read_excel("./datasets/OxCGRT_Download_latest_data.xlsx", sep=';')
oxford_raw['Date'] = pd.to_datetime(oxford_raw['Date'], format='%Y%m%d')
oxford_raw = oxford_raw[["Date","CountryName","S1_School closing","S3_Cancel public events","S7_International travel controls"]]
oxford_raw[["S1_School closing","S3_Cancel public events","S7_International travel controls"]] = oxford_raw[["S1_School closing","S3_Cancel public events","S7_International travel controls"]].fillna(method="ffill").fillna(method="bfill") 
oxford_raw[["S1_School closing","S3_Cancel public events"]] = oxford_raw[["S1_School closing","S3_Cancel public events"]] * -50
oxford_raw[["S7_International travel controls"]] = oxford_raw[["S7_International travel controls"]] * -33
oxford_raw[["S1_School closing","S3_Cancel public events","S7_International travel controls"]] = oxford_raw[["S1_School closing","S3_Cancel public events","S7_International travel controls"]].rolling(15,14).mean()

metrics = "google"
if metrics=="oxford":
  
  oxford_dataset= pd.read_csv("./{}_oxford.csv".format(date(2020,4,8).strftime("datasets/%Y_%m_%d")), parse_dates=['Date'])
  oxford_dataset = oxford_dataset[oxford_dataset["ConfirmedCases"]>0]
  columns_start_label = "Days_since_S1_School closing_1.0"
  columns_end_label ="%ConfirmedCases"
  ref_dataset = oxford_dataset.copy()
else:
  columns_start_label = "grocery/pharmacy_15days"
  columns_end_label ="Days_since_Peak_RateCases"
  ref_dataset = dataset.copy()

merged = pd.merge(all_countries, ref_dataset, on=["CountryName","Date"], how="left")

if metrics=="google": ## add oxford measures dataset
  merged = pd.merge(merged, oxford_raw, on=["CountryName","Date"], how="left")
  merged = merged.fillna(method="ffill")
  pass

merged = merged.dropna()
merged = merged[merged["Date"]<pd.to_datetime("2020-04-12",yearfirst=True)] # last date for google Data
#merged = merged.iloc[7:,:] # skip first week to have a relevance on the rolling features

print(merged.head(1),merged.tail(1))

merged_columns = list(merged.columns)
columns_start = merged_columns.index(columns_start_label) #merged_columns.index("grocery/pharmacy_15days")
columns_end = merged_columns.index(columns_end_label)#columns_start+18
columns = merged_columns[columns_start:columns_end]


#columns = columns + ["density","population","population_p65","population_p14","Tests","gdp","area","region"]
#columns = [ 'retail/recreation_15days', 'retail/recreation_10days', 'retail/recreation_5days']
#columns = columns + [ 'workplace_15days', 'workplace_10days', 'workplace_5days']
#columns = columns + ['transit_stations_15days', 'transit_stations_10days', 'transit_stations_5days']
columns = columns + ["density","population","population_p65","population_p14","gdp","area"]
columns = columns + ["S1_School closing",	"S7_International travel controls"] #"S3_Cancel public events",

country_names = ["Luxembourg","France","Germany","Spain","United kingdom","Greece","Italy","Switzerland","Latvia","Belgium","Netherlands"]
country = merged[merged["CountryName"].isin(country_names)]
non_country = merged[~merged["CountryName"].isin(country_names)]
non_country = merged

all_countries = merged["CountryName"].unique()
print(all_countries)
print(country["CountryName"].unique())
#country[['workplace_15days', 'workplace_10days', 'workplace_5days']] = country[['workplace_15days', 'workplace_10days', 'workplace_5days']] * 1.5
country[['Date','workplace']].tail(1)

X_train, y_train = non_country[columns], non_country["R"]
X_test, y_test = country[columns], country["R"]

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test) 


from copy import deepcopy
nb_iters = 10
best_perf = -1
best_model = None
for i in range(nb_iters):


    mlp = MLPRegressor((1000,50),max_iter=1500, verbose=False, solver="adam")
    mlp_clf = mlp #GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    mlp_clf.fit(X_train_scaled,y_train.values)

    """
    parameter_space = {
        'hidden_layer_sizes': [(50,100,50),(50,100,100), (50,500,50)],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }


    mlp_clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    mlp_clf.fit(X_train_scaled,y_train.values)
    means = mlp_clf.cv_results_['mean_test_score']
    stds = mlp_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, mlp_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    """
    reports, reports_train = metrics_report(X_test_scaled, y_test, mlp_clf),metrics_report(X_train_scaled, y_train, mlp_clf)
    print("ref",best_perf,reports)

    if reports["r2_score"]>best_perf:
        best_model = deepcopy(mlp_clf)
        best_perf =  reports["r2_score"]

    if reports["r2_score"]>0.8:
        break


X_train.shape, X_test_scaled.shape

y_mlp = best_model.predict(X_test_scaled)

reg = BayesianRidge(compute_score=True, tol=1e-5)

parameters = {'alpha_init':(0.2, 0.5, 1, 1.5), 'lambda_init':[1e-3, 1e-4, 1e-5,1e-6]}
srch = GridSearchCV(reg, parameters)

srch.fit(X_train, y_train)

params = srch.get_params()
reg.set_params(alpha_init=params["estimator__alpha_init"], lambda_init=params["estimator__lambda_init"]) 
reg.fit(X_train, y_train)
ymean, ystd = reg.predict(X_test, return_std=True)

folder = "./models/seirhcd/{}".format(current_dataset_date)
os.makedirs(folder, exist_ok=True)

joblib.dump(best_model, '{}/mlp.save'.format(folder))
joblib.dump(scaler, "{}/scaler.save".format(folder)) 

with open('{}/metrics.json'.format(folder), 'w') as fp:
    json.dump({"perf":reports,"std_test":list(ystd.values), "columns":columns, "countries":list(all_countries)}, fp)

merged.to_csv('{}/features.csv'.format(folder))


yvar =np.sqrt(ystd)
#yvar = ystd

plt.figure(figsize=(30,10))

plt.plot(np.arange(len(X_test)), y_mlp, color="red", label="predict mlp")
plt.plot(np.arange(len(X_test)), y_test, color="blue", label="ground truth")

plt.fill_between(np.arange(len(X_test)), y_mlp-yvar/2, y_mlp+yvar/2,
                color="pink", alpha=0.5, label="Confidence interval")
plt.tight_layout()
plt.legend()

plt.xticks(np.arange(len(X_test)),country["Date"].dt.strftime('%d/%m/%Y'), rotation=90)
plt.show()