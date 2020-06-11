import joblib
import shap
import tensorflow
import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
from datetime import date
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from helpers import metrics_report
from simulations import features, periods
from scipy.integrate import solve_ivp


# Import model and scaler

current_dataset_date = "v2_1"
folder = "./models/seirhcd/{}".format(current_dataset_date)
scaler = joblib.load("{}/scaler.save".format(folder))
mlp_clf = joblib.load("{}/mlp.save".format(folder))

# Dataset
current_dataset_date = date(2020,5,10).strftime("%Y_%m_%d")

# Change here to get the file
dataset= pd.read_csv("v2_1_google.csv", parse_dates=['Date'])
dataset = dataset.drop(["Unnamed: 0"],axis=1)
current_dataset_date = "v2_1"
all_countries= pd.read_csv("v2_1_seirhcd.csv", parse_dates=['Date'])

dataset = pd.get_dummies(dataset,prefix="day_of_week", columns=["day_of_week"])
dataset = pd.get_dummies(dataset,prefix="region", columns=["region"])
merged = pd.merge(all_countries.groupby(["CountryName","Date"]).agg("first"), dataset.groupby(["CountryName","Date"]).agg("first"),  on=["CountryName","Date"], how="inner")
merged = merged.reset_index().dropna()

merged_columns = list(merged.columns)

columns = ["{}{}".format(f,p) for p in periods for f in features]

columns = columns + ["density","population","population_p65","population_p14","gdp","area"]
columns = columns + ["day_of_week_{}".format(i) for i in range(7)]
columns = columns + ["region_{}".format(i) for i in range(10)]

merged = merged.rename(columns={'region_10':'region_9'})

test_countries = merged[(merged['CountryName'] == 'Luxembourg')|(merged['CountryName'] == 'Japan')|(merged['CountryName'] == 'Italy')]
test_countries.reset_index(inplace=True)
del test_countries['index']

lux = test_countries[test_countries['CountryName'] == 'Luxembourg']
jap = test_countries[test_countries['CountryName'] == 'Japan']
italy = test_countries[test_countries['CountryName'] == 'Italy']

# Put to scale

lux_scale = pd.DataFrame(scaler.transform(lux[columns]), columns=columns)
jap_scale = pd.DataFrame(scaler.transform(jap[columns]), columns=columns)
it_scale = pd.DataFrame(scaler.transform(italy[columns]), columns=columns)

# Use to model to make prediction

y_lift_lux = mlp_clf.predict(lux_scale)
y_lift_jap = mlp_clf.predict(jap_scale)
y_lift_it = mlp_clf.predict(it_scale)

# SHAP

shap_explainer_lux = shap.KernelExplainer(mlp_clf.predict,lux_scale)
shap_values_lux = shap_explainer_lux.shap_values(lux_scale)

shap_explainer_jap = shap.KernelExplainer(mlp_clf.predict,jap_scale)
shap_values_jap = shap_explainer_jap.shap_values(jap_scale)

shap_explainer_it = shap.KernelExplainer(mlp_clf.predict,jap_scale)
shap_values_it = shap_explainer_it.shap_values(jap_scale)

# Feature importance using SHAP
shap.summary_plot(shap_values_lux, lux_scale, show=False)
plt.savefig('shap_summary_lux_test.png')

shap.summary_plot(shap_values_jap, jap_scale, show=False)
plt.savefig('shap_summary_jap_test.png')

shap.summary_plot(shap_values_it, it_scale, show=False)
plt.savefig('shap_summary_it_test.png')

# Plot the SHAP values for the last observation
shap.force_plot(shap_explainer_lux.expected_value,shap_values_lux[-1,:], lux_scale.iloc[-1,:],
               matplotlib = True, show=False)
plt.savefig('lux_local.png', dpi=1000, bbox_inches='tight')

shap.force_plot(shap_explainer_jap.expected_value,shap_values_jap[-1,:], jap_scale.iloc[-1,:],
               matplotlib = True, show=False)
plt.savefig('jap_local.png', dpi=1000, bbox_inches='tight')

shap.force_plot(shap_explainer_it.expected_value,shap_values_it[-1,:], it_scale.iloc[-1,:],
               matplotlib = True, show=False)
plt.savefig('it_local.png', dpi=1000, bbox_inches='tight')
