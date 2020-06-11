# Module imports

import joblib
import shap
import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
from datetime import date
from sklearn import preprocessing
from simulations import features, periods
from scipy.integrate import solve_ivp
from utils import load_dataset, get_feature_columns

# Import model and scaler
scaler = joblib.load('./models/scaler_.save') 
mlp_clf = joblib.load('./models/mlp_.save') 

# Dataset
data = load_dataset()
columns = get_feature_columns()
del data['R']
lux = data[data['CountryName'] == 'Luxembourg']
jap = data[data['CountryName'] == 'Japan']
italy = data[data['CountryName'] == 'Italy']

# Use Sclaer
lux_scale = pd.DataFrame(scaler.transform(lux[columns]), columns=columns)
jap_scale = pd.DataFrame(scaler.transform(jap[columns]), columns=columns)
it_scale = pd.DataFrame(scaler.transform(italy[columns]), columns=columns)


# Generate SHAP Explainer
shap_explainer_lux = shap.KernelExplainer(mlp_clf.predict,lux_scale)
shap_values_lux = shap_explainer_lux.shap_values(lux_scale)

shap_explainer_jap = shap.KernelExplainer(mlp_clf.predict,jap_scale)
shap_values_jap = shap_explainer_jap.shap_values(jap_scale)

shap_explainer_it = shap.KernelExplainer(mlp_clf.predict,jap_scale)
shap_values_it = shap_explainer_it.shap_values(jap_scale)

# Feature importance using SHAP
## Luxembourg
shap.summary_plot(shap_values_lux, lux_scale, show=False)
plt.savefig('shap_summary_lux_test.png')

## Japan
shap.summary_plot(shap_values_jap, jap_scale, show=False)
plt.savefig('shap_summary_jap_test.png')

## Italy
shap.summary_plot(shap_values_it, it_scale, show=False)
plt.savefig('shap_summary_it_test.png')

# Plot the SHAP values for the last observation
## Luxembourg
shap.force_plot(shap_explainer_lux.expected_value,shap_values_lux[-1,:], lux_scale.iloc[-1,:],
               matplotlib = True, show=False)
plt.savefig('lux_local.png', dpi=1000, bbox_inches='tight')

## Japan
shap.force_plot(shap_explainer_jap.expected_value,shap_values_jap[-1,:], jap_scale.iloc[-1,:],
               matplotlib = True, show=False)
plt.savefig('jap_local.png', dpi=1000, bbox_inches='tight')

## Italy
shap.force_plot(shap_explainer_it.expected_value,shap_values_it[-1,:], it_scale.iloc[-1,:],
               matplotlib = True, show=False)
plt.savefig('it_local.png', dpi=1000, bbox_inches='tight')
