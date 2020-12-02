import os
import json
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import hashlib
import re

from flask_cors import CORS
from economic_sectors.simulator import EconomicSimulator

app = Flask(__name__, static_url_path="")
CORS(app)


"""
possibile_inputs = {
          "b_be": ["open", "close"],
          "b_fr": ["open", "close"],
          "b_de": ["open", "close"],
          "schools_m" : ["open", "partial", "preventive_measure", "close"],
          "public_gath":["yes", "no"],
          "social_dist": ["yes", "no"],
          "resp_gov_measure": ["yes", "no"],
          "private_gath":[1000,0,5,10,20],
          "parks_m":["yes","no"],
          "travel_m":["yes", "no"],
          "activity_restr":["open", "close", "mixed"]
        }
"""


@app.route('/predict', methods=['POST'])
def predict():
    """
    Will launch the simulation taking into account the economical indicdes.
    """
    print('[+] A simulation begins for Luxembourg')
    posted_data = request.json

    measures = posted_data['measures']
    dates = posted_data['dates']
    values = posted_data['values']

    if('date_end' in posted_data.keys()):
        end_date = posted_data['date_end']
    else:
        end_date = "2020-12-15"

    simulator = EconomicSimulator()

    simulation_results = simulator.run(dates, measures, values, end_date)

    columuns_to_keep = ['SimulationCases_ALL', 'SimulationHospital_ALL', 
                        'SimulationCritical_ALL', 'SimulationDeaths_ALL', 
                        'SimulationInfectious_ALL', 'R_ALL',
                        'inflation', 'ipcn',  'unemploy', 'export']

    filetered_simulation_results = simulation_results[columuns_to_keep].reset_index()

    return jsonify({'df': filetered_simulation_results.to_dict(orient='records')})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return 'You want path: %s' % path


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
