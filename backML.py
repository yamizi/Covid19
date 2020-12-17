import os
import json
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import hashlib
import re
from datetime import datetime

from flask_cors import CORS
from economic_sectors.simulator import EconomicSimulator

app = Flask(__name__, static_url_path="")
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """Will launch the simulation taking into account the economical indicdes.

    Returns:
        String: Json object ready to be parsed by the client.
    """


    start = datetime.now()
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

    columuns_to_keep = ['SimulationCases_ALL', 'SimulationCases_ALL_min', 'SimulationCases_ALL_max',
                        'SimulationHospital_ALL', 'SimulationHospital_ALL_min', 'SimulationHospital_ALL_max', 
                        'SimulationCritical_ALL', 'SimulationCritical_ALL_min', 'SimulationCritical_ALL_max', 
                        'SimulationDeaths_ALL', 'SimulationDeaths_ALL_min', 'SimulationDeaths_ALL_max', 
                        'R_ALL', 'R_ALL_min', 'R_ALL_max',
                        'inflation', 'inflation_min', 'inflation_max',
                        'ipcn',  'ipcn_min', 'ipcn_max',
                        'unemploy', 'unemploy_min', 'unemploy_max',
                        'export', 'export_min', 'export_max']


    filetered_simulation_results = simulation_results[columuns_to_keep]

    end = datetime.now()

    print('[+] Execution Time:', end-start)


    return jsonify({'df': filetered_simulation_results.reset_index().to_dict(orient='records')})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return 'You want path: %s' % path


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
