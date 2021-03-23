from economic_sectors.simulator import EconomicSimulator
from seir_hcd_simulations import scaler, yvar, merged, mlp_clf, columns
import os
import json
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import hashlib
import re
from datetime import datetime, timedelta

from flask_cors import CORS
from simulations import simulate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)

base_folder = "./plots/simulations"


app = Flask(__name__, static_url_path="")
CORS(app)


DATE_PAST_SHIFT = 15
SEIR_SHIFT = 14
DATE_FUTURE_SHIFT = 30

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json

    a = json.dumps(json_, sort_keys=True).encode("utf-8")
    seed = hashlib.md5(a).hexdigest()
    df = {}
    path = "{}/{}".format(base_folder, seed)
    exists = os.path.exists(path)
    nb_files = len([name for name in os.listdir(
        path) if os.path.isfile(name)]) if exists else 0

    if True or nb_files != 6:  # disable cache
        measures_to_lift = [json_.get("measures")]
        measure_values = json_.get("values")
        dates = json_.get("dates")
        measure_dates = [pd.to_datetime(d) for d in dates]
        country_name = json_.get("country_name")
        country_df = merged[merged["CountryName"] == country_name]

        end_date = pd.to_datetime("2020-9-11")
        df = simulate(country_df, measures_to_lift, 0, end_date, None, columns, yvar, mlp_clf, scaler,
                      measure_values=measure_values, base_folder=None, seed=seed, lift_date_values=measure_dates)
        df = df.to_dict(orient='records')

    # print("processed")
    return jsonify({'path': seed, 'df': df})


@app.route('/reborn_api_limit', methods=['POST'])
def get_limit_date():
    """Beacause the dataset have some limits in terms of time,
    Prediction will not be accurate... To solve this problem we must 
    add a maximum date for the user interface...

    Returns:
        Object: An object that contains the maximum date the user can ask.
    """
    simulator = EconomicSimulator()
    min_date, max_date = simulator.get_limit_dates() 

    if max_date.year > min_date.year:
        print("on est bonnnnn")
        min_date =  datetime(max_date.year, 1, 1).date()
    else:
        min_date = min_date + timedelta(days=DATE_PAST_SHIFT) + timedelta(days=SEIR_SHIFT)



    print(min_date, max_date)
    print(type(min_date), type(max_date))

    return jsonify({'min_date': min_date, 
                    'max_date': max_date })



@app.route('/predict_reborn', methods=['POST'])
def predict_reborn():
    """Will launch the simulation taking into account the economical indicdes.

    Returns:
        String: Json object ready to be parsed by the client.
    """
    posted_data = request.json
    # posted_data = {
    # 'country_name': 'Luxembourg', 
    # 'measures': [['b_be', 'b_fr', 'b_de', 'schools_m', 'public_gath', 'private_gath', 'parks_m', 'travel_m', 'activity_restr', 'resp_gov_measure', 'vaccinated_peer_week']], #
    # 'dates': ['2020-08-03'],
    # 'values':  [['open', 'open', 'open', 'open', 'yes', 1000, 'yes', 'yes', 'open', 'yes', 0]] 
    # }

    measures = posted_data['measures']
    dates = posted_data['dates']
    values = posted_data['values']

    # If the user enter no measure in inputs,
    # We should make predictions on the worst case.
    if len(measures[0]) == 0:
        measures = [['b_be', 'b_fr', 'b_de', 'schools_m', 'public_gath', 'private_gath',
                     'parks_m', 'travel_m', 'activity_restr', 'resp_gov_measure', 'vaccinated_peer_week']]
        values = [['open', 'open', 'open', 'open', 'yes', 1000,
                   'yes', 'yes', 'open', 'yes', 0]]
        dates = None

    if('date_end' in posted_data.keys()):
        end_date = posted_data['date_end']
    elif(dates is not None):
        end_date = datetime.strptime(dates[0], '%Y-%m-%d')
        init_date = end_date - timedelta(days=DATE_PAST_SHIFT) 
        end_date = end_date + timedelta(days=DATE_FUTURE_SHIFT) 
    else:
        end_date = datetime.now()
        init_date = datetime.now() - timedelta(days=DATE_PAST_SHIFT) 
        end_date = end_date + timedelta(days=DATE_FUTURE_SHIFT) 

    end_date = end_date.strftime('%Y-%m-%d')
    init_date = init_date.strftime('%Y-%m-%d')

    # print('[+] informations')
    # print(posted_data)
    # print(measures)
    # print(values)
    # print(init_date)
    # print(end_date)
    # exit()

    simulator = EconomicSimulator()
    simulation_results = simulator.run(dates, measures, values, end_date, init_date=init_date)
    simulation_results['objective'] = 1
    simulation_results = simulation_results.drop(columns=['Date'])

    # convert_code_to_real_name(simulation_results)

    # print(simulation_results.columns.to_list())
    # plt.figure()
    # simulation_results['R_A'].plot()
    # plt.legend()

    # plt.figure()
    # simulation_results['SimulationCases_A'].plot()
    # plt.legend()
    # plt.show()
    
    return jsonify({'df': simulation_results.reset_index().to_dict(orient='records')})
     

@app.route('/sims/rate/<path:sim_id>')
def send_reproduction_plots(sim_id):
    sim_id = re.sub('[\W_]+', '', sim_id)
    return send_from_directory("{}/{}".format(base_folder, sim_id), "reproduction_rate.png")


@app.route('/sims/case/<path:sim_id>')
def send_cases_plots(sim_id):
    sim_id = re.sub('[\W_]+', '', sim_id)
    return send_from_directory("{}/{}".format(base_folder, sim_id), "cases.png")


@app.route('/sims/hospital/<path:sim_id>')
def send_hospitals_plots(sim_id):
    sim_id = re.sub('[\W_]+', '', sim_id)
    return send_from_directory("{}/{}".format(base_folder, sim_id), "hospitals.png")


@app.route('/sims/critical/<path:sim_id>')
def send_scriticals_plots(sim_id):
    sim_id = re.sub('[\W_]+', '', sim_id)
    return send_from_directory("{}/{}".format(base_folder, sim_id), "criticals.png")


@app.route('/sims/death/<path:sim_id>')
def send_deaths_plots(sim_id):
    sim_id = re.sub('[\W_]+', '', sim_id)
    return send_from_directory("{}/{}".format(base_folder, sim_id), "deaths.png")


@app.route('/sims/csv/<path:sim_id>')
def send_csv(sim_id):
    sim_id = re.sub('[\W_]+', '', sim_id)
    return send_from_directory("{}/{}".format(base_folder, sim_id), "out.csv")


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return 'You want path: %s' % path


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
    # predict_reborn()
    # get_limit_date()