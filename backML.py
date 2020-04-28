import os , json 
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import hashlib

from simulations import simulate

app = Flask(__name__)
@app.route('/predict', methods=['POST'])

def predict():
     base_folder="./plots/simulations"

     json_ = request.json

     a = json.dumps(json_, sort_keys = True).encode("utf-8")
     seed = hashlib.md5(a).hexdigest()

     path = "{}/{}".format(base_folder,seed)
     exists = os.path.exists(path)
     nb_files = len([name for name in os.listdir(path) if os.path.isfile(name)]) if exists else 0

     if nb_files == 6 :
          measures_to_lift = [json_.get("measures")]
          measure_values=json_.get("values")
          dates=json_.get("dates")
          measure_dates = [pd.to_datetime(d) for d in dates]
          country_name = json_.get("country_name")
          country_df = merged[merged["CountryName"]==country_name]

          end_date = pd.to_datetime("2020-9-11")
          simulate(country_df, measures_to_lift,0,end_date,None,columns,yvar, mlp_clf, scaler,measure_values=measure_values,base_folder=base_folder, seed=seed, lift_date_values=measure_dates)

     return jsonify({'path': seed})
     
if __name__ == '__main__':
     #clf = joblib.load('modelRt.pkl')
     from seir_hcd_simulations import scaler, yvar, merged, mlp_clf, columns 
     app.run(port=8080)
