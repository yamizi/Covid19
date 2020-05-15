import os, io, json
import numpy as np
import pandas as pd
from flask import request, jsonify, Flask
from google.cloud import storage
from sklearn.externals import joblib
import hashlib
import re
import logging
from flask_cors import CORS
from simulations import simulate

GCS_BUCKET = os.environ['GCS_BUCKET']
GCS_MODEL = os.environ['GCS_MODEL']
GCS_SCALER = os.environ['GCS_SCALER']
GCS_METRICS = os.environ['GCS_METRICS']
GCS_FEATURES= os.environ['GCS_FEATURES']

base_folder="./simulations"

app = Flask(__name__)
CORS(app)

@app.before_first_request
def _load_model():
    global  scaler, yvar, merged, mlp_clf, columns

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob_model = bucket.blob(GCS_MODEL)
    blob_scaler = bucket.blob(GCS_SCALER)
    blob_features = bucket.blob(GCS_FEATURES)
    blob_metrics = bucket.blob(GCS_METRICS)

    if blob_model.exists():
        f = io.BytesIO()
        blob_model.download_to_file(f)
        mlp_clf = joblib.load(f)

        f = io.BytesIO()
        blob_scaler.download_to_file(f)
        scaler = joblib.load(f)

        #blob_features.download_to_file(f)
        merged = pd.read_csv("gs://{}/{}".format(GCS_BUCKET,GCS_FEATURES), parse_dates=["Date"])
        
        metrics = json.loads(blob_metrics.download_as_string(client=None))

        yvar = np.power(metrics["std_test"], 0.5)
        columns = metrics["columns"]

    else:
        mlp_clf , scaler , columns , yvar , columns = None, None,None, None,None


@app.route('/predict', methods=['POST'])
def predict():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    json_ = request.json

    a = json.dumps(json_, sort_keys=True).encode("utf-8")
    seed = hashlib.md5(a).hexdigest()
    df = {}
    path = "{}/{}".format(base_folder, seed)

    blob = bucket.blob(path)
    exists = blob.exists()

    if True or not exists:
        measures_to_lift = [json_.get("measures")]
        measure_values = json_.get("values")
        dates = json_.get("dates")
        measure_dates = [pd.to_datetime(d) for d in dates]
        country_name = json_.get("country_name")
        country_df = merged[merged["CountryName"] == country_name]

        end_date = pd.to_datetime("2020-12-11")
        df = simulate(country_df, measures_to_lift, 0, end_date, None, columns, yvar, mlp_clf, scaler,
                      measure_values=measure_values, base_folder=None, seed=seed, lift_date_values=measure_dates)
        df = df.to_dict(orient='records')

    # print("processed")
    return jsonify({'path': seed, 'df': df})


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
