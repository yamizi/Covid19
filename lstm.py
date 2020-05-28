import pandas as pd
import numpy as np
import re
import random
import math

import helpers

from datetime import date, timedelta

from numpy import concatenate

from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error,median_absolute_error

from keras.models import Sequential
from keras.layers import Dense, LSTM

from matplotlib import pyplot


REGION_COLUMNS = ['region_{}'.format(i) for i in range(11)]
TIME_COLUMNS = ['day_of_week_{}'.format(i) for i in range(7)]
MOBILITY_COLUMNS = ["retail/recreation", "grocery/pharmacy", "parks", "transit_stations", "workplace"]
DEMOGRAPHY_COLUMNS = ["density", "population", "population_p14", "population_p65", "gdp", "area"]


def series_to_supervised(data, input_columns, output_columns, n_in=1, n_out=1, dropnan=True):
    '''
    Converts time series to supervised formed.

    Parameters:
    data (pandas.Dataframe or list): Sequence of observations
    input_columns (list): the columns that are the feature to be learned from the past
    output_columns (list): output values
    n_in (int): Number of lag observations as input (X). Values may be between [1..len(data)]. Optional. Defaults to 1.
    n_out (int): Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
    droopnan (boolean): Whether or not to drop rows with NaN values. Optional. Defaults to True.

    Returns:
    pandas.Dataframe: Series framed for supervised learning. The new columns are renamed VarM(t-N) where M is the Mth variable and N is the lag.
    '''

    selected_data = data[input_columns + output_columns]
    n_vars = selected_data.shape[1]

    df = pd.DataFrame(selected_data)
    cols, names = list(), list()

    labels = ['input'] * len(input_columns) + ['output'] * len(output_columns)

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('{}-{}(t-{})'.format(labels[j], j+1 if labels[j] == 'input' else j+1 - len(input_columns), i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('{}-{}(t)'.format(labels[j], j+1 if labels[j] == 'input' else j+1 - len(input_columns))) for j in range(n_vars)]
        else:
            names += [('{}-{}(t+{})'.format(labels[j], j+1 if labels[j] == 'input' else j+1 - len(input_columns), i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_future_output_columns(columns):
    return [column for column in columns if re.search(r'^output\-\d+\((t\+\d+|t)\)$', column)]


def get_input_columns(dataset):
    input_columns = list(dataset.columns)
    input_columns.remove('R')
    input_columns.remove('Date')
    input_columns.remove('CountryName')

    return input_columns


def load_dataset():
    '''
    Load data from the dataset folder, all the values for the files are hardcoded.
    We first load the google mobility dataset and merge it with demography and
    epidemiologic data.

    Returns:
    pandas.Dataframe: All the value for each day for each country.
    '''

    google = pd.read_csv("./datasets/2020_04_23_google.csv", parse_dates=['Date']).drop(["Unnamed: 0"],axis=1)
    countries = pd.read_csv("./datasets/seirhcd.csv", parse_dates=['Date'])

    google = pd.get_dummies(google,prefix="day_of_week", columns=["day_of_week"])
    google = pd.get_dummies(google,prefix="region", columns=["region"])
    dataset = pd.merge(countries.groupby(["CountryName","Date"]).agg("first"), google.groupby(["CountryName","Date"]).agg("first"),  on=["CountryName","Date"], how="inner")
    dataset = dataset.reset_index().dropna()

    columns = ['R', 'CountryName', 'Date']
    columns.extend(REGION_COLUMNS)
    columns.extend(TIME_COLUMNS)
    columns.extend(MOBILITY_COLUMNS)
    columns.extend(DEMOGRAPHY_COLUMNS)

    return dataset[columns]


def split_dataset(dataset, by, split, n_in=None, n_out=None):
    '''
    Create a split of the dataset based on a date. All data before the date are used for training and the ones after are used for testing

    Parameters:
    dataset(pandas.Dataframe): the rasw input dataset
    date(string): date (inclusive for the training set) at which we want to plit the data expressing in YYYY-MM-DD

    Returns:
    pandas.Dataframe: The training set
    pandas.Datafarame: The testing set
    '''

    if by == 'date':
        training = dataset[dataset['Date'] <= pd.to_datetime(split) + timedelta(days=n_out)]
        test = dataset[dataset['Date'] > pd.to_datetime(split) - timedelta(days=n_in)]
    elif by == 'region':
        training = dataset[np.logical_or.reduce([(dataset['region_{}'.format(i)] == 1) for i in split])]
        test = dataset[np.logical_and.reduce([(dataset['region_{}'.format(i)] == 0) for i in split])]
    elif by == 'country':
        countries = list(dataset['CountryName'].unique())
        random.shuffle(countries)
        train_size = math.floor(len(countries) * split)

        training = dataset[dataset['CountryName'].isin(countries[:train_size])]
        test = dataset[dataset['CountryName'].isin(countries[train_size:])]

    return training, test


def normalize_features(dataset, scaler=None):
    '''
    All the features should be between 0 and 1
    Categorical values are first set to integers then normalized

    Parameters:
    dataset(pandas.Dataframe): Sequence of observations

    Returns:
    normalizing scaler (sklearn.preprocessing.MinMaxScaler): a scaler that defines how to normalize the data
    '''
    x_columns = ['R']
    x_columns.extend(DEMOGRAPHY_COLUMNS)
    x_columns.extend(MOBILITY_COLUMNS)
    
    if scaler:
        dataset[x_columns] = scaler.transform(dataset[x_columns])
    else:
        scaler = preprocessing.MinMaxScaler()
        dataset[x_columns] = scaler.fit_transform(dataset[x_columns])

    return scaler


def denormalize(y, scaler):
    x_columns = ['R']
    x_columns.extend(DEMOGRAPHY_COLUMNS)
    x_columns.extend(MOBILITY_COLUMNS)

    n_features = len(x_columns)

    #get a matrix of the right shape for the scaler
    m = np.zeros(shape = (y.shape[0] * y.shape[1], n_features))
    #set the output values as a column
    m[:,0] = y.reshape((y.shape[0] * y.shape[1],))
    #apply scaler and get back the y column
    inv_y = scaler.inverse_transform(m)[:,0]
    #return the output in the right shape
    return inv_y.reshape((y.shape[0], y.shape[1]))


def reframe(dataset, n_in, n_out):
    reframed = pd.DataFrame()

    input_columns = get_input_columns(dataset)
    output_columns = ['R']

    country_columns = ['CountryName', 'region']
    country_columns.extend(DEMOGRAPHY_COLUMNS)

    for country in dataset['CountryName'].unique():
        country_data = dataset.loc[dataset['CountryName'] == country]
        country_data.sort_values(by=['Date'])

        country_reframed = series_to_supervised(country_data, input_columns, output_columns, n_in, n_out)

        reframed = reframed.append(country_reframed, ignore_index=True)

    return reframed


def get_x_y(data, n_in, n_out, n_features):
    '''
    Extract X and Y matrices from the dataset
    '''

    y_columns = get_future_output_columns(data)

    data_x = data.copy()
    data_x[y_columns] = 0
    data_x = data_x.values
    data_x = data_x.reshape(data_x.shape[0], n_in + n_out, n_features)

    data_y = data[y_columns].values

    return data_x, data_y
    

def train_model(train_X, train_Y, test_X, test_Y, n_out):
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(50))    
    model.add(Dense(n_out))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_Y, epochs=150, batch_size=20, validation_data=(test_X, test_Y), verbose=2, shuffle=True)

    return model, history


def print_metrics(model, normalized_test_x, normalized_test_y, scaler, n_in, n_out, n_features):
    normalized_pred_y = model.predict(normalized_test_x)

    pred_y = denormalize(normalized_pred_y, scaler)
    test_y = denormalize(normalized_test_y, scaler)

    print('r2_score:              {}'.format(r2_score(test_y, pred_y)))
    print('mean_absolute_error:   {}'.format(mean_absolute_error(test_y, pred_y)))
    print('mean_squared_error:    {}'.format(mean_squared_error(test_y, pred_y)))
    print('median_absolute_error: {}'.format(median_absolute_error(test_y, pred_y)))
    print('RMSE:                  {}'.format(np.sqrt(mean_absolute_error(test_y, pred_y))))


def draw_prediction(history):
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    raw_dataset = load_dataset()

    n_in = 15
    n_out = 7

    # we add one to account for R which is also a feature
    n_features = len(get_input_columns(raw_dataset)) + 1

    #Date to split was selected to have a train:set ratio of about 80:20
    #raw_training_set, raw_test_set = split_dataset(raw_dataset, by='date', split="2020-04-10", n_in=n_in, n_out=n_out)

    #Regions to split were selected to have a train:set ratio about 80:20
    #raw_training_set, raw_test_set = split_dataset(raw_dataset, by='region', split=list(range(0, 7)))    

    #With the countries, we simply select them randomly until we have for the initial dataset a spit of the row 80:20
    raw_training_set, raw_test_set = split_dataset(raw_dataset, by='country', split=0.8)    

    scaler = normalize_features(raw_training_set)
    training_set = reframe(raw_training_set, n_in, n_out)
    train_x, train_y = get_x_y(training_set, n_in, n_out, n_features)

    scaler = normalize_features(raw_test_set, scaler)
    test_set = reframe(raw_test_set, n_in, n_out)
    test_x, test_y = get_x_y(test_set, n_in, n_out, n_features)

    print('shape train_x: ' + str(train_x.shape))
    print('shape train_y: ' + str(train_y.shape))
    print('shape test_x: ' + str(test_x.shape))
    print('shape test_y: ' + str(test_y.shape))

    print('ration train:test: {}:{}'.format(train_x.shape[0] / (train_x.shape[0] + test_x.shape[0]), test_x.shape[0] / (train_x.shape[0] + test_x.shape[0])))

    model, history = train_model(train_x, train_y, test_x, test_y, n_out)

    print_metrics(model, test_x, test_y, scaler, n_in, n_out, n_features)
    draw_prediction(history)