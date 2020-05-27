import pandas as pd
import numpy as np
import re

from datetime import date
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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
    return [column for column in columns if re.search('^output\-\d+\((t\+\d+|t)\)$', column)]


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


def split_dataset(dataset, date):
    '''
    Create a split of the dataset based on a date. All data before the date are used for training and the ones after are used for testing

    Parameters:
    dataset(pandas.Dataframe): the rasw input dataset
    date(string): date (inclusive for the training set) at which we want to plit the data expressing in YYYY-MM-DD

    Returns:
    pandas.Dataframe: The training set
    pandas.Datafarame: The testing set
    '''
    training = dataset[dataset['Date'] <= pd.to_datetime(date)].copy()
    test = dataset[dataset['Date'] > pd.to_datetime(date)].copy()

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
    x_columns = []
    x_columns.extend(DEMOGRAPHY_COLUMNS)
    x_columns.extend(MOBILITY_COLUMNS)
    
    if scaler:
        dataset[x_columns] = scaler.transform(dataset[x_columns])
    else:
        scaler = preprocessing.MinMaxScaler()
        dataset[x_columns] = scaler.fit_transform(dataset[x_columns])

    return scaler


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
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(n_out))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    return model.fit(train_X, train_Y, epochs=50, batch_size=7, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

def draw_prediction(model):
    pyplot.plot(model.history['loss'], label='train')
    pyplot.plot(model.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    raw_dataset = load_dataset()

    n_in = 15
    n_out = 7

    # we add one to account for R which is also a feature
    n_features = len(get_input_columns(raw_dataset)) + 1

    raw_training_set, raw_test_set = split_dataset(raw_dataset, "2020-04-01")

    scaler = normalize_features(raw_training_set)
    training_set = reframe(raw_training_set, n_in, n_out)
    train_x, train_y = get_x_y(training_set, n_in, n_out, n_features)

    # TODO: when computing values for test, use the historical data from the entire dataset
    scaler = normalize_features(raw_test_set, scaler)
    test_set = reframe(raw_test_set, n_in, n_out)
    test_x, test_y = get_x_y(test_set, n_in, n_out, n_features)

    print('shape train_x: ' + str(train_x.shape))
    print('shape train_y: ' + str(train_y.shape))
    print('shape test_x: ' + str(test_x.shape))
    print('shape test_y: ' + str(test_y.shape))

    model = train_model(train_x, train_y, test_x, test_y, n_out)

    draw_prediction(model)