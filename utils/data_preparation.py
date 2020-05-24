import pandas as pd
import numpy as np
from simulations import features, periods
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def split_sequences(sequences_x, sequences_y, sequences_ref, n_steps):
    X, y, ref = list(), list(),list()
    for i in range(len(sequences_x)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences_x):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y, seq_ref = sequences_x[i:end_ix, :], sequences_y[end_ix - 1], sequences_ref[i:end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
        ref.append(seq_ref)
    return np.array(X), np.array(y), ref


def split_features_target(dataset, steps=1, y_label=["R"]):
    dataset = pd.get_dummies(dataset, prefix="day_of_week", columns=["day_of_week"])
    region = dataset["region"]
    dataset = pd.get_dummies(dataset, prefix="region", columns=["region"])
    columns = ["{}{}".format(f, p) for p in periods for f in features]
    columns = columns + ["density", "population", "population_p65", "population_p14", "gdp", "area"]
    columns = columns + ["day_of_week_{}".format(i) for i in range(7)]
    columns = columns + ["region_{}".format(i) for i in range(10)]
    dataset = dataset.rename(columns={'region_10': 'region_9'})
    dataset["region"] = region

    X, y, ref = dataset[columns], dataset[y_label].values, dataset[["R","ConfirmedCases_y", "ConfirmedDeaths", "CountryName","region", "Date"]].index.values

    if steps > 1:
        X, y, ref = split_sequences(X, y, ref, steps)
    return X, y, ref


def split_by_countries(X, y, ref, countries):
    test_countries = ref["CountryName"].isin(countries)
    X_train = X[~test_countries]
    y_train = y[~test_countries]
    ref_train = ref[~test_countries]
    X_test = X[~test_countries]
    y_test = y[~test_countries]
    ref_test = ref[~test_countries]
    return X_train, X_test, y_train, y_test, ref_train, ref_test

def split_by_regions(X, y, ref, regions):
    test_countries = ref["region"].isin(regions)
    X_train = X[~test_countries]
    y_train = y[~test_countries]
    ref_train = ref[~test_countries]
    X_test = X[~test_countries]
    y_test = y[~test_countries]
    ref_test = ref[~test_countries]
    return X_train, X_test, y_train, y_test, ref_train, ref_test

def train_test_split_scaled(X, y, ref, by_countries=None, by_regions=None, test_size=0.2, shuffle=True):

    if by_countries is not None:
        X_train, X_test, y_train, y_test, ref_train, ref_test = split_by_countries(X, y, ref, by_countries)

    elif by_regions is not None:
        X_train, X_test, y_train, y_test, ref_train, ref_test = split_by_regions(X, y, ref, by_regions)
    else:
        X_train, X_test, y_train, y_test, ref_train, ref_test = train_test_split(X, y, ref, test_size=test_size, random_state=42, shuffle=shuffle)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, ref_train, ref_test


