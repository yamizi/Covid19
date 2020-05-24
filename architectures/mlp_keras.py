import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from utils.data_preparation import split_features_target, train_test_split_scaled
from utils.helpers import metrics_report
import numpy as np
from keras import backend as K
from utils.loss import LossFunction, seirhcd_loss, keras_auc
from keras.models import Model

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))



sierchd_dataset= pd.read_csv("../datasets/v2_1_seirhcd.csv", parse_dates=['Date'])
mobility_dataset= pd.read_csv("../datasets/v2_1_google.csv", parse_dates=['Date'])
mobility_dataset = mobility_dataset.drop(["Unnamed: 0"],axis=1)

#merge datasets
merged_dataset = pd.merge(sierchd_dataset.groupby(["CountryName","Date"]).agg("first"), mobility_dataset.groupby(["CountryName","Date"]).agg("first"),  on=["CountryName","Date"], how="inner")
merged_dataset = merged_dataset.reset_index().dropna()
#print(merged_dataset.describe())

#columns
print(merged_dataset.columns)
X, y, ref = split_features_target(merged_dataset)
print(X.shape, y.shape)

def mlp_Rt_random():

    X_train_scaled, X_test_scaled, y_train, y_test, ref_train, ref_test = train_test_split_scaled(X,y,ref)

    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_scaled, y_train, epochs=1500, verbose=1)
    return metrics_report(X_test_scaled, y_test, model),metrics_report(X_train_scaled, y_train, model)


def mlp_seirhcd_random():

    updated_dataset = merged_dataset.copy()


    X, y, ref = split_features_target(merged_dataset, y_label=["R"])
    X_train_scaled, X_test_scaled, y_train, y_test, ref_train, ref_test = train_test_split_scaled(X,y,ref,test_size=0.9,shuffle=False)

    x_ = Input(shape=(X.shape[1],))
    y_ = Dense(100,activation="relu")(x_)
    y_ = Dense(50,activation="relu")(y_)
    y_ = Dense(1)(y_)
    #y_ = SEIRLayer()(y_) #Lambda(seir_call, name='seir')(y_)

    model = Model(inputs=x_, outputs=y_)
    loss =keras_auc #seirhcd_loss(updated_dataset) # LossFunction(model, updated_dataset) #'mse'
    model.compile(optimizer='adam', loss=loss, metrics=[keras_auc])
    model.fit(X_train_scaled, ref_train, epochs=1500, verbose=2)
    print("end")
    return metrics_report(X_test_scaled, ref_test, model),metrics_report(X_train_scaled, y_train, model)

"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(X.shape[1],)),  # input shape required
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        SeirLayer(),
        tf.keras.layers.Dense(1)
    ])
"""

def mlp_Rt_countries():
    X_train_scaled, X_test_scaled, y_train, y_test, ref_train, ref_test = train_test_split_scaled(X,y,ref, by_countries=["Luxembourg","France","Germany","Spain","United kingdom","Greece","Italy","Switzerland","Belgium","Netherlands"])
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_scaled, y_train, epochs=1500, verbose=1)
    return metrics_report(X_test_scaled, y_test, model),metrics_report(X_train_scaled, y_train, model)


reports_r, reports_train_r =mlp_seirhcd_random()
#reports_r, reports_train_r =mlp_Rt_random()
#reports, reports_train = mlp_Rt_countries
print("reports random split", reports_r, reports_train_r)
#print("reports country split", reports, reports_train)