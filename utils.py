import matplotlib, os
#matplotlib.use('Agg')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,median_absolute_error


FIGURE_FOLDER = 'figures/'
EXTENSION = '.png'


REGIONS = ['region_{}'.format(i) for i in range(10)]
DAYS_OF_THE_WEEK = ['day_of_week_{}'.format(i) for i in range(7)]
DEMOGRAPHY = ["density", "population", "population_p14", "population_p65", "gdp", "area"]
MOBILITY = ["parks", "residential", "retail/recreation", "transit_stations", "workplace"]
SEIR = ['ConfirmedCases_x', 'ConfirmedCases_y', 'ConfirmedDeaths', 'Fatalities', 'HospitalizedCases','CriticalCases', 'ExposedCases', 'RecoveredCases', 'InfectiousCases', 't_hosp', 't_crit', 'm', 'c', 'f']

MOBILITY_WINDOWS = ["{}{}".format(f,p) for p in ["","_5days","_10days","_15days","_30days"] for f in MOBILITY]


def features_values():
    raw_data = pd.read_csv('data/features.csv')
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])

    return raw_data


def metrics_report(X_test, y_test, reg):
    y_pred = reg.predict(X_test)
    return {'r2_score': r2_score(y_test, y_pred), 
          'mean_absolute_error': mean_absolute_error(y_test, y_pred),
          'mean_squared_error': mean_squared_error(y_test, y_pred),
         'median_absolute_error': median_absolute_error(y_test, y_pred),
           'RMSE': np.sqrt(mean_absolute_error(y_test, y_pred))}


def load_dataset():
    '''
    Load data from the dataset folder, all the values for the files are hardcoded.
    We first load the google mobility dataset and merge it with demography and
    epidemiologic data.

    Returns:
    pandas.Dataframe: All the value for each day for each country.
    '''

    google = pd.read_csv("./data/google.csv", parse_dates=['Date']).drop(["Unnamed: 0"],axis=1)
    countries = pd.read_csv("./data/seirhcd.csv", parse_dates=['Date']).drop(["Unnamed: 0"],axis=1)

    google = pd.get_dummies(google,prefix="day_of_week", columns=["day_of_week"])
    google = pd.get_dummies(google,prefix="region", columns=["region"])
    dataset = pd.merge(countries.groupby(["CountryName","Date"]).agg("first"), google.groupby(["CountryName","Date"]).agg("first"), on=["CountryName","Date"], how="inner")
    dataset = dataset.reset_index().dropna()

    columns = ['R', 'CountryName', 'Date']
    columns.extend(REGIONS)
    columns.extend(DAYS_OF_THE_WEEK)
    columns.extend(MOBILITY_WINDOWS)
    columns.extend(DEMOGRAPHY)
    columns.extend(SEIR)

    return dataset[columns]


def get_feature_columns():
    columns = []
    columns.extend(MOBILITY_WINDOWS)
    columns.extend(REGIONS)
    columns.extend(DAYS_OF_THE_WEEK)
    columns.extend(DEMOGRAPHY)

    return columns


def color_palette(data, hue):
    n_colors = 1 if hue == None else len(data[hue].unique())
    return sns.color_palette("cubehelix", n_colors=n_colors)


def save_figure(filename, dpi=300, bbox_inches='tight'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    plt.close('all')


def plot(type, data, name, x, y, y_label, x_label='', hue=None, y_lim=None, fig_size=(6,4), legend_pos='best', style=None, show_error=False, **kwargs):
    fig = plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=1.5)

    palette = color_palette(data, hue)

    if type == 'line':
        g = sns.lineplot(x=x, y=y, hue=hue, data=data, palette=palette, legend="full", style=style, **kwargs)
        plt.ticklabel_format(style='plain', axis='y',useOffset=False)
    elif type == 'scatter':
        g = sns.scatterplot(x=x, y=y, hue=hue, data=data, palette=palette, legend="full", style=style, **kwargs)
    else:
        raise TypeError("Only line or scatter are allowed")

    fig.tight_layout()

    if not legend_pos:
        if g.legend_:
            g.legend_.remove()
    elif hue:
        handles, labels = g.get_legend_handles_labels()
        plt.legend(loc='best', prop={'size': 15}, handles=handles[1:], labels=labels[1:])

    plt.ylabel(y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.xticks(rotation=45)

    if y_lim != None and len(y_lim) == 2:
        plt.ylim(y_lim)

    if show_error:
        error_band(data, x, hue)

    save_figure('figures/' + name + EXTENSION)


def error_band(data, x, hue):
    ax = plt.gca()
    
    valid_labels = data[hue].unique()

    for line in ax.lines:
        if not line.get_label() in valid_labels:
            continue

        x_values = data.loc[data[hue] == line.get_label()][x].values
        y_min = data.loc[data[hue] == line.get_label()]['Min'].values
        y_max = data.loc[data[hue] == line.get_label()]['Max'].values

        ax.fill_between(x_values, y_min, y_max, color=line.get_color(), alpha=0.2, linewidth=0.0)
        
