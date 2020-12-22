import time, utils
import pandas as pd
import matplotlib.pyplot as plt

def plot_model_results(y_pred, train_data, valid_data=None):
    FEATURES_VALUES = utils.features_values()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(48, 10))

    ax1.set_title('Confirmed Cases')
    ax2.set_title('Fatalities')

    train_data['ConfirmedCases'].plot(label='Confirmed Cases (train)', color='g', ax=ax1)
    y_pred.loc[train_data.index, 'ConfirmedCases'].plot(label='Modeled Cases', color='r', ax=ax1)
    ax4 = y_pred['R'].plot(label='Reproduction number', color='c', linestyle='-', secondary_y=True, ax=ax1)
    ax4.set_ylabel("Reproduction number", fontsize=10, color='c');

    train_data['Fatalities'].plot(label='Fatalities (train)', color='g', ax=ax2)
    y_pred.loc[train_data.index, 'Fatalities'].plot(label='Modeled Fatalities', color='r', ax=ax2)

    if valid_data is not None:
        valid_data['ConfirmedCases'].plot(label='Confirmed Cases (valid)', color='g', linestyle=':', ax=ax1)
        valid_data['Fatalities'].plot(label='Fatalities (valid)', color='g', linestyle=':', ax=ax2)
        y_pred.loc[valid_data.index, 'ConfirmedCases'].plot(label='Modeled Cases (forecast)', color='r', linestyle=':',
                                                            ax=ax1)
        y_pred.loc[valid_data.index, 'Fatalities'].plot(label='Modeled Fatalities (forecast)', color='r', linestyle=':',
                                                        ax=ax2)
    else:
        y_pred.loc[:, 'ConfirmedCases'].plot(label='Modeled Cases (forecast)', color='r', linestyle=':', ax=ax1)
        y_pred.loc[:, 'Fatalities'].plot(label='Modeled Fatalities (forecast)', color='r', linestyle=':', ax=ax2)

        y_pred.loc[:, 'HospitalizedCases'].plot(label='Modeled Hospitalizations (forecast)', color='b', ax=ax3)
        y_pred.loc[:, 'CriticalCases'].plot(label='Modeled Critical (forecast)', color='r', linestyle=':', ax=ax3)
        ax3.set_title('Hospitalizations & Criticals')
        ax3.legend(loc='best')

    ax1.legend(loc='best')

def pretty_name(x):
    FEATURES_VALUES = utils.features_values()

    measure = x['Measures']
    
    if measure == 'grocery/pharmacy':
        return 'Grocery & Pharmacies'
    elif measure == 'workplace':
        return 'Workplace'
    elif measure == 'S1_School closing':
        return 'Schools'
    elif measure == 'parks':
        return 'Parks'
    elif measure == 'transit_stations':
        return 'Transit Stations'
    elif measure == 'retail/recreation':
        return 'Retail & Recreation'
    elif measure == 'residential':
        return 'Residential'


def draw_mobility(countries):
    FEATURES_VALUES = utils.features_values()
    value_vars = utils.MOBILITY
    data = FEATURES_VALUES.melt(id_vars=['Date', 'CountryName'], value_vars=value_vars, var_name='Measures', value_name="Value")
    data['Measures'] = data.apply(pretty_name, axis=1)

    for country in countries:
        utils.plot('line', data.loc[data['CountryName'] == country], 'data_visualization_mobility_' + country, 'Date', 'Value', 'Variation of activity [%]', hue='Measures', legend_pos=None)


def draw_death_rate(countries):
    FEATURES_VALUES = utils.features_values()
    data = FEATURES_VALUES[['Date', 'CountryName', 'ConfirmedDeaths', 'population']]
    
    data['Death Rate'] = data.apply(lambda x: x['ConfirmedDeaths'] * 1000000 / x['population'], axis=1)

    for country in countries:
         utils.plot('line', data.loc[data['CountryName'] == country], 'data_visualization_death_rate_' + country, 'Date', 'Death Rate', 'Number of death per million hab')


def draw_death_over_mobility(countries):
    FEATURES_VALUES = utils.features_values()
    value_vars = ['{}_15days'.format(mobility) for mobility in utils.MOBILITY]
    data = FEATURES_VALUES.melt(id_vars=['CountryName', 'ConfirmedDeaths', 'R'], value_vars=value_vars, var_name='Measures', value_name="Value")
    data['Measures'] = data.apply(pretty_name, axis=1)

    for country in countries:
        utils.plot('scatter', data.loc[data['CountryName'] == country], 'data_visualization_death_to_measure_' + country, 'Value', 'R', 'Rt', x_label='mobility', hue='Measures', legend_pos=None)


if __name__ == '__main__':
    t_start = time.perf_counter()

    countries = ['Belgium', 'France', 'Germany', 'Greece', 'Italy', 'Luxembourg', 'Netherlands', 'Spain', 'Switzerland', 'Brazil', 'Cameroon', 'Canada', 'Japan', 'United Kingdom']

    draw_mobility(countries)
    draw_death_rate(countries)
    draw_death_over_mobility(countries)

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------")
