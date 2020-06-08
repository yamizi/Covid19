import time, utils
import pandas as pd


FEATURES_VALUES = utils.features_values()


def pretty_name(x):
    measure = x['Measures']
    
    if measure == 'grocery/pharmacy':
        return 'Grocery & Pharmacies'
    elif measure == 'workplace':
        return 'Workplace'
    elif measure == 'parks':
        return 'Parks'
    elif measure == 'transit_stations':
        return 'Transit Stations'
    elif measure == 'retail/recreation':
        return 'Retail & Recreation'
    elif measure == 'residential':
        return 'Residential'


def draw_mobility(countries):
    value_vars = utils.MOBILITY
    data = FEATURES_VALUES.melt(id_vars=['Date', 'CountryName'], value_vars=value_vars, var_name='Measures', value_name="Value")
    data['Measures'] = data.apply(pretty_name, axis=1)

    for country in countries:
        utils.plot('line', data.loc[data['CountryName'] == country], 'data_visualization_mobility_' + country, 'Date', 'Value', 'Variation of activity [%]', hue='Measures', legend_pos=None)


def draw_death_rate(countries):
    data = FEATURES_VALUES[['Date', 'CountryName', 'ConfirmedDeaths', 'population']]
    
    data['Death Rate'] = data.apply(lambda x: x['ConfirmedDeaths'] * 1000000 / x['population'], axis=1)

    for country in countries:
         utils.plot('line', data.loc[data['CountryName'] == country], 'data_visualization_death_rate_' + country, 'Date', 'Death Rate', 'Number of death per million hab')


def draw_death_over_mobility(countries):
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