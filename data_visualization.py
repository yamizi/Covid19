import experiments_utils as utils
import time
import pandas as pd


def prepare_dataframe_for_seaborn(data, value_vars, var_name):
    return data.melt(id_vars=['Date', 'CountryName'], value_vars=value_vars, var_name=var_name, value_name="Value")

def draw_mobility(countries):
    value_vars = ['grocery/pharmacy', 'workplace', 'parks', 'transit_stations', 'retail/recreation']
    data = prepare_dataframe_for_seaborn(utils.GOOGLE_VALUES, value_vars, 'Measure')

    for country in countries:
        utils.lineplot(data.loc[data['CountryName'] == country], 'mobility_' + country, 'Date', 'Value', 'Variation of activity [%]', hue='Measure')


def draw_death_rate(countries):
    data = utils.GOOGLE_VALUES[['Date', 'CountryName', 'ConfirmedDeaths', 'population']]
    
    data['Death Rate'] = data.apply(lambda x: x['ConfirmedDeaths'] * 1000000 / x['population'], axis=1)

    for country in countries:
         utils.lineplot(data.loc[data['CountryName'] == country], 'death_rate_' + country, 'Date', 'Death Rate', 'Number of death per million hab')


if __name__ == '__main__':
    t_start = time.perf_counter()

    countries = ['Belgium', 'France', 'Germany', 'Greece', 'Italy', 'Latvia', 'Luxembourg', 'Netherlands', 'Spain', 'Switzerland', 'Brazil', 'Cameroon', 'Canada', 'Japan', 'United Kingdom']

    draw_mobility(countries)
    draw_death_rate(countries)

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------")