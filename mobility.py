import experiments_utils as utils
import time
import pandas as pd


def prepare_dataframe_for_seaborn(data):
    value_vars = ['grocery/pharmacy_15days', 'workplace_15days', 'parks_15days', 'transit_stations_15days', 'retail/recreation_15days']
    return data.melt(id_vars=['Date', 'CountryName'], value_vars=value_vars, var_name='Measure', value_name="Value")

def draw_mobility():
    for country in ['Belgium', 'France', 'Germany', 'Italy', 'Luxembourg', 'Netherlands', 'Spain', 'Brazil', 'Cameroon', 'Canada', 'United Kingdom', 'Latvia', 'Greece', 'Japan']:
        data = prepare_dataframe_for_seaborn(utils.GOOGLE_VALUES)
        utils.lineplot(data.loc[data['CountryName'] == country], 'mobility_' + country, 'Date', 'Value', 'Variation of activity [%]', 'Measure')

if __name__ == '__main__':
    t_start = time.perf_counter()

    draw_mobility()

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------")