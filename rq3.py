import pandas as pd
import numpy as np
import experiments_utils as utils
import time
import data_visualization

from seir_hcd_simulations import scaler, yvar, merged, mlp_clf, columns
from simulations import simulate
from sklearn import metrics

MEASURES = ['residential', 'workplace', 'parks', 'transit_stations', 'retail/recreation', 'grocery/pharmacy',]


def get_no_lockdown(country):
    no_lockdown = list(map(lambda x: max(x, 0), get_initial_lockdown(country)))
    #put it if you want to say that during no lockdown activity is at 100% and not over 100%
    #no_lockdown[0] = 0

    return no_lockdown


def get_initial_lockdown(country):
    initial = utils.FEATURES_VALUES.loc[(utils.FEATURES_VALUES['CountryName'] == country)]
    initial = initial.loc[initial['Date'].idxmax()]
    return [initial['residential_15days'], initial['workplace_15days'], initial['parks_15days'], initial['transit_stations_15days'], initial['retail/recreation_15days'], initial['grocery/pharmacy_15days']]


def update_soft_exit(index, value, step):
    #put it if you want to say that during no lockdown activity is at 100% and not over 100%
    #if index == 0:
    #    return min(value, max(0, value - abs(value * 0.13 * (step + 1))))

    return max(value, min(0, value + abs(value * 0.13 * (step + 1))))
    

def create_measures(n):
    measures = []
    for i in range(n):
        measures.extend(MEASURES)

    return measures


def create_dates(raw_dates):
    dates = []

    for date in raw_dates:
        dates.extend([date] * len(MEASURES))

    return dates


def create_hard_exit(country):
    dates = ['2020-05-11']
    measures = get_no_lockdown(country)

    return (dates, measures)


def create_soft_exit(country):
    dates = ['2020-05-11', '2020-05-25', '2020-06-08', '2020-06-22', '2020-07-06', '2020-07-20', '2020-08-03', '2020-08-17']
    # monthly cycles instead of every 2 weeks
    #dates = ['2020-05-11', '2020-06-08', '2020-07-06', '2020-08-03', '2020-08-31']

    measures = []
    initial = get_initial_lockdown(country)
    for i in range(len(dates)):
        measures.extend([update_soft_exit(index, e, i) for index, e in enumerate(initial)])

    return (dates, measures)


def create_no_exit(country):
    return ([], [])


def create_cyclic_exit(country):
    dates = ['2020-05-11', '2020-05-25', '2020-06-08', '2020-06-22', '2020-07-06', '2020-07-20', '2020-08-03']

    measures = []
    initial = get_initial_lockdown(country)
    for i in range(len(dates)):
        if i % 2 == 0:
            measures.extend(get_no_lockdown(country))
        else:
            measures.extend(initial)

    return (dates, measures)


def get_scenario(country, scenario_name):
    dates = []
    values = []

    if scenario_name == 'hard exit':
        dates, values = create_hard_exit(country)
    elif scenario_name == 'progressive exit':
        dates, values = create_soft_exit(country)
    elif scenario_name == 'cyclic exit':
        dates, values = create_cyclic_exit(country)
    else:
        dates, values = create_no_exit(country)

    scenario = dict()
    scenario['measures_to_lift'] = create_measures(len(dates))
    scenario['measure_values'] = values
    scenario['measure_dates'] = create_dates(dates)
    
    return scenario


def build_parameters(country, scenario):
    parameters = get_scenario(country, scenario)

    print('Run for {} with {}\n'.format(country, scenario))

    parameters['measure_dates'] = [pd.to_datetime(d) for d in parameters['measure_dates']]
    parameters['country_df'] = merged[merged["CountryName"]==country]

    return parameters


def run_simulation(countries, scenarios):
    end_date = pd.to_datetime("2020-9-30")

    results = pd.DataFrame()
    calendar = pd.DataFrame()

    for country in countries:
        for scenario in scenarios:
            parameters = build_parameters(country, scenario)

            simulation_country, calendar_country = simulate(parameters['country_df'], [parameters['measures_to_lift'],],0,end_date,None,columns,yvar, mlp_clf, scaler,measure_values=parameters['measure_values'],base_folder=None, lift_date_values=parameters['measure_dates'])
            
            simulation_country['Scenario'] = scenario
            simulation_country['Country'] = country
            results = results.append(simulation_country, ignore_index = True)
            
            calendar_country['Scenario'] = scenario
            calendar = calendar.append(calendar_country, ignore_index = True)
            
    return results, calendar


def get_min_max(date, country, scenario, column, df, min_max):
    column_to_find = '{}_{}'.format(column, min_max)
    value = df.loc[(df['Date'] == date) & (df['Date'] == date) & (df['Country'] == country) & (df['Scenario'] == scenario)][column_to_find].values[0]

    return value


def prepare_dataframe_for_seaborn(data):
    value_vars = ['SimulationCases', 'SimulationHospital', 'SimulationCritical', 'SimulationDeaths', 'R']
    melted = data.melt(id_vars=['Date', 'Country', 'Scenario'], value_vars=value_vars, var_name='Population', value_name="Value")
    melted['Min'] = melted.apply(lambda x: get_min_max(x['Date'], x['Country'], x['Scenario'], x['Population'], data, 'min'), axis=1)
    melted['Max'] = melted.apply(lambda x: get_min_max(x['Date'], x['Country'], x['Scenario'], x['Population'], data, 'max'), axis=1)

    return melted

def draw_results(raw_data):
    data = prepare_dataframe_for_seaborn(raw_data)

    for country in data['Country'].unique():
        for population in data['Population'].unique():
            data_to_draw = data.loc[(data['Country'] == country) & (data['Population'] == population)]
            name = 'scenarios_{}_{}'.format(country, population)

            legend_pos = None if not country == 'Italy' else 'best'
            utils.plot('line', data_to_draw, name, 'Date', 'Value', 'Rt', hue='Scenario', show_error=False, legend_pos=legend_pos)


def draw_mobility(calendar, countries):
    for scenario in calendar['Scenario'].unique():
        data_visualization.draw_mobility(calendar.loc[calendar['Scenario'] == scenario], countries, 'mobility_{}_'.format(scenario))


def compute_AUC(column):
    return metrics.auc(np.arange(column.shape[0]), column.values)


def print_AUC(calendar):
    results = calendar.drop(columns=['residential']).groupby(['Scenario', 'CountryName']).agg([('AUC', compute_AUC),])
    print(results)

    mean_results = results.mean(axis=1, level=1, numeric_only=True)
    print(mean_results)

def print_simulation_info(simulation):
    results = simulation[['Scenario', 'Country', 'SimulationDeaths']].groupby(['Scenario', 'Country']).agg(['max'])
    print(results)

if __name__ == '__main__':
    t_start = time.perf_counter()

    #countries = ['Belgium', 'France', 'Germany', 'Greece', 'Italy', 'Latvia', 'Luxembourg', 'Netherlands', 'Spain', 'Switzerland', 'Brazil', 'Cameroon', 'Canada', 'Japan', 'United Kingdom']
    countries = ['Japan', 'Luxembourg', 'Italy']
    scenarios = ['hard exit', 'no exit', 'progressive exit', 'cyclic exit']

    simulation, calendar = run_simulation(countries, scenarios)
    
    draw_results(simulation)
    draw_mobility(calendar, countries)
    print_AUC(calendar)
    print_simulation_info(simulation)

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------") 