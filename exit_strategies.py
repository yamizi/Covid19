import pandas as pd
from model_run import simulate, create_calendar_from_scenario, MEASURES
import utils
import time
from datetime import timedelta

FEATURES_VALUES = utils.features_values()
END_DATE = pd.to_datetime("2020-9-11")

def get_no_lockdown(country):
    return list(map(lambda x: max(x, 0), get_initial_lockdown(country)))


def get_initial_lockdown(country):
    initial = FEATURES_VALUES.loc[(FEATURES_VALUES['CountryName'] == country)]
    initial = initial.loc[initial['Date'].idxmax()]
    return [initial['workplace_15days'], initial['parks_15days'], initial['transit_stations_15days'], initial['retail/recreation_15days'], initial['residential_15days']]


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

    measures = []
    initial = get_initial_lockdown(country)
    for i in range(len(dates)):
        measures.extend([max(e, min(0, e + abs(e * 0.13 * (i + 1)))) for e in initial])

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
    scenario['measure_names'] = create_measures(len(dates))
    scenario['measure_values'] = values
    scenario['measure_dates'] = create_dates(dates)
    
    return scenario


def build_parameters(country, scenario):
    parameters = get_scenario(country, scenario)

    parameters['measure_dates'] = [pd.to_datetime(d) for d in parameters['measure_dates']]
    parameters['features_country'] = FEATURES_VALUES[FEATURES_VALUES["CountryName"]==country]
    parameters['init_date'] = parameters['features_country']["Date"].tail(1).dt.date.values[0]
    parameters['end_date'] = END_DATE

    return parameters


def run_simulation(countries, scenarios):
    results = pd.DataFrame()

    for country in countries:
        for scenario in scenarios:
            print('Run for {} with {}\n'.format(country, scenario))
            
            parameters = build_parameters(country, scenario)

            country_lift = create_calendar_from_scenario(**parameters)
            df = simulate(country_lift, 'random', parameters['init_date'], parameters['end_date'])
            df['Scenario'] = scenario
            df['Country'] = country
            results = results.append(df, ignore_index = True)
            
    return results


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

def draw_plots(raw_data):
    data = prepare_dataframe_for_seaborn(raw_data)

    for country in data['Country'].unique():
        for population in data['Population'].unique():
            data_to_draw = data.loc[(data['Country'] == country) & (data['Population'] == population)]
            name = 'scenarios_{}_{}'.format(country, population)

            legend_pos = None if not country == 'Italy' else 'best'
            utils.plot('line', data_to_draw, name, 'Date', 'Value', 'Number of habitants', hue='Scenario', show_error=False, legend_pos=legend_pos)

if __name__ == '__main__':
    t_start = time.perf_counter()

    countries = ['Japan', 'Luxembourg', 'Italy']
    scenarios = ['hard exit', 'no exit', 'progressive exit', 'cyclic exit']

    simulation = run_simulation(countries, scenarios)
    draw_plots(simulation)

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------") 