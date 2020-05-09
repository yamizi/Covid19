import pandas as pd
from seir_hcd_simulations import scaler, yvar, merged, mlp_clf, columns
from simulations import simulate
import experiments_utils as utils
import time

MEASURES = ['grocery/pharmacy', 'workplace', 'S1_School closing', 'parks', 'transit_stations', 'retail/recreation', 'S7_International travel controls']

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
    measures = [0] * len(MEASURES)

    return (dates, measures)


def create_soft_exit(country):
    dates = ['2020-05-11', '2020-05-25', '2020-06-08', '2020-06-22', '2020-07-06', '2020-07-20', '2020-08-03']
    
    measures = []
    initial = get_initial_lockdown(country)
    for i in range(len(dates)):
        measures.extend([min(0, e + abs(e * 0.15 * (i + 1))) for e in initial])

    return (dates, measures)


def create_no_exit(country):
    return ([], [])


def create_cyclic_exit(country):
    dates = ['2020-05-11', '2020-05-25', '2020-06-08', '2020-06-22', '2020-07-06', '2020-07-20', '2020-08-03']

    measures = []
    initial = get_initial_lockdown(country)
    for i in range(len(dates)):
        if i % 2 == 0:
            measures.extend([0] * len(MEASURES))
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

def get_initial_lockdown(country):
    initial = utils.GOOGLE_VALUES.loc[(utils.GOOGLE_VALUES['CountryName'] == country) & (utils.GOOGLE_VALUES['Date'] == '2020-04-01')].iloc[0]
    return [initial['grocery/pharmacy_15days'], initial['workplace_15days'], -100, initial['parks_15days'], initial['transit_stations_15days'], initial['retail/recreation_15days'], -100]

def build_parameters(country, scenario):
    parameters = get_scenario(country, scenario)
    parameters['measure_dates'] = [pd.to_datetime(d) for d in parameters['measure_dates']]
    parameters['country_df'] = merged[merged["CountryName"]==country]

    return parameters


def run_simulation(countries, scenarios):
    end_date = pd.to_datetime("2020-9-11")

    results = pd.DataFrame()

    for country in countries:
        for scenario in scenarios:
            parameters = build_parameters(country, scenario)

            df = simulate(parameters['country_df'], [parameters['measures_to_lift'],],0,end_date,None,columns,yvar, mlp_clf, scaler,measure_values=parameters['measure_values'],base_folder=None, lift_date_values=parameters['measure_dates'])
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

            utils.lineplot(data_to_draw, name, 'Date', 'Value', population, 'Scenario', show_error=True)

if __name__ == '__main__':
    t_start = time.perf_counter()

    countries = ['Belgium', 'France', 'Germany', 'Luxembourg']
    scenarios = ['hard exit', 'no exit', 'progressive exit', 'cyclic exit']

    simulation = run_simulation(countries, scenarios)
    draw_plots(simulation)

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------") 