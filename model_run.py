import json, utils, joblib
from helpers import seir
import pandas as pd
import numpy as np

from datetime import timedelta, datetime
from scipy.integrate import solve_ivp

MEASURES = ['workplace', 'parks', 'transit_stations', 'retail/recreation', 'residential']

def update_seir(df, active_date, e_date, l_date=None, confidence_interval=True):

    cols = list(df.columns)
    data = df[df["Date"] >= active_date]

    # t_hosp=7, t_crit=14, m_a=0.8, c_a=0.1, f_a=0.3
    ref_params = [7, 14, 0.7, 0.3, 0.3]
    params = [7, 14, 0.8, 0.3, 0.3]
    params_name = ("t_hosp", "t_crit", "m", "c", "f")
    for i, param in enumerate(params_name):
        if param in cols:
            params[i] = data[param].mean()
    params = ref_params
    params.append(False)


    ref_data = df[df["Date"] == active_date + timedelta(days=-7)]
    inf_data = df[df["Date"] == active_date + timedelta(days=-14)]

    if l_date is None:
        l_date = active_date
    population = data["population"].min() *0.7 # Herd immunity is assumed at 70%
    N = population
    n_infected = ref_data['ConfirmedCases_y'].iloc[0]-inf_data['ConfirmedCases_y'].iloc[0] #data['InfectiousCases'].iloc[0]
    n_exposed = data['ConfirmedCases_y'].iloc[0] - ref_data['ConfirmedCases_y'].iloc[0] #data['ExposedCases'].iloc[0]
    n_hospitalized = (1-params[2]) * n_exposed #data['HospitalizedCases'].iloc[0]*1.5
    n_exposed = params[2] * n_exposed 
    n_critical = (params[3]) * n_hospitalized #data['CriticalCases'].iloc[0]*1.5
    n_recovered =  inf_data['ConfirmedCases_y'].iloc[0] -inf_data['ConfirmedDeaths'].iloc[0]  #data['RecoveredCases'].iloc[0]
    n_deaths = data['ConfirmedDeaths'].iloc[0]
    # S, E, I, R, H, C, D
    initial_state = [(N - n_infected) / N, n_exposed / N, n_infected / N, n_recovered / N, n_hospitalized / N,
                     n_critical / N, n_deaths / N]

    # t_hosp=7, t_crit=14, m_a=0.8, c_a=0.1, f_a=0.3

    ref_params =  [7, 14, 0.8, 0.3, 0.3]
    params = [7, 14, 0.8, 0.3, 0.3]
    params_name = ("t_hosp", "t_crit", "m", "c", "f")
    for i, param in enumerate(params_name):
        if param in cols:
            params[i] = data[param].mean()
    #params = ref_params
    print("disease params", params, np.sum([params, ref_params],axis=1))
    #"decay_values"
    params.append(True)
    R_t = data['R'].values

    def time_varying_reproduction(t):
        index = np.min((int(t), len(R_t)-1))
        return R_t[index]

    args = (time_varying_reproduction, 5.6, 2.9, *params)

    init_date = datetime.combine(active_date, datetime.min.time())
    max_days = ((e_date - pd.Timestamp(init_date)) / np.timedelta64(1, 'D'))
    max_days = int(max_days)
    sol = solve_ivp(seir.model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    y_pred_cases = np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population
    y_pred_critic = np.clip(crit, 0, np.inf) * population
    y_pred_hosp = np.clip(hosp, 0, np.inf) * population
    y_pred_deaths = np.clip(deaths, 0, np.inf) * population
    y_pred_infectious = np.clip(inf, 0, np.inf) * population

    dates = data["Date"].iloc[1:]
    l = len(dates)

    # print(l,len(y_pred_cases),y_pred_hosp_max.min(),y_pred_hosp_max.max() )
    smoothing_columns = ["SimulationCases", "SimulationHospital", "SimulationCritical", "SimulationDeaths"]

    simulations = pd.DataFrame({"Date": dates, "SimulationCases": y_pred_cases.astype(int),
                                "SimulationHospital": y_pred_hosp.astype(int) + y_pred_critic.astype(int),
                                "SimulationCritical": y_pred_critic.astype(int),
                                "SimulationDeaths": y_pred_deaths.astype(int),
                               "SimulationInfectious": y_pred_infectious.astype(int)})

    simulations["R"] = data['R'].iloc[1:].values

    for e in smoothing_columns:
        simulations[e] = simulations[e].rolling(3, 2,center=True).mean().astype(int)

    if confidence_interval:

        R_t = data['R_min'].values
        args = (time_varying_reproduction, 5.6, 2.9, *params)

        sol = solve_ivp(seir.model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
        sus_min, exp_min, inf_min, rec_min, hosp_min, crit_min, deaths_min = sol.y

        R_t = data['R_max'].values
        args = (time_varying_reproduction, 5.6, 2.9, *params)
        sol = solve_ivp(seir.model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
        sus_max, exp_max, inf_max, rec_max, hosp_max, crit_max, deaths_max = sol.y

        y_pred_cases_min = np.clip(inf_min + rec_min + hosp_min + crit_min + deaths_min, 0, np.inf) * population
        y_pred_critic_min = np.clip(crit_min, 0, np.inf) * population
        y_pred_hosp_min = np.clip(hosp_min, 0, np.inf) * population
        y_pred_deaths_min = np.clip(deaths_min, 0, np.inf) * population
        y_pred_infectious_min = np.clip(inf_min, 0, np.inf) * population

        y_pred_cases_max = np.clip(inf_max + rec_max + hosp_max + crit_max + deaths_max, 0, np.inf) * population
        y_pred_critic_max = np.clip(crit_max, 0, np.inf) * population
        y_pred_hosp_max = np.clip(hosp_max, 0, np.inf) * population
        y_pred_deaths_max = np.clip(deaths_max, 0, np.inf) * population
        y_pred_infectious_max = np.clip(inf_max, 0, np.inf) * population

        simulations_min = pd.DataFrame(
            {"Date": dates[:len(y_pred_hosp_min)], "SimulationCases_min": y_pred_cases_min.astype(int),
             "SimulationHospital_min": y_pred_hosp_min.astype(int) + y_pred_critic_min.astype(int),
             "SimulationCritical_min": y_pred_critic_min.astype(int),
             "SimulationDeaths_min": y_pred_deaths_min.astype(int),
             "SimulationInfectious_min": y_pred_infectious_min.astype(int)})
        simulations_min["R_min"] = data['R_min'].iloc[1:].values

        simulations_max = pd.DataFrame(
            {"Date": dates[:len(y_pred_hosp_max)], "SimulationCases_max": y_pred_cases_max.astype(int),
             "SimulationHospital_max": y_pred_hosp_max.astype(int) + y_pred_critic_max.astype(int),
             "SimulationCritical_max": y_pred_critic_max.astype(int),
             "SimulationDeaths_max": y_pred_deaths_max.astype(int),
             "SimulationInfectious_max": y_pred_infectious_max.astype(int)})
        simulations_max["R_max"] = data['R_max'].iloc[1:].values

        for e in smoothing_columns:
            simulations_min["{}_min".format(e)] = simulations_min["{}_min".format(e)].rolling(3, 2,center=True).mean().astype(int)
            simulations_max["{}_max".format(e)] = simulations_max["{}_max".format(e)].rolling(3, 2,center=True).mean().astype(int)

        simulations = pd.merge(simulations, simulations_min, how="left", on="Date")
        simulations = pd.merge(simulations, simulations_max, how="left", on="Date")

        herd = (1-1/simulations["R_max"])* population
        min_pop = 0.05
        simulations["Herd_immunity"] = herd.clip(min_pop*population, population)

        simulations = simulations.dropna()

    return simulations


### updating means

def update_mean(df):
    df[utils.MOBILITY] = df[utils.MOBILITY].rolling(3, 2).mean()
    for f in utils.MOBILITY:
        days_15 = df[f].rolling(15, min_periods=14).mean().fillna(method="bfill")
        df["{}_15days".format(f)] = days_15

        days_10 = df[f].rolling(10, min_periods=9).mean().fillna(method="bfill")
        df["{}_10days".format(f)] = days_10

        days_5 = df[f].rolling(5, min_periods=4).mean().fillna(method="bfill")
        df["{}_5days".format(f)] = days_5

        days_30 = df[f].rolling(30, min_periods=29).mean().fillna(method="bfill")
        df["{}_30days".format(f)] = days_30

    return df


def simulate(country_lift, model_suffix, init_date, end_date):
    scaler = joblib.load('./models/scaler_{}.save'.format(model_suffix)) 
    mlp_clf = joblib.load('./models/mlp_{}.save'.format(model_suffix)) 

    with open('./models/metrics_{}.json'.format(model_suffix)) as fp:
        metrics = json.load(fp)
        y_var = np.power(metrics["std_test"],0.5)
        columns = metrics["x_columns"]
    
    X_lift = scaler.transform(country_lift[columns])
    y_lift = mlp_clf.predict(X_lift)

    country_lift["R"] = np.clip(y_lift, 0, 10)
    country_lift["R_min"] = np.clip(y_lift - y_var.mean()/2, 0, 10)
    country_lift["R_max"] = np.clip(y_lift + y_var.mean()/2, 0, 10)

    country_lift = update_seir(country_lift, init_date, end_date)

    return country_lift


def simulate_constantRt(df, end_date):
    init_date = df["Date"].tail(1).dt.date.values[0]
    current_date= init_date
    country_lift = df.copy()

    while current_date < end_date:
        current_date = current_date + timedelta(days=1)

        obj = {"Date": current_date}
        country_lift = country_lift.append(obj, ignore_index=True)

    country_lift = update_mean(country_lift.fillna(method="pad")).fillna(method="bfill")
    country_lift = update_seir(country_lift, init_date, end_date, None, confidence_interval=False)

    return country_lift


def create_calendar_from_scenario(features_country, measure_names, measure_dates, measure_values, init_date, end_date):
    country_calendar = features_country.copy()
    current_date = init_date

    while current_date < end_date:
        current_date = current_date + timedelta(days=1)

        obj = {"Date": current_date, "day_of_week":current_date.weekday}

        for i, measure in enumerate(measure_names):
            lift = current_date >= measure_dates[i] if measure_dates is not None and len(measure_dates) > i else current_date >= None
            if lift:
                obj[measure] = measure_values[i] if measure_values is not None else 0


        country_calendar = country_calendar.append(obj, ignore_index=True)

    country_calendar = update_mean(country_calendar.fillna(method="pad")).fillna(method="bfill")

    return country_calendar