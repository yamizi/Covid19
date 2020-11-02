import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error
from data_visualization import plot_model_results

from helpers.seir import model

OPTIM_DAYS = 60  # Number of days to use for the optimisation evaluation

# Use a Hill decayed reproduction number
def eval_model_decay(params, data, population, return_solution=False, forecast_days=0):
    R_0, t_hosp, t_crit, m, c, f, k, L = params
    max_days = len(data) + forecast_days

    # https://github.com/SwissTPH/openmalaria/wiki/ModelDecayFunctions
    # Hill decay. Initial values: R_0=2.2, k=2, L=50
    def time_varying_reproduction(t):
        return R_0 / (1 + (t / L) ** k)

    N = population
    n_infected = data['ConfirmedCases'].iloc[0]

    initial_state = [(N - n_infected) / N, 0, n_infected / N, 0, 0, 0, 0]
    args = (time_varying_reproduction, 5.6, 2.9, t_hosp, t_crit, m, c, f)

    sol = solve_ivp(model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))

    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    y_pred_cases = np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population
    y_true_cases = data['ConfirmedCases'].values
    y_pred_fat = np.clip(deaths, 0, np.inf) * population

    y_true_fat = data['ConfirmedDeaths'].values

    optim_days = min(OPTIM_DAYS, len(data))  # Days to optimise for
    weights = 1 / np.arange(1, optim_days + 1)[::-1]  # Recent data is more heavily weighted

    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)
    msle_fat = mean_squared_log_error(y_true_fat[-optim_days:], y_pred_fat[-optim_days:], weights)
    msle_final = np.mean([msle_cases, msle_fat])

    if return_solution:
        return msle_final, sol
    else:
        return msle_final


# Fit a model on the full dataset (i.e. no validation)
def fit_model(area_name,df, population,
              initial_guess=[3.6, 4, 14, 0.8, 0.1, 0.3, 2, 50],
              bounds=((1, 20),  # R bounds
                      (0.5, 10), (2, 20),  # transition time param bounds
                      (0.5, 1), (0, 1), (0, 1), (1, 5), (1, 100)),  # fraction time param bounds
              make_plot=True, pred_days=0):
    train_data = df[(df["CountryName"]==area_name) & (df['ConfirmedCases'] >0)]

    # begin = train_data.index[-1]
    # test_index = pd.date_range(start=train_data['Date'][begin], periods=pred_days)
    # dates_all = train_data.index.append(test_index)

    res_decay = minimize(eval_model_decay, initial_guess, bounds=bounds,
                         args=(train_data, population, False),
                         method='L-BFGS-B')

    msle, sol = eval_model_decay(res_decay.x, train_data, population, True, pred_days)
    res = res_decay

    # Calculate the R_t values
    t = np.arange(len(train_data))
    R_0, t_hosp, t_crit, m, c, f, k, L = res.x
    R_t = pd.Series(R_0 / (1 + (t / L) ** k), train_data.index)

    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    y_pred = pd.DataFrame({
        'ConfirmedCases': np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population,
        'Fatalities': np.clip(deaths, 0, np.inf) * population,
        'R': R_t,
        'HospitalizedCases': hosp * population,
        'ExposedCases': exp * population,
        'CriticalCases': crit * population,
        'RecoveredCases': rec * population,
        'InfectiousCases': inf * population,
        'Date': train_data['Date'],
        'CountryName': area_name,
        't_hosp': t_hosp,
        't_crit': t_crit,
        'm': m,
        'c': c,
        'f': f
    })

    if make_plot:
        print(f'R: {res.x[0]:0.3f}, t_hosp: {res.x[1]:0.3f}, t_crit: {res.x[2]:0.3f}, '
              f'm: {res.x[3]:0.3f}, c: {res.x[4]:0.3f}, f: {res.x[5]:0.3f}, k: {k:0.3f}, L: {L:0.3f}', msle)
        plot_model_results(y_pred, train_data)

    return y_pred
