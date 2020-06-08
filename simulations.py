import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os, json, random
from datetime import timedelta, datetime, date

from scipy.integrate import solve_ivp
from SEIR import SEIR_HCD_model

features = ["parks", "residential", "retail/recreation", "transit_stations", "workplace"] #, "grocery/pharmacy","school",      "international_transport"]
periods = ["","_5days","_10days","_15days","_30days"]


### building seir predictions:

def update_seir(df, active_date, e_date, folder=None, l_date=None, confidence_interval=True):

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
    #print("disease params", params, np.sum([params, ref_params], axis=1))
    # "decay_values"
    params.append(False)


    ref_data = df[df["Date"] == active_date + timedelta(days=-7)]
    inf_data = df[df["Date"] == active_date + timedelta(days=-14)]
    #print(data['Date'].iloc[0],ref_data.iloc[0])

    if l_date is None:
        l_date = active_date
    population = data["population"].min() *0.7 # Herd immunity is assumed at 70%
    N = population
    n_infected = data['ConfirmedCases_y'].iloc[0]-inf_data['ConfirmedCases_y'].iloc[0] #data['InfectiousCases'].iloc[0]
    n_exposed = data['ConfirmedCases_y'].iloc[0] - ref_data['ConfirmedCases_y'].iloc[0] #data['ExposedCases'].iloc[0]
    n_hospitalized = (1-params[2]) * n_exposed #data['HospitalizedCases'].iloc[0]*1.5
    n_exposed = params[2] * n_exposed 
    n_critical = (params[3]) * n_hospitalized #data['CriticalCases'].iloc[0]*1.5
    n_recovered =  inf_data['ConfirmedCases_y'].iloc[0] -inf_data['ConfirmedDeaths'].iloc[0]  #data['RecoveredCases'].iloc[0]
    n_deaths = data['ConfirmedDeaths'].iloc[0]
    # S, E, I, R, H, C, D
    initial_state = [(N - n_infected) / N, n_exposed / N, n_infected / N, n_recovered / N, n_hospitalized / N,
                     n_critical / N, n_deaths / N]

    R_t = data['R'].values

    def time_varying_reproduction(t):
        index = np.min((int(t), len(R_t)-1))
        return R_t[index]

    args = (time_varying_reproduction, 5.6, 2.9, *params)

    init_date = datetime.combine(active_date, datetime.min.time())
    max_days = ((e_date - pd.Timestamp(init_date)) / np.timedelta64(1, 'D'))
    max_days = int(max_days)
    sol = solve_ivp(SEIR_HCD_model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    y_pred_cases = np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population
    y_pred_critic = np.clip(crit, 0, np.inf) * population
    y_pred_hosp = np.clip(hosp, 0, np.inf) * population
    y_pred_deaths = np.clip(deaths, 0, np.inf) * population
    y_pred_infectious = np.clip(inf, 0, np.inf) * population

    dates = data["Date"].iloc[1:]
    l = len(dates)
    dt = np.arange(l)
    ticks = dates.values  # dt.strftime('%d/%m/%Y')
    fig_size = (20, 5)

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

        sol = solve_ivp(SEIR_HCD_model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
        sus_min, exp_min, inf_min, rec_min, hosp_min, crit_min, deaths_min = sol.y

        R_t = data['R_max'].values
        args = (time_varying_reproduction, 5.6, 2.9, *params)
        sol = solve_ivp(SEIR_HCD_model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
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

        simulations = simulations.dropna()#fillna(method='ffill')

    if folder is not None:
        simulations.to_csv("{}/out.csv".format(folder))

        fig_hospitals = plt.figure(figsize=fig_size)
        plt.plot(dt, simulations["SimulationHospital"] + simulations["SimulationCritical"],
                 label="Probable hospitalized")
        plt.plot(dt, y_pred_hosp_min + y_pred_critic_min, label="Best case hospitalized")
        plt.plot(dt, y_pred_hosp_max + y_pred_critic_max, label="Worst case hospitalized")
        plt.xticks(dt, ticks, rotation=90)
        plt.tight_layout()
        plt.legend()

        plt.savefig("{}/hospitals.png".format(folder))
        plt.close(fig_hospitals)
        fig_hospitals.clf()

        fig_critical = plt.figure(figsize=fig_size)
        plt.plot(dt, simulations["SimulationCritical"], label="Probable critical")
        plt.plot(dt, y_pred_critic_min, label="Best case critical")
        plt.plot(dt, y_pred_critic_max, label="Worst case critical")
        plt.xticks(dt, ticks, rotation=90)
        plt.tight_layout()
        plt.legend()

        plt.savefig("{}/criticals.png".format(folder))
        plt.close(fig_critical)
        fig_critical.clf()

        fig_deaths = plt.figure(figsize=fig_size)
        plt.plot(dt, simulations["SimulationDeaths"], label="Probable deaths")
        plt.plot(dt, y_pred_deaths_min, label="Best case deaths")
        plt.plot(dt, y_pred_deaths_max, label="Worst case deaths")
        plt.xticks(dt, ticks, rotation=90)
        plt.tight_layout()
        plt.legend()

        plt.savefig("{}/deaths.png".format(folder))
        plt.close(fig_deaths)
        fig_deaths.clf()

        fig_cases = plt.figure(figsize=fig_size)
        plt.plot(dt, y_pred_cases, label="Probable cases")
        plt.plot(dt, y_pred_cases_min, label="Best case cases")
        plt.plot(dt, y_pred_cases_max, label="Worst case cases")
        plt.xticks(dt, ticks, rotation=90)
        plt.tight_layout()
        plt.legend()

        plt.savefig("{}/cases.png".format(folder))
        plt.close(fig_cases)
        fig_cases.clf()

    return simulations


### updating means

def update_mean(df):
    df[features] = df[features].rolling(3, 2).mean()
    for f in features:
        days_15 = df[f].rolling(15, min_periods=14).mean().fillna(method="bfill")
        df["{}_15days".format(f)] = days_15

        days_10 = df[f].rolling(10, min_periods=9).mean().fillna(method="bfill")
        df["{}_10days".format(f)] = days_10

        days_5 = df[f].rolling(5, min_periods=4).mean().fillna(method="bfill")
        df["{}_5days".format(f)] = days_5

        days_30 = df[f].rolling(30, min_periods=29).mean().fillna(method="bfill")
        df["{}_30days".format(f)] = days_30

    #df[["S7_International travel controls"]] = df[["S7_International travel controls"]].rolling(15, 14).mean()
    #df[["S1_School closing", "S3_Cancel public events"]] = df[["S1_School closing", "S3_Cancel public events"]].rolling(7, 6).mean()

    return df


def simulate(df, measures_to_lift, measure_value, end_date, lift_date, columns, yvar, mlp_clf, scaler,
             measure_values=None, lift_date_values=None, base_folder="./plots/simulations", seed="", confidence_interval=True, filter_output=None):
    #print("Building simulation for ", measures_to_lift, df["CountryName"].unique())

    init_date = df["Date"].tail(1).dt.date.values[0]
    all_simulations = []
    for k, measure_to_lift in enumerate(measures_to_lift):
        country_lift = df.copy()
        current_date = init_date

        if base_folder is None:
            folder = None
        else:
            folder = "{}/{}".format(base_folder, "_".join(measure_to_lift).replace("/",
                                                                                   "")) if seed == "" else "./plots/simulations/{}".format(
                seed)
            os.makedirs(folder, exist_ok=True)

        while current_date < end_date:
            current_date = current_date + timedelta(days=1)

            obj = {"Date": current_date, "day_of_week":current_date.weekday}

            measures_translations = {"S7_International travel controls":"international_transport","S1_School closing":"school"}
            # df[["S1_School closing", "S3_Cancel public events"

            measure_labels = [measures_translations.get(e,e) for e in measure_to_lift]
            vals = measure_values[k] if measure_values is not None and len(measure_values) > 1 else measure_values

            for i, measure in enumerate(measure_labels):
                lift = current_date >= lift_date_values[i] if lift_date_values is not None and len(
                    lift_date_values) > i else current_date >= lift_date
                if lift:
                    obj[measure] = vals[i] if vals is not None else measure_value


            country_lift = country_lift.append(obj, ignore_index=True)

        country_lift = pd.get_dummies(country_lift, prefix="day_of_week", columns=["day_of_week"])
        country_lift = update_mean(country_lift.fillna(method="pad")).fillna(method="bfill")
        X_lift = scaler.transform(country_lift[columns])
        y_lift = mlp_clf.predict(X_lift)

        country_lift["R"] = np.clip(y_lift, 0, 10)
        country_lift["R_min"] = np.clip(y_lift - yvar.mean()/2, 0, 10)
        country_lift["R_max"] = np.clip(y_lift + yvar.mean()/2, 0, 10)

        measures_to_display = ["transit_stations","retail/recreation","school","workplace","international_transport","grocery/pharmacy", "parks"]

        if folder is not None:
            ax1 = country_lift.plot(x="Date", y="R", figsize=(20, 5), color="red")
            plt.fill_between(np.arange(len(country_lift["R"])), country_lift["R_min"], country_lift["R_max"],
                             color="pink", alpha=0.5, label="Confidence interval")
            ax1.legend(loc="lower left")
            ax2 = ax1.twinx()
            ax2.spines['right'].set_position(('axes', 1.0))
            print(country_lift.head(1).to_dict(orient='records'))
            country_lift.plot(ax=ax2, x="Date", y=measures_to_display)
            ax2.legend(loc="lower right")

            fig_R = ax1.get_figure()

            fig_R.savefig("{}/reproduction_rate.png".format(folder))
            plt.close(fig_R)
            fig_R.clf()
            plt.close("all")

        country_lift = update_seir(country_lift, init_date, end_date, folder, confidence_interval=confidence_interval)

        if filter_output:
            all_simulations.append(country_lift[filter_output])
        else:
            all_simulations.append(country_lift)

    return country_lift if len(measures_to_lift)==1 else all_simulations


def simulate_constantRt(df, end_date, Rt=None):


  init_date = df["Date"].tail(1).dt.date.values[0]
  current_date= init_date
  country_lift = df.copy()

  while current_date < end_date:
      current_date = current_date + timedelta(days=1)

      obj = {"Date": current_date}
      country_lift = country_lift.append(obj, ignore_index=True)

  if Rt is not None:
      print(country_lift[["Date","R"]])
      country_lift["R"] = Rt
  country_lift = update_mean(country_lift.fillna(method="pad")).fillna(method="bfill")
  country_lift = update_seir(country_lift, init_date, end_date, None, confidence_interval=False)

  return country_lift
