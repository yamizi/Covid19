from model_training_multisector import get_output_name
import datetime
import shutil
import os
import pandas as pd
import numpy as np
from utils import compute_sector_rt, prepare_cases, highest_density_interval, get_posteriors, case_time
from multisectors_export_simulations import export_simulations_on_real_data, make_simulations_on_reborn_inputs
import matplotlib.pyplot as plt
from model_training_multisector import train_mlp
import utils, time

import argparse


# Config global variable.
OUTPUT_SUFFIX="economic_sectors"
MODEL_FOLDER = './models'
DATA_PATH = './data/luxembourg'
BACKUP_FOLDER = './backups/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M"))



def compute_Rt_from_cases():
    sectors2_df = pd.read_excel("PJG397_Covid19_daily_workSector_Residents_IGSS.xlsx", "Daily_Infections_by_Sector")
    sectors2_df["date"] = pd.to_datetime(sectors2_df['Dates\Age Range'])
    sectors2_df.index = pd.to_datetime(sectors2_df.date)
    sectors2_df = sectors2_df.drop(["date",'Dates\Age Range'], axis=1).dropna()

    start, end = sectors2_df.index[0], sectors2_df.index[-1]
    dates = pd.date_range(start, end, freq='D')

    sectors_1 = compute_sector_rt(sectors2_df, new_provided=True, max_elements=25)


    ref_df = pd.DataFrame([],index=dates)
    
    all_sects = {}
    for k in sectors_1.keys():
        sector_A = sectors_1[k]

        A = ref_df.join(sector_A).interpolate()
        A = A.rolling(7,
                win_type='gaussian',
                min_periods=1,
                center=True).mean(std=2)
        all_sects[k] = A

    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    confirmed_df = confirmed_df.loc[lambda x:x["Province/State"].isnull()]

    confirmed_df.rename(columns={'Country/Region':'Country'}, inplace=True)
    cols = confirmed_df.keys()
    confirmed_cases = confirmed_df.copy()
    confirmed_cases.index = confirmed_cases.Country
    confirmed_cases = confirmed_cases.loc[:, cols[4]:].transpose()

    luxembourg_allcases = confirmed_cases["Luxembourg"]
    luxembourg_allcases.index = pd.to_datetime(luxembourg_allcases.index,format="%m/%d/%y")

    
    _, smoothed = prepare_cases(luxembourg_allcases, cutoff=5)
    posteriors, _ = get_posteriors(smoothed, sigma=0.05, gamma=1/7)


    hdis_90 = highest_density_interval(posteriors, p=.9)
    hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('ML')

    most_likely = most_likely.rolling(7,
            win_type='gaussian',
            min_periods=1,
            center=True).mean(std=1)

    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
    
    all_df = result[["ML"]].copy()
    all_df["date"] = all_df.index

    for k in all_sects.keys():
        rows = all_sects[k]
        all_df[k] = rows.loc[~rows.index.duplicated(keep='first')]

    all_df = all_df.fillna(0).rename({"ML":"ALL"},axis=1)

    for c in all_df.columns:
        all_df[c] = all_df[[c,"ALL"]].sum(axis=1)/2
    
    all_df = all_df.drop(["date"],axis=1)
    all_df  = all_df.loc[start: end]

    filename = "data/luxembourg/luxembourg_allsectors_rt.csv"
    print('[+] Save Rt file at:', filename)
    all_df.to_csv(filename)

    return start, end


def oxford_dataset():
    oxford_dataset = pd.read_csv("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv")
    features = ["C1_School closing","C2_Workplace closing","C3_Cancel public events","C4_Restrictions on gatherings","C5_Close public transport","C6_Stay at home requirements","C7_Restrictions on internal movement","C8_International travel controls","H1_Public information campaigns","H2_Testing policy","H3_Contact tracing"]
    short_features =[e[0:2] for e in features]
    oxford_luxembourg = oxford_dataset[oxford_dataset["CountryName"]=="Luxembourg"]
    oxford_luxembourg["Date"] = pd.to_datetime(oxford_luxembourg["Date"],format="%Y%m%d")
    oxford_luxembourg.index = oxford_luxembourg["Date"]
    oxford_luxembourg = oxford_luxembourg[features]
    oxford_luxembourg.columns = short_features
    oxford_luxembourg = oxford_luxembourg.dropna()

    oxford_luxembourg.reset_index(inplace =True)
    oxford_luxembourg['C1'] = oxford_luxembourg.apply(lambda x: case_time(x.Date), axis=1)
    oxford_luxembourg.set_index('Date', inplace=True)

    return oxford_luxembourg

def google_mobility_dataset():
    mobility_raw_csv = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv',parse_dates=["date"])
    mobility_features = ["retail_and_recreation_percent_change_from_baseline","grocery_and_pharmacy_percent_change_from_baseline","parks_percent_change_from_baseline","transit_stations_percent_change_from_baseline","workplaces_percent_change_from_baseline"]
    mobility_short_features=["retail/recreation","grocery/pharmacy","parks","transit","workplaces"]
    mobility_luxembourg = mobility_raw_csv[mobility_raw_csv["country_region_code"]=="LU"]
    mobility_luxembourg.index = mobility_luxembourg["date"]
    mobility_luxembourg  = mobility_luxembourg[mobility_features]
    mobility_luxembourg.columns= [mobility_short_features]

    smoothed_mobility_luxembourg = mobility_luxembourg.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()

    return smoothed_mobility_luxembourg


def convert_apple_dataset(apple_dataset:pd.DataFrame):
    apple_dataset = apple_dataset[apple_dataset.region == 'Luxembourg']
    apple_dataset = apple_dataset.drop(columns=['geo_type', 'region', 'alternative_name', 'sub-region', 'country'])
    apple_dataset.index = apple_dataset.transportation_type
    apple_dataset = apple_dataset.drop(columns=['transportation_type'])
    apple_dataset = apple_dataset.T
    apple_dataset.index = apple_dataset.index.rename('')
    apple_dataset = apple_dataset.rename(columns={"transit":'public_transport'})
    return apple_dataset 


def download_dataset(apple_file:str, start_date, end_date):
    print('[+] Download Oxford Dataset...')
    oxford_dset = oxford_dataset()
    oxford_dset = oxford_dset.loc[start_date: end_date]
    oxford_dset.to_csv(DATA_PATH + '/luxembourg_npi_oxford.csv')


    print('[+] Download Google Dataset...')
    google_dset = google_mobility_dataset()
    google_dset = google_dset.loc[start_date: end_date]
    google_dset.to_csv(DATA_PATH + '/luxembourg_mobility_google.csv')

    print('[+] Convert Apple Dataset...')
    apple_dset = pd.read_csv(apple_file)
    apple_dset = convert_apple_dataset(apple_dset)
    apple_dset.index = pd.to_datetime(apple_dset.index)
    apple_dset = apple_dset.loc[start_date: end_date]
    apple_dset.to_csv(DATA_PATH + '/luxembourg_mobility_apple.csv')



def backup_reborn_data():
    for root, _, files in os.walk(DATA_PATH):
        os.makedirs(BACKUP_FOLDER + root, exist_ok=True)

        for f in files: 
            print(root + '/' + f)
            shutil.copy2(root + '/' + f, BACKUP_FOLDER + root + '/' + f )

def backup_models():
    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    model_path = get_output_name(MODEL_FOLDER, 'mlp', OUTPUT_SUFFIX, 'save')
    scaler_path = get_output_name(MODEL_FOLDER, 'scaler', OUTPUT_SUFFIX, 'save')
    metrics_path = get_output_name(MODEL_FOLDER, 'metrics', OUTPUT_SUFFIX, 'json')
    features_path = get_output_name(MODEL_FOLDER, 'features', OUTPUT_SUFFIX, 'csv')

    backup_model_path = get_output_name(BACKUP_FOLDER, 'mlp', OUTPUT_SUFFIX, 'save')
    backup_scaler_path = get_output_name(BACKUP_FOLDER, 'scaler', OUTPUT_SUFFIX, 'save')
    backup_metrics_path = get_output_name(BACKUP_FOLDER, 'metrics', OUTPUT_SUFFIX, 'json')
    backup_features_path = get_output_name(BACKUP_FOLDER, 'features', OUTPUT_SUFFIX, 'csv')

    shutil.copy2(model_path, backup_model_path)
    shutil.copy2(scaler_path, backup_scaler_path)
    shutil.copy2(metrics_path, backup_metrics_path)
    shutil.copy2(features_path, backup_features_path)





def train_rt_model():
    t_start = time.perf_counter()
    data, y = utils.load_luxembourg_dataset()

    model, reports = train_mlp(data, y, output_suffix="economic_sectors")
    print(reports)

    t_stop = time.perf_counter()

    print('\n')
    print("--------------------------------------------------")
    print('Elapsed time:{:.1f} [sec]'.format(t_stop-t_start))
    print("--------------------------------------------------") 


def make_simulations():
    export_simulations_on_real_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("IGSS_dataset_path", help="the path to the IGSS CSV file (Has to be formated like the old IGSS file.)")
    parser.add_argument("apple_dataset_path", help="The path to the apple CSV file")
    args = parser.parse_args()

    print('[+] Backup models...')
    backup_models()
    print('[+] Backup data...')
    backup_reborn_data()
    print('[+] Backup done.')

    print('[+] Re-compute Rt from new cases file')
    print('[+] File location :', args.IGSS_dataset_path)
    start_date, end_date = compute_Rt_from_cases()

    print('[+] Replace IGSS cases file...')
    shutil.copy2(args.IGSS_dataset_path, DATA_PATH + '/PJG397_Covid19_daily_workSector_Residents_IGSS.xlsx')

    download_dataset(args.apple_dataset_path, start_date, end_date)

    print('[+] Train the Rt model...')
    train_rt_model()

    print('[+] Run simulations...')
    make_simulations()
    






