from pandas.core.common import SettingWithCopyWarning

from sources import (
    gmobility_import as gi,
    oxford_import as oi,
    state_info_import as si,
    jhu_import as di,
)
import pandas as pd
import warnings
import sys
import os
import errno


DEFAULT_PATH="dataset"
DEFAULT_OUTPUT_CSV = f"{DEFAULT_PATH}/features.csv"
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

selected_measures = [
    ("school", "c1_schoolclosing"),
    ("public_transport", "c5_closepublictransport"),
    ("international_transport", "c8_internationaltravel"),
]


def add_days_granularity(out, features_list):
    for f in features_list:
        days_15 = out[f].rolling(15, min_periods=14).mean().fillna(method="bfill")
        out["{}_15days".format(f)] = days_15

        days_10 = out[f].rolling(10, min_periods=9).mean().fillna(method="bfill")
        out["{}_10days".format(f)] = days_10

        days_5 = out[f].rolling(5, min_periods=4).mean().fillna(method="bfill")
        out["{}_5days".format(f)] = days_5

        days_30 = out[f].rolling(30, min_periods=29).mean().fillna(method="bfill")
        out["{}_30days".format(f)] = days_30

    return out


def build_features_list():
    mobility_features_list = [
        "retail/recreation",
        "grocery/pharmacy",
        "parks",
        "transit_stations",
        "workplace",
        "residential",
    ]
    measures_feature_list = [n for n, _ in selected_measures]
    features_list = mobility_features_list + measures_feature_list
    return features_list


def filter_countries(demographics, merged):
    pop_country_names = demographics[1]["CountryName"].unique()
    pop_country_names_p14 = demographics[3]["CountryName"].unique()
    pop_country_names_p65 = demographics[2]["CountryName"].unique()

    pop_country_names = [
        value for value in pop_country_names if value in pop_country_names_p14
    ]
    pop_country_names = [
        value for value in pop_country_names if value in pop_country_names_p65
    ]

    X_filtered = merged[merged["CountryName"].isin(pop_country_names)]

    X_filtered["density"] = X_filtered["CountryName"].replace(
        demographics[0]["CountryName"].values, demographics[0]["value"].values
    )
    X_filtered["population"] = X_filtered["CountryName"].replace(
        demographics[1]["CountryName"].values, demographics[1]["value"].values
    )
    X_filtered["population_p65"] = X_filtered["CountryName"].replace(
        demographics[2]["CountryName"].values, demographics[2]["value"].values
    )
    X_filtered["population_p14"] = X_filtered["CountryName"].replace(
        demographics[3]["CountryName"].values, demographics[3]["value"].values
    )

    return X_filtered


def merge_features(measures, cases_death, mobility):
    m = pd.merge(
        mobility,
        measures,
        left_on=["Date", "CountryName"],
        right_on=["Date", "CountryName"],
        how="inner",
    )

    merged = pd.merge(
        m,
        cases_death,
        left_on=["Date", "CountryName"],
        right_on=["Date", "CountryName"],
        how="inner",
    )

    return merged


def extract_features(out_full_path):
    """
    Main method for feature extraction and compilation.
    Extract the features from oxford csv about the post-covid measures per countries.
    Extract the features from JHU csv about the  confirmed cases and deaths per countries
    Extract the features from google about the mobility per countries.
    Add state informations (demographic & gdp)
    Add granularity with a rolling window of 5 10 15 and 30 days
    save the results to a csv
    """
    print(f"Preparing data oxford measure features...", end="")
    measures_features = oi.prepare_measure_features(selected_measures)
    print("done")
    print(f"Preparing data jhu cases & death features...", end="")
    cases_death_features = di.prepare_cases_deaths()
    print("done")
    print(f"Preparing data google mobility features...", end="")
    mobility_features = gi.prepare_mobility_features()
    print("end")

    print("First merge ...", end="")
    features = merge_features(
        measures_features, cases_death_features, mobility_features
    )
    print("done")
    print("Preparing demographic data...", end="")
    demographics = si.prepare_demographic_features()
    print("done")
    print("Second merge ...", end="")
    features = filter_countries(demographics, features)
    print("done")
    print("Preparing state info data...", end="")
    features = si.prepare_gdp_features(features)
    print("done")

    features_list = build_features_list()

    print("Adding days granularity...", end="")
    features = add_days_granularity(features, features_list)
    print("done")
    print("Writting to csv...", end="")
    if not os.path.exists(os.path.dirname(out_full_path)):
        try:
            os.makedirs(os.path.dirname(out_full_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(out_full_path, "w") as f:
        features.to_csv(f)
        print(f"done. Saved to {f.name}")


if __name__ == "__main__":
    args = sys.argv
    out_full_path = DEFAULT_OUTPUT_CSV
    if len(args) - 1 == 0:
        print(f"No output file path provided using default [{out_full_path}]")
    else:
        out_full_path = args[1]
        print(f"The output will be saved to [{out_full_path}]")

    extract_features(out_full_path)
