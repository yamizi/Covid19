import pandas as pd
import numpy as np

country_info_uri = "raw_data/metrics_country.csv"
demographic_info_path = "raw_data/population"


def prepare_demographic_features() -> list:
    """
    Method to join demographic data for each countries.
    :return: list containing demographic data for each countries
    :rtype: list
    """
    demographics = []
    demographics_label = ["density", "total", "population_p65", "population_p14"]
    demographics_files = [
        "population_density_long",
        "population_total_long",
        "population_above_age_65_percentage_long",
        "population_below_age_14_percentage_long",
    ]
    for i, e in enumerate(demographics_files):
        f = f"{demographic_info_path}/{e}.csv"
        demo = pd.read_csv(f)
        demo = demo.loc[demo.groupby("Country Name").Year.idxmax()]
        demo = demo[["Country Name", "Count"]]
        demo.columns = ["CountryName", "value"]
        demo = demo.replace("Korea, Rep.", "South Korea")
        demographics.append(demo)

    return demographics


def prepare_gdp_features(X_filtered) -> pd.DataFrame:
    """
    Method to join gdp values for each countries to the existing dataframe
    :param X_filtered: Dataframe containg all the filtered countries
    :return: Dataframe enhanced with gdp area and region indices
    """
    infos = pd.read_csv(country_info_uri)
    infos = infos.replace("Korea, South", "South Korea")
    infos["Country"] = infos["Country"].str.strip()
    # X_country_names = X_filtered["CountryName"].unique()
    test_country_names = infos["Country"].unique()
    X_filtered = X_filtered[X_filtered["CountryName"].isin(test_country_names)]
    regions_indices = infos.Region.unique()
    infos["regions_indices"] = infos["Region"].replace(
        regions_indices, np.arange(len(regions_indices))
    )
    X_filtered["gdp"] = X_filtered["CountryName"].replace(
        infos["Country"].values, infos["GDP ($ per capita)"].values
    )
    X_filtered["area"] = X_filtered["CountryName"].replace(
        infos["Country"].values, infos["Area (sq. mi.)"].values
    )
    X_filtered["region"] = X_filtered["CountryName"].replace(
        infos["Country"].values, infos["regions_indices"].values
    )

    return X_filtered
