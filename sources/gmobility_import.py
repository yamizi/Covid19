import pandas as pd

google_mobility_url = (
    "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
)


def prepare_mobility_features():
    """
    Retrieve google mobility data per countries
    :return: Dataframe containing mobility data per countries
    """
    mobility_csv = pd.read_csv(
        f"{google_mobility_url}", parse_dates=["date"], dtype={4: str}
    )

    mobility = mobility_csv[
        [
            "country_region",
            "date",
            "retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "parks_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ]
    ]
    mobility.columns = [
        "CountryName",
        "Date",
        "retail/recreation",
        "grocery/pharmacy",
        "parks",
        "transit_stations",
        "workplace",
        "residential",
    ]

    mobility = mobility.dropna()
    mobility[
        [
            "retail/recreation",
            "grocery/pharmacy",
            "parks",
            "transit_stations",
            "workplace",
            "residential",
        ]
    ] = mobility[
        [
            "retail/recreation",
            "grocery/pharmacy",
            "parks",
            "transit_stations",
            "workplace",
            "residential",
        ]
    ].clip(
        -100, 100
    )

    return mobility
