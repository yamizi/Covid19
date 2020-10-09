import pandas as pd

# Johns Hopkins University Center for Systems Science and Engineering root dataset url
JHU_CSSE_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"


def __load_confirmed_cases_jhu() -> pd.DataFrame:
    """
    Load JHU raw confirmed cases and return the (dates x countries) dataframe whose values corresponds to confirmed
    covid cases
    :rtype: Pandas.Dataframe
    """
    confirmed_df = pd.read_csv(
        f"{JHU_CSSE_url}/time_series_covid19_confirmed_global.csv"
    )
    cases_countries = confirmed_df[~confirmed_df["Province/State"].notnull()]
    cases_countries.index = cases_countries["Country/Region"]
    confirmed = cases_countries.drop(
        ["Lat", "Long", "Province/State", "Country/Region"], axis=1
    )
    return confirmed.transpose()


def __load_death_cases_jhu() -> pd.DataFrame:
    """
        Load JHU raw death cases and return the (dates x countries) dataframe whose values corresponds to covid deaths
        :rtype: Pandas.Dataframe
        """
    death_df = pd.read_csv(f"{JHU_CSSE_url}/time_series_covid19_deaths_global.csv")
    death_countries = death_df[~death_df["Province/State"].notnull()]
    death_countries.index = death_countries["Country/Region"]
    deaths = death_countries.drop(
        ["Lat", "Long", "Province/State", "Country/Region"], axis=1
    )
    return deaths.transpose()


def prepare_cases_deaths() -> pd.DataFrame:
    """
    Method to create the dataframe holding both confirmed covid cases and covid deaths for each countries for each
    available dates.
    :rtype: pd.DataFrame
    """
    cases = __load_confirmed_cases_jhu()
    deaths = __load_death_cases_jhu()
    dates = cases.index.values[2:]
    dates = [pd.to_datetime(d) for d in dates]

    cases_deaths = pd.DataFrame(
        {"Date": [], "CountryName": [], "ConfirmedCases": [], "ConfirmedDeaths": []}
    )
    countries = list(cases.columns)
    for c in countries:
        confirmed_c = cases[c].values
        death_c = deaths[c].values
        country = [c] * len(confirmed_c)
        df = list(zip(dates, country, confirmed_c, death_c))
        df = pd.DataFrame(
            df, columns=["Date", "CountryName", "ConfirmedCases", "ConfirmedDeaths"]
        )
        cases_deaths = cases_deaths.append(df, ignore_index=True)

    return cases_deaths
