import pandas as pd
import wget

xls_oxford_url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/OxCGRT_timeseries_all.xlsx"
xls_oxford_path = "./raw_data/oxford_pandemic_measures/OxCGRT_timeseries_all.xlsx"
# Measure alias and corresponding sheet name
selected_measures = [
    ("school", "c1_schoolclosing"),
    ("public_transport", "c5_closepublictransport"),
    ("international_transport", "c8_internationaltravel"),
]


def download_oxford_data():
    wget.download(f"{xls_oxford_url}", f"{xls_oxford_path}")


def __load_xls() -> dict:
    """
    Loads the corresponding excel oxford_file and returns dict of {measure_name : corresponding_df}
    :return:
    """
    xls_oxford = pd.ExcelFile(f"{xls_oxford_path}")
    res = {}
    for (df_name, s_name) in selected_measures:
        df = pd.read_excel(xls_oxford, f"{s_name}")
        df.index = df.CountryName
        df = df.drop(["CountryCode", "CountryName"], axis=1)
        df = df.transpose()
        res[f"{df_name}"] = df

    return res


def __normalize(unormalized_measures):
    normalized_measures = unormalized_measures.copy()
    for (measure_name, _) in selected_measures:
        max_value = normalized_measures[measure_name].max()
        factor = -int(100 / max_value)
        normalized_measures[measure_name] *= factor

    return normalized_measures


def prepare_measure_features(selected_measures_in=None):
    """
    Method to join all meaures value per country per dates returns dataframe (dates x (countrynames,measures))
    :return:
    """
    global selected_measures
    if selected_measures_in is None:
        selected_measures_in = selected_measures

    selected_measures = selected_measures_in

    measures_dict = __load_xls()
    measure_name, _ = selected_measures[0]
    measure = measures_dict.get(measure_name)
    dates = measure.index.values
    dates = [pd.to_datetime(d, format="%d%b%Y") for d in dates]
    countries = list(measure.columns)
    d = {"Date": [], "CountryName": []}

    for (n, _) in selected_measures:
        d[f"{n}"]: []

    oxford_measures = pd.DataFrame(d)

    for c in countries:
        country = [c] * len(measure)
        measures_df = []
        columns = [
            "Date",
            "CountryName",
        ]
        for key, val in measures_dict.items():
            measures_df.append(val[c].values)
            columns.append(f"{key}")

        df = list(zip(dates, country, *measures_df))
        df = pd.DataFrame(df, columns=columns, )
        oxford_measures = oxford_measures.append(df, ignore_index=True)

    oxford_measures = __normalize(oxford_measures)

    return oxford_measures


if __name__ == "__main__":
    download_oxford_data()
