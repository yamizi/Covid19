import sys, getopt
import pandas as pd

sys.path.append("./")

from helpers.seir_fit import fit_model


def main(argv):
    all_countries = pd.DataFrame({
        'ConfirmedCases': [],
        'Fatalities': [],
        'R': [],
        'HospitalizedCases': [],
        'CriticalCases': [],
        'Date': [],
        'CountryName': []
    })
    dataset_path = argv[0] or "./dataset/features.csv" 
    output_seirhcd_path = argv[1] or "./data/seirhcd.csv"
    dataset = pd.read_csv(dataset_path, parse_dates=['Date'])
    # dataset = pd.read_csv("./data/google.csv", parse_dates=['Date'])
    dataset = dataset.drop(["Unnamed: 0"], axis=1)

    dataset.tail(1)

    d = dataset[["CountryName", "population"]].groupby("CountryName").min()["population"]
    populations = dict(zip(list(d.index), d.values))
    countries = list(populations.keys())

    for c in countries:
        print("Country ", c)
        out = fit_model(c, dataset, populations.get(c), make_plot=False)
        all_countries = all_countries.append(out)
    all_countries.to_csv(output_seirhcd_path)


if __name__ == "__main__":
    main(sys.argv[1:])
