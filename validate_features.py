import sys
import pandas as pd


def validate_ds(features):
    if features.shape[1] != 57:
        exit(1)


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    ds = pd.read_csv(f"{path}")
    validate_ds(ds)