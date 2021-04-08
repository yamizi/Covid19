import matplotlib, os
#matplotlib.use('Agg')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import stats as sps


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,median_absolute_error


FIGURE_FOLDER = 'figures/'
EXTENSION = '.png'


REGIONS = ['region_{}'.format(i) for i in range(9)]
DAYS_OF_THE_WEEK = ['day_of_week_{}'.format(i) for i in range(7)]
DEMOGRAPHY = ["density", "population", "population_p14", "population_p65", "gdp", "area"]
MOBILITY = ["parks", "residential", "retail/recreation", "transit_stations", "workplace"]
SEIR = ['ConfirmedCases_x', 'ConfirmedCases_y', 'ConfirmedDeaths', 'Fatalities', 'HospitalizedCases','CriticalCases', 'ExposedCases', 'RecoveredCases', 'InfectiousCases', 't_hosp', 't_crit', 'm', 'c', 'f']

MOBILITY_WINDOWS = ["{}{}".format(f,p) for p in ["","_5days","_10days","_15days","_30days"] for f in MOBILITY]


def features_values(suffix='random'):
    return pd.read_csv('models/features_{}.csv'.format(suffix), parse_dates=['Date'], index_col=0)

def metrics_report(X_test, y_test, reg):
    y_pred = reg.predict(X_test)
    return {'r2_score': r2_score(y_test, y_pred), 
          'mean_absolute_error': mean_absolute_error(y_test, y_pred),
          'mean_squared_error': mean_squared_error(y_test, y_pred),
         'median_absolute_error': median_absolute_error(y_test, y_pred),
           'RMSE': np.sqrt(mean_absolute_error(y_test, y_pred))}


def extend_features_with_means(df,columns,smoothing_days):
    for c in columns:
        for p in smoothing_days:
            smoothed = df[c].astype(float).rolling(p,
                     win_type='gaussian',
                     min_periods=p-1).mean(std=2)

            df["{}_{}".format(c,p)] = smoothed.fillna(method="bfill")

    return df

def load_luxembourg_dataset(extend_features = True):
    '''
    Load data from the dataset folder, all the values for the files are hardcoded.
    We first load the google mobility dataset and merge it with demography and
    epidemiologic data.

    Returns:
    pandas.Dataframe: All the value for each day of the training set for each country.
    '''

    google = pd.read_csv("./data/luxembourg/luxembourg_mobility_google.csv")
    apple = pd.read_csv("./data/luxembourg/luxembourg_mobility_apple.csv")
    oxford = pd.read_csv("./data/luxembourg/luxembourg_npi_oxford.csv")
    sectors = pd.read_csv("./data/luxembourg/luxembourg_allsectors_rt.csv")

    google = google.rename({"Unnamed: 0": "Date"}, axis=1)
    google = google.drop(index=0)

    apple = apple.rename({"Unnamed: 0": "Date"}, axis=1)
    sectors = sectors.rename({"Unnamed: 0": "Date"}, axis=1)
    y_columns = sectors.columns[1:]

    all_df = pd.merge(sectors, google, on="Date", how="left")
    all_df = pd.merge(all_df, apple, on="Date", how="left")
    all_df = pd.merge(all_df, oxford, on="Date", how="left")

    all_df.index = pd.to_datetime(all_df["Date"], format="%Y-%m-%d")
    all_df = all_df.drop(["Date"], axis=1)

    if extend_features:
        all_columns = all_df.columns
        feature_columns = [x for x in all_columns if x not in y_columns]
        smoothing_days = [5, 10, 15]
        all_columns_extended = extend_features_with_means(all_df,feature_columns,smoothing_days)
        return all_columns_extended.fillna(0), y_columns
    return all_df.fillna(0), y_columns


def load_dataset():
    '''
    Load data from the dataset folder, all the values for the files are hardcoded.
    We first load the google mobility dataset and merge it with demography and
    epidemiologic data.

    Returns:
    pandas.Dataframe: All the value for each day of the training set for each country.
    '''

    google = pd.read_csv("./data/google.csv", parse_dates=['Date']).drop(["Unnamed: 0"],axis=1)
    countries = pd.read_csv("./data/seirhcd.csv", parse_dates=['Date']).drop(["Unnamed: 0"],axis=1)

    google = pd.get_dummies(google,prefix="day_of_week", columns=["day_of_week"])
    google = pd.get_dummies(google,prefix="region", columns=["region"])
    dataset = pd.merge(countries.groupby(["CountryName","Date"]).agg("first"), google.groupby(["CountryName","Date"]).agg("first"), on=["CountryName","Date"], how="inner")
    dataset = dataset.reset_index().dropna()

    columns = ['R', 'CountryName', 'Date']
    columns.extend(REGIONS)
    columns.extend(DAYS_OF_THE_WEEK)
    columns.extend(MOBILITY_WINDOWS)
    columns.extend(DEMOGRAPHY)
    columns.extend(SEIR)

    return dataset[columns]


def get_feature_columns():
    columns = []
    columns.extend(MOBILITY_WINDOWS)
    columns.extend(REGIONS)
    columns.extend(DAYS_OF_THE_WEEK)
    columns.extend(DEMOGRAPHY)

    return columns


def color_palette(data, hue):
    n_colors = 1 if hue == None else len(data[hue].unique())
    return sns.color_palette("cubehelix", n_colors=n_colors)


def save_figure(filename, dpi=300, bbox_inches='tight'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    plt.close('all')


def plot(type, data, name, x, y, y_label, x_label='', hue=None, y_lim=None, fig_size=(6,4), legend_pos='best', style=None, show_error=False, **kwargs):
    fig = plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=1.5)

    palette = color_palette(data, hue)

    if type == 'line':
        g = sns.lineplot(x=x, y=y, hue=hue, data=data, palette=palette, legend="full", style=style, **kwargs)
        plt.ticklabel_format(style='plain', axis='y',useOffset=False)
    elif type == 'scatter':
        g = sns.scatterplot(x=x, y=y, hue=hue, data=data, palette=palette, legend="full", style=style, **kwargs)
    else:
        raise TypeError("Only line or scatter are allowed")

    fig.tight_layout()

    if not legend_pos:
        if g.legend_:
            g.legend_.remove()
    elif hue:
        handles, labels = g.get_legend_handles_labels()
        plt.legend(loc='best', prop={'size': 15}, handles=handles[1:], labels=labels[1:])

    plt.ylabel(y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.xticks(rotation=45)

    if y_lim != None and len(y_lim) == 2:
        plt.ylim(y_lim)

    if show_error:
        error_band(data, x, hue)

    save_figure('figures/' + name + EXTENSION)


def error_band(data, x, hue):
    ax = plt.gca()
    
    valid_labels = data[hue].unique()

    for line in ax.lines:
        if not line.get_label() in valid_labels:
            continue

        x_values = data.loc[data[hue] == line.get_label()][x].values
        y_min = data.loc[data[hue] == line.get_label()]['Min'].values
        y_max = data.loc[data[hue] == line.get_label()]['Max'].values

        ax.fill_between(x_values, y_min, y_max, color=line.get_color(), alpha=0.2, linewidth=0.0)

def a12(lst1, lst2, rev=True):
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x == y:
                same += 1
            elif rev and x > y:
                more += 1
            elif not rev and x < y:
                more += 1
    return (more + 0.5*same) / (len(lst1)*len(lst2))



GAMMA = 1/10
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

def get_posteriors(sr, sigma=0.15, gamma=None):
    min_val = 1e-20
    # (1) Calculate Lambda
    if gamma is None:
      gamma=GAMMA
    lam = sr[:-1].values * np.exp(gamma * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)

    if len(r_t_range)==0:
      print("error range")
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        if denominator==0:
          #print("error denominator")
          denominator = min_val
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood

def prepare_cases(cases, cutoff=25, new_provided=False):
    if new_provided:
      new_cases = cases
    else:
      new_cases =cases.diff()
    
    new_cases[new_cases < 0] = 0

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed


def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
  
      return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows)
    #print(pmf.values)
    best = best.argmin() if len(best) else 0

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])



def compute_sector_rt(df, new_provided=False, max_elements=1):
  sectors_output={}
  for i, sector_name in tqdm(enumerate(df.columns.to_list()), total=len(df.columns.to_list())):
    if i>=max_elements:
      break


    cases = df[sector_name]*100
    
    original, smoothed = prepare_cases(cases, cutoff=5, new_provided=new_provided)
    original = pd.DataFrame(original)
    #smoothed = pd.DataFrame(smoothed)
    if len(smoothed) ==0:
      print(sector_name, "not enough cases")
      continue

    #Posteriors & Rt
    posteriors, _ = get_posteriors(smoothed, sigma=0.5)
 
    #posteriors= posteriors.fillna(1e-20)
    hdis_90 = None
    hdis_50 = None
    try:
      hdis_90 = highest_density_interval(posteriors, p=.9, debug=True)
    except Exception as e:
    #   print(e)
        pass
    
    #hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('ML')
    sectors_output[sector_name] = most_likely

  return sectors_output


def case_time(var):
    if var < pd.to_datetime("2020-03-16").date():
        return 3
    elif var >= pd.to_datetime("2020-03-16").date() and var < pd.to_datetime("2020-05-04").date():
        return 0
    elif var > pd.to_datetime("2020-07-17").date() and var < pd.to_datetime("2020-09-15").date():
        return 0
    elif var >= pd.to_datetime("2020-05-04").date() and var < pd.to_datetime("2020-06-29").date():
        return 1
    elif var >= pd.to_datetime("2020-06-29").date() and var <= pd.to_datetime("2020-07-16").date():
        return 2
    else:
        return 2

