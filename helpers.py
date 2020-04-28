
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error,median_absolute_error
import numpy as np
import matplotlib.pyplot as plt


def metrics_report(X_test, y_test, reg):
    y_pred = reg.predict(X_test)
    return {'r2_score': r2_score(y_test, y_pred), 
          'mean_absolute_error': mean_absolute_error(y_test, y_pred),
          'mean_squared_error': mean_squared_error(y_test, y_pred),
         'median_absolute_error': median_absolute_error(y_test, y_pred),
           'RMSE': np.sqrt(mean_absolute_error(y_test, y_pred))}



def plot_model(solution, title='SEIR+HCD model'):
    sus, exp, inf, rec, hosp, crit, death = solution.y
    
    cases = inf + rec + hosp + crit + death

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
    fig.suptitle(title)
    
    ax1.plot(sus, 'tab:blue', label='Susceptible');
    ax1.plot(exp, 'tab:orange', label='Exposed');
    ax1.plot(inf, 'tab:red', label='Infected');
    ax1.plot(rec, 'tab:green', label='Recovered');
    ax1.plot(hosp, 'tab:purple', label='Hospitalised');
    ax1.plot(crit, 'tab:brown', label='Critical');
    ax1.plot(death, 'tab:cyan', label='Deceased');
    
    ax1.set_xlabel("Days", fontsize=10);
    ax1.set_ylabel("Fraction of population", fontsize=10);
    ax1.legend(loc='best');
    
    ax2.plot(cases, 'tab:red', label='Cases');    
    ax2.set_xlabel("Days", fontsize=10);
    ax2.set_ylabel("Fraction of population (Cases)", fontsize=10, color='tab:red');
    
    ax3 = ax2.twinx()
    ax3.plot(death, 'tab:cyan', label='Deceased');    
    ax3.set_xlabel("Days", fontsize=10);
    ax3.set_ylabel("Fraction of population (Fatalities)", fontsize=10, color='tab:cyan');

