"""# SEIR-HCD Model
This is a working example of a [SIER](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model) model with added compartments for HCD. 
The letters stand for Susceptible, Exposed, Infected, Recovered, Hospitalized, Critical, Death

* http://gabgoh.github.io/COVID/index.html
* https://neherlab.org/covid19

## Parameters used in the model
`R_t` = reproduction number at time t. Typical 3.6* at t=0

**Transition times**
* `T_inc` = average incubation period. Typical 5.6* days
* `T_inf` = average infectious period. Typical 2.9 days
* `T_hosp` = average time a patient is in hospital before either recovering or becoming critical. Typical 4 days
* `T_crit` = average time a patient is in a critical state (either recover or die). Typical 14 days

**Fractions**
These constants are likely to be age specific (hence the subscript a):
* `m_a` = fraction of infections that are asymptomatic or mild. Assumed 80% (i.e. 20% severe)
* `c_a` = fraction of severe cases that turn critical. Assumed 10%
* `f_a` = fraction of critical cases that are fatal. Assumed 30%

*Averages and formulas taken from https://www.kaggle.com/covid-19-contributions
"""

import numpy as np

# Susceptible equation
def dS_dt(S, I, R_t, t_inf):
    return -(R_t / t_inf) * I * S


# Exposed equation
def dE_dt(S, E, I, R_t, t_inf, t_inc):
    return (R_t / t_inf) * I * S - (E / t_inc)


# Infected equation
def dI_dt(I, E, t_inc, t_inf):
    return (E / t_inc) - (I / t_inf)


# Hospialized equation
def dH_dt(I, C, H, t_inf, t_hosp, t_crit, m_a, f_a):
    return ((1 - m_a) * (I / t_inf)) + ((1 - f_a) * C / t_crit) - (H / t_hosp)


# Critical equation
def dC_dt(H, C, t_hosp, t_crit, c_a):
    return (c_a * H / t_hosp) - (C / t_crit)


# Recovered equation
def dR_dt(I, H, t_inf, t_hosp, m_a, c_a):
    return (m_a * I / t_inf) + (1 - c_a) * (H / t_hosp)


# Deaths equation
def dD_dt(C, t_crit, f_a):
    return f_a * C / t_crit


def model(t, y, R_t, t_inc=2.9, t_inf=5.2, t_hosp0=4, t_crit0=14, m_a0=0.8, c_a0=0.1, f_a0=0.3, decay_values=False):
    """
    :param t: Time step for solve_ivp
    :param y: Previous solution or initial values
    :param R_t: Reproduction number
    :param t_inc: Average incubation period. Default 5.2 days
    :param t_inf: Average infectious period. Default 2.9 days
    :param t_hosp0: Average time a patient is in hospital before either recovering or becoming critical. Default 4 days
    :param t_crit0: Average time a patient is in a critical state (either recover or die). Default 14 days
    :param m_a0: Fraction of infections that are asymptomatic or mild. Default 0.8
    :param c_a0: Fraction of severe cases that turn critical. Default 0.1
    :param f_a0: Fraction of critical cases that are fatal. Default 0.3
    :param end_values: end values of decay of t_hosp, t_crit, m_a, c_a, f_a. Default None
    :return:
    """
    if callable(R_t):
        reprod = R_t(t)
    else:
        reprod = R_t
        
    S, E, I, R, H, C, D = y

    if decay_values is False:
        t_hosp = t_hosp0
        t_crit = t_crit0
        m_a = m_a0
        c_a = c_a0
        f_a = f_a0

    else:
        lambda_ = 1e-1
        decay = np.exp(-lambda_*t)

        t_hosp = decay*t_hosp0 + (1-decay)*4
        t_crit = decay*t_crit0 + (1-decay)*14
        m_a = decay*m_a0 + (1-decay) *0.8
        c_a = decay*c_a0 + (1-decay) *0.1
        f_a = decay*f_a0 + (1-decay) *0.3

    
    S_out = dS_dt(S, I, reprod, t_inf)
    E_out = dE_dt(S, E, I, reprod, t_inf, t_inc)
    I_out = dI_dt(I, E, t_inc, t_inf)
    R_out = dR_dt(I, H, t_inf, t_hosp, m_a, c_a)
    H_out = dH_dt(I, C, H, t_inf, t_hosp, t_crit, m_a, f_a)
    C_out = dC_dt(H, C, t_hosp, t_crit, c_a)
    D_out = dD_dt(C, t_crit, f_a)

    return [S_out, E_out, I_out, R_out, H_out, C_out, D_out]