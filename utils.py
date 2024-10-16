import numpy as np
from scipy import stats, interpolate
import pickle
import os
import pandas as pd
import datetime as dt

def save_object(obj, name, filename='data_store.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    data[name] = obj
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Object '{name}' has been saved to {filename}")

def load_objects(filename='data_store.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"File {filename} does not exist.")
        return {}


def temperature(x, a, b, c, varphi):
    omega = 2 * np.pi / 365.25
    y_hat = a + b * x + c * np.sin(omega * x + varphi)
    return y_hat

def temperature_derivative(x, a, b, c, varphi):
    omega = 2 * np.pi / 365.25
    derivative = b + c * omega * np.cos(omega * x + varphi)
    return derivative

def volatility_model(k, indep, dep):
    indep_prime = np.linspace(0, 1, k + 2)[1: - 1]
    knot, coeff, degree = interpolate.splrep(indep, dep, t = np.quantile(indep, indep_prime))
    dep_fit = interpolate.BSpline(knot, coeff, degree)(indep)
    return dep_fit

def coefficients(x, parameters):
    a, b, a1, b1 = parameters
    omega = 2 * np.pi / 365.25
    varphi = np.arctan(a1 / b1) - np.pi
    c = np.sqrt( (a1 ** 2) + (b1 ** 2))
    print('Amsterdam Parameters:\n a {0:0.3}\n b {1:.2e}\n c {2:0.3}\n varphi {3:0.3}'.format(a, b, c, varphi))
    y_hat = a + b * x + c * np.sin(omega * x + varphi)
    return y_hat

def coefficients_fit(x, a, b, a1, b1):
    omega = np.pi * 2 / 365.25
    y_pred = a + b * x + a1 * np.cos(omega * x) + b1 * np.sin(omega * x)
    return y_pred

def expectation(seasonal, vol, time, lmd, kappa):
    grid = 1 / 365.25
    n = len(time)
    second_term = - vol * (1 - np.exp(- kappa * grid)) / kappa
    mean = 18 * n - (np.sum(seasonal) - lmd * np.sum(second_term))
    return mean

def variance(vol, time, kappa):
    grid = 1 / 365.25
    var = np.power(vol, 2)
    first_term = var / (2 * kappa) * (1 - np.exp(- 2 * kappa * grid))
    cov = 0
    for y, ty in enumerate(time):
        for z, tz in enumerate(time):
            if z > y:
                cov += np.exp(- kappa * (tz - ty)) * first_term[y]
    return np.sum(first_term) + 2 * cov

def rn_mean(time_arr, vol_arr, Tbars, lamda, kappa):
    dt = 1/365.25
    N = len(time_arr)
    mean_intervals = -vol_arr*(1 - np.exp(-kappa*dt))/kappa
    return 18*N - (np.sum(Tbars) - lamda*np.sum(mean_intervals))

def rn_var(time_arr, vol_arr, kappa):
    dt = 1/365.25 
    var_arr = np.power(vol_arr,2) 
    var_intervals = var_arr/(2*kappa)*(1-np.exp(-2*kappa*dt))
    cov_sum = 0
    for i, ti in enumerate(time_arr):
        for j, tj in enumerate(time_arr):
            if j > i:
                cov_sum += np.exp(-kappa*(tj-ti)) * var_intervals[i]
    return np.sum(var_intervals) + 2*cov_sum

def price(dates, seasonal_params, vol_model, ordinal, lmd, kappa):
    if isinstance(dates, pd.DatetimeIndex):
        date = dates.map(dt.datetime.toordinal)
    seasonals = temperature(date - ordinal, * seasonal_params)
    seasonals_derivative = temperature_derivative(date - ordinal, * seasonal_params)
    temps = pd.DataFrame(data = np.array([seasonals, seasonals_derivative]).T,
                         index = dates, columns = ['seasonal', 'seasonal_derivative'])
    temps['day'] = temps.index.dayofyear
    temps['vol'] = vol_model[temps['day'] - 1]
    time = np.array([x / 365.25 for x in range(1, len(dates) + 1)])
    vol = temps['vol'].values
    exp = expectation(seasonals, vol, time, lmd, kappa)
    var = variance(vol, time, kappa)
    return exp, var

def option(dates, r, seasonal_params, vol, ordinal, kappa, alpha, K, tau, option = str, lmd = 0):
    amst_exp, amst_var = price(dates, seasonal_params, vol, ordinal, lmd, kappa)
    discount = np.exp(-r*tau)
    amst_vol = np.sqrt(amst_var)
    xi = (K - amst_exp)/amst_vol
    exponential = np.exp(-xi**2/2)
    if option == 'call':
        predicted_price = alpha*discount*((amst_exp - K)*stats.norm.cdf(-xi) + amst_vol*exponential/np.sqrt(2*np.pi))
    elif option == 'put':
        exponential2 = np.exp(-amst_exp**2/(2*amst_vol**2))
        predicted_price = alpha*discount*((K - amst_exp)*(stats.norm.cdf(xi) - stats.norm.cdf(-amst_exp/amst_vol)) +
                                           amst_vol/np.sqrt(2*np.pi)*(exponential-exponential2))
        
    print('The option price is:', predicted_price)
