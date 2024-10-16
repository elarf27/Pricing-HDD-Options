import numpy as np
from scipy import stats
from scipy.optimize import newton
from utils import price


def OptimalPrice(dates, r, alpha, K, tau, seasonal_params, volatility, ordinal, kappa, option = 'put', market_price = None):
    def price_of_risk(lmd):
        amst_exp, amst_var = price(dates, seasonal_params, volatility, ordinal, lmd, kappa)
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
            
        return market_price - predicted_price
    
    initial_lambda = 0.0  
    
    optimal_lambda = newton(price_of_risk, x0 = initial_lambda, maxiter = 2000)
    
    amst_exp, amst_var = price(dates, seasonal_params, volatility, ordinal, optimal_lambda, kappa)
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
    
    return predicted_price, optimal_lambda