import scipy
import numpy as np

def power_law_model(x, a, b):
    return a * np.power(x, b)

def fit_power_law(x_data, y_data, std=1.0):
    if std is not None:
        sigma = np.ones_like(x_data) * std  # constant standard deviation
    else:
        sigma=None

    # mask = x_data > 0  # avoid zero or negative values for power-law
    # x_filtered = x_data[mask]
    # y_filtered = y_data[mask]
    # sigma_filtered = sigma[mask]
    
    # Initial guess: a ~ 1, b ~ 1
    p0 = [1.0, 1.5]
    
    popt, pcov = scipy.optimize.curve_fit(
        power_law_model, x_data, y_data, p0=p0,
        sigma=sigma, absolute_sigma=True, maxfev=10000
    )
    
    a_fit, b_fit = popt
    perr = np.sqrt(np.diag(pcov))
    
    return a_fit, b_fit, perr

def linear_model(t, a, b):
    return a*t + b

def fit_linear(t_data, N_data, std):
    # sigma = [std] * len(t_data)  # constant standard deviation
    mask = N_data >= 0  # optionally filter negative values
    t_filtered = t_data[mask]
    N_filtered = N_data[mask]
    sigma=std
    
    # Initial guess: slope ~0, intercept = first value
    p0 = [0.0, N_filtered[0]]
    
    popt, pcov = scipy.optimize.curve_fit(linear_model, t_filtered, N_filtered, p0=p0,
                                          sigma=sigma, absolute_sigma=True)
    a_fit, b_fit = popt
    perr = np.sqrt(np.diag(pcov))
    
    return a_fit, b_fit, perr


def exponential_model(t, N0, r):
    return N0 * np.exp(r * t)

def fit_exponential(t_data, N_data, std):
    # Avoid log of zero by filtering out zero or negative counts
    
    # relative_error = 0.4
    # sigma = relative_error * N_data
    # sigma = np.maximum(sigma, 10)
    # sigma = [std]*len(t_data)
    sigma=np.maximum(std, 10)
    
    mask = N_data > 0
    t_filtered = t_data[mask]
    N_filtered = N_data[mask]

    # Initial guess: N0 is first count, r is small positive number
    p0 = [N_filtered[0], 0.001]

    popt, pcov = scipy.optimize.curve_fit(exponential_model, t_filtered, N_filtered, p0=p0, sigma=sigma, absolute_sigma=True)
    N0_fit, r_fit = popt
    perr = np.sqrt(np.diag(pcov))  # parameter std errors
    return N0_fit, r_fit, perr

def logistic_model(t, N0, r, K):
    return K*N0 / (N0 + (K-N0)*np.exp(-r*t))

def fit_logistic(t_data, N_data, std):
        # Avoid log of zero by filtering out zero or negative counts
        
    # relative_error = 0.2
    # sigma = relative_error * N_data
    # sigma = np.maximum(sigma, 10)
    # sigma = [20]*len(t_data)
    # sigma = 0.01
    # sigma = [std]*len(t_data)
    # sigma=np.std
    sigma=np.maximum(std, 10)
    mask = N_data > 0
    t_filtered = t_data[mask]
    N_filtered = N_data[mask]

    # Initial guess: N0 is first count, r is small positive number
    p0 = [N_filtered[0], 0.2, max(N_filtered)]

    popt, pcov = scipy.optimize.curve_fit(logistic_model, t_filtered, N_filtered, p0=p0, sigma=sigma, absolute_sigma=True, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    N0_fit, r_fit, K_fit = popt
    perr = np.sqrt(np.diag(pcov))  # parameter std errors
    return N0_fit, r_fit, K_fit, perr

def gompertz_model(t, r, t0, K):
    """Gompertz growth model: K * exp(-exp(-r*(t - t0)))"""
    return K * np.exp(-np.exp(-r * (t - t0)))

def fit_gompertz(t_data, N_data, std):
    # Filter out zero or negative values
    mask = N_data > 0
    t_filtered = t_data[mask]
    N_filtered = N_data[mask]

    # Handle uncertainty
    # relative_error = 0.1
    # sigma = relative_error * N_filtered
    # sigma = np.maximum(sigma, 10)  # Avoid tiny sigma
    # sigma = [std]*len(t_data)
    # sigma=0.01
    sigma=np.maximum(std, 10)

    # Initial guesses:
    K_init = np.max(N_filtered)
    # t0_init = t_filtered[np.argmax(np.diff(N_filtered))] if len(N_filtered) > 1 else t_filtered[0]
    t0_init = np.max(t_filtered)
    r_init = 0.01

    p0 = [r_init, t0_init, K_init]

    # Bounds: r, t0, K
    bounds = ([0, 0, 0], [np.inf, t0_init*3, np.inf])

    popt, pcov = scipy.optimize.curve_fit(
        gompertz_model, t_filtered, N_filtered,
        p0=p0, sigma=sigma, absolute_sigma=True,
        bounds=bounds, maxfev=10000
    )

    r_fit, t0_fit, K_fit = popt
    perr = np.sqrt(np.diag(pcov))  # standard errors

    return r_fit, t0_fit, K_fit, perr