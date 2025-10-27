import scipy
import numpy as np

def exponential_model(t, N0, r):
    return N0 * np.exp(r * t)

def fit_exponential(t_data, N_data):
    # Avoid log of zero by filtering out zero or negative counts
    mask = N_data > 0
    t_filtered = t_data[mask]
    N_filtered = N_data[mask]

    # Initial guess: N0 is first count, r is small positive number
    p0 = [N_filtered[0], 0.001]

    popt, pcov = scipy.optimize.curve_fit(exponential_model, t_filtered, N_filtered, p0=p0)
    N0_fit, r_fit = popt
    perr = np.sqrt(np.diag(pcov))  # parameter std errors
    return N0_fit, r_fit, perr

def logistic_model(t, N0, r, K):
    return K*N0 / (N0 + (K-N0)*np.exp(-r*t))

def fit_logistic(t_data, N_data):
        # Avoid log of zero by filtering out zero or negative counts
        
    relative_error = 0.2
    sigma = relative_error * N_data
    sigma = np.maximum(sigma, 10)
    # sigma = [20]*len(t_data)
    # sigma = 0.01
    
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

def fit_gompertz(t_data, N_data):
    # Filter out zero or negative values
    mask = N_data > 0
    t_filtered = t_data[mask]
    N_filtered = N_data[mask]

    # Handle uncertainty
    relative_error = 0.1
    sigma = relative_error * N_filtered
    sigma = np.maximum(sigma, 10)  # Avoid tiny sigma
    # sigma=0.01

    # Initial guesses:
    K_init = np.max(N_filtered)
    # t0_init = t_filtered[np.argmax(np.diff(N_filtered))] if len(N_filtered) > 1 else t_filtered[0]
    t0_init = np.max(t_filtered) / 2
    r_init = 0.01

    p0 = [r_init, t0_init, K_init]

    # Bounds: r, t0, K
    bounds = ([0, 0, 0], [np.inf, np.max(t_filtered), np.inf])

    popt, pcov = scipy.optimize.curve_fit(
        gompertz_model, t_filtered, N_filtered,
        p0=p0, sigma=sigma, absolute_sigma=True,
        bounds=bounds
    )

    r_fit, t0_fit, K_fit = popt
    perr = np.sqrt(np.diag(pcov))  # standard errors

    return r_fit, t0_fit, K_fit, perr