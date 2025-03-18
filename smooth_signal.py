# Base libraries
import numpy as np

# Plotting libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Decomposition libraries
from PyEMD import EMD
from scipy.fft import fft
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.feature_selection import mutual_info_regression

def __plot_components__(signal:np.array,
                        t:np.array,
                        stochastic_component:np.array,
                        deterministic_component:np.array):
    """
    Function for the plotting of the original signal, stochastic and deterministic components of the time series

    Inputs:
    ----------
    signal : np.array
        Time series for decomposition
    t : np.array
        Index of time series for plotting
    stochastic_component : np.array
        Stochastic component of the time series
    deterministic_component : np.array
        Deterministic component of the time series

    Plots:
    ----------
    Plots of original time series, stohastic and determenistic components
    """

    # Creating grid of subplots
    fig = make_subplots(rows = 3, cols = 1, subplot_titles = ['Original Signal', 'Stochastic Component', 'Deterministic Component'])
    
    # Scattering signal and components
    fig.add_trace(go.Scatter(x = t, y = signal, mode = 'lines', name = 'Original Signal'), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = t, y = stochastic_component, mode = 'lines', name = 'Stochastic Component'), row = 2, col = 1)
    fig.add_trace(go.Scatter(x = t, y = deterministic_component, mode = 'lines', name = 'Deterministic Component'), row = 3, col = 1)

    # Update layout
    fig.update_layout(
        showlegend = False,
        font = dict(size = 20),
        height = 1200,
        width = 2000
    )
    fig.show()

#---------------------------------------------------------------------------------------------------------------------------------------

def __phase_spectrum__(imfs:np.array):
    """
    Function for the calculation of the time series' phase spectrum
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    imfs : np.array
        Decomposed time series

    Returns:
    ----------
    imfs_p : np.array
        Phase spectrum of decomposed time series
    """

    # Iterate over decomposed timer series to calculate each ones phase spectrum
    imfs_p = []
    for imf in imfs:
        trans = fft(imf)
        imf_p = np.arctan(trans.imag / trans.real)
        imfs_p.append(imf_p)

    return imfs_p

#---------------------------------------------------------------------------------------------------------------------------------------

def __phase_mi__(phases:np.array):
    """
    Function for the calculation of mutual information in the phases
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    phases : array
        Phase spectrum of decomposed time series
    
    Returns:
    ----------
    mis : array
        Mutual information of phase spectrums of decomposed time series
    """

    # Iterate over phases to calculate mutual info
    mis = []
    for i in range(len(phases) - 1):
        mis.append(mutual_info_regression(phases[i].reshape(-1, 1), phases[i + 1])[0])
        
    return np.array(mis)

#---------------------------------------------------------------------------------------------------------------------------------------

def __divide_signal__(signal, 
                      t, 
                      imfs, 
                      mis, 
                      cutoff:float = 0.05, 
                      plot = False):
    """
    Function for the final separation to the stohastic and determenistic components
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    signal : array
        Time series for decomposition
    t : array
        Index of time series for plotting
    imfs : array
        Decomposed time series
    mis : array
        Mutual information of phase spectrums of decomposed time series
    cutoff : float = 0.05
        Border of separation between stohastic and determenistic components
    plot : bool = False
        Flag whether to plot original time series, stohastic and determenistic components

    Plots:
    ----------
    Plots of original time series, stohastic and determenistic components if plot == True

    Returns:
    ----------
    stochastic_component : array
        Sum of time series components that are considered stohastic
    deterministic_component : array
        Sum of time series components that are considered deterministic
    """

    # Separate time series to stohastic and deterministic components 
    cut_point = np.where(mis > cutoff)[0][0]    
    stochastic_component = np.sum(imfs[:cut_point], axis=0)
    deterministic_component = np.sum(imfs[cut_point:], axis=0)

    if plot:
        __plot_components__(signal, t, stochastic_component, deterministic_component)
    
    return stochastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

def emd_wrapper(signal:np.array,
        t:np.array,
        plot_signals:bool = False,
        plot_components:bool = False) -> tuple:
    """
    Function for the decomposition of time series to the several components until the last one is monotonous
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    signal : array
        Time series for decomposition (values, not array itself)
    t : array
        Index of time series for plotting
    plot_signals : bool = False
        Flag whether the plot of the detailed components is needed
    plot_components : bool = False
        Flag whether the plot of the original, stochastic and deterministic time series is needed

    Plots:
    ----------
    Plots of original time series and its decomposed parts if plot == True

    Returns:
    ----------
    stochastic_component : array
        Stochastic component of the time series
    deterministic_component : array
        Deterministic component of the time series
    """
    
    # Separate time series into components
    emd = EMD(DTYPE = np.float16, spline_kind = 'akima')
    imfs = emd(signal)
    N = imfs.shape[0]
    
    if plot_signals:
        # Creating grid of subplots
        fig = make_subplots(rows = N + 1, cols = 1, subplot_titles = ['Original Signal'] + [f'IMF {i}' for i in range(N)])

        # Scattering signal and IMFs
        fig.add_trace(go.Scatter(x = t, y = signal, mode = 'lines', name = 'Original Signal'), row = 1, col = 1)
        for i, imf in enumerate(imfs):
            fig.add_trace(go.Scatter(x = t, y = imf, mode = 'lines', name = f'IMF {i}'), row = i + 2, col = 1)

        # Update layout
        fig.update_layout(
            showlegend = False,
            font = dict(size = 20),
            height = 400 * (N + 1),
            width = 2000
        )
        fig.show()

    imfs_p = __phase_spectrum__(imfs)
    mis = __phase_mi__(imfs_p)
    rmse_min = np.inf
    best_cut = 0.5
    # for cut in np.linspace(0.25, 3, 12):
    #     try:
    #         stochastic_component, deterministic_component = divide_signal(signal, t, imfs, mis, cutoff = cut)
    #         if mse(deterministic_component, preds['orig'], squared = False) < rmse_min:
    #             best_cut = cut
    #     except:
    #         pass
    stochastic_component, deterministic_component = __divide_signal__(signal, t, imfs, mis, cutoff = best_cut)

    
    
    return stochastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

