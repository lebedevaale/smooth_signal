# Base libraries
import numpy as np

# Plotting libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Decomposition and scoring libraries
from PyEMD import EMD # pip install EMD-signal

import pywt

from scipy.fft import fft
from scipy.signal import savgol_filter as sgf

# from filterpy.common import Saver
# from filterpy.kalman import KalmanFilter as kf
# from filterpy.common import Q_discrete_white_noise

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_percentage_error as mape

from statsmodels.tsa.stattools import adfuller

#---------------------------------------------------------------------------------------------------------------------------------------

def __plot_components__(signal:np.array,
                        t:np.array,
                        stochastic_component:np.array,
                        deterministic_component:np.array):
    """
    Function for the plotting of the original signal, stochastic and deterministic components of the time series

    Inputs
    ----------
    signal : np.array
        Time series for decomposition
    t : np.array
        Index of time series for plotting
    stochastic_component : np.array
        Stochastic component of the time series
    deterministic_component : np.array
        Deterministic component of the time series

    Plots
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

def __phase_spectrum__(imfs:np.array) -> np.array:
    """
    Function for the calculation of the time series' phase spectrum
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs
    ----------
    imfs : np.array
        Decomposed time series

    Returns
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

def __phase_mi__(phases:np.array) -> np.array:
    """
    Function for the calculation of mutual information in the phases
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs
    ----------
    phases : array
        Phase spectrum of decomposed time series
    
    Returns
    ----------
    mis : array
        Mutual information of phase spectrums of decomposed time series
    """

    # Iterate over phases to calculate mutual info
    mis = np.array([])
    for i in range(len(phases) - 1):
        mis = np.append(mis, mutual_info_regression(phases[i].reshape(-1, 1), phases[i + 1])[0])
        
    return mis

#---------------------------------------------------------------------------------------------------------------------------------------

def __divide_signal__(signal:np.array, 
                      t:np.array, 
                      imfs:np.array, 
                      mis:np.array, 
                      cutoff:float = 0.05, 
                      plot:bool = False) -> tuple:
    """
    Function for the final separation to the stohastic and determenistic components
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs
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

    Plots
    ----------
    Plots of original time series, stohastic and determenistic components if plot == True

    Returns
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

    # Plot components if needed
    if plot == True:
        __plot_components__(signal, t, stochastic_component, deterministic_component)
    
    return stochastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

def emd_wrapper(signal:np.array,
                t:np.array,
                mape_max:float = None,
                plot_signals:bool = False,
                plot_components:bool = False) -> tuple:
    """
    Function for the decomposition of time series to the several components until the last one is monotonous
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs
    ----------
    signal : array
        Time series for decomposition (values, not array itself)
    t : array
        Index of time series for plotting
    mape_min : float = None
        Highest expected mean absolute percentage error
    plot_signals : bool = False
        Flag whether the plot of the detailed components is needed
    plot_components : bool = False
        Flag whether the plot of the original, stochastic and deterministic time series is needed

    Plots
    ----------
    Plots of original time series and its decomposed parts if plot == True

    Returns
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

    imfs_p = __phase_spectrum__(imfs)
    mis = __phase_mi__(imfs_p)

    # Search for the best cut to separate stohastic and deterministic components based on MAPE
    best_cut = 0.5
    if mape_max != None:
        for cut in np.linspace(0.25, 3, 12):
            try:
                stochastic_component, deterministic_component = __divide_signal__(signal, t, imfs, mis, cutoff = cut)
                
                mape_cut = mape(deterministic_component, signal)
                print(f'Cut: {cut}, MAPE: {round(mape_cut, 4)}')
                if mape_cut <= mape_max:
                    best_cut = cut.copy()
                else:
                    break
            except:
                pass

    if (plot_signals == True) & (plot_components == True):
        plot_components = False
        combine_plots = True
    else:
        combine_plots = False

    # Calculated the final separation
    stochastic_component, deterministic_component = __divide_signal__(signal, t, imfs, mis, cutoff = best_cut, plot = plot_components)
 
    # Combined plotting of the signals and components
    if plot_signals == True:
        # Creating grid of subplots
        if combine_plots == True:
            starting_point = 4
            fig = make_subplots(rows = N + starting_point - 1, cols = 1, subplot_titles = ['Original Signal',
                                                                                        'Deterministic Component',
                                                                                        'Stochastic Component']\
                                                                                        + [f'IMF {i}' for i in range(N)])
            
        else:
            starting_point = 2
            fig = make_subplots(rows = N + starting_point - 1, cols = 1, subplot_titles = ['Original Signal'] + [f'IMF {i}' for i in range(N)])

        # Scattering signal and IMFs
        fig.add_trace(go.Scatter(x = t, y = signal, mode = 'lines', name = 'Original Signal'), row = 1, col = 1)
        if combine_plots == True:
            fig.add_trace(go.Scatter(x = t, y = deterministic_component, mode = 'lines', name = 'Deterministic Component'), row = 2, col = 1)
            fig.add_trace(go.Scatter(x = t, y = stochastic_component, mode = 'lines', name = 'Stochastic Component'), row = 3, col = 1)
        for i, imf in enumerate(imfs):
            fig.add_trace(go.Scatter(x = t, y = imf, mode = 'lines', name = f'IMF {i}'), row = i + starting_point, col = 1)

        # Update layout
        fig.update_layout(
            showlegend = False,
            font = dict(size = 20),
            height = 300 * (N + starting_point - 1),
            width = 1500
        )
        fig.show()

    return stochastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

def sav_gol_wrapper(signal:np.array,
                    t:np.array, 
                    window_width:int = 15,
                    window_polynomial:int = 2,
                    plot_components:bool = False) -> tuple:
    """
    Wrapper function for the Savitzky-Golay filter

    Inputs
    ----------
    signal : array
        Time series for decomposition
    t : array
        Index of time series for plotting
    window_width : int = 15
        Window width of the Savitzky-Golay filter
    window_polynomial : int = 2
        Polynomial order of the Savitzky-Golay filter
    plot_components : bool = False
        Flag whether the plot of the original, stochastic and deterministic time series is needed

    Returns
    ----------
    stochastic_component : array
        Stochastic component of the time series
    deterministic_component : array
        Deterministic component of the time series
    """

    # Apply Savitzky-Golay filter
    deterministic_component = sgf(signal, window_length = window_width, polyorder = window_polynomial)
    stohastic_component = signal - deterministic_component

    # Plot components if needed
    if plot_components == True:
        __plot_components__(signal, t, stohastic_component, deterministic_component)

    return stohastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

def wavelet_wrapper(signal:np.array,
                    t:np.array,
                    wavelet_type:str = 'haar',
                    sigma:float = 0.6,
                    plot_components:bool = False) -> tuple:
    """
    Wrapper function for the wavelet decomposition
    Source: https://github.com/CSchoel/learn-wavelets/blob/main/wavelet-denoising.ipynb

    Inputs
    ----------
    signal : array
        Time series for decomposition
    t : array
        Index of time series for plotting
    wavelet : str
        Type of wavelet from pywt. Possible variants: 
            haar family: haar
            db family: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
            sym family: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20
            coif family: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
            bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
            rbio family: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8
            dmey family: dmey
    sigma : float = 0.6
        Defines the level of noise separation. The higher, the more noise is separated
    plot_components : bool = False
        Flag whether the plot of the original, stochastic and deterministic time series is needed

    Returns
    ----------
    stochastic_component : array
        Stochastic component of the time series
    deterministic_component : array
        Deterministic component of the time series
    """

    def neigh_block(details,
                    n:int, sigma):
        res = []
        L0 = int(np.log2(n) // 2)
        L1 = max(1, L0 // 2)
        L = L0 + 2 * L1

        def nb_beta(sigma, L, detail):
            S2 = np.sum(detail ** 2)
            lmbd = 4.50524 # solution of lmbd - log(lmbd) = 3
            beta = (1 - lmbd * L * sigma**2 / S2)
            return max(0, beta)
        
        for d in details:
            d2 = d.copy()
            for start_b in range(0, len(d2), L0):
                end_b = min(len(d2), start_b + L0)
                start_B = start_b - L1
                end_B = start_B + L
                if start_B < 0:
                    end_B -= start_B
                    start_B = 0
                elif end_B > len(d2):
                    start_B -= end_B - len(d2)
                    end_B = len(d2)
                assert end_B - start_B == L
                d2[start_b:end_b] *= nb_beta(sigma, L, d2[start_B:end_B])
            res.append(d2)
        return res

    # Check if wavelet is discrete
    # (because their interfaces are different and backward composition is still implemented only for it)
    if pywt.DiscreteContinuousWavelet(wavelet_type).family_name not in ['Haar', 'Daubechies', 'Symlets',
                                                                        'Coiflets', 'Biorthogonal',
                                                                        'Reverse biorthogonal',
                                                                        'Discrete Meyer (FIR Approximation)']:
        raise ValueError(f'Wavelet {wavelet_type} is not discrete')

    # Apply wavelet decomposition based on the discrete or continuous wavelet
    wavelet = pywt.Wavelet(wavelet_type)
    coeffs = pywt.dwt(signal, wavelet)
        
    # Reconstruct time series
    approx = coeffs[0]
    details = coeffs[1:]
    details_nb = neigh_block(details, len(signal), sigma)
    deterministic_component = pywt.waverec([approx] + details_nb, wavelet_type)
    stohastic_component = signal - deterministic_component

    # Plot components if needed
    if plot_components == True:
        __plot_components__(signal, t, stohastic_component, deterministic_component)

    return stohastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

# def kalman_wrapper(signal:np.array,
#                    t:np.array,
#                    dt:float = 1,
#                    measurement_noise:float = 1,
#                    process_noise:float = 0.05,
#                    plot_components:bool = False) -> tuple:
#     """
#     Wrapper function for the second orderKalman filter
#     Source: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/08-Designing-Kalman-Filters.ipynb

#     Inputs
#     ----------
#     signal : array
#         Time series for decomposition
#     t : array
#         Index of time series for plotting
#     plot_components : bool = False
#         Flag whether the plot of the original, stochastic and deterministic time series is needed

#     Returns
#     ----------
#     stochastic_component : array
#         Stochastic component of the time series
#     deterministic_component : array
#         Deterministic component of the time series
#     """

#     def SecondOrderKF(R_std:float,
#                       Q:float,
#                       dt:int = 1,
#                       P = 10):
#         """ Create second order Kalman filter."""
        
#         # Create filter
#         filter = kf(dim_x = 3, dim_z = 1)
        
#         # Current state
#         filter.x = np.zeros(3)

#         # Covariance matrix
#         filter.P[0, 0] = P
#         filter.P[1, 1] = 1
#         filter.P[2, 2] = 1

#         # Measurement noise
#         filter.R *= R_std**2

#         # Process noise
#         filter.Q = Q_discrete_white_noise(3, dt, Q)

#         # State transition matrix
#         filter.F = np.array([[1., dt, .5*dt*dt],
#                             [0., 1.,       dt],
#                             [0., 0.,       1.]])
        
#         # Measurement function
#         filter.H = np.array([[1., 0., 0.]])

#         return filter
    
#     def filter_data(filter, zs):
#         s = Saver(filter)
#         filter.batch_filter(zs, saver = s)
#         s.to_array()
#         return s

#     # Apply Kalman filter
#     kf2 = SecondOrderKF(measurement_noise, process_noise, dt = dt)
#     deterministic_component = filter_data(kf2, signal)
#     stohastic_component = signal - deterministic_component

#     # Plot components if needed
#     if plot_components == True:
#         __plot_components__(signal, t, stohastic_component, deterministic_component)

#     return stohastic_component, deterministic_component