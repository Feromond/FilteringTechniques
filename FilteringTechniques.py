import numpy as np
from numpy import complex128
from scipy.linalg import toeplitz

def freqAxis(dt:float, nt:int) -> np.ndarray:
    """
    Generates the frequency axis for a fft output array, 
    centered at 0 Hz
    Parameters:
        dt (float): Sampling rate   
        nt (int): Total number of samples
    Returns:
        np.ndarray: Frequency axis values
    """
    f = np.arange((-nt)/2,(nt-1)/2)
    return (f/(dt*nt))

def getPaddedArrays(w:list[float], r:np.ndarray) -> tuple[np.ndarray,np.ndarray, int]:
    """
    Compares two arrays and pads the shorter array with leading
    & trailing zeros to match the size. 
    Parameters:
        w (list[float]): Input array 1 (possible wavelet)
        r (np.ndarray): Input array 2 (possible reflectivity series)
    Returns:
        wc (np.ndarray): Ouput array 1 (padded possible wavelet)
        rc (np.ndarray): Output array 2 (padded possible reflectivity)
        n (int): Maximum length of the two input vectors
    """
    wc = w.copy(); rc = r.copy()
    n = max(len(wc),len(rc))
    padw = (n-len(wc))//2
    padr = (n-len(rc))//2
    if len(wc)>len(rc):
        rc = np.pad(rc,(padr,padr),'constant')
    elif len(w)<len(rc):
        wc = np.pad(wc,(padw,padw),'constant')
    return wc,rc,n

def f_from_t(times:np.ndarray, n:int) -> np.ndarray:
    """
    Computes the FFT of a time series input array
    Parameters:
        times (np.ndarry): Descrete time domain values
        n (int): Number of elements expected for potential padding.
    Returns:
        np.ndarray: The frequency domain centered at 0 Hz
    """
    return (np.fft.fftshift(np.fft.fft(times,n)))

def t_from_f(freqs: np.ndarray) -> np.ndarray[complex128]:
    """
    Computes the iFFT of a frequency domain input array
    Parameters:
        freqs (np.ndarray): Frequency domain values, centered at 0Hz
    Returns:
        np.ndarray[complex128]: The decrete time domain series
    """
    return (np.fft.ifft(np.fft.ifftshift(freqs)))

def conv_FreqDom(wc: np.ndarray,rc: np.ndarray) -> np.ndarray[complex128]:
    """
    Convolves two arrays using frequency domain multiplication
    Parameters:
        wc (np.ndarray): Input array 1 (potentially a wavelet)
        rc (np.ndarray): Input array 2 (potentially a reflectivity series)
    Returns:
        np.ndarray[complex128]: Convolved time domain array
    """
    w = wc.copy(); r = rc.copy()
    w,r,n = getPaddedArrays(w,r)
    return t_from_f(f_from_t(w,n)*f_from_t(r,n))

def ormsbyWavelet(fparams:np.ndarray[int], freqs:np.ndarray) -> np.ndarray[float]:
    """
    Generates an Ormsby Wavelet in the frequency domain, 
    with defined lowcut, highpass, lowpass, highcut 
    frequencies (For both + & - frequencies)
    Parameters:
        fparams (np.ndarray[int]): Array of length 4, containing values of [fLc, fHp, fLp, fHc]
        freqs (np.ndarray): Frequency axis array from data the wavelet is to be applied on
    Returns:
        ormsby_wavelet (np.ndarray[float]): Array containing the frequency domain ormsby filter
    """
    ormsby_wavelet = np.zeros(len(freqs))
    lowCut = np.where(np.round(freqs) == fparams[0])
    lowPass = np.where(np.round(freqs) == fparams[1])
    highPass = np.where(np.round(freqs) == fparams[2])
    highCut = np.where(np.round(freqs) == fparams[3])
    nlowCut = np.where(np.round(freqs) == -fparams[0])
    nlowPass = np.where(np.round(freqs) == -fparams[1])
    nhighPass = np.where(np.round(freqs) == -fparams[2])
    nhighCut = np.where(np.round(freqs) == -fparams[3])
    ampchange1 = np.abs((1/(lowPass[0][0] - lowCut[0][0])))
    ampchange2 = np.abs((1/(highCut[0][0] - highPass[0][0])))
    slope1 = 0; slope2 = 1
    nampchange1 = np.abs((1/(nlowPass[0][0] - nlowCut[0][0])))
    nampchange2 = np.abs((1/(nhighCut[0][0] - nhighPass[0][0])))
    nslope1 = 1; nslope2 = 0
    for i in range(len(ormsby_wavelet)):
        if freqs[i] > fparams[0] and freqs[i] <= fparams[1]:
            ormsby_wavelet[i] = slope1
            slope1 += ampchange1
        if freqs[i] > fparams[1] and freqs[i] <= fparams[2]:
            ormsby_wavelet[i] = 1
        if freqs[i] > fparams[2] and freqs[i] <= fparams[3]:
            ormsby_wavelet[i] = slope2
            slope2 -= ampchange2
        if freqs[i] < -fparams[0] and freqs[i] >= -fparams[1]:
            ormsby_wavelet[i] = nslope1
            nslope1 -= nampchange1
        if freqs[i] < -fparams[1] and freqs[i] >= -fparams[2]:
            ormsby_wavelet[i] = 1
        if freqs[i] < -fparams[2] and freqs[i] >= -fparams[3]:
            ormsby_wavelet[i] = nslope2
            nslope2 += nampchange2
    return ormsby_wavelet

def prediction_Filter(signal: list[float] | np.ndarray, n_filt: int) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Creates a prediction filter of a select size. Uses the
    signal index n and prediction index n+1 at each step
    Parameters:
        signal (list[float] | np.ndarray): Array of data to be used for building the filter
        n_filt (int): Desired length of prediction filter
    Returns:
        pred_filter (tuple[np.ndarray, np.ndarray, int, np.ndarray]): Prediction filter based on input signal
    """
    matrix_first_column = signal[:len(signal)-1]
    matrix_first_row = np.zeros(n_filt)
    matrix_first_row[0] = signal[0]
    s_matrix_half = toeplitz(matrix_first_column,matrix_first_row)
    s_target = signal[1:]
    pred_filter = np.linalg.lstsq(np.transpose(s_matrix_half).dot(s_matrix_half),np.transpose(s_matrix_half).dot(s_target),rcond=None)[0]
    return pred_filter

def getToplitzZeroForm(column: list[float] | np.ndarray, n_col: int) -> np.ndarray:
    """
    Generates a toplitzx matrix using the input of the 
    first column and the number of desired columns
    Parameters:
        column (list[float] | np.ndarray): The array of values to be in the first column
        n_col (int): The number of columns in the matrix
    Returns:
        np.ndarray: Toeplitz matrix of the data with the desired number of columns
    """
    matrix_first_column = column
    matrix_first_row = np.zeros(n_col)
    matrix_first_row[0] = matrix_first_column[0]
    return toeplitz(matrix_first_column,matrix_first_row)

def getMatchFilter(wavelet:list[float], data:list[float]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Creates a match filter based on an input wavelet 
    and the data it will be applied to
    Parameters:
        wavelet (list[float]): Input wavelet for the filter to match
        data (list[float]): The dataset where the filter will be applied
    Returns:
        tuple:
        match_filter (np.ndarray): The designed match filter for the dataset
        
        d_padded (np.ndarray): (Optional use) The dataset of matching length to the filter.
        
        n_match (int): The length of the filter
    """
    wavelet_flipped = np.flip(wavelet)
    w_padded, d_padded,n_match = getPaddedArrays(wavelet_flipped,data)
    match_filter = np.fft.fftshift(w_padded)
    return match_filter,d_padded,n_match

def applyMatchFilter(filter:np.ndarray, data:list[float], n:int) -> np.ndarray[complex128]:
    """
    Applies an input match filter to an input dataset
    Parameters:
        filter (np.ndarray): The input match filter to be applied
        data (list[float]): The data to apply the filter on
        n (int): Length of the filter and data
    Returns:
        match_filtered_data (np.ndarray[complex128]): Data after applying the filter
    """
    f_freq = f_from_t(filter,n)
    d_freq = f_from_t(data,n)
    f_freq = np.abs(f_freq)/np.abs(np.max(f_freq))
    match_filter_freq = f_freq * d_freq
    match_filtered_data = t_from_f(match_filter_freq)
    return match_filtered_data

