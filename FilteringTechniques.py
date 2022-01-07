import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import toeplitz

def freqAxis(dt,nt):
    """
    Generates the frequency axis for a fft output array, 
        centered at 0 Hz
    Inputs:
        dt: Sampling rate   
        nt: Total number of samples
    Returns:
        Frequency axis values
    """
    f = np.arange((-nt)/2,(nt-1)/2)
    return (f/(dt*nt))

def getPaddedArrays(w,r):
    """
    Compares two arrays and pads the shorter array with leading
        & trailing zeros to match the size. 
    Inputs:
        w: Input array 1 (possible wavelet)
        r: Input array 2 (possible reflectivity series)
    Returns:
        wc: Ouput array 1 (padded possible wavelet)
        rc: Output array 2 (padded possible reflectivity)
        n: Maximum length of the two input vectors
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

def f_from_t(times,n):
    """
    Computes the FFT of a time series input array
    Inputs:
        times: Descrete time domain values
        n: Number of elements expected for potential padding.
    Returns:
        The frequency domain centered at 0 Hz
    """
    return (np.fft.fftshift(np.fft.fft(times,n)))

def t_from_f(freqs):
    """
    Computes the iFFT of a frequency domain input array
    Inputs:
        freqs: Frequency domain values, centered at 0Hz
    Returns:
        The decrete time domain series
    """
    return (np.fft.ifft(np.fft.ifftshift(freqs)))

def conv_FreqDom(wc,rc):
    """
    Convolves two arrays using frequency domain multiplication
    Inputs:
        wc: Input array 1 (potentially a wavelet)
        rc: Input array 2 (potentially a reflectivity series)
    Returns:
        Convolved time domain array
    """
    w = wc.copy(); r = rc.copy()
    w,r,n = getPaddedArrays(w,r)
    return t_from_f(f_from_t(w,n)*f_from_t(r,n))

def ormsbyWavelet(fparams,freqs):
    """
    Generates an Ormsby Wavelet in the frequency domain, 
        with defined lowcut, highpass, lowpass, highcut 
        frequencies (For both + & - frequencies)
    Inputs:
        fparams: Array of length 4, containing values of [fLc, fHp, fLp, fHc]
        freqs: Frequency axis array from data the wavelet is to be applied on
    Returns:
        ormsby_wavelet: Array containing the frequency domain ormsby filter
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

def prediction_Filter(signal,n_filt):
    """
    Creates a prediction filter of a select size. Uses the
        signal index n and prediction index n+1 at each step
    Inputs:
        signal: Array of data to be used for building the filter
        n_filt: Desired length of prediction filter
    Returns:
        pred_filter: Prediction filter based on input signal
    """
    matrix_first_column = signal[:len(signal)-1]
    matrix_first_row = np.zeros(n_filt)
    matrix_first_row[0] = signal[0]
    s_matrix_half = toeplitz(matrix_first_column,matrix_first_row)
    s_target = signal[1:]
    pred_filter = np.linalg.lstsq(np.transpose(s_matrix_half).dot(s_matrix_half),np.transpose(s_matrix_half).dot(s_target),rcond=None)[0]
    return pred_filter

def getToplitzZeroForm(column,n_col):
    """
    Generates a toplitzx matrix using the input of the 
        first column and the number of desired columns
    Inputs:
        column: The array of values to be in the first column
        n_col: The number of columns in the matrix
    Returns:
        Toeplitz matrix of the data with the desired number of columns
    """
    matrix_first_column = column
    matrix_first_row = np.zeros(n_col)
    matrix_first_row[0] = matrix_first_column[0]
    return toeplitz(matrix_first_column,matrix_first_row)

def getMatchFilter(wavelet,data):
    """
    Creates a match filter based on an input wavelet 
        and the data it will be applied to
    Inputs:
        wavelet: Input wavelet for the filter to match
        data: The dataset where the filter will be applied
    Returns:
        match_filter: The designed match filter for the dataset
        d_padded: (Optional use) The dataset of matching length to the filter.
        n_match: The length of the filter
    """
    wavelet_flipped = np.flip(wavelet)
    w_padded, d_padded,n_match = getPaddedArrays(wavelet_flipped,data)
    match_filter = np.fft.fftshift(w_padded)
    return match_filter,d_padded,n_match

def applyMatchFilter(filter,data,n):
    """
    Applies an input match filter to an input dataset
    Inputs:
        filter: The input match filter to be applied
        data: The data to apply the filter on
        n: Length of the filter and data
    Returns:
        match_filtered_data: Data after applying the filter
    """
    f_freq = f_from_t(filter,n)
    d_freq = f_from_t(data,n)
    f_freq = np.abs(f_freq)/np.abs(np.max(f_freq))
    match_filter_freq = f_freq * d_freq
    match_filtered_data = t_from_f(match_filter_freq)
    return match_filtered_data



"""
Importing the Measured Data and Wavelet from Files
"""
data_file = open('rawdata.txt','r')
wavelet_file = open('waveletData.txt','r')

measured_data = []; measured_time = []; wavelet = []; wavelet_time = []

for line in data_file:          # Reading the data file contents
    column = line.strip().split(',')
    measured_data.append(float(column[0]))
    measured_time.append(float(column[1]))
data_file.close()

for line in wavelet_file:       # Reading the wavelet file contents
    column = line.strip().split(',')
    wavelet.append(float(column[0]))
    wavelet_time.append(float(column[1]))
wavelet_file.close()

# Time between each measurement sample (sampling rate)
dt = abs(wavelet_time[1]-wavelet_time[0])

"""
Plotting the provided wavelet and raw data
"""
figure_RawData, (axe0,axe1) = plt.subplots(2,1,num='Raw Data')
axe0.plot(measured_time,measured_data)
axe0.set_xlabel('Time (sec)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Measured Data',fontweight='bold')
axe1.plot(wavelet_time,wavelet)
axe1.set_xlabel('Time (sec)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Wavelet',fontweight='bold')
plt.tight_layout()

"""
Creating a delta function and convolving it with the input wavelet.
    Testing convolution via multiplication in the frequency domain.
"""
delta_func = signal.unit_impulse(500,'mid')     # Delta spike located at the center
wavelet_delta_padded, delta_func_padded, n = getPaddedArrays(wavelet,delta_func)
wavelet_delta_shifted = np.fft.fftshift(wavelet_delta_padded)
delta_time = np.arange(0,dt*len(delta_func),dt)
convolved_delta_wavelet = conv_FreqDom(wavelet_delta_shifted,delta_func_padded)

figure_DeltaSpike, (axe0,axe1) = plt.subplots(2,1,num='Delta Spike & Convolved Spike w/ Wavelet')
axe0.plot(delta_time,delta_func)                        # Plotting the delta spike
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Delta Spike',fontweight='bold')
axe1.plot(delta_time,np.real(convolved_delta_wavelet),'r')  # Plotting convolved d-spike & wavelet
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Convolved Wavelet & Delta Spike',fontweight='bold')
plt.tight_layout()

wavelet_copy = wavelet.copy()
wavelet_shifted = np.fft.fftshift(wavelet_copy)
n_wavelet = len(wavelet_shifted); n_data = len(measured_data)

"""
Plotting the frequency domain of the wavelet and the raw-data
"""
figure_FreqDomain, (axe0,axe1) = plt.subplots(2,1,num='Frequency Domain of Data & Wavelet')
axe0.plot(freqAxis(dt,n_wavelet),np.abs(f_from_t(wavelet_shifted,n_wavelet)))
axe0.set_xlabel('Frequency (Hz)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Wavelet Frequency Domain',fontweight='bold')
axe1.plot(freqAxis(dt,n_data),np.abs(f_from_t(measured_data,n_data)))
axe1.set_xlabel('Frequency (Hz)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Measured Data Frequency Domain',fontweight='bold')
plt.tight_layout(pad=0.25)


"""
Designing a Butterworth Band-pass Filter
"""
fc1 = 75 ; fc2 = 27     # High & Low Cut Freqs
b1,a1 = signal.butter(5,fc1,btype='lowpass',fs=(1/dt))
b2,a2 = signal.butter(2,fc2,btype='highpass',fs=(1/dt))
w1,h1 = signal.freqz(b1,a1) ; w2,h2 = signal.freqz(b2,a2)
measured_data_copy = measured_data.copy()
butter_filtered_data_init = signal.filtfilt(b1,a1,measured_data_copy)  # Filtered data using butter-filter lowpass
butter_filtered_data = signal.filtfilt(b2,a2,butter_filtered_data_init) # Filtered data using butte-filtered highpass

"""
Designing a Ormsby Band-pass Filter
"""
fparams = np.array([0,40,65,115])   # Low-Cut, Highpass, Lowpass, High-cut
ormsby_freqs = freqAxis(dt,n_data)
ormsby_wavelet = ormsbyWavelet(fparams,ormsby_freqs)
ormsby_filtered_data = np.fft.ifft(np.fft.ifftshift(ormsby_wavelet * f_from_t(measured_data,n_data)))

"""
Plotting the raw data, ormsby filtered data, and butterworth filtered data
"""
figure_Filters, (axe0,axe1,axe2) = plt.subplots(3,1,num='Bandpass Filters')
axe0.plot(measured_time,measured_data)
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Measured Data',fontweight='bold')
axe1.plot(measured_time,np.real(ormsby_filtered_data),'blue')
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Ormsby Filtered Data',fontweight='bold')
axe2.plot(measured_time,np.real(butter_filtered_data),'red')
axe2.set_xlabel('Time (s)',fontweight='bold')
axe2.set_ylabel('Amplitude',fontweight='bold')
axe2.set_title('Butterworth Filtered Data',fontweight='bold')
plt.tight_layout(pad=0.001)

"""
Creating and applying prediction filter of length 30 and 
    length 2000 on raw data
"""
measured_data_copy = measured_data.copy()
data_30 = getToplitzZeroForm(measured_data_copy,30)
data_2000 = getToplitzZeroForm(measured_data_copy,2000)
pred_filter_30 = prediction_Filter(measured_data_copy[:int(len(measured_data_copy)/2)],30)
pred_filter_2000 = prediction_Filter(measured_data_copy[:int(len(measured_data_copy)/2)],2000)
prediction_30 = data_30.dot(pred_filter_30)
prediction_2000 = data_2000.dot(pred_filter_2000)

"""
Plotting the raw-data prediction filteres along with the 
    raw data and the butterworth bandpass filter for comparison
"""
figure_Predictions_MeasuredData, (axe0,axe1,axe2,axe3) = plt.subplots(4,1,num='Raw-Data Prediction Filters')
axe0.plot(measured_time,measured_data)
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Measured Data',fontweight='bold')
axe1.plot(measured_time,np.real(butter_filtered_data))
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Butterworth Filtered Data',fontweight='bold')
axe2.plot(measured_time,prediction_30)
axe2.set_xlabel('Time (s)',fontweight='bold')
axe2.set_ylabel('Amplitude',fontweight='bold')
axe2.set_title('Prediction Filter - Length 30',fontweight='bold')
axe3.plot(measured_time,prediction_2000)
axe3.set_xlabel('Time (s)',fontweight='bold')
axe3.set_ylabel('Amplitude',fontweight='bold')
axe3.set_title('Prediction Filter - Length 2000',fontweight='bold')
plt.tight_layout(pad=0.05)

"""
Creating and applying prediction filter of length 30 and 
    length 2000 on butterworth filtered data
"""
butter_filtered_data_copy = butter_filtered_data.copy()
bandpassed_30 = getToplitzZeroForm(butter_filtered_data_copy,30)
bandpassed_2000 = getToplitzZeroForm(butter_filtered_data_copy,2000)
pred_filter_30_bandpassed = prediction_Filter(butter_filtered_data_copy[:int(len(butter_filtered_data_copy)/2)],30)
pred_filter_2000_bandpassed = prediction_Filter(butter_filtered_data_copy[:int(len(butter_filtered_data_copy)/2)],2000)
prediction_30_bandpassed = bandpassed_30.dot(pred_filter_30_bandpassed)
prediction_2000_bandpassed = bandpassed_2000.dot(pred_filter_2000_bandpassed)

"""
Plotting the butterworth-filtered-data prediction filteres along with the 
    raw data and the butterworth bandpass filter for comparison
"""
figure_Predictions_BandPassed, (axe0,axe1,axe2,axe3) = plt.subplots(4,1,num='Filtered-Data Prediction Filters')
axe0.plot(measured_time,measured_data)
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Measured Data',fontweight='bold')
axe1.plot(measured_time,np.real(butter_filtered_data))
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Butterworth Filtered Data',fontweight='bold')
axe2.plot(measured_time,prediction_30_bandpassed)
axe2.set_xlabel('Time (s)',fontweight='bold')
axe2.set_ylabel('Amplitude',fontweight='bold')
axe2.set_title('Bandpassed Prediction Filter - Length 30',fontweight='bold')
axe3.plot(measured_time,prediction_2000_bandpassed)
axe3.set_xlabel('Time (s)',fontweight='bold')
axe3.set_ylabel('Amplitude',fontweight='bold')
axe3.set_title('Bandpassed Prediction Filter - Length 2000',fontweight='bold')
plt.tight_layout(pad=0.05)

"""
Creating and applying a match filter on the raw data
"""
wavelet_copy = wavelet.copy()
measured_data_copy = measured_data.copy()
match_filter, measured_data_copy, n_match = getMatchFilter(wavelet_copy,measured_data_copy)
match_filtered_data = applyMatchFilter(match_filter,measured_data_copy,n_match)

"""
Plotting the raw data, butterworth filtered data, 
    and the match filtered raw-data
"""
figure_Matched_Filter, (axe0,axe1,axe2) = plt.subplots(3,1,num='Raw-Data Match Filter')
axe0.plot(measured_time,measured_data)
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Measured Data',fontweight='bold')
axe1.plot(measured_time,np.real(butter_filtered_data))
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Butterworth Filtered Data',fontweight='bold')
axe2.plot(measured_time,np.real(match_filtered_data))
axe2.set_xlabel('Time (s)',fontweight='bold')
axe2.set_ylabel('Amplitude',fontweight='bold')
axe2.set_title('Match Filtered Data',fontweight='bold')
plt.tight_layout(pad=0.25)

"""
Creating and applying a match filter on the butterworth filtered data
"""
wavelet_copy = wavelet.copy()
butter_filtered_data_copy = butter_filtered_data.copy()
match_filter_bandpassed, butter_filtered_data_copy, n_match = getMatchFilter(wavelet_copy,butter_filtered_data_copy)
match_filtered_data_bandpassed = applyMatchFilter(match_filter_bandpassed,butter_filtered_data_copy,n_match)

"""
Plotting the raw data, butterworth filtered data, and 
    the match filtered butterworth-filtered-data
"""
figure_Matched_Filter_Bandpassed, (axe0,axe1,axe2) = plt.subplots(3,1,num='Filtered-Data Match Filter')
axe0.plot(measured_time,measured_data)
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Measured Data',fontweight='bold')
axe1.plot(measured_time,np.real(butter_filtered_data))
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Butterworth Filtered Data',fontweight='bold')
axe2.plot(measured_time,np.real(match_filtered_data_bandpassed))
axe2.set_xlabel('Time (s)',fontweight='bold')
axe2.set_ylabel('Amplitude',fontweight='bold')
axe2.set_title('Bandpassed Match Filtered Data',fontweight='bold')
plt.tight_layout(pad=0.25)


"""
Additional Figures
"""
figure_Filter_Freqs, (axe0,axe1) = plt.subplots(2,1,num='Bandpass Filter Frequency Domains')
axe0.plot(freqAxis(dt,n_data),np.abs(ormsby_wavelet),'r--',label='Ormsby Filter')
axe0.plot(freqAxis(dt,n_wavelet),np.abs(f_from_t(wavelet_shifted,n_wavelet))/np.max(np.abs(f_from_t(wavelet_shifted,n_wavelet))),label='Wavelet',color='blue')
axe0.set_xlim(-150,150)
axe0.set_xlabel('Frequency (Hz)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Ormsby & Wavelet Frequency Domain',fontweight='bold')
axe0.legend()

axe1.plot(((1/dt) * 0.5 / np.pi) * w1, abs(h1),'r--',label='Butterworth Lowpass')
axe1.plot(((1/dt) * 0.5 / np.pi) * w2, abs(h2),'g--',label='Butterworth Highpass')
axe1.plot(freqAxis(dt,n_wavelet),np.abs(f_from_t(wavelet_shifted,n_wavelet))/np.max(np.abs(f_from_t(wavelet_shifted,n_wavelet))),label='Wavelet',color='blue')
axe1.set_xlim(0,200)
axe1.set_xlabel('Frequency (Hz)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Butterworth & Wavelet Frequency Domain',fontweight='bold')
axe1.legend()
plt.tight_layout()


figure_predictionFilter30_2000_raw, (axe0,axe1) = plt.subplots(2,1,num='Prediction Filters')
axe0.plot(np.arange(0,dt*len(pred_filter_30),dt),pred_filter_30)
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Prediction Filter of Raw-Data - Length 30',fontweight='bold')

axe1.plot(np.arange(0,dt*len(pred_filter_2000),dt),pred_filter_2000)
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Prediction Filter of Raw-Data - Length 2000',fontweight='bold')
plt.tight_layout()


figure_predictionFilter30_2000_filtered, (axe0,axe1) = plt.subplots(2,1,num='Prediction Filters Bandpassed')
axe0.plot(np.arange(0,dt*len(pred_filter_30_bandpassed),dt),pred_filter_30_bandpassed,'r')
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Prediction Filter of Bandpassed-Data - Length 30',fontweight='bold')

axe1.plot(np.arange(0,dt*len(pred_filter_2000_bandpassed),dt),pred_filter_2000_bandpassed,'r')
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Prediction Filter of Bandpassed-Data - Length 2000',fontweight='bold')
plt.tight_layout()


figure_Match_Filter, (axe0,axe1) = plt.subplots(2,1,num='Match Filters')
axe0.plot(measured_time,match_filter)
axe0.set_xlabel('Time (s)',fontweight='bold')
axe0.set_ylabel('Amplitude',fontweight='bold')
axe0.set_title('Raw-Data Match Filter',fontweight='bold')

axe1.plot(measured_time,match_filter_bandpassed)
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Bandpassed-Data Match Filter',fontweight='bold')
plt.tight_layout()


figure_Filters2, (axe1,axe2) = plt.subplots(2,1,num='Bandpass Filters2')
axe1.plot(measured_time,np.real(ormsby_filtered_data),'blue')
axe1.set_xlim(0.6,0.8)
axe1.set_xlabel('Time (s)',fontweight='bold')
axe1.set_ylabel('Amplitude',fontweight='bold')
axe1.set_title('Ormsby Filtered Data',fontweight='bold')
axe2.plot(measured_time,np.real(butter_filtered_data),'red')
axe2.set_xlim(0.6,0.8)
axe2.set_xlabel('Time (s)',fontweight='bold')
axe2.set_ylabel('Amplitude',fontweight='bold')
axe2.set_title('Butterworth Filtered Data',fontweight='bold')
plt.tight_layout()

plt.show()
plt.close()
