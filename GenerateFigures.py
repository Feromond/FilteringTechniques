import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from FilteringTechniques import *
import pandas as pd

def load_data(file_path:str, delimiter:str=',') -> tuple[list[float],list[float]]:
    """
    Reads a file containing two-column numerical data (time and signal).
    Handles files with or without headers and supports various delimiters.
    
    Parameters:
        file_path (str): Path to the input file.
        
    Returns:
        tuple[list[float],list[float]]: Two lists of floats - time and signal data (returns the file data in order of col0, then col1).
    """
    data = pd.read_csv(file_path, sep=delimiter, header=None)
    # Check for headers (assume headers if non-numeric in the first row)
    if not pd.api.types.is_numeric_dtype(data.iloc[0, 0]):
        data.columns = data.iloc[0]
        data = data[1:].reset_index(drop=True)
    if data.shape[1] < 2:
        raise ValueError(f"File {file_path} must have at least two columns.")
    col0 = data.iloc[:, 0].astype(float).tolist()
    col1 = data.iloc[:, 1].astype(float).tolist()
    return col0, col1

"""
Importing the Measured Data and Wavelet from Files
"""

wavelet, wavelet_time = load_data("waveletData.txt", ',')
measured_data, measured_time  = load_data("rawdata.txt", ',')

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
