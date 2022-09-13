# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:15:16 2022

@author: ws4
"""

#%% import libs

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds #LogLevels,
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations # AggOperations, WindowFunctions, 
import time
# import os
import matplotlib.pyplot as plt
import mne

#%% Connect to brainflow

params = BrainFlowInputParams()
# board_id = BoardIds.SYNTHETIC_BOARD.value
board_id = BoardIds.MUSE_S_BOARD.value
sf= sampling_rate = BoardShim.get_sampling_rate(board_id)
board_descr = BoardShim.get_board_descr(board_id)
time_channel = BoardShim.get_timestamp_channel(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)
board = BoardShim(board_id, params)

board.prepare_session()
board.start_stream()

print('\nPlotting real time data')
time.sleep(2)
# data = board.get_current_board_data (256) # get latest 256 samples or less, doesnt remove them from internal buffer
data = board.get_board_data()  # get all data and removes it from internal buffer


#############################################################################################################
#%% Visualize continuous real time data without appending it - explore different filters
channel=3
# mini_sample_size = int(sf/8)
mini_sample_size = 20
time.sleep(2)

plt.figure()
count = 0
wait_max=10
start_time, current_time = time.time(), time.time()
while time.time() < (start_time + wait_max):
    plt.cla() 
    newest_data = board.get_current_board_data(sf)[channel]
    count+=1
    d1=newest_data.copy()
    d2=newest_data.copy()
    d3=newest_data.copy()
    d4=newest_data.copy()
    # DataFilter.detrend(newest_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(d1, sampling_rate, 15.0, 28.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(d2, sampling_rate, 1.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_lowpass(d2, sampling_rate, 29.0, 2,FilterTypes.BUTTERWORTH.value, 0)    
    DataFilter.perform_lowpass(d3, BoardShim.get_sampling_rate(board_id), 29.0, 2,
                                       FilterTypes.CHEBYSHEV_TYPE_1.value, 1)
    DataFilter.perform_highpass(d3, BoardShim.get_sampling_rate(board_id), 1.0, 2,
                                       FilterTypes.CHEBYSHEV_TYPE_1.value, 1)
    mne.filter.filter_data(d4, sampling_rate,l_freq=None,h_freq=40., method='iir')
    mne.filter.filter_data(d4, sampling_rate,l_freq=1,h_freq=None, method='iir')
    # time.sleep(0.001)
    # # plt.plot(times,data_2s)
    plt.plot(d1,'r')
    plt.plot(d2,'g')
    plt.plot(d3,'b')
    plt.plot(d4,'k')
    # plt.draw()
    plt.axvline(x=newest_data.shape[0]-mini_sample_size,color='r')
    # # plt.ylim(-50,50)
    plt.pause(0.5)
print(count)


#############################################################################################################
#%% End

# board.stop_stream()
# board.release_session()

print(' Stopped streaming  !')


