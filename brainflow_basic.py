# -*- coding: utf-8 -*-
"""
Created on Fri May  6 13:01:15 2022

@author: Sruthi Kuriakose

https://brainflow.readthedocs.io/en/stable/UserAPI.html#python-api-reference
"""

import time
import os
import matplotlib.pyplot as plt


from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, DetrendOperations

#%% Connect to brainflow

params = BrainFlowInputParams()
board_id = BoardIds.SYNTHETIC_BOARD.value
# board_id = BoardIds.MUSE_S_BOARD.value
sf= sampling_rate = BoardShim.get_sampling_rate(board_id)
board_descr = BoardShim.get_board_descr(board_id)
time_channel = BoardShim.get_timestamp_channel(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)
# nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
board = BoardShim(board_id, params)

board.prepare_session()

# path_w=os.path.abspath('C:\\Users\ws4\Documents\Sruthi_JRF\Brainflow-muse\data_muse.csv')
# board.start_stream(1000,'file://'+path_w+':w')
board.start_stream()

# start_stream(num_samples: int = 450000, streamer_params: str = None) → None
# Start streaming data, this methods stores data in ringbuffer
# num_samples (int) – size of ring buffer to keep data
# parameter to stream data from brainflow, supported vals (streamer_params) – “file://%file_name%:w”, “file://%file_name%:a”, “streaming_board://%multicast_group_ip%:%port%”. Range for multicast addresses is from “224.0.0.0” to “239.255.255.255”

time.sleep(2)
data = board.get_current_board_data (256) # get latest 256 samples or less, doesnt remove them from internal buffer
# data = board.get_board_data()  # get all data and remove it from internal buffer
print(data)


for i in range(10):
    time.sleep(1)
    board.insert_marker(i + 1)
data = board.get_board_data()

# board.insert_marker(i + 1)

# board.stop_stream()
# board.release_session()

# df = pd.DataFrame(np.transpose(data))
# df[eeg_channels].plot(subplots=True)
# plt.show()


# # use first eeg channel for demo
# # second channel of synthetic board is a sine wave at 10 Hz, should see big 'alpha'
# eeg_channel = eeg_channels[1]
# plt.plot(data[2])

#%% #%% plot the 4 eeg channels 

data = board.get_board_data() #clear buffer
fig, axs = plt.subplots(len(eeg_channels))
time.sleep(3)
# dataz = board.get_current_board_data(sf*3)
dataz = board.get_board_data()

for i in range(4):
    channel = i+1
    DataFilter.detrend(dataz[channel], DetrendOperations.CONSTANT.value) # DetrendOperations.LINEAR.value
    DataFilter.perform_bandpass(dataz[channel], sampling_rate, 15.0, 28.0, 2,FilterTypes.BUTTERWORTH.value, 0) # filter (1,29)
    DataFilter.perform_bandstop(dataz[channel], sampling_rate, 50.0, 4.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    axs[i].plot(dataz[time_channel],dataz[i+1]) # dataz[14] - time channel

# psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate, WindowFunctions.HANNING.value)
# plt.plot(psd[1][:60], psd[0][:60])
# plt.show()
# plt.savefig('psd.png')

# alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
# beta = DataFilter.get_band_power(psd, 14.0, 30.0)
# print("Alpha/Beta Ratio is: %f" % (alpha / beta))


#%% #%% MNE integration
import mne
eeg_channels = BoardShim.get_eeg_channels(board_id)
eeg_data = data[eeg_channels, :]
eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(board_id)
sfreq = BoardShim.get_sampling_rate(board_id)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_data, info)
raw.plot()

#%%

#dir(board)