# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:34:18 2022

@author: Sruthi Kuriakose

Qs - 
- how much data will the brainflow buffer hold?
optimize window_size to  minimum required for filtering

Algorithm
- pause time of 0.01 if no of samples we get from buffer every call is low - unnecessary
- get real time eeg data of n samples - maybe 2s for filtering ? - optimize
- of last 20 samples, find peaks - Rpeak acc to threshold - 10 didnt work in finding peaks :/
- If Rpeak, produce sound
- check delay between sound and rpeak in saved ecg data
- check if all rpeaks are being detected - real-time filtering may cause some to be missed?


"""

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, DetrendOperations

#%% Connect to brainflow

params = BrainFlowInputParams()
# board_id = BoardIds.SYNTHETIC_BOARD.value
board_id = BoardIds.MUSE_S_BOARD.value
sf= sampling_rate = BoardShim.get_sampling_rate(board_id)
board_descr = BoardShim.get_board_descr(board_id)
time_channel = BoardShim.get_timestamp_channel(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)
marker_ch = BoardShim.get_marker_channel(board_id)
# nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
board = BoardShim(board_id, params)

board.prepare_session()

# path_w=os.path.abspath('C:\\Users\ws4\Documents\Sruthi_JRF\Brainflow-muse\data_muse_ecg.csv')
# board.start_stream(10,'file://'+path_w+':w')
board.start_stream()

time.sleep(2)
data = board.get_current_board_data (256) # get latest 256 samples or less, doesnt remove them from internal buffer
# data = board.get_board_data()  # get all data and remove it from internal buffer
print(data)


# board.stop_stream()
# board.release_session()


#%% #%% plot the 4 eeg channels - check best

data = board.get_board_data() #clear buffer
fig, axs = plt.subplots(len(eeg_channels))
time.sleep(3)
# dataz = board.get_current_board_data(sf*3)
dataz = board.get_board_data()

for i in range(len(eeg_channels)):
    channel = i+1
    DataFilter.detrend(dataz[channel], DetrendOperations.CONSTANT.value) # DetrendOperations.LINEAR.value
    DataFilter.perform_bandpass(dataz[channel], sampling_rate, 15.0, 28.0, 2,FilterTypes.BUTTERWORTH.value, 0) # filter (1,29)
    axs[i].plot(dataz[time_channel],dataz[i+1]) 

plt.show()

#%% Find min samples and pause_time to call from buffer without any delay 

def testdelay(mini_sample_size,t):
    count = 0
    wait_max=1
    start_time, current_time = time.time(), time.time()
    while time.time() < (start_time + wait_max):
        newest_data = board.get_current_board_data (mini_sample_size)[channel] 
        count+=1
        time.sleep(t)
        # plt.pause(0.01)
    print("sample_size : ",mini_sample_size, ", time ",t, ", count",count)
    # print(data_1s)
    
mini_sample_sizes = [250,5]    
# mini_sample_sizes = [int(sf/10),24,20,16,15,12,10,8,4]
times = [0,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
combined = [(f,s) for f in mini_sample_sizes for s in times]
for samples,t in combined: 
    testdelay(samples,t)

# testdelay(10,0.001)  # COUNT = 554

# depends on time.sleep(t)
# In 1 second
# no time.sleep -> count= 40k for sample size 5 , around 30k for 250 samples
# time.sleep(t) 
#t=0.1, count = 10  -> so for mini_sample_size=5, we miss samples since only 50 samples collected of 256 sf
# sample_size :  5 , time  0.1 , count 10
# sample_size :  5 , time  0.05 , count 20
# sample_size :  5 , time  0.01 , count 97
# sample_size :  5 , time  0.005 , count 186
# sample_size :  5 , time  0.001 , count 566
# sample_size :  5 , time  0.0005 , count 563
# sample_size :  5 , time  0.0001 , count 568

# t=0.001 optimal time if pause needed to get max count

#########################################################################################################
#%% Visualize continuous real time ecg
channel=1

# mini_sample_size = int(sf/8)
mini_sample_size = 20

data_2s = board.get_board_data()
time.sleep(2)
data_2s = board.get_board_data()[channel]
whole_data = data_2s.copy()
DataFilter.detrend(whole_data, DetrendOperations.CONSTANT.value)
DataFilter.perform_bandpass(whole_data, sampling_rate,15, 28, 2,FilterTypes.BUTTERWORTH.value, 0)  
# times=np.linspace(0,3,data_2s.shape[0])

# plt.figure()
fig, axs = plt.subplots(2)
count = 0
wait_max=5
start_time, current_time = time.time(), time.time()
while time.time() < (start_time + wait_max):
    axs[0].cla()
    newest_data = board.get_current_board_data(mini_sample_size)
    count+=1
    new_data=newest_data[channel]
    data_2s= data_2s[mini_sample_size:]
    data_2s=np.append(data_2s,new_data)
    whole_data = data_2s.copy()
    # DataFilter.detrend(whole_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(whole_data, sampling_rate, 15.0, 28.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    
    # # time.sleep(0.001)
    # plt.plot(times,data_2s)
    axs[0].plot(whole_data)
    # plt.draw()
    axs[0].axvline(x=data_2s.shape[0]-mini_sample_size,color='r')
    # plt.show()
    # plt.ylim(-50,50)
    plt.pause(0.1)
    
end=time.time()
print("totaltime:",end-start_time)
print("sample_size : ",mini_sample_size)
print("count",count)

print(wait_max*sf,"<", mini_sample_size*count,"? if yes, samples not missing")

#########################################################################################################
#%% record baseline, find peak prominence cutoff
import scipy.signal as signal

#baseline 5 seconds
channel=4
baseline_data = board.get_board_data()
time.sleep(5)
baseline_data = board.get_board_data()[channel]
DataFilter.detrend(baseline_data, DetrendOperations.CONSTANT.value)
DataFilter.perform_bandpass(baseline_data, sampling_rate, 15.0, 24.0, 2,FilterTypes.BUTTERWORTH.value, 0)

#get peak threshold from baseline filtered data         --------- Testing - data_baseline = ecg_data_array[:2000]
peaks, _ = signal.find_peaks(baseline_data)
prominences = signal.peak_prominences(baseline_data, peaks)[0]
prom_threshold=np.percentile(prominences, 96) *0.95
peak_baseline = signal.find_peaks(baseline_data,prominence=prom_threshold)[0] 

plt.figure(figsize=(20,5))
plt.plot(baseline_data)
plt.scatter(peak_baseline,baseline_data[peak_baseline],color="red")

# def findpeak(data):
#     peaks=signal.find_peaks(data,prominence=prom_threshold)[0]
#     # if peaks.size!=0:# makesound()
#     return peaks

#########################################################################################################
#%%
import pygame
pygame.mixer.init()
pygame.mixer.music.load(r"C:\Users\ws4\Documents\Sruthi_JRF\IITG\440Hz.wav")
pygame.mixer.music.set_volume(0.2)

#%%
board.get_board_data() # to clear buffer
time.sleep(2)

mini_sample_size = 10
time_trial = 10
window_size = 2*sf
# smarkers,trial_time = find_rpeak(time_trial=time_trial,sync=True)
count = 0 # no of buffer calls
total_peaks = 0
soundmarkers = []

pygame.mixer.music.play()
time.sleep(1)
sample_start = board.get_board_data_count()
start_trial = time.time()
start_time, current_time = time.time(), time.time()

eeg_chunk_list = []

while current_time < (start_time + time_trial):
    # plt.cla()
    pygame.mixer.music.set_volume(0.1)
    count+=1
    # since filtering required, take longer data from buffer instead of appending operations
    window_data = board.get_current_board_data(window_size)[channel]
    DataFilter.detrend(window_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(window_data, sampling_rate, 15.0, 28.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    
    board.insert_marker(count)
    eeg_chunk_list.append(window_data)
    # Find all prominent peaks in longer window data - produce sound only if peak is there in last 10 samples
    peak_window = signal.find_peaks(window_data,prominence=prom_threshold)[0]
    peak_curr=peak_window[(peak_window > window_data.shape[0]-mini_sample_size)]
    # peak_curr = signal.find_peaks(data_curr,prominence=prom_thres)[0]
    print(peak_window.shape)
    print(peak_curr.shape)
    
    if peak_curr.shape[0]!=0:
        total_peaks+=1
        print('r-wave')
        pygame.mixer.music.set_volume(1)
        soundmarkers.append(board.get_board_data_count())  #no of samples
        board.insert_marker(1)
	time.sleep(0.4) # no ecg peak in next 400ms
        
    time.sleep(0.02) # try removing - more count ie no of buffer calls
    
    # # remove plotting
    # plt.plot(window_data)
    # plt.scatter(peak_window,window_data[peak_window],color="red")
    # plt.draw()
    # plt.axvline(x=window_data.shape[0]-mini_sample_size,color='r')
    # plt.pause(0.001)
    
    current_time=time.time() 

pygame.mixer.music.stop()
print("count ",count)   
# end_trial=time.time()
# trial_time_taken = np.round(end_trial-start_trial,3)
# print(trial_time_taken)
print("total_peaks ",total_peaks)
# soundmarkers = np.unique(soundmarkers)

#########################################################################################################
#%% 'Save complete data from buffer'
#dir(board)
soundmarkers = np.unique(soundmarkers)

sample_end = board.get_board_data_count()
print(sample_end)  
# data_whole_rec =  board.get_current_board_data(sample_end-sample_start) #.reshape(-1).astype('float64') 
# will only work in above line is called immediately - should miss samples - better to take whole data
# cross check buffer size
data_whole_rec = board.get_board_data()
df = pd.DataFrame(np.transpose(data_whole_rec))
# DataFilter.write_file(data_whole_rec, r"C:\Users\ws4\Documents\Sruthi_JRF\IITG\ecg_muse_test/ecg_muse.csv", 'w')  # use 'a' for append mode

df.iloc[soundmarkers,:] #- verify that markers have been recorded in marker_ch

paths = r"C:\Users\ws4\Documents\Sruthi_JRF\ecg_muse_test/"
df.to_csv(paths+'saved_ecg_240822_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.csv')

# soundmarkers_ecg = [np.array(sm)-sample_start for sm in soundmarkers]  # only if saved data is using get_current_board_data
print("sound markers : ",soundmarkers) #_ecg
joblib.dump(soundmarkers,paths+'soundmarkers_ecg_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.pkl')
# joblib.dump(trial_choice,paths+'true_markers.pkl')
# joblib.dump(keyboard_responses,paths+'responses.pkl')

# r=joblib.load('D:/CCS_Users/sruthi/ecg/'+'responses.pkl')

# End buffer
#board.stop_stream()
#board.release_session()

# data_1=pd.DataFrame(np.transpose(data))
# print(data_1.head(10))

joblib.dump(eeg_chunk_list ,paths+'chunk_list_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.pkl')

#########################################################################################################
#%% analyze delay between Rpeak and soundmarker

#no saved data
# data_whole_rec = board.get_board_data()
data_pres = pd.DataFrame(np.transpose(data_whole_rec[channel]))
sample_start1,sample_end1 = soundmarkers[0]-200,soundmarkers[-1]+200
data_pres = data_pres[sample_start1:sample_end1][0]


#%%FROM SAVED DATA
ecg_data_rec = pd.read_csv(paths+'saved_ecg_240822_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.csv',index_col=0)
soundmarkers_rec=joblib.load(paths+'soundmarkers_ecg_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.pkl')

marker_data = ecg_data_rec.iloc[:,marker_ch]
soundmarkers_rec2 = marker_data[marker_data == 1].index
# soundmarkers = soundmarkers[1:]

ecg_data_rec = ecg_data_rec.iloc[:,channel]
# plot soundmarkers
plt.plot(ecg_data_rec,'g')
plt.scatter(soundmarkers_rec,ecg_data_rec[soundmarkers],color='b',zorder=3)

sample_start2,sample_end2 = soundmarkers_rec[0]-200,soundmarkers_rec[-1]+200
ecg_data_rec = ecg_data_rec[sample_start2:sample_end2]

#%%find rpeaks

# #non saved
# ecg_data_array = data_pres
# soundmarkers_delayed = soundmarkers

#saved
ecg_data_array = ecg_data_rec
soundmarkers_delayed = soundmarkers_rec

peaks, _ = signal.find_peaks(ecg_data_array)
prominences = signal.peak_prominences(ecg_data_array, peaks)[0]
prom_thres=np.percentile(prominences, 99)*0.7
signal_rpeaks = signal.find_peaks(ecg_data_array,prominence=prom_thres)[0] +ecg_data_array.index[0]

plt.figure
plt.plot(ecg_data_array,'g')
plt.scatter(signal_rpeaks,ecg_data_array[signal_rpeaks],color='r')
plt.scatter(soundmarkers_delayed,ecg_data_array[soundmarkers_delayed],color='b',zorder=3)
plt.title(str(mini_sample_size)+'samples- rtplot '+str(count)+' count')

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx #array[idx]

#find delay in no of samples bw rpeak and soundmarker
sample_diffs = []
for soundmk in soundmarkers_delayed:
    nearest_rpeaks_idx = find_nearest_idx(signal_rpeaks,soundmk)
    sample_diff = soundmk - signal_rpeaks[nearest_rpeaks_idx]
    sample_diffs.append(sample_diff)
print(sample_diffs)
    

#%%

board.stop_stream()
board.release_session()
    
#########################################################################################################
#%% Compare filtered chunks to original - whether effect of filter is removing rpeaks

# Reasons why only taking 10 samples may miss peaks
# 1 filter is creating phase delay so peak is missing
# 2 peak detection algo not good enough

mini_sample_size= 20 
count = 198


mini_sample_size= 10 
count = 476
paths = r"C:\Users\ws4\Documents\Sruthi_JRF\IITG\ecg_muse_test/"

eeg_chunks = joblib.load(paths+'chunk_list_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.pkl')
ecg_data_rec = pd.read_csv(paths+'saved_ecg_240822_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.csv',index_col=0)
# ecg_data_rec = ecg_data_rec.iloc[:,channel]
soundmarkers_rec=joblib.load(paths+'soundmarkers_ecg_'+str(mini_sample_size)+'samples_plotrt_'+str(count)+'count.pkl')


#check ecg_data_rec dataframe - column 15 "marker"
marker_data = ecg_data_rec.iloc[:,marker_ch]
idx_buffercall = marker_data[marker_data != 0].index
idx_diffs = np.diff(idx_buffercall)

for i in range(6,40):
    plt.plot(eeg_chunks[i])
    # plt.pause(1)

eeg_chunks = np.vstack(eeg_chunks)
eeg_unique_unordered = [list(x) for x in set(tuple(x) for x in eeg_chunks)] # orderig not preserved
eeg_unique_np = np.unique(np.array(eeg_chunks), axis=0)  # orderig not preserved

seq = [1,3,5,7,6,4,7,4,2,9,2]
eeg_noDupes = []
[eeg_noDupes.append(i) for i in seq if not eeg_noDupes.count(i)]

idfun=None
if idfun is None:
    def idfun(x): return x
seen = {}
result = []
for item in seq:
    marker = idfun(item)
    # in old Python versions:
    # if seen.has_key(marker)
    # but in new ones:
    if marker in seen: continue
    seen[marker] = 1
    result.append(item)
print(result)

########################################################################################################

#For 1s (count ~ no of buffer calls
# time.sleep(0.001) - count = 550
#Plotting with cla, count = 20
# only plt.cla, count = 54  
# without any plotting or time sleeping- 20k
# How to check delay?
# Get rpeaks, produce sound, verify sample differences between soundmarker and rpeak
