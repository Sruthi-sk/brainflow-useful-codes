# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 16:02:49 2022

@author: Sruthi Kuriakose

Real-Time streaming 
(Using matplotlib's FuncAnimation)
"""


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds #LogLevels,
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations # AggOperations, WindowFunctions, 
import time
# import os
# import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print('Check which eeg channel is best !')

#%% Connect to brainflow

params = BrainFlowInputParams()
board_id = BoardIds.SYNTHETIC_BOARD.value
# board_id = BoardIds.MUSE_S_BOARD.value
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


#%%

class brainflowAnimation:
    def __init__(self,pause_t=10):
        figure, axs = plt.subplots(4,figsize=(4,3))
        figure.suptitle(' Real-time streaming of four channels')
        x = np.linspace(0, 2, sf*2)
        y = board.get_current_board_data(2*sf)[0]
        self.linech1, = axs[0].plot(x, y)
        self.linech2, = axs[1].plot(x, y)
        self.linech3, = axs[2].plot(x, y)
        self.linech4, = axs[3].plot(x, y,color='g')
        # self.line = [linech1, linech2,linech3,linech4]
        self.pause_time=pause_t

        for i in range(4): #same axes initializations
            axs[i].axis([0, 2, -300, 600]) # xmin,xmax,ymin,ymax
            
        self.animation = FuncAnimation(
            figure, self.run, self.data_gen, interval=500, blit=True,repeat=False) #frames=200 or 10 - 10fps, dpi, interval in ms
        self.paused = False

        figure.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused


    def data_gen(self):
        start_time=time.time()
        while time.time() < start_time+ self.pause_time:
            y = board.get_current_board_data(2*sf)
            yield y

    def run(self,data):
        # update the data
        self.linech1.set_ydata(data[4])
        self.linech2.set_ydata(data[11])
        self.linech3.set_ydata(data[14])
        self.linech4.set_ydata(data[1])    
        return self.linech1,self.linech2,self.linech3,self.linech4,

rtstream = brainflowAnimation(pause_t=5)
plt.show()
