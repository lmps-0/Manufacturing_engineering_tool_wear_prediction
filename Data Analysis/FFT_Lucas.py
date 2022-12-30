# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:29:05 2020

@author: pnh_lm
"""
#import os
#import numpy as np
#from numpy import loadtxt
#import pandas as pd
#import nptdms
#from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
#import re
#import statistics as st
#from scipy import interpolate
from numpy.fft import fftfreq

#%% - first a function to convert series in an array

#def file_conv(tdms_file[u],columnname=u):
#    obj = tdmsFile.object(u)


#%%

def PlotFFT(w, amplitudes, columnname, name):
    #fig = plt.figure(figsize=(20,8)) 
    #plt.scatter(w, amplitudes,s=5)   
    
    #columnname : sensor
    #name : file name 
    
    fig = plt.figure(figsize=(16/2.54,10/2.54))
    plt.rcParams.update({'font.size': 12})
    plt.plot(w, amplitudes,'-')   
    
    title = name+'_'+columnname
    
    #plt.plot(w, amplitudes)
    plt.xlim([50, 400])
    ########################
    ########################
    ########################
    plt.ylim([0, 8]) # Aufnahme
    ########################
    #plt.ylim([0, 8000])   # Futter
    ########################
    ########################
    #ax.legend(loc=1); # upper left corner
    ##########################################################################
    ##########################################################################
    # English:
    #plt.xlabel('Frequency f / kHz')
    #plt.ylabel('Amplitude A / mV')
    #plt.ylabel(u'Amplitude A / \u03bcV')
    # Deutsch:
    plt.xlabel('Frequenz f / kHz')
    plt.ylabel('Amplitude A / mV')
    ##########################################################################
    ##########################################################################
    
    #plt.ticklabel_format(axis='y', style='sci',scilimits=(-3,3))
    plt.minorticks_on()
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    #plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ############################################################################################
    ############################################################################################
    #fig.savefig(r"U:\Bohrer_Titan_2020\26042020\FFT_plots\Bohrer8_"+name+"_"+bohrung_str+"_FFT.png")
    #fig.savefig(r"U:\Bohrer_Titan_2020\26042020\FFT_plots\Bohrer9_"+name+"_"+bohrung_str+"_FFT.png")
    #fig.savefig(r"Y:\04 Versuche MärzApril 2020\04 Summary\plots_FFT\Plot_FFT_"+name+"_"+columnname+".png")
    #English:
    #fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_FFT\plot_"+title+".jpeg",dpi=300)
    #Deutsch:
    fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_FFT_german\plot_"+title+".jpeg",dpi=300)
    ############################################################################################
    ############################################################################################
    plt.close();
    #plt.show(block='False')

#%%
    
def signalFFT(dt, Data):
    amplitudes = np.abs(np.fft.rfft(Data))/len(Data)
    w = np.fft.rfftfreq(len(Data), d=dt)
    #w_o = w; amplitudes_o= amplitudes;
    freqBoundsFFT = (w>50000) & (w < 400000)
    # Units
    #amplitudes = amplitudes[freqBoundsFFT]/1e-6 # [uV]
    amplitudes = amplitudes[freqBoundsFFT]/1e-3 # [mV]
    w = w[freqBoundsFFT]/1e3 
    return w, amplitudes
