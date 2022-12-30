# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:31:37 2020

@author: ags
"""
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import pandas as pd
# from functions_edit import wavelet_calculator,stat_calculator
from tkinter.filedialog import askdirectory,askopenfilenames
from tkinter import Tk

from tsfresh import extract_features

from numpy import fft
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

#%%

labellist=pd.read_excel(r"D:\00 Messdaten\pnh\00 Cut Data\Drehen_NZXBohrer20200421.xlsx")

wz=labellist["Werkzeug"].dropna().reset_index(drop=True).tolist()
# vc=labellist["Schnittgeschwindigkeit [m/min]"].dropna().reset_index(drop=True).tolist()
# ff=labellist["Vorschub f [mm]"].dropna().reset_index(drop=True).tolist()
# ap=labellist["Schnitttiefe ap [mm]"].dropna().reset_index(drop=True).tolist()
# lc=labellist["Drehweg [m]"].dropna().reset_index(drop=True).tolist()

# vbmax=labellist["Vbmax [Pixel]"].dropna().reset_index(drop=True).tolist()
vbarea=labellist["Messung"].dropna().reset_index(drop=True).tolist()

feature_container = pd.DataFrame()
feature_container_CSD = pd.DataFrame()
feature_container_CH = pd.DataFrame()

#%% Paths
#Create list to store series and labels into
Tk().withdraw()
paths=askopenfilenames()

#%% Container

i=0
container = pd.DataFrame()
while i<  len(paths):
    tdms_file = TdmsFile(paths[i]).as_dataframe()
    tdms_file['id'] = i
    tdms_file['time'] = tdms_file.index
    tdms_file = tdms_file.drop(columns = ["/'Untitled'/'AE_RMS_Futter'", "/'Untitled'/'Mz'","/'Untitled'/'Fz'","/'Untitled'/'AE_RMS_Aufnahme'"])
    tdms_file = tdms_file.dropna()
    # tdms_file = tdms_file[0:6000]
    container = container.append(tdms_file)
    print(i)
    i=i+1  
    

#%% Cross Power Spectral density
sample_rate = 1000000
npsg = 256
ids = container["id"].unique()

plt.close("all")

f, Pxy = signal.csd(container[container["id"] == 0]["/'Untitled'/'AE_RAW_Futter'"], container[container["id"] == 1]["/'Untitled'/'AE_RAW_Futter'"], sample_rate, nperseg=npsg)

features_CSD = pd.DataFrame(columns = f)
features_CH = pd.DataFrame(columns = f)
features_CSD.loc[0] = np.nan # Da erste Messung Referenz ist, kann sie nicht ausgewertet werden
features_CH.loc[0] = np.nan # Da erste Messung Referenz ist, kann sie nicht ausgewertet werden

for ii in range(len(ids)-1):
    print(ii)
    
    f, Pxy = signal.csd(container[container["id"] == 0]["/'Untitled'/'AE_RAW_Futter'"], container[container["id"] == ii+1]["/'Untitled'/'AE_RAW_Futter'"], sample_rate, nperseg=npsg)
    
    plt.figure(0)
    
    plt.semilogy(f, np.abs(Pxy))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('CSD [V**2/Hz]')
    plt.legend(ids)
    plt.show()
    
    features_CSD.loc[ii+1] = np.abs(Pxy)

    f, Cxy = signal.coherence(container[container["id"] == 0]["/'Untitled'/'AE_RAW_Futter'"], container[container["id"] == ii+1]["/'Untitled'/'AE_RAW_Futter'"], sample_rate, nperseg=npsg)
    
    plt.figure(1)
    
    plt.semilogy(f, np.abs(Cxy))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('CSD [V**2/Hz]')
    plt.legend(ids)
    plt.show()
    
    features_CH.loc[ii+1] = Cxy
    

feature_container_CSD = feature_container_CSD.append(features_CSD)
feature_container_CH = feature_container_CH.append(features_CH)


#%% Abspeichern

# feature_container_CSD = feature_container_CSD.add_suffix('_CSD')
# feature_container_CH = feature_container_CH.add_suffix('_CH')

feature_container = pd.concat([feature_container_CSD,feature_container_CH],axis=1)

finalframe=feature_container.reset_index(drop=True)
#finalframe=pd.concat(extracted_features).reset_index(drop=True)
#finalframe=finalframe.dropna(axis='columns')        
# finalframe["vbmax"]=vbmax
finalframe["vbarea"]=vbarea
# finalframe["Schnittgeschwindigkeit [m/min]"]=vc
# finalframe["Vorschub f [mm]"]=ff
# finalframe["Schnitttiefe ap [mm]"]=ap
# finalframe["Drehweg [m]"]=lc
finalframe["Werkzeug"]=wz
        
finalframe.to_csv("dataDrill_short_beginning_csd_ch_B3_7.csv", index=False)


finalframe.to_csv("TEST.csv", index=False)

# #%% FFT
# sample_rate = 1000000

# signal1 = container[container["id"] == 0]["/'Messdaten'/'AE Roh'"]


# freq = fft.rfftfreq(len(signal1), 1/sample_rate)

# sp= abs(fft.rfft(signal1))/len(signal1) # Normalisierung


# # freq[:] = freq[(freq[:]>50000) & freq[:]<400000)]

# df = pd.DataFrame({'sp':sp, 'freq':freq})

# df = df.drop(df[(df.freq< 50000) | (df.freq> 400000)].index)


# plt.plot(df.freq, df.sp)

