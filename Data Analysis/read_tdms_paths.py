from nptdms import TdmsFile

import nptdms

import pandas as pd
import numpy as np
# from functions import wavelet_calculator,stat_calculator
from tkinter.filedialog import askdirectory,askopenfilenames
from tkinter import Tk

import os
from natsort import natsorted
#%%

#labellist=pd.read_excel(r"D:\00 Messdaten\pnh\00 Cut Data\NZXBohrer20200505_ags2.xlsx")
labellist=pd.read_excel(r"D:\users\pnh_lm\11052020\NZXBohrer20200505_ags2.xlsx")

wz=labellist["Werkzeug"].dropna().reset_index(drop=True).tolist()
vc=labellist["Schnittgeschwindigkeit [m/min]"].dropna().reset_index(drop=True).tolist()
eb = labellist["Eingriffsbereich"].dropna().reset_index(drop=True).tolist()
ver=labellist["Versuch"].dropna().reset_index(drop=True).tolist()
rf = labellist["Reihenfolge"].dropna().reset_index(drop=True).tolist()
bt = labellist["Bohrtiefe [mm]"].dropna().reset_index(drop=True).tolist()

#filenames = labellist["Datensatz Auswertung"]
filenames = labellist["Dateienname original"]


#%%
# #Create list to store series and labels into
# Tk().withdraw()
# paths=askopenfilenames()

# #paths=askdirectory()


#%% - Lucas 15.05.2020

#path = r'D:\00 Messdaten\pnh\00 Cut Data\00 short all'
#path = r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\00 short all'
path = r'Y:\04 Versuche MärzApril 2020\00 Messdaten komplett\Messungen\Daten'

#from functions_edit import wavelet_calculator,stat_calculator
increment=9.9999999999997E-07 # s
increment_fz=0.0001 # s
time_frame = 4.525 # s

resultlist=[]
i=0
while i<len(filenames):
    paths = path + "\\" + filenames[i] + ".tdms"
    tdms_file = TdmsFile(paths).as_dataframe()
    #tdms_file_nptdms_obj = nptdms.TdmsFile(paths)
    
    #for Signal plotting reasons only # 07.08.2020
    #t_thresh_index =tdms_file[tdms_file["/'Untitled'/'Fz'"].values>=200].index[0]
    #t_thresh = t_thresh_index*increment_fz
    #t_end =  t_thresh + time_frame
    
    # Aufnahme | Futter
    #tdms_file = tdms_file.drop(columns = ["/'Untitled'/'AE_RMS_Futter'", "/'Untitled'/'Mz'","/'Untitled'/'Fz'","/'Untitled'/'AE_RMS_Aufnahme'"])
    #dt=increment
    
    #includes Fz
    #tdms_file = tdms_file.drop(columns = ["/'Untitled'/'AE_RMS_Futter'","/'Untitled'/'AE_RMS_Aufnahme'"])
    
    # only Fz
    #tdms_file = tdms_file.drop(columns = ["/'Untitled'/'AE_RMS_Futter'","/'Untitled'/'AE_RAW_Futter'", "/'Untitled'/'Mz'","/'Untitled'/'AE_RMS_Aufnahme'","/'Untitled'/'AE_RAW_Aufnahme'"])
    dt=increment_fz

    # only Mz
    tdms_file = tdms_file.drop(columns = ["/'Untitled'/'AE_RMS_Futter'","/'Untitled'/'AE_RAW_Futter'", "/'Untitled'/'Fz'","/'Untitled'/'AE_RMS_Aufnahme'","/'Untitled'/'AE_RAW_Aufnahme'"])


    columns=tdms_file.columns.tolist()
    featurelist=[]
    
    
    for u in columns:
        
        #w, amplitudes = signalFFT(dt=increment, Data=tdms_file[u])
        #PlotFFT(w, amplitudes, columnname=u[12:], name=filenames[i])
        
        #plotSignal(tdms_file[u],columnname=u[12:], name=filenames[i],dt=dt)
        #only fft features
    
        wavelet_set=wavelet_calculator(tdms_file[u],columnname=u)
        #stat_set=stat_calculator(tdms_file[u],columnname=u+'_signal_')
        #stat_set_fft=stat_calculator(amplitudes,columnname=u+'_fft_')

        featurelist.append(wavelet_set)
        #featurelist.append(stat_set)
        #featurelist.append(stat_set_fft)
        
        
    result = pd.concat(featurelist, axis=1)
    result["Filename"] = filenames[i]
    resultlist.append(result)
    
    print(i)
    i=i+1    
    
        
#%%
    

    
finalframe=pd.concat(resultlist).reset_index(drop=True)
finalframe=finalframe.dropna(axis='columns')        

finalframe["Werkzeug"]=wz
finalframe["Versuch"]=ver
finalframe["Eingriffsbereich"]=eb
finalframe["Schnittgeschwindigkeit [m/min]"]=vc
finalframe["Bohrtiefe"]=bt
finalframe["Reihenfolge"]=rf
finalframe["vbarea"]=0

# eliminating Stufenbohren
finalframe=finalframe[finalframe.Versuch == 'Vollbohrung']
# eliminating Speed 60 m/min
finalframe=finalframe[finalframe["Schnittgeschwindigkeit [m/min]"] == 80]
# eliminating experiments with index > 80
finalframe=finalframe[finalframe.Reihenfolge <= 80]


finalframe.to_csv("Test_Signal_&_FFT_features.csv",index=False)
#finalframe.to_csv("Test_Fz_label_FFT_fts.csv",index=False)
#finalframe.to_csv("Test_Fz_label_only_FFT_fts.csv",index=False)
#finalframe.to_csv("Test_only_FFT_fts.csv",index=False)

#df=finalframe
#df[df.name != 'Tina']

#%% - Lucas 15.05.2020 - This part is not working - investigate later ...

#List of all samples and their features including the label
featureslist=[]

#calculate wavelet features and statistical features from the series
i=0
while i <len(subsequence_series_list):
    wavelet_set=wavelet_calculator(subsequence_series_list[i])
    stat_set=stat_calculator(subsequence_series_list[i])
    result = pd.concat([wavelet_set, stat_set], axis=1, sort=False)
    result["label"]=corresponding_label_list[i]
    featureslist.append(result)
    i=i+1
    
#Concatenate all dataframe into one and save to file    
dataset=pd.concat(featureslist)    
dataset.to_csv(r"ae_features.csv")

