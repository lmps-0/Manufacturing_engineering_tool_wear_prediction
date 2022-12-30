from nptdms import TdmsFile
import pandas as pd
# from functions import wavelet_calculator,stat_calculator
from tkinter.filedialog import askdirectory,askopenfilenames
from tkinter import Tk
#%%

#labellist=pd.read_excel(r"D:\00 Messdaten\pnh\00 Cut Data\NZXBohrer20200505_ags2.xlsx")
labellist=pd.read_excel(r"D:\users\pnh_lm\11052020\NZXBohrer20200505_ags2.xlsx")

wz=labellist["Werkzeug"].dropna().reset_index(drop=True).tolist()
# vc=labellist["Schnittgeschwindigkeit [m/min]"].dropna().reset_index(drop=True).tolist()
# f=labellist["Vorschub f [mm]"].dropna().reset_index(drop=True).tolist()
# ap=labellist["Schnitttiefe ap [mm]"].dropna().reset_index(drop=True).tolist()
# lc=labellist["Drehweg [m]"].dropna().reset_index(drop=True).tolist()

# vbmax=labellist["Vbmax [Pixel]"].dropna().reset_index(drop=True).tolist()
# vbarea=labellist["Area [Pixel]"].dropna().reset_index(drop=True).tolist()
messung = labellist["Reihenfolge"].dropna().reset_index(drop=True).tolist()

#%%
#Create list to store series and labels into
Tk().withdraw()
paths=askopenfilenames()

#paths=askdirectory()

#%%
resultlist=[]
i=0
while i<len(paths):
    tdms_file = TdmsFile(paths[i]).as_dataframe()
    # tdms_file = tdms_file.drop(columns = ["/'Untitled'/'AE_RMS_Futter'", "/'Untitled'/'Mz'","/'Untitled'/'Fz'","/'Untitled'/'AE_RMS_Aufnahme'"])
    columns=tdms_file.columns.tolist()
    featurelist=[]
    for u in columns:
        wavelet_set=wavelet_calculator(tdms_file[u],columnname=u)
        stat_set=stat_calculator(tdms_file[u],columnname=u)
        featurelist.append(wavelet_set)
        featurelist.append(stat_set)
        
    result = pd.concat(featurelist, axis=1)
    resultlist.append(result)
    print(i)
    i=i+1    
        
       
#%%
finalframe=pd.concat(resultlist).reset_index(drop=True)
finalframe=finalframe.dropna(axis='columns')        
# finalframe["vbmax"]=vbmax
# finalframe["vbarea"]=vbarea
# finalframe["Schnittgeschwindigkeit [m/min]"]=vc
# finalframe["Vorschub f [mm]"]=f
# finalframe["Schnitttiefe ap [mm]"]=ap
# finalframe["Drehweg [m]"]=lc
finalframe["Werkzeug"]=wz
finalframe["vbarea"]=messung

        
finalframe.to_csv("dataDrill_short_beginning_specfeat_B3_8.csv",index=False)
#%%
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

#%% 
print("FINISHED")