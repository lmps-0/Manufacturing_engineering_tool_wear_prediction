from nptdms import TdmsFile
import pandas as pd
# from functions_edit import wavelet_calculator,stat_calculator
from tkinter.filedialog import askdirectory,askopenfilenames
from tkinter import Tk

from tsfresh import extract_features


#%%

#labellist=pd.read_excel(r"D:\00 Messdaten\pnh\00 Cut Data\Drehen_NZXBohrer20200421.xlsx")
labellist=pd.read_excel(r"D:\users\pnh_lm\11052020\NZXBohrer20200505_ags2.xlsx")

wz=labellist["Werkzeug"].dropna().reset_index(drop=True).tolist()
# vc=labellist["Schnittgeschwindigkeit [m/min]"].dropna().reset_index(drop=True).tolist()
# f=labellist["Vorschub f [mm]"].dropna().reset_index(drop=True).tolist()
# ap=labellist["Schnitttiefe ap [mm]"].dropna().reset_index(drop=True).tolist()
# lc=labellist["Drehweg [m]"].dropna().reset_index(drop=True).tolist()

# vbmax=labellist["Vbmax [Pixel]"].dropna().reset_index(drop=True).tolist()
vbarea=labellist["Messung"].dropna().reset_index(drop=True).tolist()


#%%
#Create list to store series and labels into
Tk().withdraw()
paths=askopenfilenames()

#paths=askdirectory()

#%% Fill Container

i=0
container = pd.DataFrame()
while i<  len(paths):
    tdms_file = TdmsFile(paths[i]).as_dataframe()
    tdms_file['id'] = i
    tdms_file['time'] = tdms_file.index
    # try: 
    #     tdms_file = tdms_file.drop(columns = ["/'Messdaten'/'Passivkraft'", "/'Messdaten'/'Schnittkraft'","/'Messdaten'/'Vorschubkraft'"])
    # except:
    #     pass
    tdms_file = tdms_file.dropna()
    tdms_file = tdms_file[0:6000]
    container = container.append(tdms_file)
    print(i)
    i=i+1  
  

#%% Run Feature Extractor    IN CONSOLE KOPIEREN

extracted_features = extract_features(container, column_id="id", column_sort="time", n_jobs = 4)

      
#%%

finalframe=extracted_features.reset_index(drop=True)
# finalframe=pd.concat(extracted_features).reset_index(drop=True)
finalframe=finalframe.dropna(axis='columns')        
# finalframe["vbmax"]=vbmax
finalframe["vbarea"]=vbarea
# finalframe["Schnittgeschwindigkeit [m/min]"]=vc
# finalframe["Vorschub f [mm]"]=f
# finalframe["Schnitttiefe ap [mm]"]=ap
# finalframe["Drehweg [m]"]=lc
finalframe["Werkzeug"]=wz
        
finalframe.to_csv("dataDRILL_short_beginning_tsfresh.csv", index=False)

