# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:40:21 2020

@author: pnh_lm
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%

def plotSignal(ae_series,columnname,name,dt):
    
    fig = plt.figure(figsize=(16/2.54,10/2.54))
    plt.rcParams.update({'font.size': 12})
    
    #Offset correction:
    # 0 s - 1.5 s
    x_t = 1.5 # s
    x_t_index = int( x_t / dt)
    offset_y = np.mean(ae_series[0:x_t_index])
    print(offset_y)
    ae_series=ae_series-offset_y
    
    plt.plot(ae_series.index*dt,ae_series,'-')   

    title = name+'_'+columnname
    
    #left=t_thresh
    #right=t_end
    #plt.xlim([left, right])
    plt.xlim([1.5, 7])
    plt.xlabel('Time t / s') #En
    #plt.xlabel('Zeit t / s') #De
    
    # Fz 
    #plt.ylim([-50, 1200])
    #plt.ylabel('Force $F_{z}$ / N') #En
    #plt.ylabel('Kraft $F_{z}$ / N') #De

    # Mz
    plt.ylim([-2, 4])
    plt.ylabel('Torque $M_{z}$ / Nm') #En
    #plt.ylabel('Drehmoment $M_{z}$ / Nm') #De
    
    # Signal
    #plt.ylim([-4, 4])
    #plt.xlabel('Time t / s')
    #plt.ylabel('Amplitude A / V')
    
    major=['major','-','0.5','black']
    plt.grid(major)
    plt.tight_layout()
    fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_Signal\plot_"+title+".jpeg",dpi=300) #En
    #fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_Signal_german\plot_"+title+".jpeg",dpi=300) #De

    plt.close();
    #columnname : sensor
    #name : file name 

#%% 31.07.2020 : Plotting Features x Tools

channel_option=4 # Aufnahme
#channel_option=5 # Futter
#channel_option=6 # Aufnahme FFT
#channel_option=7 # Futter FFT

tools_option=1 # Tools 3,4,5,6
#tools_option=2 # Tools 7,8,9,10

dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\Test_Signal_&_FFT_features.csv') 

feat_list=[]

# English:
'''
# nur channel 4:
#feat_list.append(["Reihenfolge", "Werkzeug", "Number of drilling holes n / - ","Force $F_{z}$ / N", "/'Untitled'/'Fz'_signal_mean"])
#feat_list.append(["Reihenfolge", "Werkzeug", "Number of drilling holes n / - ","Torque $M_{z}$ / Nm", "/'Untitled'/'Mz'_signal_mean"])
#
feat_list.append(["Reihenfolge", "Werkzeug", "Number of drilling holes n / - ","Mean x\u0305 / V","/'Untitled'/'AE_RAW_Aufnahme'_signal_mean", "/'Untitled'/'AE_RAW_Futter'_signal_mean","/'Untitled'/'AE_RAW_Aufnahme'_fft_mean","/'Untitled'/'AE_RAW_Futter'_fft_mean"])
feat_list.append(["Reihenfolge","Werkzeug", "Number of drilling holes n / - ","Standard Deviation σ / V", "/'Untitled'/'AE_RAW_Aufnahme'_signal_standarddeviation","/'Untitled'/'AE_RAW_Futter'_signal_standarddeviation","/'Untitled'/'AE_RAW_Aufnahme'_fft_standarddeviation","/'Untitled'/'AE_RAW_Futter'_fft_standarddeviation"])
feat_list.append(["Reihenfolge","Werkzeug", "Number of drilling holes n / - ","Variance σ² / V²", "/'Untitled'/'AE_RAW_Aufnahme'_signal_variance","/'Untitled'/'AE_RAW_Futter'_signal_variance","/'Untitled'/'AE_RAW_Aufnahme'_fft_variance","/'Untitled'/'AE_RAW_Futter'_fft_variance"])
feat_list.append(["Reihenfolge","Werkzeug", "Number of drilling holes n / - ","Kurtosis K / -", "/'Untitled'/'AE_RAW_Aufnahme'_signal_kurtosis","/'Untitled'/'AE_RAW_Futter'_signal_kurtosis","/'Untitled'/'AE_RAW_Aufnahme'_fft_kurtosis","/'Untitled'/'AE_RAW_Futter'_fft_kurtosis"])
feat_list.append(["Reihenfolge","Werkzeug", "Number of drilling holes n / - ","Abs. Energy $E_{s}$ / V²s", "/'Untitled'/'AE_RAW_Aufnahme'_signal_energy","/'Untitled'/'AE_RAW_Futter'_signal_energy","/'Untitled'/'AE_RAW_Aufnahme'_fft_energy","/'Untitled'/'AE_RAW_Futter'_fft_energy"])
feat_list.append(["Reihenfolge","Werkzeug", "Number of drilling holes n / - ","Median x\u0303 / V", "/'Untitled'/'AE_RAW_Aufnahme'_signal_median","/'Untitled'/'AE_RAW_Futter'_signal_median","/'Untitled'/'AE_RAW_Aufnahme'_fft_median","/'Untitled'/'AE_RAW_Futter'_fft_median"])
feat_list.append(["Reihenfolge","Werkzeug", "Number of drilling holes n / - ","Skewness $S_{k}$ / -", "/'Untitled'/'AE_RAW_Aufnahme'_signal_skewness","/'Untitled'/'AE_RAW_Futter'_signal_skewness","/'Untitled'/'AE_RAW_Aufnahme'_fft_skewness","/'Untitled'/'AE_RAW_Futter'_fft_skewness"])
feat_list.append(["Reihenfolge","Werkzeug", "Number of drilling holes n / - ","Autocorrelation $R_{xx}$ / -", "/'Untitled'/'AE_RAW_Aufnahme'_signal_autocorrelation","/'Untitled'/'AE_RAW_Futter'_signal_autocorrelation","/'Untitled'/'AE_RAW_Aufnahme'_fft_autocorrelation","/'Untitled'/'AE_RAW_Futter'_fft_autocorrelation"])

#result_feat_list = pd.concat(feat_list, axis=1)
tool_en_de="Tool "
'''

# Deutsch:

# nur channel 4:
#feat_list.append(["Reihenfolge", "Werkzeug", "Anzahl Bohrungen n / - ","Kraft $F_{z}$ / N", "/'Untitled'/'Fz'_signal_mean"])
#feat_list.append(["Reihenfolge", "Werkzeug", "Anzahl Bohrungen n / - ","Drehmoment $M_{z}$ / Nm", "/'Untitled'/'Mz'_signal_mean"])
#
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Mittelwert x\u0305 / V","/'Untitled'/'AE_RAW_Aufnahme'_signal_mean", "/'Untitled'/'AE_RAW_Futter'_signal_mean","/'Untitled'/'AE_RAW_Aufnahme'_fft_mean","/'Untitled'/'AE_RAW_Futter'_fft_mean"])
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Standardabweichung σ / V", "/'Untitled'/'AE_RAW_Aufnahme'_signal_standarddeviation","/'Untitled'/'AE_RAW_Futter'_signal_standarddeviation","/'Untitled'/'AE_RAW_Aufnahme'_fft_standarddeviation","/'Untitled'/'AE_RAW_Futter'_fft_standarddeviation"])
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Varianz σ² / V²", "/'Untitled'/'AE_RAW_Aufnahme'_signal_variance","/'Untitled'/'AE_RAW_Futter'_signal_variance","/'Untitled'/'AE_RAW_Aufnahme'_fft_variance","/'Untitled'/'AE_RAW_Futter'_fft_variance"])
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Wölbung w / -", "/'Untitled'/'AE_RAW_Aufnahme'_signal_kurtosis","/'Untitled'/'AE_RAW_Futter'_signal_kurtosis","/'Untitled'/'AE_RAW_Aufnahme'_fft_kurtosis","/'Untitled'/'AE_RAW_Futter'_fft_kurtosis"])
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Energie $E_{s}$ / V²s", "/'Untitled'/'AE_RAW_Aufnahme'_signal_energy","/'Untitled'/'AE_RAW_Futter'_signal_energy","/'Untitled'/'AE_RAW_Aufnahme'_fft_energy","/'Untitled'/'AE_RAW_Futter'_fft_energy"])
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Median x\u0303 / V", "/'Untitled'/'AE_RAW_Aufnahme'_signal_median","/'Untitled'/'AE_RAW_Futter'_signal_median","/'Untitled'/'AE_RAW_Aufnahme'_fft_median","/'Untitled'/'AE_RAW_Futter'_fft_median"])
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Schiefe $S_{k}$ / -", "/'Untitled'/'AE_RAW_Aufnahme'_signal_skewness","/'Untitled'/'AE_RAW_Futter'_signal_skewness","/'Untitled'/'AE_RAW_Aufnahme'_fft_skewness","/'Untitled'/'AE_RAW_Futter'_fft_skewness"])
feat_list.append(["Reihenfolge","Werkzeug", "Anzahl Bohrungen n / - ","Automatische Korrelation $R_{xx}$ / -", "/'Untitled'/'AE_RAW_Aufnahme'_signal_autocorrelation","/'Untitled'/'AE_RAW_Futter'_signal_autocorrelation","/'Untitled'/'AE_RAW_Aufnahme'_fft_autocorrelation","/'Untitled'/'AE_RAW_Futter'_fft_autocorrelation"])
#result_feat_list = pd.concat(feat_list, axis=1)
tool_en_de="Bohrer"
#%% 31.07.2020

for u in feat_list:
    feat=u
    fig = plt.figure(figsize=(16/2.54,10/2.54))
    plt.rcParams.update({'font.size': 12})
    title = feat[channel_option].split("/")[2]
    
    if tools_option==1:
        dataset_tool_3=dataset[dataset["Werkzeug"].values==3]
        dataset_tool_4=dataset[dataset["Werkzeug"].values==4]
        dataset_tool_5=dataset[dataset["Werkzeug"].values==5]
        dataset_tool_6=dataset[dataset["Werkzeug"].values==6]
    
        plt.plot(dataset_tool_3[feat[0]],dataset_tool_3[feat[channel_option]],"+",label=tool_en_de+" 3", color=(0/256,84/256,159/256)) # big problem... vector is not in order
        plt.plot(dataset_tool_4[feat[0]],dataset_tool_4[feat[channel_option]],"+",label=tool_en_de+" 4", color=(142/256,186/256,229/256)) # big problem... vector is not in order
        plt.plot(dataset_tool_5[feat[0]],dataset_tool_5[feat[channel_option]],"+",label=tool_en_de+" 5", color=(0/256,0/256,0/256)) # big problem... vector is not in order
        plt.plot(dataset_tool_6[feat[0]],dataset_tool_6[feat[channel_option]],"+",label=tool_en_de+" 6", color=(156/256,158/256,159/256)) # big problem... vector is not in order
        title = title+"_"+tool_en_de+"_3_4_5_6"
        #plt.legend(loc=3, bbox_to_anchor=(-0.02, -0.5),ncol=4)
        plt.legend(loc=3, bbox_to_anchor=(-0.2, -0.4),ncol=4) #DE

    if tools_option==2:
        dataset_tool_7=dataset[dataset["Werkzeug"].values==7]
        dataset_tool_8=dataset[dataset["Werkzeug"].values==8]
        dataset_tool_9=dataset[dataset["Werkzeug"].values==9]
        dataset_tool_10=dataset[dataset["Werkzeug"].values==10]
        
        plt.plot(dataset_tool_7[feat[0]],dataset_tool_7[feat[channel_option]],"+",label=tool_en_de+" 7", color=(0/256,84/256,159/256)) # big problem... vector is not in order
        plt.plot(dataset_tool_8[feat[0]],dataset_tool_8[feat[channel_option]],"+",label=tool_en_de+" 8", color=(142/256,186/256,229/256)) # big problem... vector is not in order
        plt.plot(dataset_tool_9[feat[0]],dataset_tool_9[feat[channel_option]],"+",label=tool_en_de+" 9", color=(0/256,0/256,0/256)) # big problem... vector is not in order
        plt.plot(dataset_tool_10[feat[0]],dataset_tool_10[feat[channel_option]],"+",label=tool_en_de+" 10", color=(156/256,158/256,159/256)) # big problem... vector is not in order
        title = title+"_"+tool_en_de+"_7_8_9_10"
        #plt.legend(loc=3, bbox_to_anchor=(-0.04, -0.4),ncol=4)
        plt.legend(loc=3, bbox_to_anchor=(-0.2, -0.4),ncol=4) #DE

        
    plt.ticklabel_format(axis='y', style='sci',scilimits=(-1,1))
    plt.xlabel(feat[2])
    plt.ylabel(feat[3])
    plt.xlim(left=0)
    top=dataset[feat[channel_option]].max()+(0.05*dataset[feat[channel_option]].max())
    #########################################################
    #top=0.062
    #plt.ylim(bottom=0,top=1100)
    plt.xlim(left=0,right=82)
    
    #########################################################
    major=['major','-','0.5','black']
    plt.grid(major)
    plt.tight_layout()
    #fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_features\plot_"+title+"_y_min_850.jpeg",dpi=300)
    # English :
    #fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_features\plot_"+title+".jpeg",dpi=300)
    # Deutsch :
    fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_features_german\plot_"+title+".jpeg",dpi=300)

#%%
    
    