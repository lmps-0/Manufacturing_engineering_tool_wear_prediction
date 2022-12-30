import pandas as pd
import pywt
from pywt import wavedec
import tsfresh.feature_extraction.feature_calculators as feat

def stat_calculator(ae_series,columnname=""):
    mean=pd.Series([])
    std_dev=pd.Series([])
    var=pd.Series([])
    kurt=pd.Series([])
    abseng=pd.Series([])
    med=pd.Series([])
    skew=pd.Series([])
    corr=pd.Series([])
    
    a=pd.Series([feat.mean(ae_series)])
    mean=pd.concat([mean,a])
    b=pd.Series([feat.standard_deviation(ae_series)])
    std_dev=pd.concat([std_dev,b])
    c=pd.Series([feat.variance(ae_series)])
    var=pd.concat([var,c])
    d=pd.Series([feat.kurtosis(ae_series)])
    kurt=pd.concat([kurt,d])
    e=pd.Series([feat.abs_energy(ae_series)])
    abseng=pd.concat([abseng,e])
    f=pd.Series([feat.median(ae_series)])
    med=pd.concat([med,f])
    g=pd.Series([feat.skewness(ae_series)])
    skew=pd.concat([skew,g])
    h=pd.Series([feat.autocorrelation(ae_series,1)])
    corr=pd.concat([corr,h])
    
    
    statistics={columnname+'mean':mean,columnname+'standarddeviation':std_dev,columnname+'variance':var,columnname+'kurtosis':kurt,columnname+'energy':abseng,columnname+'median':med,columnname+'skewness':skew,columnname+'autocorrelation':corr}
    sdf=pd.DataFrame(data=statistics)
    sdf=sdf.reset_index(drop=True)
    return sdf

    

def wavelet_calculator(ae_series,wavelettype="db1",columnname=""):
    
    
    w=pywt.Wavelet(wavelettype)
    maxlevel=pywt.dwt_max_level(len(ae_series),w)
    ecA10=pd.Series([])
    ecD10=pd.Series([])
    ecD9=pd.Series([])
    ecD8=pd.Series([])
    ecD7=pd.Series([])
    ecD6=pd.Series([])
    ecD5=pd.Series([])
    ecD4=pd.Series([])
    ecD3=pd.Series([])
    ecD2=pd.Series([])
    ecD1=pd.Series([])
    
    coeffs=wavedec(ae_series,wavelettype,level=maxlevel)
    
    cA10=pd.Series([feat.abs_energy(coeffs[0])])
    ecA10=pd.concat([ecA10,cA10])
    
    cD10=pd.Series([feat.abs_energy(coeffs[1])])
    ecD10=pd.concat([ecD10,cD10])
    
    cD9=pd.Series([feat.abs_energy(coeffs[2])])
    ecD9=pd.concat([ecD9,cD9])
    
    cD8=pd.Series([feat.abs_energy(coeffs[3])])
    ecD8=pd.concat([ecD8,cD8])
    
    cD7=pd.Series([feat.abs_energy(coeffs[4])])
    ecD7=pd.concat([ecD7,cD7])
    
    cD6=pd.Series([feat.abs_energy(coeffs[5])])
    ecD6=pd.concat([ecD6,cD6])
    
    cD5=pd.Series([feat.abs_energy(coeffs[6])])
    ecD5=pd.concat([ecD5,cD5])
    
    cD4=pd.Series([feat.abs_energy(coeffs[7])])
    ecD4=pd.concat([ecD4,cD4])
    
    cD3=pd.Series([feat.abs_energy(coeffs[8])])
    ecD3=pd.concat([ecD3,cD3])
    
    cD2=pd.Series([feat.abs_energy(coeffs[9])])
    ecD2=pd.concat([ecD2,cD2])
    
    cD1=pd.Series([feat.abs_energy(coeffs[10])])
    ecD1=pd.concat([ecD1,cD1])
    
    
    wavelet_header={columnname+'ecA10':ecA10,columnname+'ecD10':ecD10,columnname+'ecD9':ecD9,columnname+'ecD8':ecD8,columnname+'ecD7':ecD7,columnname+'ecD6':ecD6,columnname+'ecD5':ecD5,columnname+'ecD4':ecD4,columnname+'ecD3':ecD3,columnname+'ecD2':ecD2,columnname+'ecD1':ecD1}
    wavelet=pd.DataFrame(data=wavelet_header)
    wavelet=wavelet.reset_index(drop=True)
    return wavelet