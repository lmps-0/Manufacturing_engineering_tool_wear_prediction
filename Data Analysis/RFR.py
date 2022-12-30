# coding: utf-8
# In[1]:
# ranking features with the Random Forest Regressor
# In[2]:
# in Aanaconda Prompt:
# (base) U:\first_task_Scheibe> pushd \\wzl-archive1\pnh$
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[3]:
# Here we are going to try to apply "Random Forest" for selecting or ranking the Features.
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score
from sklearn.model_selection import train_test_split,cross_validate,ShuffleSplit
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,RandomForestRegressor
from sklearn import preprocessing

# Vollbohrung
dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\Test_Signal_&_FFT_features.csv') 

#%% Choose the Tool:

# Choosing Tool 
#dataset=dataset[dataset["Werkzeug"].values==7]

#%% Label definition

y=dataset["Reihenfolge"]
#y=dataset["/'Untitled'/'Fz'_signal_mean"]
#y=dataset["/'Untitled'/'Mz'_signal_mean"]
#y=dataset.index

#%%
X=dataset

#%% Dataset definition - Features From:

X=X.drop(["Filename"],axis =1)
X=X.drop(["Versuch"],axis =1)
X=X.drop(["Eingriffsbereich"],axis =1)
X=X.drop(["Reihenfolge"],axis =1)
X=X.drop(["Werkzeug"],axis =1)

X=X.drop(["Bohrtiefe"],axis =1)
X=X.drop(["Schnittgeschwindigkeit [m/min]"],axis =1)
X=X.drop(["vbarea"],axis =1)


#X=X.drop(["/'Untitled'/'Fz'mean"],axis =1)
#X=X.drop(["/'Untitled'/'Fz'standarddeviation"],axis =1)
#X=X.drop(["/'Untitled'/'Fz'variance"],axis =1)
#X=X.drop(["/'Untitled'/'Fz'kurtosis"],axis =1)
#X=X.drop(["/'Untitled'/'Fz'abs_energy"],axis =1)
#X=X.drop(["/'Untitled'/'Fz'median"],axis =1)
#X=X.drop(["/'Untitled'/'Fz'skewness"],axis =1)
#X=X.drop(["/'Untitled'/'Fz'autocorrelation"],axis =1)

X = X[X.columns.drop(list(X.filter(regex="'Fz'_signal")))]
X = X[X.columns.drop(list(X.filter(regex="'Mz'_signal")))]

# nur FFTs
#X = X[X.columns.drop(list(X.filter(regex="signal")))]
#X = X[X.columns.drop(list(X.filter(regex='ec')))]

#y=dataset['wear_measurements_interpolated']
#X=X.drop('wear_measurements_interpolated',axis=1)
#X=X.drop('wear_rate',axis=1)
#X=X.drop('median',axis=1)
#X=X.drop('signal__quantile__q_0.6',axis=1)
#y=dataset['wear_measurements_interpolated']

le = preprocessing.LabelEncoder()
y=le.fit_transform(y)
scaler=StandardScaler()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
X_train[X_train.columns]=scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns]=scaler.transform(X_test[X_test.columns])

print(np.unique(y))

#Feature selection
feat_labels=X.columns[:].copy()
#Regression
forest = RandomForestRegressor(n_estimators=100,random_state=0,n_jobs=-1)
#Classification forest = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

OrderedFeatures=[]
OrderedImportances=[]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f+1,feat_labels[indices[f]], importances[indices[f]])
    OrderedFeatures.append(feat_labels[indices[f]])
    OrderedImportances.append(importances[indices[f]])


'''
TestScore=[]
scoringlist=[]
#scoring_classification=["accuracy","f1_micro","f1_macro","f1_weighted"]
reg_metric="r2" #"neg_mean_squared_error" #r2
scoring_regression=[reg_metric]
cv=ShuffleSplit(n_splits=3,test_size=0.1,random_state=0)
i=1
while i<=len(OrderedFeatures):
    
    X1=X_train[OrderedFeatures[:i]]
    X2=X_test[OrderedFeatures[:i]]
    
    #Regression
    clf=RandomForestRegressor(n_jobs=-1,n_estimators=100)
    #Classification clf=RandomForestClassifier(n_jobs=-1,n_estimators=100,criterion="gini")
    scores=cross_validate(clf,X1,y_train,scoring=scoring_regression,cv=cv,return_train_score=True)
    scoringlist.append(scores) 
    
    clf.fit(X1,y_train)
    y_pred=clf.predict(X2)
    #test_score=accuracy_score(y_test,y_pred)
    test_score=r2_score(y_test,y_pred)
    TestScore.append(test_score)
    print(i)
    
    i=i+1
'''
OrderedFeatures_OrderedImportances = pd.DataFrame({
        "OrderedFeatures" : OrderedFeatures,
        "OrderedImportances" : OrderedImportances,
})
# In[6]: Creating new DataSet with the 9th Features most ranked
# first: drop repeted features
#X=X.drop("mean",axis=1)
#X=X.drop("median",axis=1)
X=X[X.columns[indices[0:9]]]
#newDataset = pd.concat([X, dataset['wear_measurements_interpolated']], axis=1) 
#newDataset = pd.concat([X], axis=1) 

# Adding Label to the new Dataset
newDataset = pd.concat([X, dataset["Reihenfolge"]], axis=1) 
#newDataset = pd.concat([X, dataset["/'Untitled'/'Fz'_signal_mean"]], axis=1) 
#newDataset = pd.concat([X, dataset["/'Untitled'/'Mz'_signal_mean"]], axis=1) 

# In[7]:

#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_2_fz_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_3_index_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_4_fz_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_1-HQN_index_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_2-HQN_index_label_fft_features.csv')

#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_1_vollbohrung_index_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_2_vollbohrung_fz_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_3_vollbohrung_index_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_4_vollbohrung_fz_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_5_vollbohrung_index_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_6_vollbohrung_fz_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_7_vollbohrung_index_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_8_vollbohrung_fz_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_9_vollbohrung_index_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_10_vollbohrung_fz_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_11_vollbohrung_index_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_12_vollbohrung_fz_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_13_vollbohrung_mz_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_14_vollbohrung_mz_label_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_15_vollbohrung_mz_label_only_fft_features.csv')
#newDataset.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_16_vollbohrung_mz_label_only_fft_features.csv')


#max_dt=dataset["/'Untitled'/'Fz'mean"].max()  
#max_dt=dataset["/'Untitled'/'Mz'_signal_mean"].max()

#%%
# Plotting the ranked features:

plt.close("all")

fig = plt.figure(figsize=(20,8))

plt.tight_layout()
#plt.grid()

ax1 = fig.add_subplot(3,4,1)
ax2 = fig.add_subplot(3,4,5, sharex=ax1)
ax3 = fig.add_subplot(3,4,9, sharex=ax1)
ax4 = fig.add_subplot(3,4,2, sharex=ax1)
ax5 = fig.add_subplot(3,4,6, sharex=ax1)
ax6 = fig.add_subplot(3,4,10, sharex=ax1)
ax7 = fig.add_subplot(3,4,3, sharex=ax1)
ax8 = fig.add_subplot(3,4,7, sharex=ax1)
ax9 = fig.add_subplot(3,4,11, sharex=ax1)
ax10 = fig.add_subplot(3,4,4, sharex=ax1)
ax11 = fig.add_subplot(3,4,8, sharex=ax1)
#ax12 = fig.add_subplot(3,4,12, sharex=ax1)

# plt.subplots_adjust(hspace=.0)

ax1.title.set_text(OrderedFeatures[0][12:])
ax2.title.set_text(OrderedFeatures[1][12:])
ax3.title.set_text(OrderedFeatures[2][12:])
ax4.title.set_text(OrderedFeatures[3][12:])
ax5.title.set_text(OrderedFeatures[4][12:])
ax6.title.set_text(OrderedFeatures[5][12:])
ax7.title.set_text(OrderedFeatures[6][12:])
ax8.title.set_text(OrderedFeatures[7][12:])
ax9.title.set_text(OrderedFeatures[8][12:])
ax10.title.set_text('Fz')
ax11.title.set_text('Mz')
#ax12.title.set_text('Mz')

#dataset=dataset[dataset["Werkzeug"].values==tool]
feat_HQN=dataset[dataset["Eingriffsbereich"].values=='HQN']
feat_HQ=dataset[dataset["Eingriffsbereich"].values=='HQ']
feat_H=dataset[dataset["Eingriffsbereich"].values=='H']
ax1.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[0]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[0]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[0]],"m+")
ax2.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[1]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[1]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[1]],"m+")
ax3.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[2]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[2]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[2]],"m+")
ax4.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[3]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[3]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[3]],"m+")
ax5.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[4]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[4]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[4]],"m+")
ax6.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[5]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[5]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[5]],"m+")
ax7.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[6]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[6]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[6]],"m+")
ax8.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[7]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[7]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[7]],"m+")
ax9.plot(feat_HQN["Reihenfolge"],feat_HQN[OrderedFeatures[8]],"b+",feat_HQ["Reihenfolge"],feat_HQ[OrderedFeatures[8]],"g+",feat_H["Reihenfolge"],feat_H[OrderedFeatures[8]],"m+")
ax10.plot(dataset["Reihenfolge"],dataset["/'Untitled'/'Fz'_signal_mean"],"r+")
ax11.plot(dataset["Reihenfolge"],dataset["/'Untitled'/'Mz'_signal_mean"],"k+")
#ax12.plot(newDataset["Reihenfolge"],dataset["/'Untitled'/'Mz'_signal_mean"],"k+")

ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()
ax6.minorticks_on()
ax7.minorticks_on()
ax8.minorticks_on()
ax9.minorticks_on()
ax10.minorticks_on()
ax11.minorticks_on()
#ax12.minorticks_on()
major=['major','-','0.5','black']
minor=['minor',':','0.5','black']
ax1.grid(major)
ax1.grid(minor)
ax2.grid(major)
ax2.grid(minor)
ax3.grid(major)
ax3.grid(minor)
ax4.grid(major)
ax4.grid(minor)
ax5.grid(major)
ax5.grid(minor)
ax6.grid(major)
ax6.grid(minor)
ax7.grid(major)
ax7.grid(minor)
ax8.grid(major)
ax8.grid(minor)
ax9.grid(major)
ax9.grid(minor)
ax10.grid(major)
ax10.grid(minor)
ax11.grid(major)
ax11.grid(minor)
#ax12.grid(major)
#ax12.grid(minor)
plt.tight_layout()

fig.savefig(r"Y:\04 Versuche MärzApril 2020\04 Summary\plots\Ansys_16_features_for_ML.png")


# In[8]:

#alternative
#PLOT BARS
plt.figure()
#plt.title("feature selection for label "+labs+" highest accuracy "+str(np.max(meanlist).round(4)))
plt.bar(range(len(OrderedFeatures)), OrderedImportances,color="grey", align="center") #yerr=std[indices[:nof]],
plt.xticks(range(len(OrderedFeatures)), OrderedFeatures,rotation=90,fontsize=7)
plt.ylabel("relative importance")

#PLOT TEST 
meanlist=[]
stdlist=[]
for i in scoringlist:

    #Regression
    a=np.mean(i["test_"+reg_metric])
    b=np.std(i["test_"+reg_metric])
    #a=np.mean(i["test_accuracy"])
    #b=np.std(i["test_accuracy"])
    meanlist.append(a)
    stdlist.append(b)

a_upperline=np.array(meanlist)+np.array(stdlist)
a_lowerline=np.array(meanlist)-np.array(stdlist)
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(meanlist,color="black",marker=".")#,label="highest accuracy"+str(np.max(meanlist).round(4)))
ax2.plot(a_upperline,color="red",label="Test")
ax2.plot(a_lowerline,color="red")
ax2.fill_between(OrderedFeatures,a_lowerline,a_upperline,color="red", alpha=0.3)
ax2.set_ylabel("cumulated accuracy")
ax2.set_xlabel("ordered features")
ax2.plot(TestScore,color="green",label="Validierung",marker=".")

#Plot train
meanlisttrain=[]
stdlisttrain=[]
for i in scoringlist:

    #Regression
    at=np.mean(i["train_"+reg_metric])
    bt=np.std(i["train_"+reg_metric])
    #at=np.mean(i["train_accuracy"])
    #bt=np.std(i["train_accuracy"])
    meanlisttrain.append(at)
    stdlisttrain.append(bt)

a_upperlinet=np.array(meanlisttrain)+np.array(stdlisttrain)
a_lowerlinet=np.array(meanlisttrain)-np.array(stdlisttrain)
ax2.plot(meanlisttrain,color="black",marker=".")
ax2.plot(a_upperlinet,color="blue",label="Train")
ax2.plot(a_lowerlinet,color="blue")
ax2.fill_between(OrderedFeatures,a_lowerlinet,a_upperlinet,color="blue", alpha=0.3)

plt.title("Feature importance highest testaccuracy "+str(np.max(meanlist).round(4)))
plt.autoscale(enable=True, axis='x', tight=True)
#plt.tight_layout()
plt.savefig("featureimportance.svg")
ax2.legend(loc="best")

plt.savefig('U:\\first_task_Scheibe\\16012020\\dataset_complete_nur_10_fts.png')

#"""**Führe das Training aus. Parameter max_trials definiert die Anzahl unterschiedlicher Keras Modelle die verwendet werden sollen**"""

# DeepLearning Autokeras #Cross validated training
import autokeras as ak
#Classification 
#clf = ak.StructuredDataClassifier(name="wzl",max_trials=10,metrics=["acc","f1"])
#Regression
clf=ak.StructuredDataRegressor(name="wzl",max_trials=50,metrics=["mse"])
clf.fit(x=X_train, y=y_train,epochs=1 ,validation_data=(X_test,y_test))

#testlist=[]
#clf.fit(X,y)
#y_pred=clf.predict(X_test)
#test_acc=accuracy_score(y_test,y_pred)
#testlist.append(test_acc)

#MachineLearning ScikitLearn #Cross validated training
classifiers =[RandomForestClassifier(n_jobs=-1, criterion="entropy",n_estimators=1000),GradientBoostingClassifier(),LogisticRegression(C=10000),MLPClassifier(),QDA(),SGDClassifier()]
cv_list=[]

for clf in classifiers:
    cv=ShuffleSplit(n_splits=3, random_state=0, test_size=0.1)
    cv_results = cross_validate(clf, X=data_pca2, y=labellist, cv=cv,n_jobs=-1,return_train_score=True) 
    
    cv_list.append(cv_results)
    
#%%
meanscorelisttrain=[]
stdscorelisttrain=[]

for i in cv_list:
    meanscore=np.mean(i["train_score"], dtype=np.float64)
    stdscore=np.std(i["train_score"],dtype=np.float64)
    meanscorelisttrain.append(meanscore)
    stdscorelisttrain.append(stdscore)

roundedmeanlisttrain=[]
for i in meanscorelisttrain:
    d=i.round(3)
    roundedmeanlisttrain.append(d)

meanscorelisttest=[]
stdscorelisttest=[]

for i in cv_list:
    meanscore=np.mean(i["test_score"], dtype=np.float64)
    stdscore=np.std(i["test_score"],dtype=np.float64)
    meanscorelisttest.append(meanscore)
    stdscorelisttest.append(stdscore)

roundedmeanlisttest=[]
for i in meanscorelisttest:
    d=i.round(3)
    roundedmeanlisttest.append(d)
    

modelname=[]
for i in classifiers:
    c=type(i).__name__
    modelname.append(c)
    
width=0.35  
fig, ax = plt.subplots()
b=np.arange(len(classifiers))
rect=ax.bar(b- width/2,roundedmeanlisttrain,width=width,color="blue")
rect1=ax.bar(b+width/2,roundedmeanlisttest,width=width,color="red")
plt.xticks(b, modelname)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rect)
autolabel(rect1)
plt.xlabel("Classifier")
plt.ylabel("Accuracies")
plt.show()


fig.savefig('U:\\first_task_Scheibe\\16012020\\dataset_complete_nur_10_fts.png')

#%%


# In[ ]:


help(extract_features)

