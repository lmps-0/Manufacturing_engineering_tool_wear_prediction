# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:50:42 2020

@author: Dr. Thorsten Augspurger
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute  import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt

import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree

from mpl_toolkits.mplot3d import Axes3D

import xgboost as xgb

from sklearn import metrics

import warnings
#%% - Lucas 15.05.2020

data = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\TEST.csv')
#%% - Lucas 15.05.2020 - I dont need this:
'''
# runcell('Read csv and create Pandas Data Frame', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('Combine Features', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('Auswahl eines Werkzeugs od. Prozessparameter', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('Drop or keep defined features', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
# runcell('Choose Specific Features 1', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')

runcell('Include Wear rate', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')


#runcell('Correlation Matrix', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
#runcell('PCA', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')

runcell('Prepare Data Set', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('Transformation Pipline', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')


runcell('Linear Regression', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('Decision Tree Regressor', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('RandomForestRegressor', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('Support Vector Maschine Regressor', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('AdaBoost Regressor', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
runcell('XGBoost Regressor', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')

runcell('GridSearch for Random Forest Regressor', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')
# runcell('GridSearch for SVR', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')


runcell('Cross Validation', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')

runcell('Evaluate Test DataSet', 'D:/WZL_ags/12 Skit AE Pipe/Bohrer/DataPrep.py')

'''
#%% Read csv and create Pandas Data Frame
# data2 = pd.read_csv(r'D:\WZL_ags\12 Skit AE Pipe\Bohrer\dataDrill_short_beginning_specfeat_B3_7.csv')
data2 = pd.read_csv(r'D:\WZL_ags\12 Skit AE Pipe\Bohrer\Test.csv')


data2.to_excel("output.xlsx",

             sheet_name='Sheet_name_1')  

data2 = data2.iloc[:, ::-1]


data2 = data2.dropna(axis="rows")

#%% Combine Features

data2 = pd.read_csv(r'D:\WZL_ags\12 Skit AE Pipe\Bohrer\dataDrill_short_beginning_ae_csd_ch_B3_7.csv')

# data2 = pd.read_csv(r'D:\WZL_ags\12 Skit AE Pipe\Bohrer\TEST.csv')

test = np.abs(data2)

#Für csd Dateien mit 128 Frequenzbändern

# data2 = data2.drop(data2.iloc[:, 0:24], axis = 1) 

# data2 = data2.drop(data2.iloc[:,16:105], axis = 1) #Eingrenzen des Frequenzbereichs


data1 = pd.read_csv(r'D:\WZL_ags\12 Skit AE Pipe\Bohrer\dataDrill_short_beginning_specfeat_B3_7.csv')

data1 = data1.drop(["vbarea", "Werkzeug"], axis=1)

data2 = pd.concat([data2,data1],axis=1)

#%%  ##Auswahl eines Werkzeugs od. Prozessparameter

data2 = data2[data2['Werkzeug'] == 4]

# # data2 = data2[(data2['Schnittgeschwindigkeit [m/min]'] == 400) & (data2['Schnitttiefe ap [mm]'] == 2)]

# # data2 = data2[data2['vbarea'] < 15000]

# # ##Drop bestimmter Werkzeuge

# # indexNames = data2[(data2['Werkzeug'] == 6) | (data2['Werkzeug'] == 6)].index

# # data2.drop(indexNames , inplace=True)


#%% Drop or keep defined features

# data2 = data2[list(data2.filter(regex='(CH|vbarea|Werkzeug)'))]

data2 = data2[data2.columns.drop(list(data2.filter(regex='CSD')))]

data2 = data2[data2.columns.drop(list(data2.filter(regex='CH')))]

data2 = data2[data2.columns.drop(list(data2.filter(regex='Mz')))]

data2 = data2[data2.columns.drop(list(data2.filter(regex='Fz')))]

# data2 = data2[data2.columns.drop(list(data2.filter(regex='_Aufnahme')))]

# data2 = data2[data2.columns.drop(list(data2.filter(regex='_Futter')))]

#%% Choose Specific Features 1

feat =["vbarea", "Werkzeug", "/'Untitled'/'AE_RAW_Futter'median"]
feat1 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Futter'energy"]
feat2 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Futter'mean"]
feat3 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Futter'standarddeviation"]
feat4 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Futter'variance"]
feat5 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Futter'ecD1"]


plt.close("all")


fig = plt.figure(0)

plt.tight_layout()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322, sharex=ax1)
ax3 = fig.add_subplot(323, sharex=ax1)
ax4 = fig.add_subplot(324, sharex=ax1)
ax5 = fig.add_subplot(325, sharex=ax1)
ax6 = fig.add_subplot(326, sharex=ax1)

# plt.subplots_adjust(hspace=.0)


ax1.title.set_text('median')
ax2.title.set_text('energy')
ax3.title.set_text('mean')
ax4.title.set_text('standarddeviation')
ax5.title.set_text('variance')
ax6.title.set_text('ecD1')


ax1.plot(data2[feat[0]],data2[feat[2]],"b+")
ax2.plot(data2[feat1[0]],data2[feat1[2]],"b+")
ax3.plot(data2[feat2[0]],data2[feat2[2]],"b+")
ax4.plot(data2[feat3[0]],data2[feat3[2]],"b+")
ax5.plot(data2[feat4[0]],data2[feat4[2]],"b+")
ax6.plot(data2[feat5[0]],data2[feat5[2]],"b+")


feat =["vbarea", "Werkzeug", "/'Untitled'/'AE_RAW_Aufnahme'median"]
feat1 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Aufnahme'energy"]
feat2 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Aufnahme'mean"]
feat3 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Aufnahme'standarddeviation"]
feat4 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Aufnahme'variance"]
feat5 =["vbarea","Werkzeug", "/'Untitled'/'AE_RAW_Aufnahme'ecD1"]


ax1.plot(data2[feat[0]],data2[feat[2]],"r+")
ax2.plot(data2[feat1[0]],data2[feat1[2]],"r+")
ax3.plot(data2[feat2[0]],data2[feat2[2]],"r+")
ax4.plot(data2[feat3[0]],data2[feat3[2]],"r+")
ax5.plot(data2[feat4[0]],data2[feat4[2]],"r+")
ax6.plot(data2[feat5[0]],data2[feat5[2]],"r+")

data2 =pd.DataFrame(data2[feat]).copy()


#%% Include Wear rate

# data2["delta_vbarea"] = 1

data2.insert(0,"delta_vbarea",0)

for i in range(len(data2)-1):
    data2["delta_vbarea"][i+1+data2.index[0]] = data2["vbarea"][i+1+data2.index[0]]-data2["vbarea"][i+data2.index[0]]
       
# data2.delta_vbarea[data2['Drehweg [m]'] == 150] = 0  #Messintervalle
# data2.delta_vbarea[data2['Drehweg [m]'] == 75] = 0

data2.delta_vbarea[data2['delta_vbarea'] < 0] = 0



#%% Correlation Matrix

# Create correlation matrix
corr_matrix = data2.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper.drop(["vbarea","delta_vbarea"])[column] > 0.95)]
# Drop features
data2 = data2.drop(to_drop, axis =1)


#%%Prepare Data Set

train_set, test_set = train_test_split(data2, test_size=0.2, random_state = 22)

# ##Split Labels
# data_train = train_set.drop(["vbarea","vbmax"], axis=1)
# data_train_labels =train_set["vbarea"].copy()

# data_test = test_set.drop(["vbarea","vbmax"], axis=1)
# data_test_labels = test_set["vbarea"].copy()

##Split Labels DELTA_WEAR
data_train = train_set.drop(["vbarea","delta_vbarea", "Werkzeug"], axis=1)
data_train_labels =train_set["vbarea"].copy()

data_test = test_set.drop(["vbarea","delta_vbarea", "Werkzeug"], axis=1)
data_test_labels = test_set["vbarea"].copy()

# ##Split Labels Extended
# data_train = train_set.drop(["vbarea","vbmax","delta_vbarea","Schnittgeschwindigkeit [m/min]", "Vorschub f [mm]", "Schnitttiefe ap [mm]", "Drehweg [m]", "Werkzeug"], axis=1)
# data_train_labels =train_set["vbarea"].copy()

# data_test = test_set.drop(["vbarea","vbmax","delta_vbarea","Schnittgeschwindigkeit [m/min]", "Vorschub f [mm]", "Schnitttiefe ap [mm]", "Drehweg [m]", "Werkzeug"], axis=1)
# data_test_labels = test_set["vbarea"].copy()


#%% Choose Specific Features 2

# train_set, test_set = train_test_split(data2, test_size=0.1, random_state=42)

# # feat =["/'Messdaten'/'AE Roh'__abs_energy", "/'Messdaten'/'AE Roh'__mean", "/'Messdaten'/'AE Roh'__quantile__q_0.9", "/'Messdaten'/'AE Roh'__absolute_sum_of_changes"]
# # feat =["/'Messdaten'/'AE Roh'energy", "/'Messdaten'/'AE Roh'mean", "/'Messdaten'/'AE Roh'variance"]

# feat =["/'Messdaten'/'AE Roh'energy", "/'Messdaten'/'AE Roh'variance", "/'Messdaten'/'Schnittkraft'mean"]
# feat =["/'Messdaten'/'AE Roh'variance", "/'Messdaten'/'Schnittkraft'mean"]

# data_train =pd.DataFrame(train_set[feat]).copy()
# data_train_labels =pd.DataFrame(train_set["vbarea"].copy())

# data_test =pd.DataFrame(test_set[feat].copy())
# data_test_labels = pd.DataFrame(test_set["vbarea"].copy())

#%% Transformation Pipline

#class Convert Pandas Data Frame to NumpyArray
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
num_attribs = list(data_train)
## Feature Scaling

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)), #use Pandas DataFrames as Input into Pipline
        ('imputer', SimpleImputer(strategy="median")), # median for missin g vlaues
        ])

num_pipeline_tr = Pipeline([
        ('selector', DataFrameSelector(num_attribs)), #use Pandas DataFrames as Input into Pipline
        ('imputer', SimpleImputer(strategy="median")), # median for missin g vlaues
        ('std_scaler', StandardScaler()), # Standardization!!!
        ])

num_pipeline_tr2 = Pipeline([
        ('selector', DataFrameSelector(num_attribs)), #use Pandas DataFrames as Input into Pipline
        ('imputer', SimpleImputer(strategy="median")), # median for missin g vlaues
        ('std_scaler', RobustScaler()), # Standardization!!!
        ])


data_train_tr = num_pipeline_tr.fit_transform(data_train) 
data_test_tr = num_pipeline_tr.transform(data_test)
# data_train_tr = num_pipeline_tr.fit_transform(data_train)    ###Transformierter Trainingsdatensatz


#%% Transformation pipline with top feature selector
#OPTIONAL!!!!

# def indices_of_top_k(arr, k):
#     return np.sort(np.argpartition(np.array(arr), -k)[-k:])

# class TopFeatureSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, feature_importances, k):
#         self.feature_importances = feature_importances
#         self.k = k
#     def fit(self, X, y=None):
#         self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
#         return self
#     def transform(self, X):
#         return X[:, self.feature_indices_]


# k = 5
# preparation_and_feature_selection_pipeline = Pipeline([
#     ('preparation', num_pipeline),
#     ('feature_selection', TopFeatureSelector(feature_importances, k))
# ])

# data_train_tr_k = preparation_and_feature_selection_pipeline.fit_transform(data)

#%% PCA
num_attribs = list(data_train)

pca = PCA(n_components = 10)
X2D = pca.fit_transform(data_train)
print(pca.explained_variance_ratio_)

data_train_tr = X2D

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X2D[:,0],X2D[:,1],X2D[:,2])

 
#%% kPCA

num_attribs = list(data_train)

rbf_pca = KernelPCA(n_components = 10, kernel="linear", gamma=0.04)
X_reduced = rbf_pca.fit_transform(data_train)

data_train_tr = X_reduced
 

#%% Simple Pearson Korrealtionsmatrix
def histogram_intersection(a, b):

    v = np.minimum(a, b).sum().round(decimals=1)

    return v

res2 = data2.corr(method='pearson')

#%% Linear Regression 
lin_reg = LinearRegression()
lin_reg.fit(data_train_tr, data_train_labels)

# some_data = data.iloc[:10]
# some_labels = data_train_labels.iloc[:10]
# some_data_tr = num_pipeline_tr.fit_transform(some_data)    

# print("Predictions:", lin_reg.predict(some_data_tr))
# print("Labels:", list(some_labels))

data_predictions = lin_reg.predict(data_train_tr)
lin_mse = mean_squared_error(data_train_labels, data_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

#%% Decision Tree Regressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_train_tr, data_train_labels)

data_predictions = tree_reg.predict(data_train_tr)
tree_mse = mean_squared_error(data_train_labels, data_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


# # Visualizing Decision Tree Model
# #Funktioniert nicht


# num_attribs = list(data_train)
# dot_data = tree.export_graphviz(tree_reg,out_file=None,feature_names=num_attribs,class_names=data_train_labels)
# graph = pydotplus.graph_from_dot_data(dot_data)
# # Show graph
# Image(graph.create_png())


#%% RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(data_train_tr, data_train_labels)

data_predictions = forest_reg.predict(data_train_tr)
forest_mse = mean_squared_error(data_train_labels, data_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

print(forest_reg.score(data_train_tr, data_train_labels))


feature_importances0 = forest_reg.feature_importances_
feature_importances0

attributes0 = num_attribs
print ("----------------FEATURE IMPORTANCES-------------------------------")
print(sorted(zip(feature_importances0, attributes0), reverse=True)[0:10])


#%% Support Vector Maschine Regressor

svm_reg = SVR()
svm_reg.fit(data_train_tr, data_train_labels)

data_predictions = svm_reg.predict(data_train_tr)
svm_mse = mean_squared_error(data_train_labels, data_predictions)
svm_rmse = np.sqrt(svm_mse)
print(svm_rmse)

#%% AdaBoost Regressor

ada_clf = AdaBoostRegressor(
DecisionTreeRegressor(max_depth=2), n_estimators=200, learning_rate=0.5)
ada_clf.fit(data_train_tr, data_train_labels)

data_predictions = ada_clf.predict(data_train_tr)
ada_mse = mean_squared_error(data_train_labels, data_predictions)
ada_rmse = np.sqrt(ada_mse)
print(ada_rmse)

#%% XGBoost Regressor

dtrain = xgb.DMatrix(data_train_tr, label=data_train_labels)

dtest = xgb.DMatrix(data_test_tr, label=data_test_labels)

param = {'max_depth': 2, 'eta': 1,}
param['nthread'] = 4
param['eval_metric'] = 'auc'

param['eval_metric'] = ['auc', 'ams@0']

# alternatively:
# plst = param.items()
# plst += [('eval_metric', 'ams@0')]

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 10
bst = xgb.train(param, dtrain, num_round)

ypred = bst.predict(dtest)

# xgb.plot_importance(bst)
# # xgb.plot_tree(bst, num_trees=2)
# # xgb.to_graphviz(bst, num_trees=2)

data_predictions = bst.predict(dtrain)
xgb_mse = mean_squared_error(data_train_labels, data_predictions)
xgb_rmse = np.sqrt(xgb_mse)
print(xgb_rmse)

#%% Cross Validation
# Cross Validation of DataSet for Regression

warnings.filterwarnings("ignore")


k = 10

sc = "neg_mean_squared_error"
# sc = "r2"   ##Funktioniert nicht

def calc_scores(scores):
    if sc == "neg_mean_squared_error":
        lin_rmse_scores = np.sqrt(-scores)
    elif sc == "r2":
        lin_rmse_scores = scores
    return lin_rmse_scores

scores = cross_val_score(lin_reg, data_train_tr, data_train_labels,
                         scoring= sc, cv=k)
lin_rmse_scores = calc_scores(scores)

def display_scores(scores,Model):
    print(Model)
    print("Reg_Scores:", scores)
    print("MEAN:", scores.mean())
    print("Standard deviation:", scores.std())
print ("---------------------CROSS VALIDATION-------------------------------")
display_scores(lin_rmse_scores,"---------------------Linear Regression---------------------")



# Cross Validation of DataSet for Decission Tree
scores = cross_val_score(tree_reg, data_train_tr, data_train_labels,
                         scoring=sc, cv=k)
tree_rmse_scores = np.sqrt(-scores)

display_scores(tree_rmse_scores,"---------------------Decission Tree---------------------")



# Cross Validation RandomForestRegressor
scores = cross_val_score(forest_reg, data_train_tr, data_train_labels,
                         scoring=sc, cv=k)

forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores,"---------------------Random Forest Regressor---------------------")



# Cross Validation SVR
scores = cross_val_score(svm_reg, data_train_tr, data_train_labels,
                         scoring=sc, cv=k)
svm_rmse_scores = np.sqrt(-scores)

display_scores(svm_rmse_scores, "---------------------Support Vector Regression---------------------")



# Cross Validation AdaBoost 

scores = cross_val_score(ada_clf, data_train_tr, data_train_labels,
                         scoring=sc, cv=k)
ada_rmse_scores = np.sqrt(-scores)

display_scores(ada_rmse_scores, "---------------------AdaBoost Regression---------------------")


# Cross Validation XGBoost  ## FUNKTIONIERT NICHT

print("---------------------XGBoost Regression---------------------")


res = xgb.cv(param, dtrain, num_boost_round=10, nfold=k,
             metrics={'error'}, seed=0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                        xgb.callback.early_stop(3)])

print(res)
print('running cross validation, with preprocessing function')


#%% Rekursive Feature elimination

attributes = num_attribs

# esti = lin_reg  #  choose estimator
# esti = forest_reg  #  choose estimator
esti = tree_reg

rfecv = RFECV(estimator=esti, step=1, scoring="neg_mean_squared_error")
rfecv.fit(data_train_tr, data_train_labels)
data_train_tr_red = rfecv.transform(data_train_tr)
rfecv.n_features_

features = pd.concat([pd.DataFrame(attributes),pd.DataFrame(rfecv.support_)], axis =1)
features.columns = ['feature','bool']

features = features[features['bool'] == True]  #relevnate Features extrahieren

print ("----------------REMAINING FEATURES-------------------------------")
print (features)

#%% GridSearch for Random Forest Regressor

param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
               {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [1,2, 3, 4]},
               ]
forest_reg2 = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg2, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(data_train_tr, data_train_labels)
grid_search.best_params_

cvres = grid_search.cv_results_
print ("----------------GRID SEARCH RESULTS-------------------------------")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

attributes = num_attribs


best_forest = grid_search.best_estimator_


print ("----------------FEATURE IMPORTANCES-------------------------------")
print(sorted(zip(feature_importances, attributes), reverse=True))


#%%Feature Reduction for most important features for Random Forest

k =10

# model = best_forest
model = forest_reg


# Create object that selects features with importance greater
# than or equal to a threshold The threshold value to use for feature selection. 
#Features whose importance is greater or equal are kept while the others are discarded.
selector = SelectFromModel(model, threshold=0.4)
# Feature new feature matrix using selector
features_important = selector.fit_transform(data_train_tr, data_train_labels)
# Train random forest using most important featres
red_forest = forest_reg.fit(features_important, data_train_labels)

data_predictions = red_forest.predict(features_important)
forest_mse = mean_squared_error(data_train_labels, data_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

print ("----------------CROSS VALIDATION-------------------------------")
# Cross Validation RandomForestRegressor
def display_scores(scores,Model):
    print(Model)
    print("Reg_Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
sc = "neg_mean_squared_error"
scores = cross_val_score(red_forest, features_important, data_train_labels,
                         scoring=sc, cv=k)

forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores,"RandomForestRegressor")


#%% GridSearch for SVR

param_grid2 = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
svm_reg2 = SVR() 
grid_search2 = GridSearchCV(svm_reg2, param_grid2, cv=5,
scoring='neg_mean_squared_error')
grid_search2.fit(data_train_tr, data_train_labels)
grid_search2.best_params_

cvres = grid_search2.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


best_svm = grid_search2.best_estimator_

negative_mse = grid_search2.best_score_
rmse = np.sqrt(-negative_mse)
rmse

#%% GRIdsearch Linear Regression kPca

clf = Pipeline([
("kpca", KernelPCA(n_components=5)),
("lin_reg", LinearRegression())
])
param_grid = [{
"kpca__gamma": np.linspace(0.03, 0.05, 10),
"kpca__kernel": ["rbf", "sigmoid"]
}]
grid_search3 = GridSearchCV(clf, param_grid, cv=5)
grid_search3.fit(data_train, data_train_labels)

cvres = grid_search3.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


best_kPCA = grid_search3.best_estimator_

negative_mse = grid_search3.best_score_
rmse = np.sqrt(-negative_mse)
rmse

#%% Evaluate Test DataSet

final_predictions = lin_reg.predict(data_test_tr) # Linear Regression
final_mse = mean_squared_error(data_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
data_test_labels_mod = data_test_labels.reset_index(drop = True)
result2 = pd.concat([pd.DataFrame(final_predictions), pd.DataFrame(data_test_labels_mod)],axis=1)
print("Linear Regression RMSE "+str(final_rmse))
print(result2)

final_predictions = svm_reg.predict(data_test_tr) # Support Vector Regression
final_mse = mean_squared_error(data_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
data_test_labels_mod = data_test_labels.reset_index(drop = True)
result2 = pd.concat([pd.DataFrame(final_predictions), pd.DataFrame(data_test_labels_mod)],axis=1)
print("Support Vector Regression RMSE "+str(final_rmse))
print(result2)

# final_predictions = best_svm.predict(data_test_tr) # Best Support Vector Regression
# final_mse = mean_squared_error(data_test_labels, final_predictions)
# final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
# data_test_labels_mod = data_test_labels.reset_index(drop = True)
# result2 = pd.concat([pd.DataFrame(final_predictions), pd.DataFrame(data_test_labels_mod)],axis=1)
# print("Best Support Vector Regression RMSE "+str(final_rmse))
# print(result2)

final_predictions = best_forest.predict(data_test_tr) # Best RandomForest Regression
final_mse = mean_squared_error(data_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
data_test_labels_mod = data_test_labels.reset_index(drop = True)
result2 = pd.concat([pd.DataFrame(final_predictions), pd.DataFrame(data_test_labels_mod)],axis=1)
print("Random Forest RMSE "+str(final_rmse))
print(result2)


final_predictions = forest_reg.predict(data_test_tr) # RadmonForest Regression
final_mse = mean_squared_error(data_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
data_test_labels_mod = data_test_labels.reset_index(drop = True)
result2 = pd.concat([pd.DataFrame(final_predictions), pd.DataFrame(data_test_labels_mod)],axis=1)
print("Best Random Forest RMSE "+str(final_rmse))
print(result2)

final_predictions = ada_clf.predict(data_test_tr) # AdaBoost Regression 
final_mse = mean_squared_error(data_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
data_test_labels_mod = data_test_labels.reset_index(drop = True)
result2 = pd.concat([pd.DataFrame(final_predictions), pd.DataFrame(data_test_labels_mod)],axis=1)
print("AdaBoost RMSE "+str(final_rmse))
print(result2)




##ÜBERARBEITEN
# final_predictions = bst.predict(data_test_tr) # XGBoost Regression
# final_mse = mean_squared_error(data_test_labels, final_predictions)
# final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0
# print("XGBoost RMSE"+str(final_rmse))




