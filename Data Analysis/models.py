#!/usr/bin/env python
# coding: utf-8

# models scripts # 30.05.2020

# In[58]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score
from sklearn.model_selection import train_test_split,cross_validate,ShuffleSplit
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,RandomForestRegressor
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC

#%% simplifying


filenames_csv = [
        'ansys_1_vollbohrung_index_label_fft_features.csv',
        'ansys_2_vollbohrung_fz_label_fft_features.csv',
        'ansys_3_vollbohrung_index_label_only_fft_features.csv',
        'ansys_4_vollbohrung_fz_label_only_fft_features.csv',
        'ansys_5_vollbohrung_index_label_fft_features.csv',
        'ansys_6_vollbohrung_fz_label_fft_features.csv',
        'ansys_7_vollbohrung_index_label_only_fft_features.csv',
        'ansys_8_vollbohrung_fz_label_only_fft_features.csv',
        'ansys_9_vollbohrung_index_label_fft_features.csv',
        'ansys_10_vollbohrung_fz_label_fft_features.csv',
        'ansys_11_vollbohrung_index_label_only_fft_features.csv',
        'ansys_12_vollbohrung_fz_label_only_fft_features.csv',
        'ansys_13_vollbohrung_mz_label_fft_features.csv',
        'ansys_14_vollbohrung_mz_label_fft_features.csv',
        'ansys_15_vollbohrung_mz_label_only_fft_features.csv',
        'ansys_16_vollbohrung_mz_label_only_fft_features.csv'
        
        ]

#filenames_csv = ['ansys_2_vollbohrung_fz_label_fft_features.csv','ansys_6_vollbohrung_fz_label_fft_features.csv']

ansys_cont=1

ansys_number_list=[]
least_error_model_name_list=[]
least_error_model_mean_list=[]
least_error_model_std_list=[]
min_index_list=[]
least_error_model_test_data_list=[]
least_error_model_perc_list=[]
least_error_model_perc_test_data_list=[]
training_size_list=[]


for ansys_name in filenames_csv:
    
    # Importing and preparing Dataset
    dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_'+ansys_name,index_col=0)
    
    # Defining Label:
    X=dataset
    X = X.iloc[:, :-1] # Deleting the last column
    y=dataset.iloc[:,-1]    

    # Different sizing of the training Data [0.5, 0.6, 0.7, 0.8, 0.9]
    train_size_list=[0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    for train_size in train_size_list:
    
        # Splitting DATA
        #X_train_o, X_test_o, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_o, X_test_o, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
    
        X_train = X_train_o
        X_test = X_test_o
        scaler = StandardScaler()  # doctest: +SKIP
        scaler.fit(X_train)  # doctest: +SKIP
        X_train = scaler.transform(X_train)  # doctest: +SKIP
        # apply same transformation to test data
        X_test = scaler.transform(X_test)
        
        # Training Models
        # Linear Regression
        from sklearn.linear_model import LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        from sklearn.metrics import mean_squared_error
        lin_label_predictions = lin_reg.predict(X_train)
        lin_mse = mean_squared_error(y_train, lin_label_predictions)
        lin_rmse = np.sqrt(lin_mse)
        #print("Linear RMSE: ",lin_rmse)
        # Decision Tree Regressor
        from sklearn.tree import DecisionTreeRegressor
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(X_train, y_train)
        tree_label_predictions = tree_reg.predict(X_train)
        tree_mse = mean_squared_error(y_train, tree_label_predictions)
        tree_rmse = np.sqrt(tree_mse)
        #print("Decision Tree RMSE: ",tree_rmse)
        # Random Forests Regressor
        from sklearn.ensemble import RandomForestRegressor
        forest_reg = RandomForestRegressor()
        forest_reg.fit(X_train, y_train)
        forest_label_predictions = forest_reg.predict(X_train)
        forest_mse = mean_squared_error(y_train, forest_label_predictions)
        forest_rmse = np.sqrt(forest_mse)
        #print("Random Forest RMSE: ",forest_rmse)
        # Multi-layer Perceptron Regressor
        from sklearn.neural_network import MLPRegressor
        ##nn_reg = MLPRegressor(max_iter=3200)
        nn_reg = MLPRegressor(max_iter=3000)
        nn_reg.fit(X_train, y_train)
        nn_label_predictions = nn_reg.predict(X_train)
        nn_mse = mean_squared_error(y_train, nn_label_predictions)
        nn_rmse = np.sqrt(nn_mse)
        #print("Multi-layer Regressor RMSE: ",nn_rmse)
        # Support Vector Machines (SVMs)
        from sklearn.svm import LinearSVR
        svm_reg = LinearSVR(epsilon=1.5)
        svm_reg.fit(X_train, y_train)
        svm_label_predictions = svm_reg.predict(X_train)
        svm_mse = mean_squared_error(y_train, svm_label_predictions)
        svm_rmse = np.sqrt(svm_mse)
        #print("SVM RMSE: ",svm_rmse)
        
        # Cross-Validation - [Train Data]
        cont=0
        #for cont in range(0,3): # 3 times - repetition
            
            
        from sklearn.model_selection import cross_val_score
        models_name=[]
        models_mean=[]
        models_std=[]
        ansys_list=[]
        #
        lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        models_mean.append(lin_rmse_scores.mean())
        models_std.append(lin_rmse_scores.std())
        models_name.append("Linear Regressor")
        #
        tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        tree_rmse_scores = np.sqrt(-tree_scores)
        models_mean.append(tree_rmse_scores.mean())
        models_std.append(tree_rmse_scores.std())
        models_name.append("Decision Tree Regressor")
        #
        forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        forest_rmse_scores = np.sqrt(-forest_scores)
        models_mean.append(forest_rmse_scores.mean())
        models_std.append(forest_rmse_scores.std())
        models_name.append("Random Forest Regressor")
        #
        nn_scores = cross_val_score(nn_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        nn_rmse_scores = np.sqrt(-nn_scores)
        models_mean.append(nn_rmse_scores.mean())
        models_std.append(nn_rmse_scores.std())
        models_name.append("Multi-layer Regressor")
        #
        svm_scores = cross_val_score(svm_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        svm_rmse_scores = np.sqrt(-svm_scores)
        models_mean.append(svm_rmse_scores.mean())
        models_std.append(svm_rmse_scores.std())
        models_name.append("SVM Regressor")
        #
    
        # the least error model
        least_error_model_mean      = min(models_mean)
        min_index                   = models_mean.index(least_error_model_mean) # 0 1 2 3 4 
        least_error_model_name      = models_name[min_index]
        least_error_model_std       = models_std[min_index]
        y_max=max(y)
        least_error_model_perc      = (least_error_model_mean/ y_max)*100
        
        ansys_number_list.append(ansys_cont)
        
        training_size_list.append(train_size)
        
        least_error_model_mean_list.append(least_error_model_mean)
        min_index_list.append(min_index) # 0 1 2 3 4 
        least_error_model_name_list.append(least_error_model_name)
        least_error_model_std_list.append(least_error_model_std)
        least_error_model_perc_list.append(least_error_model_perc)
    
        # [Test Data]
        
        if min_index==0:
            # Linear Regression
            #from sklearn.linear_model import LinearRegression
            #lin_reg = LinearRegression()
            #lin_reg.fit(X_train, y_train)
            from sklearn.metrics import mean_squared_error
            # Test Set Data here
            lin_label_predictions = lin_reg.predict(X_test)
            lin_mse = mean_squared_error(y_test, lin_label_predictions)
            lin_rmse = np.sqrt(lin_mse)
            #
            least_error_model=lin_rmse
        if min_index==1:
            # Decision Tree Regressor
            #from sklearn.tree import DecisionTreeRegressor
            #tree_reg = DecisionTreeRegressor()
            #tree_reg.fit(X_train, y_train)
            # Test Set Data here
            tree_label_predictions = tree_reg.predict(X_test)
            tree_mse = mean_squared_error(y_test, tree_label_predictions)
            tree_rmse = np.sqrt(tree_mse)
            #
            least_error_model=tree_rmse
        if min_index==2:
            # Random Forests Regressor
            #from sklearn.ensemble import RandomForestRegressor
            #forest_reg = RandomForestRegressor()
            #forest_reg.fit(X_train, y_train)
            # Test Set Data here
            forest_label_predictions = forest_reg.predict(X_test)
            forest_mse = mean_squared_error(y_test, forest_label_predictions)
            forest_rmse = np.sqrt(forest_mse)
            #
            least_error_model=forest_rmse        
        if min_index==3:
            # Multi-layer Perceptron Regressor
            from sklearn.neural_network import MLPRegressor
            ##nn_reg = MLPRegressor(max_iter=3200)
            #nn_reg = MLPRegressor(max_iter=3000)
            #nn_reg.fit(X_train, y_train)
            # Test Set Data here
            nn_label_predictions = nn_reg.predict(X_test)
            nn_mse = mean_squared_error(y_test, nn_label_predictions)
            nn_rmse = np.sqrt(nn_mse)
            #
            least_error_model=nn_rmse        
        if min_index==4:
            # Support Vector Machines (SVMs)
            #from sklearn.svm import LinearSVR
            #svm_reg = LinearSVR(epsilon=1.5)
            #svm_reg.fit(X_train, y_train)
            # Test Set Data here
            svm_label_predictions = svm_reg.predict(X_test)
            svm_mse = mean_squared_error(y_test, svm_label_predictions)
            svm_rmse = np.sqrt(svm_mse)
            #
            least_error_model=svm_rmse
            
            
        least_error_model_test_data_list.append(least_error_model)
        least_error_model_perc_test_data_list.append((least_error_model/ y_max)*100)
            # end of for
                
            
    
            
            #ansys_list.append(ansys_df)
        
            #result = pd.concat(result,ansys_df)
            #result["Filename"] = filenames[i]
           # resultlist.append(result)  
          
        # end of the traing size for
    ##############
    ansys_cont=ansys_cont+1


# Here: create the Dataframe
ansys_df = pd.DataFrame({
                'Analysis'                               : ansys_number_list,
                'Training Size'                          : training_size_list,
                '[Train Data] - The least error ML model': least_error_model_name_list,
                '[Train Data] - RMSE (mean of cv=10)'    : least_error_model_mean_list,
                '[Train Data] - STD (mean of cv=10)'     : least_error_model_std_list,
                '[Train Data] - RMSE(%) (mean of cv=10)' : least_error_model_perc_list,
                '[Test Data] - The least error ML model' : least_error_model_name_list,
                '[Test Data] - RMSE (cv=1)'              : least_error_model_test_data_list,
                '[Test Data] - RMSE(%) (cv=1)'           : least_error_model_perc_test_data_list
                
                })
        
#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_results_vollbohrung.csv')
#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\Ansys_2_&_6_results_vollbohrung_with_diff_train_sizes.csv')
#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_results_vollbohrung_with_diff_train_sizes.csv')
#

#%% for plotting all models together - test data

# 16.09.2020 changes + k-neighbours model
    
filenames_csv = [
        'ansys_1_vollbohrung_index_label_fft_features.csv',
        'ansys_2_vollbohrung_fz_label_fft_features.csv',
        'ansys_3_vollbohrung_index_label_only_fft_features.csv',
        'ansys_4_vollbohrung_fz_label_only_fft_features.csv',
        'ansys_5_vollbohrung_index_label_fft_features.csv',
        'ansys_6_vollbohrung_fz_label_fft_features.csv',
        'ansys_7_vollbohrung_index_label_only_fft_features.csv',
        'ansys_8_vollbohrung_fz_label_only_fft_features.csv',
        'ansys_9_vollbohrung_index_label_fft_features.csv',
        'ansys_10_vollbohrung_fz_label_fft_features.csv',
        'ansys_11_vollbohrung_index_label_only_fft_features.csv',
        'ansys_12_vollbohrung_fz_label_only_fft_features.csv',
        'ansys_13_vollbohrung_mz_label_fft_features.csv',
        'ansys_14_vollbohrung_mz_label_fft_features.csv',
        'ansys_15_vollbohrung_mz_label_only_fft_features.csv',
        'ansys_16_vollbohrung_mz_label_only_fft_features.csv'
        
        ]

#filenames_csv = ['ansys_2_vollbohrung_fz_label_fft_features.csv']

ansys_cont=1

ansys_number_list=[]
#least_error_model_name_list=[]
#least_error_model_mean_list=[]
#least_error_model_std_list=[]
#min_index_list=[]
#least_error_model_test_data_list=[]
#least_error_model_perc_list=[]
#least_error_model_perc_test_data_list=[]
training_size_list=[]
test_lin_rmse_list=[]
test_lin_rmse_perc_list=[]
test_KN_rmse_list=[]
test_KN_rmse_perc_list=[]
test_tree_rmse_list=[]
test_tree_rmse_perc_list=[]
test_forest_rmse_list=[]
test_forest_rmse_perc_list=[]
test_nn_rmse_list=[]
test_nn_rmse_perc_list=[]
test_svm_rmse_list=[]
test_svm_rmse_perc_list=[]

for ansys_name in filenames_csv:
    
    # Importing and preparing Dataset
    dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_'+ansys_name,index_col=0)
    
    # Defining Label:
    X=dataset
    X = X.iloc[:, :-1] # Deleting the last column
    y=dataset.iloc[:,-1]    

    # Different sizing of the training Data [0.5, 0.6, 0.7, 0.8, 0.9]
    train_size_list=[0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    for train_size in train_size_list:
    
        # Splitting DATA
        #X_train_o, X_test_o, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_o, X_test_o, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
    
        X_train = X_train_o
        X_test = X_test_o
        scaler = StandardScaler()  # doctest: +SKIP
        scaler.fit(X_train)  # doctest: +SKIP
        X_train = scaler.transform(X_train)  # doctest: +SKIP
        # apply same transformation to test data
        X_test = scaler.transform(X_test)
        
        # Training Models
        # Linear Regression
        from sklearn.linear_model import LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        from sklearn.metrics import mean_squared_error
        lin_label_predictions = lin_reg.predict(X_train)
        lin_mse = mean_squared_error(y_train, lin_label_predictions)
        lin_rmse = np.sqrt(lin_mse)
        #print("Linear RMSE: ",lin_rmse)
        
        # KNeighborRegressor - 23.09.2020
        from sklearn.neighbors import KNeighborsRegressor
        KN_reg = KNeighborsRegressor()
        KN_reg.fit(X_train, y_train)
        from sklearn.metrics import mean_squared_error
        KN_label_predictions = KN_reg.predict(X_train)
        KN_mse = mean_squared_error(y_train, KN_label_predictions)
        KN_rmse = np.sqrt(KN_mse)
        #print("Linear RMSE: ",lin_rmse)
        
        # Decision Tree Regressor
        from sklearn.tree import DecisionTreeRegressor
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(X_train, y_train)
        tree_label_predictions = tree_reg.predict(X_train)
        tree_mse = mean_squared_error(y_train, tree_label_predictions)
        tree_rmse = np.sqrt(tree_mse)
        #print("Decision Tree RMSE: ",tree_rmse)
        # Random Forests Regressor
        from sklearn.ensemble import RandomForestRegressor
        forest_reg = RandomForestRegressor()
        forest_reg.fit(X_train, y_train)
        forest_label_predictions = forest_reg.predict(X_train)
        forest_mse = mean_squared_error(y_train, forest_label_predictions)
        forest_rmse = np.sqrt(forest_mse)
        #print("Random Forest RMSE: ",forest_rmse)
        # Multi-layer Perceptron Regressor
        from sklearn.neural_network import MLPRegressor
        ##nn_reg = MLPRegressor(max_iter=3200)
        nn_reg = MLPRegressor(max_iter=3000)
        nn_reg.fit(X_train, y_train)
        nn_label_predictions = nn_reg.predict(X_train)
        nn_mse = mean_squared_error(y_train, nn_label_predictions)
        nn_rmse = np.sqrt(nn_mse)
        #print("Multi-layer Regressor RMSE: ",nn_rmse)
        # Support Vector Machines (SVMs)
        from sklearn.svm import LinearSVR
        #svm_reg = LinearSVR(epsilon=1.5)
        svm_reg = SVR(kernel='rbf',C=100,gamma=0.1,epsilon=.5)
        svm_reg.fit(X_train, y_train)
        svm_label_predictions = svm_reg.predict(X_train)
        svm_mse = mean_squared_error(y_train, svm_label_predictions)
        svm_rmse = np.sqrt(svm_mse)
        #print("SVM RMSE: ",svm_rmse)
        
        # Cross-Validation - [Train Data]
        #cont=0
        #for cont in range(0,3): # 3 times - repetition
            
            
        #from sklearn.model_selection import cross_val_score
        models_name=[]
        #models_mean=[]
        #models_std=[]
        ansys_list=[]
        #
        #lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        #lin_rmse_scores = np.sqrt(-lin_scores)
        #models_mean.append(lin_rmse_scores.mean())
        #models_std.append(lin_rmse_scores.std())
        models_name.append("Linear Regressor")
        #
        models_name.append("KNeighbor Regressor")
        #
        #tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        #tree_rmse_scores = np.sqrt(-tree_scores)
        #models_mean.append(tree_rmse_scores.mean())
        #models_std.append(tree_rmse_scores.std())
        models_name.append("Decision Tree Regressor")
        #
        #forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        #forest_rmse_scores = np.sqrt(-forest_scores)
        #models_mean.append(forest_rmse_scores.mean())
        #models_std.append(forest_rmse_scores.std())
        models_name.append("Random Forest Regressor")
        #
        #nn_scores = cross_val_score(nn_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        #nn_rmse_scores = np.sqrt(-nn_scores)
        #models_mean.append(nn_rmse_scores.mean())
        #models_std.append(nn_rmse_scores.std())
        models_name.append("Multi-layer Regressor")
        #
        #svm_scores = cross_val_score(svm_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
        #svm_rmse_scores = np.sqrt(-svm_scores)
        #models_mean.append(svm_rmse_scores.mean())
        #models_std.append(svm_rmse_scores.std())
        models_name.append("SVM Regressor")
        #
        

        # the least error model
        #least_error_model_mean      = min(models_mean)
        #min_index                   = models_mean.index(least_error_model_mean) # 0 1 2 3 4 
        #least_error_model_name      = models_name[min_index]
        #least_error_model_std       = models_std[min_index]
        y_max=max(y)
        #least_error_model_perc      = (least_error_model_mean/ y_max)*100
        
        ansys_number_list.append(ansys_cont)
        
        training_size_list.append(train_size)
        
        #least_error_model_mean_list.append(least_error_model_mean)
        #min_index_list.append(min_index) # 0 1 2 3 4 
        #least_error_model_name_list.append(least_error_model_name)
        #least_error_model_std_list.append(least_error_model_std)
        #least_error_model_perc_list.append(least_error_model_perc)
    
        # [Test Data]
        
        #if min_index==0:
        # Linear Regression
        #from sklearn.linear_model import LinearRegression
        #lin_reg = LinearRegression()
        #lin_reg.fit(X_train, y_train)
        from sklearn.metrics import mean_squared_error
        # Test Set Data here
        test_lin_label_predictions = lin_reg.predict(X_test)
        test_lin_mse = mean_squared_error(y_test, test_lin_label_predictions)
        test_lin_rmse = np.sqrt(test_lin_mse)
        #
        #least_error_model=lin_rmse
        #if min_index==1:
        
        # KNeighbor Regressor - 23.09.2020
        
        # Test Set Data here
        test_KN_label_predictions = KN_reg.predict(X_test)
        test_KN_mse = mean_squared_error(y_test, test_KN_label_predictions)
        test_KN_rmse = np.sqrt(test_KN_mse)
        
        # Decision Tree Regressor
        #from sklearn.tree import DecisionTreeRegressor
        #tree_reg = DecisionTreeRegressor()
        #tree_reg.fit(X_train, y_train)
        # Test Set Data here
        test_tree_label_predictions = tree_reg.predict(X_test)
        test_tree_mse = mean_squared_error(y_test, test_tree_label_predictions)
        test_tree_rmse = np.sqrt(test_tree_mse)
        #
        #least_error_model=tree_rmse
        #if min_index==2:
        
        # Random Forests Regressor
        #from sklearn.ensemble import RandomForestRegressor
        #forest_reg = RandomForestRegressor()
        #forest_reg.fit(X_train, y_train)
        # Test Set Data here
        test_forest_label_predictions = forest_reg.predict(X_test)
        test_forest_mse = mean_squared_error(y_test, test_forest_label_predictions)
        test_forest_rmse = np.sqrt(test_forest_mse)
        #
        #least_error_model=forest_rmse        
        #if min_index==3:
        
        # Multi-layer Perceptron Regressor
        from sklearn.neural_network import MLPRegressor
        ##nn_reg = MLPRegressor(max_iter=3200)
        #nn_reg = MLPRegressor(max_iter=3000)
        #nn_reg.fit(X_train, y_train)
        # Test Set Data here
        test_nn_label_predictions = nn_reg.predict(X_test)
        test_nn_mse = mean_squared_error(y_test, test_nn_label_predictions)
        test_nn_rmse = np.sqrt(test_nn_mse)
        #
        #least_error_model=nn_rmse        
        #if min_index==4:
        
        # Support Vector Machines (SVMs)
        #from sklearn.svm import LinearSVR
        #svm_reg = LinearSVR(epsilon=1.5)
        #svm_reg.fit(X_train, y_train)
        # Test Set Data here
        test_svm_label_predictions = svm_reg.predict(X_test)
        test_svm_mse = mean_squared_error(y_test, test_svm_label_predictions)
        test_svm_rmse = np.sqrt(test_svm_mse)
        #
        #least_error_model=svm_rmse
        
        test_lin_rmse_list.append(test_lin_rmse)
        test_lin_rmse_perc_list.append((test_lin_rmse/ y_max)*100)

        test_KN_rmse_list.append(test_KN_rmse)
        test_KN_rmse_perc_list.append((test_KN_rmse/ y_max)*100)

        test_tree_rmse_list.append(test_tree_rmse)
        test_tree_rmse_perc_list.append((test_tree_rmse/ y_max)*100)

        test_forest_rmse_list.append(test_forest_rmse)
        test_forest_rmse_perc_list.append((test_forest_rmse/ y_max)*100)

        test_nn_rmse_list.append(test_nn_rmse)
        test_nn_rmse_perc_list.append((test_nn_rmse/ y_max)*100)

        test_svm_rmse_list.append(test_svm_rmse)
        test_svm_rmse_perc_list.append((test_svm_rmse/ y_max)*100)

        #least_error_model_test_data_list.append(least_error_model)
        #least_error_model_perc_test_data_list.append((least_error_model/ y_max)*100)
            # end of for
            
            #ansys_list.append(ansys_df)
        
            #result = pd.concat(result,ansys_df)
            #result["Filename"] = filenames[i]
           # resultlist.append(result)  
          
        # end of the traing size for
    ##############
    ansys_cont=ansys_cont+1


# Here: create the Dataframe
ansys_df = pd.DataFrame({
                'Analysis'                                 : ansys_number_list,
                'Training Size'                            : training_size_list,
                'Linear Regressor - RMSE (cv=1)'           : test_lin_rmse_list,
                'Linear Regressor - RMSE(%) (cv=1)'        : test_lin_rmse_perc_list,
                'K-Neighbor Regressor - RMSE (cv=1)'       : test_KN_rmse_list,
                'K-Neighbor Regressor - RMSE(%) (cv=1)'    : test_KN_rmse_perc_list,
                'Decision Tree Regressor - RMSE (cv=1)'    : test_tree_rmse_list,
                'Decision Tree Regressor - RMSE(%) (cv=1)' : test_tree_rmse_perc_list,
                'Random Forest Regressor - RMSE (cv=1)'    : test_forest_rmse_list,
                'Random Forest Regressor - RMSE(%) (cv=1)' : test_forest_rmse_perc_list,
                'Multi-layer Regressor - RMSE (cv=1)'      : test_nn_rmse_list,
                'Multi-layer Regressor - RMSE(%) (cv=1)'   : test_nn_rmse_perc_list,
                'SVM Regressor -  RMSE (cv=1)'             : test_svm_rmse_list,
                'SVM Regressor - RMSE(%) (cv=1)'           : test_svm_rmse_perc_list
                })
        
#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_results_vollbohrung.csv')
#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\Ansys_2_&_6_results_vollbohrung_with_diff_train_sizes.csv')
#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_results_vollbohrung_with_diff_train_sizes.csv')
#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_all_models_test_data_results_vollbohrung_with_diff_train_sizes.csv')
ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_all_models_test_data_results_vollbohrung_with_diff_train_sizes_23_09_20.csv')

#%% plotting - 16.09.2020 - 23.09.20
#ansys_df_plot = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_all_models_test_data_results_vollbohrung_with_diff_train_sizes.csv',index_col=0)
ansys_df_plot = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_all_models_test_data_results_vollbohrung_with_diff_train_sizes_23_09_20.csv',index_col=0)

#ansys_df.to_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_all_models_test_data_results_vollbohrung_with_diff_train_sizes_23_09_20.csv')

# Filtering the data:
dataset=ansys_df_plot[ansys_df_plot["Training Size"].values==0.80]

# only ansys 1, 2 
#plot_x_labels=['All Tools - Index as label', 'All Tools - $F_{z}$ mean as label']
#dataset=dataset.iloc[0:2]

# ansys 5-8 : tool n° 4
#plot_x_labels=['Index', '$F_{z}$ mean', 'FFT - Index', 'FFT - $F_{z}$ mean']
#title='Tool n°4 - RMSE(%) - Index and Fz mean labels'
#dataset=dataset.iloc[4:8]

# ansys 9-12 : tool n° 7
#plot_x_labels=['Index', '$F_{z}$ mean', 'FFT - Index', 'FFT - $F_{z}$ mean']
#title='Tool n°7 - RMSE(%) - Index and Fz mean labels'
#dataset=dataset.iloc[8:12]

# ansys 2, 6, 10 
plot_x_labels=['All Tools', 'Tool n° 4', 'Tool n° 7']
title='$F_{z}$ mean as label'
dataset=dataset.iloc[[1,5,9]]


fig = plt.figure(figsize=(16/2.54,10/2.54))
plt.rcParams.update({'font.size': 12})

#plot_x_labels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
#plot_x_labels=['1','2','3','4','5','6','7','8','9','10']

x=np.arange(len(plot_x_labels))
width = 0.15  # the width of the bars

y_lin=dataset['Linear Regressor - RMSE (cv=1)']
y_dt=dataset['Decision Tree Regressor - RMSE (cv=1)']
y_rf=dataset['Random Forest Regressor - RMSE (cv=1)']
y_nn=dataset['Multi-layer Regressor - RMSE (cv=1)']
#y_svm=dataset['SVM Regressor - RMSE (cv=1)']
y_lin=dataset['Linear Regressor - RMSE(%) (cv=1)']
y_dt=dataset['Decision Tree Regressor - RMSE(%) (cv=1)']
y_rf=dataset['Random Forest Regressor - RMSE(%) (cv=1)']
y_nn=dataset['Multi-layer Regressor - RMSE(%) (cv=1)']
y_svm=dataset['SVM Regressor - RMSE(%) (cv=1)']

   
color1 = color=(0/256,84/256,159/256)
color2 = color=(156/256,158/256,159/256)  
color3 = color=(64/256,127/256,183/256)  
color4 = color=(199/256,221/256,242/256)  
color5 = color=(142/256,186/256,229/256)  

   
plt.bar(x - (5*width)/2, y_lin, width, label='Linear Regressor', color=color1)
plt.bar(x - (3*width)/2, y_dt, width, label='Decision Tree Regressor', color =color2)
plt.bar(x - width/2, y_rf, width, label='Random Forest Regressor', color=color3)
plt.bar(x + width/2, y_nn, width, label='Multi-layer Regressor', color =color4)
plt.bar(x + (3*width)/2, y_svm, width, label='SVM Regressor', color =color5)


#title = 'RMSE_TRAIN_SIZE_'+ansys_name.split('.')[0]
#plt.plot(ansys_df_plot['Training Size'][tsl_i:tsl+1],ansys_df_plot['[Test Data] - RMSE (cv=1)'][tsl_i:tsl+1],'-o', color=(0/256,84/256,159/256))
#plt.ylabel('Mittlerer quadratischer Fehler E') # DE
#plt.ylabel('Mittlerer quadratischer Fehler E / %') # DE
plt.ylabel('Root Mean Square Error E / %') # en


plt.xticks(x,plot_x_labels)
#plt.xticklabels(plot_x_labels)
plt.legend(loc=2, bbox_to_anchor=(-0.06, -0.181),ncol=2) #DE

#plt.legend(loc='upper left', prop={'size': 12})
#plt.xlabel('Nummer der Analyse n / - ', fontsize=12)
#plt.xlabel('Analysis number / - ', fontsize=12) #en
plt.title(title, fontsize=12)

#title='RMSE(%)_vs_models_vs_analysis'

#major=['major','-','0.5','black']
plt.grid()
plt.tight_layout()
#fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)
#fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_error(%)_ansysy_1_&_2.jpeg",dpi=300)
#fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)

#%% 23.09.20



#%%
ansys_df_plot = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\All_analysis_results_vollbohrung_with_diff_train_sizes.csv',index_col=0)


# Plot errors / Traing size
ansys_cont_plot=0
tsl_i=0


d_cont=len(filenames_csv)

tsl=divmod(len(training_size_list),len(filenames_csv))[0]
tsl=tsl-1


for ansys_name in filenames_csv:
    
    #ansys_cont_plot=ansys_cont_plot+1    
    
    
    fig = plt.figure(figsize=(16/2.54,10/2.54))
    plt.rcParams.update({'font.size': 12})
    title = 'RMSE_TRAIN_SIZE_'+ansys_name.split('.')[0]
    plt.plot(ansys_df_plot['Training Size'][tsl_i:tsl+1],ansys_df_plot['[Test Data] - RMSE (cv=1)'][tsl_i:tsl+1],'-o', color=(0/256,84/256,159/256))
    plt.xlabel('Training Size') #En
    plt.ylabel('[Test Data] - RMSE') #En
    major=['major','-','0.5','black']
    plt.grid(major)
    plt.tight_layout()
    fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)
    
    
    # Percentages
    fig = plt.figure(figsize=(16/2.54,10/2.54))
    plt.rcParams.update({'font.size': 12})
    title = 'RMSE(%)_TRAIN_SIZE_'+ansys_name.split('.')[0]
    plt.plot(ansys_df_plot['Training Size'][tsl_i:tsl+1],ansys_df_plot['[Test Data] - RMSE(%) (cv=1)'][tsl_i:tsl+1],'-o', color=(0/256,84/256,159/256))
    plt.xlabel('Training Size') #En
    plt.ylabel('[Test Data] - RMSE(%)') #En
    major=['major','-','0.5','black']
    plt.grid(major)
    plt.tight_layout()
    fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)
    
    tsl_i=tsl+1
    tsl=tsl+divmod(len(training_size_list),len(filenames_csv))[0]
    #d_cont=d_cont-1


#%% Plot errors / Traing size

fig = plt.figure(figsize=(16/2.54,10/2.54))
plt.rcParams.update({'font.size': 12})
title = 'Error_Test_Data_ansys_2'
tsl=divmod(len(training_size_list),2)[0]
plt.plot(ansys_df['Training Size'][0:tsl],ansys_df['[Test Data] - RMSE (cv=1)'][0:tsl],'-o', color=(0/256,84/256,159/256))
plt.xlabel('Training Size') #En
plt.ylabel('[Test Data] - RMSE') #En
major=['major','-','0.5','black']
plt.grid(major)
plt.tight_layout()
fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)


fig = plt.figure(figsize=(16/2.54,10/2.54))
plt.rcParams.update({'font.size': 12})
title = 'Error_Test_Data_ansys_6'
plt.plot(ansys_df['Training Size'][tsl:2*tsl],ansys_df['[Test Data] - RMSE (cv=1)'][tsl:2*tsl],'-o', color=(0/256,84/256,159/256))
plt.xlabel('Training Size') #En
plt.ylabel('[Test Data] - RMSE') #En
major=['major','-','0.5','black']
plt.grid(major)
plt.tight_layout()
fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)
#######################################
#%%
# Percentages
fig = plt.figure(figsize=(16/2.54,10/2.54))
plt.rcParams.update({'font.size': 12})
title = 'Error(%)_Test_Data_ansys_2'
plt.plot(ansys_df['Training Size'][0:tsl],ansys_df['[Test Data] - RMSE(%) (cv=1)'][0:tsl],'-o', color=(0/256,84/256,159/256))
plt.xlabel('Training Size') #En
plt.ylabel('[Test Data] - RMSE(%)') #En
major=['major','-','0.5','black']
plt.grid(major)
plt.tight_layout()
fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)

fig = plt.figure(figsize=(16/2.54,10/2.54))
plt.rcParams.update({'font.size': 12})
title = 'Error(%)_Test_Data_ansys_6'
plt.plot(ansys_df['Training Size'][tsl:2*tsl],ansys_df['[Test Data] - RMSE(%) (cv=1)'][tsl:2*tsl],'-o', color=(0/256,84/256,159/256))
plt.xlabel('Training Size') #En
plt.ylabel('[Test Data] - RMSE(%)') #En
major=['major','-','0.5','black']
plt.grid(major)
plt.tight_layout()
fig.savefig(r"D:\users\pnh_lm\11052020\plots_for_thesis_ML\plot_"+title+".jpeg",dpi=300)
#######################################
    
# In[86]:

# importing and preparing Dataset


#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_1_index_label_fft_features.csv',index_col=0) 
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_2_fz_label_fft_features.csv',index_col=0) 
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_3_index_label_only_fft_features.csv',index_col=0) 
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_4_fz_label_only_fft_features.csv',index_col=0) 
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\02 Data Analysis ags\dataset_ansys_1-HQN_index_label_fft_features.csv',index_col=0)

#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_1_vollbohrung_index_label_fft_features.csv',index_col=0)
dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_2_vollbohrung_fz_label_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_3_vollbohrung_index_label_only_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_4_vollbohrung_fz_label_only_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_5_vollbohrung_index_label_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_6_vollbohrung_fz_label_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_7_vollbohrung_index_label_only_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_8_vollbohrung_fz_label_only_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_9_vollbohrung_index_label_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_10_vollbohrung_fz_label_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_11_vollbohrung_index_label_only_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_12_vollbohrung_fz_label_only_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_13_vollbohrung_mz_label_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_14_vollbohrung_mz_label_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_15_vollbohrung_mz_label_only_fft_features.csv',index_col=0)
#dataset = pd.read_csv(r'D:\users\pnh_lm\11052020\01 Cut Data ags_lk\Python\03 Data Analysis pnh_lm\dataset_ansys_16_vollbohrung_mz_label_only_fft_features.csv',index_col=0)


# the most ranked 10th features in RFR
#dataset=dataset.drop(dataset.columns[100:],axis=1)
#dataset=dataset.drop(dataset.index[600:])
#dataset = pd.read_csv('U:\\first_task_Scheibe\\13022020\\all_features_label.csv',index_col=0)

# using index as a Feature
#dataset['index1'] = dataset.index

# Look at the Data Structure :

#dataset.head()
#dataset.info()
#dataset.describe()
#dataset.hist(bins=50, figsize=(20,15))
#plt.show()

corr_matrix = dataset.corr()
#corr_matrix[dataset.index].sort_values(ascending=False)
#corr_matrix["wear_rate"].sort_values(ascending=False)
corr_matrix

#%% Defining Label:

X=dataset
#X=X.drop(["Reihenfolge"],axis =1)
X=X.drop(["/'Untitled'/'Fz'_signal_mean"],axis =1)
#X=X.drop(["/'Untitled'/'Mz'_signal_mean"],axis =1)

#X=X.drop('wear_measurements_interpolated',axis=1)
#X=X.drop('wear_rate',axis=1)

#X=X.drop(X.columns[700:],axis=1)
#y=dataset['wear_measurements_interpolated']
#y=dataset['wear_rate']
#y=dataset.index

# Defining Label:

#y=dataset["Reihenfolge"]
y=dataset["/'Untitled'/'Fz'_signal_mean"]
#y=dataset["/'Untitled'/'Mz'_signal_mean"]

#%% Splitting DATA

X_train_o, X_test_o, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train_o_index=X_train_o.index
#y_train_index=y_train.index

X_train = X_train_o
X_test = X_test_o

scaler = StandardScaler()  # doctest: +SKIP
scaler.fit(X_train)  # doctest: +SKIP
X_train = scaler.transform(X_train)  # doctest: +SKIP
# apply same transformation to test data
X_test = scaler.transform(X_test)

# In[88]:
y
#y.max() 
# In[89]:

# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# exemple with the first 5 instances
#some_data = X_train[:5]
#some_labels = y_train[:5]
#print("Predictions: ", lin_reg.predict(some_data))
#print("Labels: ", list(some_labels))
from sklearn.metrics import mean_squared_error
lin_wear_predictions = lin_reg.predict(X_train)
lin_mse = mean_squared_error(y_train, lin_wear_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear RMSE: ",lin_rmse)


# In[90]:


# Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
# exemple with the first 5 instances
#some_data = X_train[:5]
#some_labels = y_train[:5]
#print("Predictions: ", tree_reg.predict(some_data))
#print("Labels: ", list(some_labels))
tree_wear_predictions = tree_reg.predict(X_train)
tree_mse = mean_squared_error(y_train, tree_wear_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree RMSE: ",tree_rmse)


# In[91]:


# Random Forests Regressor

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
forest_wear_predictions = forest_reg.predict(X_train)
forest_mse = mean_squared_error(y_train, forest_wear_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest RMSE: ",forest_rmse)


# In[92]:


# Multi-layer Perceptron Regressor

from sklearn.neural_network import MLPRegressor
##nn_reg = MLPRegressor(max_iter=3200)
nn_reg = MLPRegressor(max_iter=3000)
nn_reg.fit(X_train, y_train)
nn_wear_predictions = nn_reg.predict(X_train)
nn_mse = mean_squared_error(y_train, nn_wear_predictions)
nn_rmse = np.sqrt(nn_mse)
print("Multi-layer Regressor RMSE: ",nn_rmse)


# In[93]:


# Support Vector Machines (SVMs)

from sklearn.svm import LinearSVR
svm_reg = SVR(kernel='rbf',C=100,gamma=0.1,epsilon=.5)
#svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X_train, y_train)
svm_wear_predictions = svm_reg.predict(X_train)
svm_mse = mean_squared_error(y_train, svm_wear_predictions)
svm_rmse = np.sqrt(svm_mse)
print("SVM RMSE: ",svm_rmse)


# In[94]:

for cont in range(0,1):
    
    # Cross-Validation
    text="Ansys16."+str(cont)+".voll"
    print(text)
    
    from sklearn.model_selection import cross_val_score
    def display_scores(scores, model_name):
        print(model_name)
        print("Scores: ",scores)
        print("Mean: ",scores.mean())
        print("Standard deviation: ",scores.std(),end='\n\n')
    
    lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    #nn_scores = cross_val_score(nn_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
    #nn_rmse_scores = np.sqrt(-nn_scores)
    svm_scores = cross_val_score(svm_reg, X_train, y_train, scoring="neg_mean_squared_error",cv=10)
    svm_rmse_scores = np.sqrt(-svm_scores)
    
    display_scores(lin_rmse_scores, model_name="Linear")
    display_scores(tree_rmse_scores, model_name="Decision Tree")
    display_scores(forest_rmse_scores, model_name="Random Forest")
    #display_scores(nn_rmse_scores, model_name="Multi-layer")
    display_scores(svm_rmse_scores, model_name="SVM")
    
    
    print("RMSE: \nStd: \nRMSE(%): \nModel: \n")
    
    # saving models for later ...
    #from sklearn.externals import joblib
    #joblib.dump(lin_reg,"U:\\first_task_Scheibe\\13022020\\saved_models\\lin_reg.pkl")
    #joblib.dump(tree_reg,"U:\\first_task_Scheibe\\13022020\\saved_models\\tree_reg.pkl")
    #joblib.dump(forest_reg,"U:\\first_task_Scheibe\\13022020\\saved_models\\forest_reg.pkl")
    
    #lin_reg_loaded = joblib.load(lin_reg)
    #tree_reg_loaded = joblib.load(tree_reg)
    #forest_reg_loaded = joblib.load(forest_reg)


# In[35]:


# Evaluating the Test Set in the 'best' model: 

# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
# Test Set Data here
wear_predictions = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, wear_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest RMSE: ",forest_rmse)


# In[37]:


# Plots for all the models with the wear interpolated

y_train_index=y_train.index
y_test_index=y_test.index

predictions_train_data = pd.DataFrame({
                'index': y_train.index,
                'y_train':y_train,
                'lin_wear_predictions':lin_wear_predictions,
                'tree_wear_predictions':tree_wear_predictions,
                'forest_wear_predictions':forest_wear_predictions,
                'nn_wear_predictions':nn_wear_predictions,
                'svm_wear_predictions':svm_wear_predictions,                
        })

fig = plt.figure(figsize=(18,10))
plt.plot(dataset["wear_measurements_interpolated"],'orange',label="wear_measurements_interpolated",linewidth=1)
#a=y_train_index; b=y[y_train_index];
#plt.scatter(a[0:100],b[0:100], color='k',label="Train Data", marker=".",s=100)
#a=y_test_index; b=y[y_test_index];
#plt.scatter(a[0:100],b[0:100], color='b',label="Test Data", marker=".",s=100)

a=predictions_train_data["index"]
#bounds=np.arange(0,200)

b=predictions_train_data["lin_wear_predictions"]
plt.scatter(a,b,color='r',label="Linear Regressor Model", marker=".",s=25)
b=predictions_train_data["tree_wear_predictions"]
plt.scatter(a,b,color='k',label="Decision Tree Regressor Model", marker=".",s=5)
b=predictions_train_data["forest_wear_predictions"]
plt.scatter(a,b,color='g',label="Random Forest Regressor Model", marker=".",s=25)
b=predictions_train_data["nn_wear_predictions"]
plt.scatter(a,b,color='m',label="Multi-layer NN Regressor Model", marker=".",s=25)
b=predictions_train_data["svm_wear_predictions"]
plt.scatter(a,b,color='b',label="SVM Regressor Model", marker=".",s=25)



plt.legend(loc=2)
#plt.title('Linear Regressor Model')
#plt.title('Decision Tree Regressor Model')
#plt.title('Random Forest Regressor Model')
#plt.title('Multi-layer NN Regressor Model')
#plt.title('SVM Regressor Model')
plt.title('All models')


plt.xlabel('Experients number')
plt.ylabel('Wear [um]')
plt.grid()
plt.show()
fig.savefig(r'U:\first_task_Scheibe\13022020\model_predictions_&_wear\models_wear_predictions_newDataset_5.png')


# In[106]:


predictions_train_data


# In[101]:


bounds


# In[82]:


# Grid Search

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]},
]
###
#%%
print("Linear Regressor: ")
lin_reg = LinearRegression()
grid_search = GridSearchCV(lin_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#grid_search.best_params_
#grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score),params)

print("Decision Tree Regressor: ")
tree_reg = DecisionTreeRegressor()
grid_search = GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#grid_search.best_params_
#grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score),params)
#%%
print("Random Forest Regressor: ")
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#grid_search.best_params_
#grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score),params)
    
#%% 23.09.2020   
    
    
param_grid = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 15, 100],'gamma': [1e-7, 1e-4, 1e-1],'epsilon':[0.05,0.1,0.5]}

print("SVM Regressor: ")
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#grid_search.best_params_
#grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score),params)
    
#%%
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.05,0.1,0.5,0.9,1,1.5]}
svr = svm.SVR()
#clf = grid_search.grid_scores_(svr, parameters)
#clf.fit(X,y)
#clf.best_params_

#%%    
    
'''
param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            #'max_iter' : 2500,
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
             ]
        }
       ]
print("Multi-layer Regressor: ")
nn_reg = MLPRegressor(max_iter=3000)
grid_search = GridSearchCV(nn_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#grid_search.best_params_
#grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score),params)
    
    
# print("Best parameters set found on development set:")
# print(clf.best_params_)
'''


# In[ ]:


param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            #'max_iter' : 2500,
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
             ]
        }
       ]
print("Multi-layer Regressor: ")
nn_reg = MLPRegressor(max_iter=3000)
grid_search = GridSearchCV(nn_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#grid_search.best_params_
#grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score),params)


# In[ ]:


grid_search.best_estimator_


# In[ ]:



# In[38]:


# simple Neural Network
'''
###
y=y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
###

clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=4000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

y_test_results=clf.predict(X_test)-y_test
#len(y_test), len(y_test_results[np.where(y_test_results==0)])
np.set_printoptions(precision=2, suppress=True)
a=y_test_results/y_test
print(a*100)
# Deleting the value Zero for percentages calculation.
exclude=np.where(y_test==0)
a=np.delete(a, exclude)


sum(abs(a))/len(a)

# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X_train)
X2D

'''


# In[ ]:





# In[38]:


'''
I have experience in Data Analysis of Monitoring Systems, using Machine Learning Algorithms and models (Decision Tree, Random Forest, Neural Network, etc.) . 
I also have experience in the subjects: simulation, finite element method, FEM, CAE system software, engineering statics, computer programming and CAD system software.
I currently work as a student assistant at Werkzeugmaschinenlabor WZL der RWTH Aachen, Germany,  analysing data from experiments of manufacturing processes and production systems.
I worked as an undergraduate teaching assistant (UTA) in engineering statics at UFSC, Brazil.
I worked as a student assistant with Simulation using Finite Elements Analysis (FEA) and Metallographic techniques in the Residual Stresses Analysis at LMP and LabMetro laboratories, UFSC, Brazil.
I was in a student exchange program at INSA – Strasbourg, France, where I developed skills in manufacturing systems.
Professional skills: Python,  Matlab, Ansys (APDL&Workbench), CAD (SolidWorks, AutoCAD)
'''


# In[ ]:




# In[ ]:




