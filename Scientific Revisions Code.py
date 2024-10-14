# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:45:56 2024

@author: Nour
"""

# Importing Libraries
import sys
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Loading Data
Well_File_Path=r'D:/PhD/2- PhD/PhD/PhD Work/My Codes/RSS_PDM/'

Wells_List=['X_3', 'C_201', 'A_3_X']

Path_List=[]

for i in Wells_List:
    Path_List.append(Well_File_Path+i+'.xlsx')

ML_1_Dict={}
for i in range(len(Path_List)):
    ML_1_Dict[i]=pd.read_excel(Path_List[i], sheet_name='ML_1')
ML_1_Raw=pd.concat([ML_1_Dict[0], ML_1_Dict[1],ML_1_Dict[2]], axis=0)
raw_data=ML_1_Raw.reset_index(drop=True)
raw_data.drop('Unnamed: 0', axis=1, inplace=True)
raw_data.drop(raw_data.loc[raw_data['DD_Tech']=='Slick'].index, inplace=True)
raw_data.drop(raw_data.loc[raw_data['ROP']<=0].index, inplace=True)
raw_data=raw_data.reset_index(drop=True)
raw_data

# EDA Function

def EDA(df):
    print('\n','Shape of this dataframe is:',df.shape,'\n')
    print('\n','This dataframe has ',len(df),' rows','\n')
    print('\n','This dataframe has ',df.shape[1],' columns','\n')
    print('\n','Column names of this dataframe are ','\n', df.columns,'\n')
    print('\n','non-Null, Counts & Data Type ','\n',df.info(),'\n')
    print('\n','Some statistical data: ','\n', df.describe(include='all').T,'\n')
    print('\n','Null Values ','\n', df.isnull().sum(),'\n')
    print('\n','CORRELATIONS','\n',df.corr(numeric_only=True),'\n')
    categorical = [var for var in df.columns if df[var].dtype=='O']
    print('The categorical variables are :\n\n', categorical,'\n','With this distribution','\n')
    for var in categorical: 
        print(df[var].value_counts())
        print(100*df[var].value_counts()/float(len(df)),' %','\n')
    print('\n','Two first rows: ','\n')
    print(df.head(2))
    print('\n','Two last rows: ','\n',)
    print(df.tail(2))
    print('\n','BOX PLOT','\n')
    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False,figsize=(10, 10))
    plt.show()
    print('\n','HISTOGRAM','\n')
    df.hist(figsize=(10, 10))
    plt.show()
    print('\n','PAIR PLOT','\n')
    sns.pairplot(df,diag_kind='kde')
    plt.show()
    print('\n','CORRELATION HEATMAP','\n')
    fig = plt.figure(figsize=(7, 7))
    sns.heatmap(df.corr(numeric_only=True),center=0, vmin=-1, vmax=1, square=True, annot=True,cmap='vlag_r',cbar=True)
    plt.show()
    
# SciKit Learn Work

# Split data based on representative features sampling

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(raw_data, raw_data[['DD_Tech','Sub_Field','HS','M_Type','FMT','B_Class', 'B_Cond']]):
    strat_train_set = raw_data.iloc[train_index]
    strat_test_set = raw_data.iloc[test_index]
    
# For Training Purpose
ML1=strat_train_set.drop('ROP', axis=1)
ML1_labels=strat_train_set['ROP'].copy()

# Function to transform prepared DF into a prepared array fo ML work
def full_transformer(df):
    ''' For a Full Transformation of any data
    Either Categorial or Numerical
    into final prepared form for ML work'''
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    # determine categorical and numerical features
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler()),])
    num_attribs = list(df.select_dtypes(include=['int64', 'float64']).columns)
    cat_attribs = list(df.select_dtypes(include=['object', 'bool']).columns)
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])
    prepared_data = full_pipeline.fit_transform(df)
    ordered_columns=pd.get_dummies(df).columns
    print('NEW ORDERED COLUMNS ARE:',ordered_columns)
    print('First row of prepared data:', prepared_data[0])
    return prepared_data

ML1_prepared=full_transformer(ML1)

## A Function to determine accuracy and CV
def accuracy(model,X,Y):
    # Takes model:model name (e.g. lin_reg), X:Actual Predictors (Prepared array) & Y:Actual labels (e.g. ROP)
    # Note to take the new prepared X for polynomial
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_score
    def display_scores(scores):
        print("CV Scores:", scores)
        print("CV Scores Mean:", scores.mean())
        print("CV Scores Standard deviation:", scores.std())
    predictions=model.predict(X)
    mse = mean_squared_error(Y, predictions)
    rmse = np.sqrt(mse)
    MAE = mean_absolute_error(Y, predictions)
    mse_scores = cross_val_score(model, X, Y,scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-mse_scores)
    scores_2 = cross_val_score(model, X, Y,scoring="neg_mean_absolute_error", cv=10)
    mae_scores = -scores_2
    print('\n')
    print('ACCURACY SCORES OF '+str(model))
    print('\n')
    print('Score (Classifier_Accuracy or Regression_R2): ',model.score(X,Y))
    print('rmse:', rmse)
    print('MAE:', MAE)
    print('\n')
    print('rmse Cross Validation:')
    display_scores(rmse_scores)
    print('\n')
    print('MAE Cross Validation:')
    display_scores(mae_scores)
    

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost
g_search.best_estimator_



best_xgb_reg = xgboost.XGBRegressor(
    base_score=0.5, booster='gbtree', colsample_bylevel=1,
    colsample_bynode=1, colsample_bytree=0.5, enable_categorical=False,
    gamma=0, gpu_id=-1, importance_type=None, interaction_constraints='',
    learning_rate=0.1, max_delta_step=0, max_depth=10, min_child_weight=1, 
    monotone_constraints='()', n_estimators=200, n_jobs=4,
    num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
    scale_pos_weight=1, subsample=0.7, tree_method='exact', verbosity=None
)

# Fit the model
best_xgb_reg.fit(ML1_prepared, ML1_labels)
accuracy(best_xgb_reg,ML1_prepared, ML1_labels)


final_model = best_xgb_reg

# For Testing Purpose
ML1_Test=strat_test_set.drop('ROP', axis=1)
ML1_labels_Test=strat_test_set['ROP'].copy()

ML1_test_prepared = full_transformer(ML1_Test)


final_predictions = final_model.predict(ML1_test_prepared)
accuracy(best_xgb_reg,ML1_test_prepared,ML1_labels_Test )

final_mse = mean_squared_error(ML1_labels_Test, final_predictions)
#final_mse
final_rmse = np.sqrt(final_mse)
final_rmse
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - ML1_labels_Test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))


plot=sns.jointplot(x=final_predictions, y=ML1_labels_Test, kind='reg')
#MIN=min(final_predictions.min(),ML2_labels_Test.min())-5
MIN=0
MAX=max(final_predictions.max(),ML1_labels_Test.max())+5
plt.xlabel('Predicted Bit ROP')
plt.ylabel('Actual Bit ROP')
lims=[MAX,MIN]
plot.ax_joint.plot(lims,lims, '-r')
#plt.gca().set_aspect('equal' adjustable='box')
plt.legend(title='Model Performance on Test Set', loc='upper left', labels=['Prediction vs Actual', 'Idendity Line'])
plot.ax_marg_x.set_xlim(MIN, MAX)
plot.ax_marg_y.set_ylim(MIN, MAX)
plt.savefig('New_ML1_Performance.png', dpi=600)
plt.savefig('New_ML1_Performance.svg')

plt.show()

#====================================================#
#Model-1 Finished#
#====================================================#



#====================================================#
# Start of Model-2#
#====================================================#

# Enter Containing Directory Here
Well_File_Path=r'D:/PhD/2- PhD/PhD/PhD Work/My Codes/RSS_PDM/'

Wells_List=['X_3', 'C_201', 'A_3_X']

Path_List=[]

for i in Wells_List:
    Path_List.append(Well_File_Path+i+'.xlsx')

ML_2_Dict={}
for i in range(len(Path_List)):
    ML_2_Dict[i]=pd.read_excel(Path_List[i], sheet_name='ML_2')
ML_2_Raw=pd.concat([ML_2_Dict[0], ML_2_Dict[1],ML_2_Dict[2]], axis=0)
raw_2=ML_2_Raw.reset_index(drop=True)
raw_2.drop('Unnamed: 0', axis=1, inplace=True)
raw_2.drop(raw_2.loc[raw_2['DD_Tech']=='Slick'].index, inplace=True)
raw_2.drop(raw_2.loc[raw_2['AVG_Bit_ROP']<=raw_2['AVG_ROP']].index, inplace=True)
raw_2=raw_2.drop_duplicates(keep='first')
raw_2=raw_2.reset_index(drop=True)
raw_2

## EDA Function

def EDA(df):
    print('\n','Shape of this dataframe is:',df.shape,'\n')
    print('\n','This dataframe has ',len(df),' rows','\n')
    print('\n','This dataframe has ',df.shape[1],' columns','\n')
    print('\n','Column names of this dataframe are ','\n', df.columns,'\n')
    print('\n','non-Null, Counts & Data Type ','\n',df.info(),'\n')
    print('\n','Some statistical data: ','\n', df.describe(include='all').T,'\n')
    print('\n','Null Values ','\n', df.isnull().sum(),'\n')
    print('\n','CORRELATIONS','\n',df.corr(numeric_only=True),'\n')
    categorical = [var for var in df.columns if df[var].dtype=='O']
    print('The categorical variables are :\n\n', categorical,'\n','With this distribution','\n')
    for var in categorical: 
        print(df[var].value_counts())
        print(100*df[var].value_counts()/float(len(df)),' %','\n')
    print('\n','Two first rows: ','\n')
    print(df.head(2))
    print('\n','Two last rows: ','\n',)
    print(df.tail(2))
    print('\n','BOX PLOT','\n')
    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False,figsize=(10, 10))
    plt.show()
    print('\n','HISTOGRAM','\n')
    df.hist(figsize=(10, 10))
    plt.show()
    print('\n','PAIR PLOT','\n')
    sns.pairplot(df,diag_kind='kde')
    plt.show()
    print('\n','CORRELATION HEATMAP','\n')
    fig = plt.figure(figsize=(7, 7))
    sns.heatmap(df.corr(numeric_only=True),center=0, vmin=-1, vmax=1, square=True, annot=True,cmap='vlag_r',cbar=True)
    plt.show()
    
   # SciKit Learn Work
   # Split data based on representative features sampling

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(raw_2, raw_2[[ 'M_Type', 'DD_Tech', 'B_Class', 'Rig']]):
    strat_train_set = raw_2.iloc[train_index]
    strat_test_set = raw_2.iloc[test_index]
    
# For Training Purpose
ML2=strat_train_set.drop('AVG_ROP', axis=1)
ML2_labels=strat_train_set['AVG_ROP'].copy()

# Function to transform prepared DF into a prepared array fo ML work
def full_transformer(df):
    ''' For a Full Transformation of any data
    Either Categorial or Numerical
    into final prepared form for ML work'''
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    # determine categorical and numerical features
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler()),])
    num_attribs = list(df.select_dtypes(include=['int64', 'float64']).columns)
    cat_attribs = list(df.select_dtypes(include=['object', 'bool']).columns)
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])
    prepared_data = full_pipeline.fit_transform(df)
    ordered_columns=pd.get_dummies(df).columns
    print('NEW ORDERED COLUMNS ARE:',ordered_columns)
    print('First row of prepared data:', prepared_data[0])
    return prepared_data

ML2_prepared=full_transformer(ML2)

from sklearn.model_selection import cross_val_score
## A Function to determine accuracy and CV
def accuracy(model,X,Y):
    # Takes model:model name (e.g. lin_reg), X:Actual Predictors (Prepared array) & Y:Actual labels (e.g. ROP)
    # Note to take the new prepared X for polynomial
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_score
    def display_scores(scores):
        print("CV Scores:", scores)
        print("CV Scores Mean:", scores.mean())
        print("CV Scores Standard deviation:", scores.std())
    predictions=model.predict(X)
    mse = mean_squared_error(Y, predictions)
    rmse = np.sqrt(mse)
    MAE = mean_absolute_error(Y, predictions)
    mse_scores = cross_val_score(model, X, Y,scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-mse_scores)
    scores_2 = cross_val_score(model, X, Y,scoring="neg_mean_absolute_error", cv=10)
    mae_scores = -scores_2
    print('\n')
    print('ACCURACY SCORES OF '+str(model))
    print('\n')
    print('Score (Classifier_Accuracy or Regression_R2): ',model.score(X,Y))
    print('rmse:', rmse)
    print('MAE:', MAE)
    print('\n')
    print('rmse Cross Validation:')
    display_scores(rmse_scores)
    print('\n')
    print('MAE Cross Validation:')
    display_scores(mae_scores)
    

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }
xgb_tuned = xgboost.XGBRegressor()
g_search = GridSearchCV(estimator = xgb_tuned, param_grid = param_tuning, cv = 5,n_jobs = -1,verbose = 1)

g_search.fit(ML2_prepared, ML2_labels)


g_search.best_estimator_

accuracy(g_search.best_estimator_,ML2_prepared, ML2_labels)


best_xgb_reg=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, 
             monotone_constraints='()', n_estimators=500, n_jobs=4,
             num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=0.7,
             tree_method='exact', validate_parameters=1, verbosity=None)
best_xgb_reg.fit(ML2_prepared, ML2_labels)

accuracy(best_xgb_reg,ML2_prepared, ML2_labels)

final_model = best_xgb_reg

# For Testing Purpose
ML2_Test=strat_test_set.drop('ROP', axis=1)
ML2_labels_Test=strat_test_set['ROP'].copy()

ML2_test_prepared = full_transformer(ML2_Test)


final_predictions = final_model.predict(ML2_test_prepared)
accuracy(best_xgb_reg,ML2_test_prepared,ML2_labels_Test )

final_mse = mean_squared_error(ML2_labels_Test, final_predictions)
#final_mse
final_rmse = np.sqrt(final_mse)
final_rmse
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - ML2_labels_Test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))


#====================================================#
#Model-2 Finished#
#====================================================#



#====================================================#
# Start of Model-3#
#====================================================#

# Loading Data
Well_File_Path=r'D:/PhD/2- PhD/PhD/PhD Work/My Codes/RSS_PDM/'

Wells_List=['X_3', 'C_201', 'A_3_X']

Path_List=[]

for i in Wells_List:
    Path_List.append(Well_File_Path+i+'.xlsx')

ML_3_PO_Dict={}
for i in range(len(Path_List)):
    ML_3_PO_Dict[i]=pd.read_excel(Path_List[i], sheet_name='ML_3_PO')
ML_3_PO=pd.concat([ML_3_PO_Dict[0], ML_3_PO_Dict[1],ML_3_PO_Dict[2]], axis=0)
raw_3_PO=ML_3_PO.reset_index(drop=True)
raw_3_PO.drop('Unnamed: 0', axis=1, inplace=True)
raw_3_PO.drop(raw_3_PO.loc[raw_3_PO['DD_Tech']=='Slick'].index, inplace=True)
raw_3_PO=raw_3_PO.drop_duplicates(keep='first')
raw_3_PO=raw_3_PO.reset_index(drop=True)
raw_3_PO['Direction']='PO'

ML_3_RO_Dict={}
for i in range(len(Path_List)):
    ML_3_RO_Dict[i]=pd.read_excel(Path_List[i], sheet_name='ML_3_RO')
ML_3_RO=pd.concat([ML_3_RO_Dict[0], ML_3_RO_Dict[1],ML_3_RO_Dict[2]], axis=0)
raw_3_RO=ML_3_RO.reset_index(drop=True)
raw_3_RO.drop('Unnamed: 0', axis=1, inplace=True)
raw_3_RO.drop(raw_3_RO.loc[raw_3_RO['DD_Tech']=='Slick'].index, inplace=True)
raw_3_RO=raw_3_RO.drop_duplicates(keep='first')
raw_3_RO=raw_3_RO.reset_index(drop=True)
raw_3_RO['Direction']='RO'

ML_3_PO_2_Dict={}
for i in range(len(Path_List)):
    ML_3_PO_2_Dict[i]=pd.read_excel(Path_List[i], sheet_name='ML_3_PO_2')
ML_3_PO_2=pd.concat([ML_3_PO_2_Dict[0], ML_3_PO_2_Dict[1],ML_3_PO_2_Dict[2]], axis=0)
raw_3_PO_2=ML_3_PO_2.reset_index(drop=True)
raw_3_PO_2.drop('Unnamed: 0', axis=1, inplace=True)
raw_3_PO_2.drop(raw_3_PO_2.loc[raw_3_PO_2['DD_Tech']=='Slick'].index, inplace=True)
raw_3_PO_2=raw_3_PO_2.drop_duplicates(keep='first')
raw_3_PO_2=raw_3_PO_2.reset_index(drop=True)
raw_3_PO_2['Direction']='PO_2'

ML_3_C_Dict={}
for i in range(len(Path_List)):
    ML_3_C_Dict[i]=pd.read_excel(Path_List[i], sheet_name='ML_3_C')
ML_3_C=pd.concat([ML_3_C_Dict[0], ML_3_C_Dict[1],ML_3_C_Dict[2]], axis=0)
raw_3_C=ML_3_C.reset_index(drop=True)
raw_3_C.drop('Unnamed: 0', axis=1, inplace=True)
raw_3_C.drop(raw_3_C.loc[raw_3_C['DD_Tech']=='Slick'].index, inplace=True)
raw_3_C=raw_3_C.drop_duplicates(keep='first')
raw_3_C=raw_3_C.reset_index(drop=True)
raw_3_C['Direction']='C'

ML_3_L_Dict={}
for i in range(len(Path_List)):
    ML_3_L_Dict[i]=pd.read_excel(Path_List[i], sheet_name='ML_3_L')
ML_3_L=pd.concat([ML_3_L_Dict[0], ML_3_L_Dict[1],ML_3_L_Dict[2]], axis=0)
raw_3_L=ML_3_L.reset_index(drop=True)
raw_3_L.drop('Unnamed: 0', axis=1, inplace=True)
raw_3_L.drop(raw_3_L.loc[raw_3_L['DD_Tech']=='Slick'].index, inplace=True)
raw_3_L=raw_3_L.drop_duplicates(keep='first')
raw_3_L=raw_3_L.reset_index(drop=True)
raw_3_L['Direction']='L'

raw_3_PO=raw_3_PO.rename(columns={"S_PO": "S"})
raw_3_RO=raw_3_RO.rename(columns={"S_RO": "S"})
raw_3_PO_2=raw_3_PO_2.rename(columns={"S_PO_2": "S"})
raw_3_C=raw_3_C.rename(columns={"S_C": "S"})
raw_3_L=raw_3_L.rename(columns={"S_L": "S"})

raw_3=pd.concat([raw_3_PO,raw_3_RO,raw_3_PO_2,raw_3_C,raw_3_L], axis=0)
raw_3=raw_3.reset_index(drop=True)

## EDA Function

def EDA(df):
    print('\n','Shape of this dataframe is:',df.shape,'\n')
    print('\n','This dataframe has ',len(df),' rows','\n')
    print('\n','This dataframe has ',df.shape[1],' columns','\n')
    print('\n','Column names of this dataframe are ','\n', df.columns,'\n')
    print('\n','non-Null, Counts & Data Type ','\n',df.info(),'\n')
    print('\n','Some statistical data: ','\n', df.describe(include='all').T,'\n')
    print('\n','Null Values ','\n', df.isnull().sum(),'\n')
    print('\n','CORRELATIONS','\n',df.corr(numeric_only=True),'\n')
    categorical = [var for var in df.columns if df[var].dtype=='O']
    print('The categorical variables are :\n\n', categorical,'\n','With this distribution','\n')
    for var in categorical: 
        print(df[var].value_counts())
        print(100*df[var].value_counts()/float(len(df)),' %','\n')
    print('\n','Two first rows: ','\n')
    print(df.head(2))
    print('\n','Two last rows: ','\n',)
    print(df.tail(2))
    print('\n','BOX PLOT','\n')
    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False,figsize=(10, 10))
    plt.show()
    print('\n','HISTOGRAM','\n')
    df.hist(figsize=(10, 10))
    plt.show()
    print('\n','PAIR PLOT','\n')
    sns.pairplot(df,diag_kind='kde')
    plt.show()
    print('\n','CORRELATION HEATMAP','\n')
    fig = plt.figure(figsize=(7, 7))
    sns.heatmap(df.corr(numeric_only=True),center=0, vmin=-1, vmax=1, square=True, annot=True,cmap='vlag_r',cbar=True)
    plt.show()
    
#SciKit Learn Work
# Split data based on representative features sampling

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(raw_3, raw_3[['Direction']]):
    strat_train_set = raw_3.iloc[train_index]
    strat_test_set = raw_3.iloc[test_index]
    
# For Training Purpose
ML3=strat_train_set.drop('S', axis=1)
ML3_labels=strat_train_set['S'].copy()

# For Testing Purpose
ML3_Test=strat_test_set.drop('S', axis=1)
ML3_labels_Test=strat_test_set['S'].copy()

# Function to transform prepared DF into a prepared array fo ML work
def full_transformer(df):
    ''' For a Full Transformation of any data
    Either Categorial or Numerical
    into final prepared form for ML work'''
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    # determine categorical and numerical features
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler()),])
    num_attribs = list(df.select_dtypes(include=['int64', 'float64']).columns)
    cat_attribs = list(df.select_dtypes(include=['object', 'bool']).columns)
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])
    prepared_data = full_pipeline.fit_transform(df)
    ordered_columns=pd.get_dummies(df).columns
    print('NEW ORDERED COLUMNS ARE:',ordered_columns)
    print('First row of prepared data:', prepared_data[0])
    return prepared_data

ML3_prepared=full_transformer(ML3)
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }
xgb_tuned = xgboost.XGBRegressor()
g_search = GridSearchCV(estimator = xgb_tuned, param_grid = param_tuning, cv = 5,n_jobs = -1,verbose = 1)

g_search.fit(ML3_prepared, ML3_labels)

g_search.best_estimator_

accuracy(g_search.best_estimator_,ML3_prepared, ML3_labels)


final_model = best_xgb_reg
ML3_Test
ML3_test_prepared = full_transformer(ML3_Test)

final_model = best_xgb_reg

# For Testing Purpose
ML3_test_prepared = full_transformer(ML3_Test)


final_predictions = final_model.predict(ML3_test_prepared)
accuracy(best_xgb_reg,ML3_test_prepared,ML3_labels_Test )

final_mse = mean_squared_error(ML3_labels_Test, final_predictions)
#final_mse
final_rmse = np.sqrt(final_mse)
final_rmse
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - ML3_labels_Test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))

#END OF CODE#