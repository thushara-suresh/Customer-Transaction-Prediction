# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:39:34 2022

@author: Thushara S
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# This function takes the test data provided by user and predicts the probabilities of the 4 classes against each query point using the model provided as a parameter.
# These probabilities are then concatanated with the test dataset to provide 4 additional features. 
# The Dataset is further standardized and PCA is applied on it to obtain 5 more features. 
# These 5 features from PCA are also concenated with the Dataset and the final Dataset with 209 features is returned

def featurization (multi_class_model, x_test):
    #predict the class label using multiclass
    #y_pred_prob contains four columns with probablities of the datapoint belonging to each of the 4 classes
    y_pred_prob = multi_class_model.predict_proba(x_test)

    #Concatenate this y_pred_prob with the X dataset to obtain 204 features
    X_data_new = np.concatenate((x_test,y_pred_prob), axis=1)
    
    #Standardize this new dataset
    scaler = StandardScaler()
    X_data_std = scaler.fit_transform(X_data_new)
    
    #Apply PCA on this data to obtain 5 new features. These 5 features will be concatenated with the dataset
    pca = PCA(n_components = 5)
    pca_features = pca.fit_transform(X_data_std)
    X_data_new = np.concatenate((X_data_new, pca_features), axis=1)
    
    return X_data_new


def checkErrors(x_test):
    errorDetails = dict(
        errorNum = 0,
        errorMessage = '',
        errorStatus = False,
        errorAutoCorrect = True
        )
    
    x_test_cols = x_test.columns
    
    #1. Checking if the test data contains any null values
    null_check = x_test.isnull().sum()
    null_columns = []
    for (index, value) in null_check.items():
        if value != 0:
            null_columns.append(index)
    if len(null_columns) > 0 : 
        errorDetails['errorNum'] += 1
        errorDetails['errorMessage']= errorDetails['errorMessage'] + " %d. NULL values in the dataset in column(s) : %s\n "%( errorDetails['errorNum'], str(null_columns))
        errorDetails['errorStatus'] = True
        
    #Check if there are total of 201 columns ( i.e. ID_code and 200 features) and if the column name 'ID_code' is present as the first column  
    
    if len(x_test_cols) != 201:
        num_missing_cols = 201 - len(x_test_cols)
        errorDetails['errorNum'] += 1
        errorDetails['errorMessage'] = errorDetails['errorMessage'] + " %s. Found %s missing column(s) \n"%( str(errorDetails['errorNum']), str(num_missing_cols))
        errorDetails['errorStatus'] = True
        errorDetails['errorAutoCorrect'] = False
        
    if x_test_cols[0] != 'ID_code':
        errorDetails['errorNum'] += 1
        errorDetails['errorMessage'] = errorDetails['errorMessage'] + " %s. Name of first column should be 'ID_code'.\n"%( str(errorDetails['errorNum']))
        errorDetails['errorStatus'] = True
        errorDetails['errorAutoCorrect'] = False
    
    #pd.to_numeric(df['column'], errors='coerce').notnull().all()
    error_cols =[]
    for col in x_test_cols[1:]: 
        if x_test[col].dtypes not in ['float64', 'int64']:
            error_cols.append(col)
    if len(error_cols)!= 0:      
        errorDetails['errorNum'] += 1
        errorDetails['errorMessage'] = errorDetails['errorMessage'] + " %s. All values of %s should be of type int or float'.\n"%( str(errorDetails['errorNum']), str(error_cols))
        errorDetails['errorStatus'] = True
        errorDetails['errorAutoCorrect'] = False
                
    return errorDetails
    
            

def predict(df):
    my_bar = st.progress(0)
    x_test_ID = df['ID_code']
    x_test = df.drop(labels=['ID_code'], axis=1)
    x_test = preprocessing(x_test)
    my_bar.progress(5)
    x_test_final = featurization(lgbm_multiclass, x_test)
    
    my_bar.progress(50)
    #final prediction of the customer's likelihood of making a transaction based on the 209 features
    y_featurized_pred = lgbm_final.predict(x_test_final)
    
    my_bar.progress(99)
    y_pred = pd.DataFrame( {'ID_code':x_test_ID, 'target': y_featurized_pred})
              
    my_bar.progress(100)
    
    my_bar.empty()
    y_pred_file = y_pred
    st.write("Predictions:")
    st.write(y_pred)
        
    #y_pred_file = y_pred
    y_pred_file =y_pred_file.to_csv(index=False).encode('utf-8')

    st.download_button(label = 'Download Results',
                       data = y_pred_file,
                       file_name = 'predictions.csv', 
                       mime = 'text/csv')

def preprocessing(df):
    column_medians = df.median()
    df = df.fillna(column_medians)
    return df


lgbm_multiclass = joblib.load("lgbm_multiclass")
lgbm_final = joblib.load("lgbm_final_binary")

test_template = pd.read_csv("test_template.csv")
test_data = test_template.to_csv(index = False).encode('utf-8')

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
            <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color:rgb(34 79 109);">
              <div class="navbar-brand" target="_blank" style="margin: auto">Santander Bank - Customer Transaction Prediction</div>
              
            </nav>
            """, unsafe_allow_html=True)
msg = None
col1, col2 = st.columns((2,1))

#st.title("Customer Transaction Prediction")
with st.sidebar.expander("Sample Template"):
        st.write(test_template)
        st.download_button(label = 'Download',
                                data = test_data,
                                file_name = 'test_template.csv', 
                                mime = 'text/csv')

#with col1:        
    
uploaded_file = st.file_uploader("Upload a csv file containing test data")
if uploaded_file is not None:
     df = pd.read_csv(uploaded_file)
     msg = st.success("File Uploaded Successfully")

         
     with st.expander(uploaded_file.name):
         st.write(df.head())
if msg is not None:
      
    if st.button("Make Predictions"):  
        errorDetails = checkErrors(df)
          
        if errorDetails['errorStatus']:
           # with col2:
            st.warning("%s Error(s) found :\n\n %s"%(str(errorDetails['errorNum']),errorDetails['errorMessage']))
            st.write("You may re-upload the csv file according to the sample template.")
           
            if errorDetails['errorAutoCorrect']:
                st.write(" However, we have handled these errors for you.")
                predict(df)
    
        else:
            predict(df)

