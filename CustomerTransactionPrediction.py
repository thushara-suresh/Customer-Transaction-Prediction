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



def checkErrors():
    pass



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

    
'''  '''
lgbm_multiclass = joblib.load("lgbm_multiclass")
lgbm_final = joblib.load("lgbm_final_binary")

test_template = pd.read_csv("test_template.csv")

st.header("Customer Transaction Prediction")


st.write("")

with st.expander("Input dataset - Sample Template"):
    st.write(test_template)
    if st.button("Download Template (test_template.csv)"):
        test_template.to_csv("test_template.csv", index=False)
    

uploaded_file = st.file_uploader("Upload a csv file containing test data")
if uploaded_file is not None:
     df = pd.read_csv(uploaded_file)
     st.success("File Uploaded Successfully")

         
     with st.expander(uploaded_file.name):
         st.write(df.head())
         
     if st.button("Predict"):    
         my_bar = st.progress(0)
         x_test_ID = df['ID_code']
         x_test = df.drop(labels=['ID_code'], axis=1)
         
         my_bar.progress(5)
         x_test_final = featurization(lgbm_multiclass, x_test)
         
         my_bar.progress(50)
         #final prediction of the customer's likelihood of making a transaction based on the 209 features
         y_featurized_pred = lgbm_final.predict(x_test_final)
         
         my_bar.progress(99)
         y_pred = pd.DataFrame( {'ID_code':x_test_ID, 'target': y_featurized_pred})            
         my_bar.progress(100)
         st.write(y_pred)
         
         my_bar.empty()
         #if st.button("Download Results"):
        
         y_pred_file = y_pred.to_csv("predictions.csv", index=False).encode('utf-8')
         #st.download_button(label='📥 Download Result',
                               # data=y_pred ,
                               # file_name= 'predictions.csv')

         st.download_button(label = 'Download Results',
                            data = y_pred_file,
                            file_name = 'predictions.csv', 
                            mime = 'text/csv')

