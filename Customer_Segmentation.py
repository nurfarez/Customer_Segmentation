# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:24:14 2022

@author: nurul
"""
#%%
#Import section

import os
import pickle
import datetime
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay


from Customer_Segmentation_Module import EDA
from Customer_Segmentation_Module import ModelEvaluation
from Customer_Segmentation_Module import cramax

from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization


#%% Path
#dataset_path
CSV_PATH = os.path.join(os.getcwd(),'dataset','train.csv')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
#model_path
mms_path=os.path.join(os.getcwd(),'model','mms.pkl')
OHE_PATH=os.path.join(os.getcwd(),'model','ohe.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model.h5')



#%%
#step 1) Data Loading
#read data using os method
df = pd.read_csv(CSV_PATH)

#step 2) Data Inspection

df.info()
df.head()

df.isna().sum() # check up the nan/missing value

#customer_age,marital,balance,personal_loan,last_contact_duration,num_contacts_in_campaign,days_since_prev_campaign_contact
#are having Nan issue

df.columns=['id', 'customer_age', 'job_type', 'marital', 'education', 'default',
       'balance', 'housing_loan', 'personal_loan', 'communication_type',
       'day_of_month', 'month', 'last_contact_duration',
       'num_contacts_in_campaign', 'days_since_prev_campaign_contact',
       'num_contacts_prev_campaign', 'prev_campaign_outcome',
       'term_deposit_subscribed']
    

#cat var
cat_col = ['job_type','marital','education','default','housing_loan',
              'personal_loan','communication_type','month',
              'prev_campaign_outcome','term_deposit_subscribed']
#col var
con_col = ['customer_age','balance','day_of_month','last_contact_duration',
               'num_contacts_in_campaign','days_since_prev_campaign_contact',
               'num_contacts_prev_campaign']


#Visualize the categorical and continuous columns

eda=EDA()
eda.visualization(con_col,cat_col,df)

# For this dataset, our target is term_deposit_subscribed: in order to 
# get better understanding and relationship of dataset, we could compare against 
#categorical columns

df.groupby(['term_deposit_subscribed','marital']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','education']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','job_type']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')

#married couple tends to have the highest possibilities not to subscribe on term deposit 
# while, by education: people who got secondary has the highest chance of getting term deposit or not.

# the better way to plot target against  categorical columns by comparison graph 

eda=EDA()
eda.countplot_graph(cat_col, df)


#%%
#step 3) Data Cleaning

#drop unrelated column for this model (id,prev_campaign_outcome) 
df = df.drop(labels=['id','prev_campaign_outcome'],axis=1)


#1) check duplicates
df.duplicated().sum()
#there are no duplicated dataset

#2) remove nan/missing values

#lets make a copy before any of changes towards the dataset
df_demo= df.copy()

#since we need to encode categorical columns into int, however we need to cater with
#Nan issue first in avoid Nan being duplicated during the enconding process

le = LabelEncoder()
# prev_campaign_outcome is col that been dropped during the early data cleaning process.
cat_col_new = ['job_type','marital','education','default','housing_loan',
              'personal_loan','communication_type','month',
             'term_deposit_subscribed']

#Saving the label encoding file to be used later 

for i in cat_col_new:
    if i=='term_deposit_subscribed':
        continue
    else:
        le=LabelEncoder()
        temp=df_demo[i]
        temp[temp.notnull()]=le.fit_transform(temp[df_demo[i].notnull()])
        df_demo[i]=pd.to_numeric(df_demo[i],errors='coerce')
        
PICKLE_SAVE_PATH=os.path.join(os.getcwd(),'model',i+'_encoder.pkl')
        
        
#checking up the Nan/missing values
df_demo.isna().sum()

#there are 7 columns that has the presence of Nan:
#1)customer_age
#2)marital
#3)balance
#4)personal_loan
#5)last_contact_duration
#6)num_contacts_in_campaign
#7)days_since_prev_campaign_contact 

# Handling Nan with KNNImputer

#Con_var/using knn imputer

knn_i=KNNImputer()
df_demo=knn_i.fit_transform(df_demo)
df_demo=pd.DataFrame(df_demo) # to convert array to df
df_demo.columns = df.columns

#checking back Nan value
df_demo.isna().sum()
#there are no more presence of nan

#to ensure that all categorical columns in float types
df.info()
df_demo.info()

#1) remove outliers
df_demo.boxplot()

#there's no need to remove outliers in this dataset as it is common for people nowadays  didnt have housing loan as
#they more likely rented due to limited budget of expenses

#step 4) Features Selection

#Continous vs Categorical using logistic regresion

#Con_col vs Target:term_deposit_subscribed
#logistic Regression

selected_features=[]

for i in con_col:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df_demo[i],axis=-1),df_demo['term_deposit_subscribed']) # X(continous), Y(Categorical)
    print(lr.score(np.expand_dims(df_demo[i],axis=-1),df_demo['term_deposit_subscribed']))
    if lr.score(np.expand_dims(df_demo[i],axis=-1),df_demo['term_deposit_subscribed']) > 0.7:
        selected_features.append(i)
    
        
print(selected_features)
    
#Categorical vs Categorical
#To find the correlation of categorical columns against target:term_deposit_subscribed.
#used crames'v

c=cramax()

for i in cat_col_new:
    print(i)
    matrix=pd.crosstab(df_demo[i],df_demo['term_deposit_subscribed']).to_numpy()
    print(c.cramers_corrected_stat(matrix))
    if  c.cramers_corrected_stat(matrix) > 0.3:
        selected_features.append(i)
        
print(selected_features)

#For the conclusion,  
#('customer_age', 'balance', 'day_of_month', 
#'last_contact_duration', 'num_contacts_in_campaign', 
#'days_since_prev_campaign_contact', 
#'num_contacts_prev_campaign') were been picked up for this model

df_demo= df_demo.loc[:,
 selected_features]
X=df_demo.drop(labels='term_deposit_subscribed',axis=1)
y=df_demo['term_deposit_subscribed'].astype(int)
#%%
#step 5) Data Preprocessing
#mms scaler

mms=MinMaxScaler()
X=mms.fit_transform(X)

#deeplearning using OHE for our target

ohe=OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=-1))

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,

                                                   random_state=123)
#%%
#Model Development

#for classification issue/can used the np.unique

nb_class=len(np.unique(y,axis=0))

model=Sequential()
model.add(Input(shape=np.shape(X_train)[1:]))
model.add(Dense(128,activation='relu',name='1st_hidden_layer'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu',name='2nd_hidden_layer'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(nb_class,activation='softmax'))#output_layer
model.summary()

#callbacks
tensorboard_callback=TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
early_callback=EarlyStopping(monitor='val_acc',patience=5)

#model_compiler
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

hist=model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test),callbacks = [tensorboard_callback,early_callback])

#plotting the model architecture
plot_model(model,show_shapes=True,show_layer_names=(True))
#%%Model Evaluation

print(hist.history.keys())

me=ModelEvaluation()
#Accgraph
me.plot_Acc_hist_graph(hist)
#lossgraph
me.plot_loss_hist_graph(hist)
#confusionmatrix
pred_y=model.predict(X_test)
pred_y=np.argmax(pred_y,axis=1)
true_y=np.argmax(y_test,axis=1)

cr=classification_report(true_y, pred_y)
cm=confusion_matrix(true_y,pred_y)
#displaytheconfusionmatrix
labels=["0","1"]
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
#printthematrix
print(cr)


#%% Model saving

#model_h5_saving
model.save(MODEL_SAVE_PATH)

#pickle_saving
with open(PICKLE_SAVE_PATH,'wb') as file:
    pickle.dump(le,file)

#ohe saving
with open(OHE_PATH,'wb') as file:
    pickle.dump(OHE_PATH,file)

#mms saving
with open(mms_path,'wb') as file:
    pickle.dump(mms,file)

#%% Conclusion

# The model shows that it able to train for accuracy at 90%
# The model accuracy can be improve with adding more layers or nodes