#!/usr/bin/env python
# coding: utf-8

# # AirBnB BenchMark Code
# https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

# In[52]:

import re
import gc
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from scipy.stats import mode
import scipy.stats as stats
from sklearn.tree import export_graphviz
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from lightgbm import plot_importance
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[53]:


train=pd.read_csv("../input/train_users_2.csv")
test=pd.read_csv("../input/test_users.csv")


# In[54]:



test["country_destination"]=np.nan
whole=pd.concat([train,test],axis=0)


# In[55]:


whole["Year_account_created"]=whole['date_account_created'].apply(lambda x:x[:4])
whole["Month_account_created"]=whole['date_account_created'].apply(lambda x:int(x[5:7]))
whole["Day_account_created"]=whole['date_account_created'].apply(lambda x:int(x[8:]))

whole.timestamp_first_active=whole.timestamp_first_active.apply(str)
whole["Year_first_active"]=whole['timestamp_first_active'].apply(lambda x:x[:4])
whole["Month_first_active"]=whole['timestamp_first_active'].apply(lambda x:int(x[4:6]))
whole["Day_first_active"]=whole['timestamp_first_active'].apply(lambda x:int(x[6:8]))
whole["Time_first_active"]=whole['timestamp_first_active'].apply(lambda x:int(x[8:10]))


# In[56]:


category_columns=["gender","signup_method","language","affiliate_channel","affiliate_provider",
                  "first_affiliate_tracked","signup_app","first_device_type","first_browser"]
whole = pd.get_dummies(whole, columns=category_columns)


# In[57]:


whole.age=whole.age.fillna(whole.age.mean())

# In[59]:

# train=whole[whole.timestamp_first_active<="20140630235824"]
# test=whole[whole.timestamp_first_active>"20140630235824"]
train=whole[:213451]
test=whole[213451:]
test_id = test['id']


# In[60]:


X_train=train.drop(["id","date_account_created","timestamp_first_active",
                    "date_first_booking","country_destination"],axis=1)
Y_train=train.country_destination


# In[61]:


X_test=test.drop(["id","date_account_created","timestamp_first_active",
              "date_first_booking","country_destination"],axis=1)


# In[62]:


country_unique=list(set(Y_train))
country_dict={country_unique[i]:i for i in range(len(country_unique))}
country_dict


# In[63]:


Y_train_map=Y_train.map(country_dict)

country_idx = sorted(list(country_dict.values()))
country_idx = np.array(country_idx)


# In[64]:


X_train_ar = np.array(X_train)
Y_train_map_ar = np.array(Y_train_map)
X_test_ar = np.array(X_test)


# In[21]:


country_dict_reverse={v:k for k,v in country_dict.items()}
country_dict_reverse


# In[67]:


clf = RandomForestClassifier()
clf.fit(X_train_ar, Y_train_map_ar)
ypred_proba = clf.predict_proba(X_test_ar)

print("creating random_forest benchmark ...")

submission_data = []
for user_id, proba_each in zip(test_id, ypred_proba):
    top5_idx = country_idx[np.argsort(proba_each)][::-1][:5]
    top5_country = [country_dict_reverse[idx] for idx in top5_idx]
    for each_country in top5_country:
        tmp_ = [user_id, each_country]
        submission_data.append(tmp_)

submission_df = pd.DataFrame(submission_data, columns=['id', 'country'])

submission_df.to_csv("random_forest_benchmark.csv",index=False)


# In[ ]:


import xgboost as xgb

train_data = xgb.DMatrix(data = X_train_ar, label = Y_train_map_ar)

param = {
    'max_depth': 10,
    'learning_rate': 0.1,
    'n_estimators': 5,
    'objective': 'multi:softprob',
    'num_class': 12,
    'gamma': 0,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'base_score': 0.5,
    'missing': None,
    'silent': True,
    'nthread': 4,
    'seed': 42
}


print("creating xgb benchmark...")

num_round = 100
model = xgb.train(param, train_data, num_boost_round=num_round)


test_xgb = xgb.DMatrix(X_test_ar)
test_pred = np.array(model.predict(test_xgb))

submission_data = []
for user_id, proba_each in zip(test_id, test_pred):
    top5_idx = country_idx[np.argsort(proba_each)][::-1][:5]  
    top5_country = [country_dict_reverse[idx] for idx in top5_idx]
    for each_country in top5_country:
        tmp_ = [user_id, each_country]
        submission_data.append(tmp_)

submission_df = pd.DataFrame(submission_data, columns=['id', 'country'])
submission_df.to_csv("xgb_benchmark.csv",index=False)


whole = whole.drop(['Year_account_created', 'Year_first_active'], axis = 1)

train=whole[:213451]
test=whole[213451:]
test_id = test['id']


X_train=train.drop(["id","date_account_created","timestamp_first_active",
                    "date_first_booking","country_destination"],axis=1)

Y_train=train.country_destination

X_test=test.drop(["id","date_account_created","timestamp_first_active",
              "date_first_booking","country_destination"],axis=1)

def submit_kaggle(df_train, df_test, target, reg_alpha, reg_lambda, learning_rate, n_estimators):
    
    le = LabelEncoder()

    y_train = le.fit_transform(target)
 
    model = lgb.LGBMClassifier(boosting_type= 'gbdt',nthread=3, n_jobs=-1, reg_alpha=reg_alpha, reg_lambda=reg_lambda, max_depth=-1, learning_rate=learning_rate, n_estimators=n_estimators)

    print("model fitting starting ...")
    
    model = model.fit(df_train, y_train)
       
    print("model fitting completed ...")
    print()
    
    predic_proba = model.predict_proba(df_test)
    
    df_submit = pd.DataFrame(columns=["id", "country"])
    ids = []
    cts = []
    for i in range(len(test_id)):
        idx = test_id.iloc[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(predic_proba[i])[::-1])[:5].tolist()
        
    df_submit["id"] = ids
    df_submit["country"] = cts
    df_submit.to_csv('light_gbm_benchmark.csv', index = False)
    
    gc.collect()
    
    return model, df_train, df_test, target

model, df_train, df_test, target = submit_kaggle(X_train, X_test, Y_train, 
                                                      reg_alpha=1, 
                                                      reg_lambda=0, 
                                                      learning_rate=0.05, 
                                                      n_estimators=400)
