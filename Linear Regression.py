
# coding: utf-8

# In[ ]:

import pandas as pd
import math as m
import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn import linear_model
from sklearn.preprocessing  import scale
from sklearn import cluster


# In[ ]:

location = r'C:\Users\Vinay\Documents\Python Scripts'
os.chdir(location)


# In[ ]:

# Importing Data

loc = r'LR_Case.xlsx'
lr = pd.read_excel(loc, 0)


# In[ ]:

# Exploring Data

lr.dtypes


# In[ ]:

# List of all Variables

vars = ["region","townsize","gender","age","agecat","ed","edcat","jobcat","union","employ",           
           "empcat","retire","income","lninc","inccat","debtinc","creddebt","lncreddebt",       
           "othdebt","lnothdebt","default","jobsat","marital","spoused","spousedcat","reside",           
           "pets","pets_cats","pets_dogs","pets_birds","pets_reptiles","pets_small","pets_saltfish","pets_freshfish",   
           "homeown","hometype","address","addresscat","cars","carown","cartype","carvalue",         
           "carcatvalue","carbought","carbuy","commute","commutecat","commutetime","commutecar","commutemotorcycle",
           "commutecarpool","commutebus","commuterail","commutepublic","commutebike","commutewalk","commutenonmotor",
           "telecommute","reason","polview","polparty","polcontrib", "vote","card","cardtype","cardbenefit",      
           "cardfee","cardtenure","cardtenurecat","card2","card2type","card2benefit","card2fee","card2tenure",      
           "card2tenurecat","carditems","cardspent","card2items","card2spent","active","bfast","tenure",           
           "churn","longmon","lnlongmon","longten","lnlongten","tollfree","tollmon","lntollmon",        
           "tollten","lntollten","equip","equipmon","lnequipmon","equipten","lnequipten","callcard",        
           "cardmon","lncardmon","cardten","lncardten","wireless","wiremon","lnwiremon","wireten",          
           "lnwireten","multline","voice","pager","internet","callid","callwait","forward",
           "confer","ebill","owntv","hourstv","ownvcr","owndvd","owncd","ownpda",
           "ownpc","ownipod","owngame","ownfax", "news","response_01","response_02","response_03"]


# In[ ]:

# Non-Categorical Variables

vars1 = ["age","ed","income","lninc","debtinc","creddebt","lncreddebt","othdebt","lnothdebt","spoused",
         "reside","pets","pets_cats","pets_dogs","pets_birds","pets_reptiles","pets_small","pets_saltfish","pets_freshfish",
         "carvalue","commutetime","carditems","card2items","tenure","longmon","lnlongmon","longten","lnlongten","tollmon",
         "tollten","equipmon","equipten","cardmon","cardten","lncardten","wiremon","lnwiremon","wireten","lnwireten","hourstv"]


# In[ ]:

# Categorical Variables

vars2 = ["region","townsize","gender","agecat","edcat", "birthmonth", "jobcat","union","employ", "empcat","retire",
           "inccat","default","jobsat","marital","spousedcat","homeown","hometype", "address", "addresscat",
           "cars","carown","cartype","carcatvalue","carbought","carbuy", "commute", "commutecat", "commutecar",
           "commutemotorcycle", "commutecarpool", "commutebus", "commuterail", "commutepublic", "commutebike",
           "commutenonmotor", "telecommute", "reason", "polview", "polparty", "polcontrib", "vote", "card", "cardtype", 
           "cardbenefit", "cardfee", "cardtenure", "cardtenurecat","card2", "card2type","card2benefit","card2fee", "card2tenure",
           "card2tenurecat", "active","bfast","churn","tollfree", "equip","callcard","wireless","multline","voice",
           "pager","internet","callid","callwait", "forward","confer","ebill","owntv","ownvcr","owndvd","owncd","ownpda",
           "ownpc","ownipod", "owngame","ownfax","news","response_01","response_02","response_03"]





# In[ ]:

# Identifying Outliers

def my_stats(x):
    stats = {'Variable': [], 'N': [], 'NMiss': [], 'Mean': [], 'Min': [], 'Max': [], 'STD': [], 'P99': []}
    for i in x:
        stats['Variable'].append(i)
        stats['N'].append(lr[i].notnull().sum())                          # N is number of observations
        stats['NMiss'].append(lr[i].isnull().sum())                       # NMiss is number of missing observations
        stats['Mean'].append(lr[i].mean())
        stats['Min'].append(lr[i].min())
        stats['Max'].append(lr[i].max())
        stats['STD'].append(lr[i].std())
        stats['P99'].append(np.percentile(lr[i][lr[i].notnull()], 99))    # P99 is 99th percentile value  
    return stats

diag_stats = my_stats(vars)
diag_stats = pd.DataFrame(OrderedDict(diag_stats))
diag_stats.to_excel('Diag_Stats.xlsx', index = False)  


# In[ ]:

# Outlier Treatment by capping at 99th percentile

var_cap = ['ed', 'employ', 'income', 'lninc', 'debtinc', 'creddebt', 'lncreddebt', 'othdebt', 'lnothdebt', 'spoused', 'reside', 'pets', 'pets_cats',
           'pets_dogs', 'pets_birds', 'pets_reptiles', 'pets_small', 'pets_saltfish', 'pets_freshfish', 'address', 'cars', 'carvalue', 'commutetime',
           'carditems', 'cardspent', 'card2items', 'card2spent', 'longmon', 'lnlongmon', 'longten', 'lnlongten', 'tollmon', 'lntollmon', 'tollten',
           'lntollten', 'equipmon', 'lnequipmon', 'equipten', 'lnequipten', 'cardmon', 'lncardmon', 'cardten', 'lncardten', 'wiremon', 'lnwiremon',
           'wireten', 'lnwireten', 'hourstv']

cap_value = [21, 39, 272.5, 5.6076369, 29.2, 14.29792, 2.6613666, 24.153048, 3.1881546, 20, 6, 13, 3, 3, 3, 2, 3, 2, 11, 48, 6, 92.05, 41, 
  19, 1216.16, 11, 713.59, 65.25, 4.1782258, 4721.85, 8.459956, 58.875, 4.1934355, 3983.18, 8.4305345, 63.325, 4.2696974, 3679.83,
  8.3690412, 64.25, 4.2449174, 4050, 8.393895, 78.5, 4.5813897, 4534.4, 8.70118, 31]

# Function for capping values at 99th Percentile

def cap(x, y):
    for i, j in zip(x, y):
        lr.loc[lr[i] > j, i] = j
        
cap(var_cap, cap_value)        


# In[ ]:

# Missing Value Treatment for variables with less number of missing values

lr = lr[lr['townsize'].notnull()]
lr = lr[lr['lncreddebt'].notnull()]
lr = lr[lr['lnothdebt'].notnull()]
lr = lr[lr['commutetime'].notnull()]
lr = lr[lr['longten'].notnull()]
lr = lr[lr['lnlongten'].notnull()]
lr = lr[lr['cardten'].notnull()]


# In[ ]:

# Mean Value Imputation

def impute(x):
    for i in x:
        lr.loc[lr[i].isnull(), i] = lr[i].mean()
        
impute(vars)   


# In[ ]:

# Creating Dependent Variable

lr['Total_Amt_Spent'] = lr['cardspent'] + lr['card2spent']


# In[ ]:

# Transforming Dependent Variable to make it Normal

lr['ln_totalamt'] = np.log(lr['Total_Amt_Spent'])      # Taking log of Total_Amt_Spent 

lr['Total_Amt_Spent'].plot.hist()                      # Plotting Histogram to see normality of Total_Amt_Spent
     


# In[ ]:

lr['ln_totalamt'].plot.hist()                          # Plotting Histogram to see normality of ln_totalamt


# In[ ]:

# Converting Categorical variable into dummies

# List of Dummy Variable Prefixes
vars2_dummy = ["dummy_region","dummy_townsize","dummy_gender","dummy_agecat","dummy_edcat", "dummy_birthmonth", "dummy_jobcat","dummy_union","dummy_employ", "dummy_empcat","dummy_retire",
           "dummy_inccat","dummy_default","dummy_jobsat","dummy_marital","dummy_spousedcat","dummy_homeown","dummy_hometype", "dummy_address", "dummy_addresscat",
           "dummy_cars","dummy_carown","dummy_cartype","dummy_carcatvalue","dummy_carbought","dummy_carbuy", "dummy_commute", "dummy_commutecat", "dummy_commutecar",
           "dummy_commutemotorcycle", "dummy_commutecarpool", "dummy_commutebus", "dummy_commuterail", "dummy_commutepublic", "dummy_commutebike",
           "dummy_commutenonmotor", "dummy_telecommute", "dummy_reason", "dummy_polview", "dummy_polparty", "dummy_polcontrib", "dummy_vote", "dummy_card", "dummy_cardtype", 
           "dummy_cardbenefit", "dummy_cardfee", "dummy_cardtenure", "dummy_cardtenurecat","dummy_card2", "dummy_card2type","dummy_card2benefit","dummy_card2fee", "dummy_card2tenure",
           "dummy_card2tenurecat", "dummy_active","dummy_bfast","dummy_churn","dummy_tollfree", "dummy_equip","dummy_callcard","dummy_wireless","dummy_multline","dummy_voice",
           "dummy_pager","dummy_internet","dummy_callid","dummy_callwait", "dummy_forward","dummy_confer","dummy_ebill","dummy_owntv","dummy_ownvcr","dummy_owndvd","dummy_owncd","dummy_ownpda",
           "dummy_ownpc","dummy_ownipod", "dummy_owngame","dummy_ownfax","dummy_news","dummy_response_01","dummy_response_02","dummy_response_03"]

lr_new = pd.get_dummies(lr, prefix = vars2_dummy, columns = vars2)


# In[ ]:

# Correlation of all independent variables with dependent variable

corr = lr_new.drop(['custid', 'Total_Amt_Spent', 'ln_totalamt'], axis = 1).corrwith(lr_new['ln_totalamt'])
corr = pd.DataFrame(corr)
corr.to_excel('Corr.xlsx')


# In[ ]:

# List of variables that are correlated with target variable

vars_model = ["ed", "income", "lninc", "creddebt", "lncreddebt", "othdebt", "lnothdebt", "carvalue", "carditems", "cardspent", 
              "card2items", "card2spent", "lnlongten", "tollmon", "lntollmon", "tollten", "lntollten", "equipmon", "lnequipmon", 
              "equipten", "lnequipten", "wiremon", "wireten", "lnwireten", "dummy_gender_0.0", "dummy_gender_1.0", 
              "dummy_agecat_2.0", "dummy_agecat_4.0", "dummy_agecat_5.0", "dummy_agecat_6.0", "dummy_edcat_1.0", 
              "dummy_edcat_2.0", "dummy_edcat_4.0", "dummy_jobcat_1.0", "dummy_jobcat_2.0", "dummy_employ_0.0", 
              "dummy_employ_1.0", "dummy_employ_17.0", "dummy_empcat_1.0", "dummy_empcat_5.0", "dummy_retire_0.0",
              "dummy_retire_1.0", "dummy_inccat_1.0", "dummy_inccat_2.0", "dummy_inccat_3.0", "dummy_inccat_4.0",
              "dummy_inccat_5.0", "dummy_jobsat_1.0", "dummy_homeown_0.0", "dummy_homeown_1.0", "dummy_hometype_2.0",
              "dummy_address_0.0", "dummy_addresscat_1.0", "dummy_addresscat_4.0", "dummy_carown_0.0", "dummy_carown_1.0",
              "dummy_carcatvalue_1.0", "dummy_carcatvalue_2.0", "dummy_carcatvalue_3.0", "dummy_reason_1.0", "dummy_reason_2.0",
              "dummy_vote_0.0", "dummy_vote_1.0", "dummy_card_1.0", "dummy_card_2.0", "dummy_card_3.0", "dummy_card_4.0", 
              "dummy_cardtenurecat_1.0", "dummy_card2_1.0", "dummy_card2_2.0", "dummy_card2_3.0", "dummy_card2_4.0", 
              "dummy_card2tenure_0.0", "dummy_card2tenurecat_1.0", "dummy_tollfree_0.0", "dummy_tollfree_1.0", 
              "dummy_equip_0.0", "dummy_equip_1.0", "dummy_wireless_0.0", "dummy_wireless_1.0", "dummy_pager_0.0",
              "dummy_pager_1.0", "dummy_internet_0.0", "dummy_internet_4.0", "dummy_callid_0.0", "dummy_callid_1.0",
              "dummy_callwait_0.0", "dummy_callwait_1.0", "dummy_forward_0.0", "dummy_forward_1.0", "dummy_confer_0.0",
              "dummy_confer_1.0", "dummy_owntv_0.0", "dummy_owntv_1.0", "dummy_ownvcr_0.0", "dummy_ownvcr_1.0", 
              "dummy_owndvd_0.0", "dummy_owndvd_1.0", "dummy_owncd_0.0", "dummy_owncd_1.0", "dummy_ownpda_0.0", 
              "dummy_ownpda_1.0", "dummy_ownfax_0.0", "dummy_ownfax_1.0", "dummy_response_03_0.0", "dummy_response_03_1.0"]


# In[ ]:

# Correlation Matrix for Reducing Variables

correlation = lr_new[vars_model].corr()
correlation.to_excel('Correlation.xlsx')


# In[ ]:

# List of Reduced Variables

vars_model1 = ["lnlongten", "lnlongmon", "income", "lncreddebt", "equipten", "card2items", "carditems", "dummy_gender_0.0", "dummy_gender_1.0", "dummy_edcat_1.0", 
               "dummy_edcat_4.0", "dummy_inccat_1.0", "dummy_inccat_3.0", "dummy_reason_1.0", "dummy_reason_2.0", "dummy_card_1.0", 
               "dummy_card2_1.0", "dummy_wireless_0.0", "dummy_owndvd_0.0", "dummy_response_03_0.0"]


# In[ ]:

# Splitting Data into Development and Validation sample (70:30)

samp = np.random.rand(len(lr_new)) < 0.70

development = lr_new[samp]
validation = lr_new[~samp]


# In[ ]:

# Regression Modelling

# Create linear regression object
model = linear_model.LinearRegression(normalize = True)

# Train the model using the Development Dataset
model.fit(development[vars_model1], development['ln_totalamt'])


# In[ ]:

# The coefficients
print('Coefficients: \n', model.coef_)


# In[ ]:

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((model.predict(development[vars_model1]) - development['ln_totalamt']) ** 2))


# In[ ]:

# Explained variance score
print('Variance score: %.2f' % model.score(development[vars_model1], development['ln_totalamt']))


# In[ ]:

# Validation

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((model.predict(validation[vars_model1]) - validation['ln_totalamt']) ** 2))


# In[ ]:

# Explained variance score
print('Variance score: %.2f' % model.score(validation[vars_model1], validation['ln_totalamt']))


# In[ ]:

#############   CLUSTERING BY KMEANS ALGORITHM  ##############


# In[ ]:

# Standardizing the data

cluster_data = lr_new[vars_model1]
clus = scale(cluster_data)
clus = np.asmatrix(clus)


# In[ ]:

# Clustering through KMeans

# For 3 clusters
km_3 = cluster.KMeans(n_clusters = 3)
km_3.fit(clus)
clus_3 = km_3.labels_    # Labels for Cluster 

# For 4 clusters
km_4 = cluster.KMeans(n_clusters = 4)
km_4.fit(clus)
clus_4 = km_4.labels_    # Labels for Cluster 

# For 5 clusters
km_5 = cluster.KMeans(n_clusters = 5)
km_5.fit(clus)
clus_5 = km_5.labels_    # Labels for Cluster 

# For 6 clusters
km_6 = cluster.KMeans(n_clusters = 6)
km_6.fit(clus)
clus_6 = km_6.labels_    # Labels for Cluster 


# In[ ]:

# Binding Columns for cluster labels in Cluster Data

clus_3 = pd.DataFrame(clus_3, columns = ['clus_3'])
clus_4 = pd.DataFrame(clus_4, columns = ['clus_4'])
clus_5 = pd.DataFrame(clus_5, columns = ['clus_5'])
clus_6 = pd.DataFrame(clus_6, columns = ['clus_6'])

Profile_Data = pd.concat([cluster_data, clus_3, clus_4, clus_5, clus_6], axis = 1)

