import pandas as pd
from pandas import DataFrame
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from GD import MSE
from preprocess import preprocess
from feature_extraction import feature_extraction
from Linear_regression import Linear_regression




# Data preprocessing

# Load data in json format and store in a dataframe
with open("proj1_data.json") as fp:
    data = json.load(fp)
    df = pd.DataFrame(data)

# Convert true-false to 1-0
df["is_root"] = df["is_root"].astype(int)

# Convert all text to lower cases
df["text"]= [x.lower() for x in df["text"]]

# Parse text where there's a space
df["text"]= [x.split() for x in df["text"]]

# First 10000 data points as training set
train = df.iloc[0:10000,:]

# 10000 to 11000 as validation set
validation = df.iloc[10000:11000,:]
validation.index -= 10000

# Last 1000 as test set
test = df.iloc[11000:12000,:]
test.index -= 11000

# Preprocess the data
train, m_c_words = preprocess(train,160)
train = train.values.astype(float)
validation = feature_extraction(validation, m_c_words).values.astype(float)
test = feature_extraction(test, m_c_words).values.astype(float)

# Train Model

# Closed form VS. GD
X_0 = train[:,0:4] # item start from 0 and end at (4-1)
y = train[:,-1]
extra_features = train[:,164:169]

# Closed Form
start_cf = time.time()
Linear_regression(X_0, y, method=0)
end_cf = time.time()
time_cf = end_cf - start_cf
MSE_cf = Linear_regression(X_0, y, method=0)[1]

# GD
start_gd = time.time()
Linear_regression(X_0, y, None, alpha_0 = 1e-06, b = 0, eps = 1e-6)
end_gd = time.time()
time_gd = end_gd - start_gd
MSE_gd = Linear_regression(X_0, y, None, alpha_0 = 1e-06, b = 0, eps = 1e-6)[1]


# Compare different learning rate/beta for GD / plot
step_sizes = [5e-05, 1e-05, 5e-06, 1e-06, 5e-07, 1e-07, 5e-08, 1e-08]
initial_weights = np.random.uniform(-1, 1, (10, 4))
results = np.zeros(shape = (len(step_sizes), initial_weights.shape[0]))
for i in range(len(step_sizes)):
    for j in range(initial_weights.shape[0]):
        results[i][j]=Linear_regression(X_0, y, w_0 = initial_weights[j,:], alpha_0=step_sizes[i], b=0, eps= 1e-10)[1]
    
# Compare 0-60-160 most common words
w_optim_0 = Linear_regression(X_0, y,method = 0)[0]
MSE_0 = Linear_regression(X_0, y,method = 0)[1]
#w_optim_0 = Linear_regression(X_0, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[0]
#MSE_0 = Linear_regression(X_0, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[1]


X_60 = train[:,0:64]
w_optim_60 = Linear_regression(X_60, y,method = 0)[0]
MSE_60 = Linear_regression(X_60, y,method = 0)[1]
#w_optim_60 = Linear_regression(X_60, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[0]
#MSE_60 = Linear_regression(X_60, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[1]


X_160 = train[:,0:164]
w_optim_160 = Linear_regression(X_160, y,method = 0)[0]
MSE_160 = Linear_regression(X_160, y,method = 0)[1]
#w_optim_160 = Linear_regression(X_160, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[0]
#MSE_160 = Linear_regression(X_160, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[1]


# Run training set with (60 most common words + added features)/(160 most common words + added features)
X_60_extra = np.concatenate((X_60, extra_features), axis=1)
w_optim_60_extra = Linear_regression(X_60_extra, y,method = 0)[0]
MSE_60_extra = Linear_regression(X_60_extra, y,method = 0)[1]
#w_optim_60_extra = Linear_regression(X_60_extra, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[0]
#MSE_60_extra = Linear_regression(X_60_extra, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[1]

X_all = train[:,0:169]
w_optim = Linear_regression(X_all, y,method = 0)[0]
MSE_all = Linear_regression(X_all, y,method = 0)[1]
#w_optim = Linear_regression(X_all, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[0]
#MSE_all = Linear_regression(X_all, y, w_0=None,alpha_0=1e-06, b=0, eps= 1e-10)[1]

# Select validation set features
val_0 = validation[:,0:4]
val_60 = validation[:,0:64]
val_160 = validation[:,0:164]
val_all = validation[:,0:169]
val_extra_features = validation[:,164:169]
val_y = validation[:,170]

val_60_extra_val = np.concatenate((val_60, val_extra_features), axis=1)

# Mean squared error on validation sets
MSE_0_val = MSE(val_0, val_y, w_optim_0)
MSE_60_val = MSE(val_60, val_y, w_optim_60)
MSE_160_val = MSE(val_160, val_y, w_optim_160)
MSE_all_val = MSE(val_all, val_y, w_optim)

MSE_60_extra_val = MSE(val_60_extra_val, val_y, w_optim_60_extra)

# Run best model on test set
testset = np.concatenate((test[:,0:64],test[:,164:169]),axis=1)
test_y = test[:,170]
MSE_test = MSE(testset, test_y, w_optim_60_extra)

# Runtime comparison table
runtime_table = DataFrame(
    [[time_cf, time_gd], [MSE_cf, MSE_gd]], columns = ['Closed Form', 'Gradient Descent'], 
                          index=['Runtime','MSE'])

# MSE comparison table
MSEs = [[MSE_0, MSE_0_val],[MSE_60, MSE_60_val],[MSE_160, MSE_160_val], [MSE_60_extra, MSE_60_extra_val], [MSE_all, MSE_all_val]]
MSE_table = DataFrame(data=MSEs, columns = ['Train Performance','Validation Performance'], 
                     index = ['No text feature','60 MCW', '160 MCW','60 MCW with extra features',
                              'All features'])