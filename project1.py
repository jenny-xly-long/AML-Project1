import pandas as pd
import numpy as np
import json
import re
import time
import matplotlib
from collections import Counter
from spellchecker import SpellChecker

from preprocess import preprocess
from feature_extraction import feature_extraction
import Linear_regression


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
train, words = preprocess(train,160)
train = train.values.astype(float)
validation = feature_extraction(validation, words).values.astype(float)
test = feature_extraction(test, words).values.astype(float)

# Train Model

# Runtime for closed form VS. GD
start_cf = time.time()
Linear_regression(X, y)
end_cf = time.time()

start_gd = time.time()
Linear_regression(X,y, None, alpha_0 = 1e-06, b = 0, eps = 1e-10)
end_gd = time.time()

# Compare different learning rate/beta for GD / plot

step_sizes = [5e-05, 1e-05, 5e-06, 1e-06, 5e-07, 1e-07, 5e-08, 1e-08]
initial_weights = np.random(-1, 1, (10, 4))
X = train[:,0:3]
y = train[:,-1]
results = list(list())
for i in range(step_sizes.shape[0]):
    for j in range(initial_weights.shape[0]):
        results[i][j]=Linear_regression(X, y, initial_weights[j,], alpha_0=step_sizes[i], b=0, eps= 1e-10)[1]
    
# Compare 0-60-160 most common words
X_0 = train[:,0:3]
w_optim_0 = Linear_regression(X, y)

X_60 = train[:,0:63]
w_optim_60 = Linear_regression(X, y)

X_160 = train[:,0:163]
w_optim_160 = Linear_regression(X, y)

# Run model with added features on validation set
X = train[:,:]
w_optim = Linear_regression(X, y)
# Run best model on test set