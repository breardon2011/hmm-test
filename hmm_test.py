import numpy as np
import pandas as pd
from hmmlearn import hmm

# Load stock prices
data = pd.read_csv('SPY.csv', index_col='Date') 

# Train test split  
train = data[:'2020']
test = data['2021':]

# Create and fit Gaussian HMM 
model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=100)
model.fit(train)

# Predict hidden states
Z = model.predict(test)

# Get most likely state sequence  
hidden_states = model.predict(test)
print(hidden_states)

# Access model attributes
print(model.transmat_) 
print(model.means_)  
print(model.covars_)