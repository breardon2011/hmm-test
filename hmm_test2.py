import pandas as pd  
import numpy as np
from hmmlearn import hmm

# Load stock prices
df = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)  

# Extract close price  
close = df['Close']   

# Train and test split
train = close[:'2019'].values.reshape(-1, 1)
test = close['2020':].values.reshape(-1, 1)

# Create and fit Gaussian HMM
model = hmm.GaussianHMM(n_components=3, covariance_type='diag')
model.fit(train)

# Define state names
states = ['Low', 'Medium', 'High']  

# Predict hidden states 
hidden_states = model.predict(test)
hidden_states = hidden_states.squeeze()
hidden_states = np.clip(hidden_states, 0, len(states) - 1)

# Print most likely states
print(states[np.argmax(hidden_states, axis=0)])

# Evaluate model

#get original index
index = df.index
# Slice index based on test indices
test_index = index[len(train):]

#convert predicted states to series 
predicted = pd.Series(hidden_states, index=test_index)
returns = predicted.diff()[1:].fillna(0)
correct = (returns > 0) & (predicted == 2) | (returns < 0) & (predicted == 0)   
print('Directional accuracy: %.2f' % correct.mean())