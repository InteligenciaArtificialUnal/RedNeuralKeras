#IMPORTS FOR PROJECT
# Import pandas
import pandas as pd
# Import matplotlib
import matplotlib.pyplot as plt
# Import numpy
import numpy as np

# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler


# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
np.random.seed(570)


# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

# Specify the data
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array
y= np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Define the scaler
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)


#
# # Model output shape
# print("output_shape")
# model.output_shape
#
# # Model summary
# print("summary")
# model.summary()
#
# # Model config
# print("get_config")
# model.get_config()
#
# # List all weight tensors

#print("get_weights")
#model.get_weights()